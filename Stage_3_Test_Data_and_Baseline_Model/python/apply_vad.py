#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) implementation.

Usage:
    python perform_vad.py in_wav out_txt

Output format (start and end times in seconds):
    start1 end1
    start2 end2
    ...
"""

import argparse
import sys
from math import exp, sqrt, log, atan

import numpy as np
import scipy.signal as sig
import soundfile as sf


# =============================================================================
# Filter Bank
# =============================================================================

def _linear_to_mel(freq: float) -> float:
    """Convert a frequency in linear scale to Mel scale."""
    return 1127 * log(1 + freq / 700)


def _linear_to_bark(freq: float) -> float:
    """Convert a frequency in linear scale to Bark scale."""
    norm_freq_1 = 0.00076 * freq
    norm_freq_2 = freq / 7500
    return 13 * atan(norm_freq_1) + 3.5 * atan(norm_freq_2 ** 2)


def spectral_filter_bank(sampling_rate: float,
                         num_bins: int, num_bands: int,
                         min_frequency: float, max_frequency: float,
                         spacing: str, shape: str) -> np.ndarray:
    """
    Create a matrix containing a spectral filter bank.

    Parameters
    ----------
    sampling_rate : float
        The sampling rate of the audio signal, in Hz.
    num_bins : int
        The number of input spectral bins.
    num_bands : int
        The number of output spectral bands.
    min_frequency : float
        The minimal frequency to consider, in Hz.
    max_frequency : float
        The maximal frequency to consider, in Hz.
    spacing : str
        The spacing between bands: 'linear', 'mel' or 'bark'.
    shape : str
        The shape of the filters: 'rectangular' or 'triangular'.

    Returns
    -------
    filters_mat : np.ndarray
        A matrix of shape (num_bands, num_bins) containing the filter coefficients.
    """
    if sampling_rate <= 0:
        raise ValueError(f'Invalid sampling rate {sampling_rate} - must be positive.')
    if num_bins <= 0:
        raise ValueError(f'Invalid number of spectral bins {num_bins} - must be positive.')
    if num_bands <= 0:
        raise ValueError(f'Invalid number of spectral bands {num_bands} - must be positive.')

    rect_filter = shape.lower().startswith('rect')
    nyquist_freq = 0.5 * sampling_rate

    if min_frequency < 0 or min_frequency >= nyquist_freq:
        raise ValueError(f'Invalid minimal frequency {min_frequency}')
    if max_frequency < 0:
        max_frequency += nyquist_freq
    if max_frequency <= min_frequency or max_frequency > nyquist_freq:
        raise ValueError(f'Invalid maximal frequency {max_frequency}')

    bin_width = nyquist_freq / num_bins
    num_intervals = num_bands if rect_filter else num_bands + 1
    bin_freqs = [0.0] * num_bins

    if spacing.lower() == 'linear':
        base_freq = min_frequency
        delta_freq = (max_frequency - min_frequency) / num_intervals
        for bin_ind in range(num_bins):
            bin_freqs[bin_ind] = bin_ind * bin_width

    elif spacing.lower() == 'mel':
        min_mel_freq = _linear_to_mel(min_frequency)
        max_mel_freq = _linear_to_mel(max_frequency)
        base_freq = min_mel_freq
        delta_freq = (max_mel_freq - min_mel_freq) / num_intervals
        for bin_ind in range(num_bins):
            bin_freqs[bin_ind] = _linear_to_mel(bin_ind * bin_width)

    elif spacing.lower() == 'bark':
        min_bark_freq = _linear_to_bark(min_frequency)
        max_bark_freq = _linear_to_bark(max_frequency)
        base_freq = min_bark_freq
        delta_freq = (max_bark_freq - min_bark_freq) / num_intervals
        for bin_ind in range(num_bins):
            bin_freqs[bin_ind] = _linear_to_bark(bin_ind * bin_width)
    else:
        raise ValueError(f'Invalid spacing option "{spacing}".')

    filters_mat = np.zeros((num_bands, num_bins))

    if rect_filter:
        for band_ind in range(num_bands):
            min_band_freq = base_freq + band_ind * delta_freq
            max_band_freq = min_band_freq + delta_freq
            for bin_ind in range(num_bins):
                bin_freq = bin_freqs[bin_ind]
                if min_band_freq <= bin_freq < max_band_freq:
                    filters_mat[band_ind, bin_ind] = 1
    else:
        for band_ind in range(num_bands):
            min_band_freq = base_freq + band_ind * delta_freq
            central_band_freq = min_band_freq + delta_freq
            max_band_freq = central_band_freq + delta_freq
            for bin_ind in range(num_bins):
                bin_freq = bin_freqs[bin_ind]
                if min_band_freq < bin_freq < max_band_freq:
                    if bin_freq <= central_band_freq:
                        coeff = (bin_freq - min_band_freq) / (central_band_freq - min_band_freq)
                    else:
                        coeff = (max_band_freq - bin_freq) / (max_band_freq - central_band_freq)
                    filters_mat[band_ind, bin_ind] = coeff

    return filters_mat


# =============================================================================
# VAD Segment
# =============================================================================

class VadSegment:
    """Representation of a VAD segment."""

    def __init__(self, channel: int, start: float, end: float,
                 conf: float, power: float, snr: float):
        self.channel_id = channel
        self.start_time = start
        self.end_time = end
        self.confidence = conf
        self.power_spl = power
        self.snr_db = snr

    def __lt__(self, other) -> bool:
        if abs(self.start_time - other.start_time) >= 0.005:
            return self.start_time < other.start_time
        if self.channel_id != other.channel_id:
            return self.channel_id < other.channel_id
        return self.end_time < other.end_time


# =============================================================================
# VAD Parameters 
# =============================================================================

def _interpolate_float(x_min: int, y_min: float, x_max: int, y_max: float, x: int) -> float:
    slope = (y_max - y_min) / (x_max - x_min)
    return y_min + slope * (x - x_min)


def _interpolate_int(x_min: int, y_min: int, x_max: int, y_max: int, x: int) -> int:
    y = _interpolate_float(x_min, float(y_min), x_max, float(y_max), x)
    return int(y + 0.5)


class VadParameters:
    """Parameters for the VAD algorithm."""

    # Constants for sensitivity levels
    _offline_vad_low_max_inactive_prob = 0.25
    _offline_vad_low_min_active_prob = 0.45
    _offline_vad_low_min_active_frames = 12
    _offline_vad_low_min_segment_frames = 40
    _offline_vad_low_boost_environment_size = 1

    _offline_vad_high_max_inactive_prob = 0.15
    _offline_vad_high_min_active_prob = 0.35
    _offline_vad_high_min_active_frames = 8
    _offline_vad_high_min_segment_frames = 20
    _offline_vad_high_boost_environment_size = 5

    _offline_vad_frag_min_inactive_frames = 12
    _offline_vad_frag_margin_frames = 20
    _offline_vad_frag_alpha_power = 0.56

    _offline_vad_cont_min_inactive_frames = 28
    _offline_vad_cont_margin_frames = 30
    _offline_vad_cont_alpha_power = 0.64

    _offline_vad_max_segment_frames = 4000
    _offline_vad_estimator_interval_size = 3000

    def __init__(self, sampling_rate: float, sensitivity_level: int = 3, segmentation_level: int = 3):
        """
        Constructor.

        Parameters
        ----------
        sampling_rate : float
            The sampling rate, in Hz.
        sensitivity_level : int
            VAD sensitivity level (1-5, default 3).
        segmentation_level : int
            VAD fragmentation level (1-5, default 3).
        """
        self.sampling_rate = sampling_rate
        self.window_size = int(sampling_rate * 0.025 + 0.5)
        self.window_slide = int(sampling_rate * 0.01 + 0.5)

        low_level, high_level = 1, 5
        sense_level = max(min(sensitivity_level, high_level), low_level)
        frag_mode, cont_mode = 1, 5
        seg_mode = max(min(segmentation_level, cont_mode), frag_mode)

        self.max_inactive_probability = _interpolate_float(
            low_level, self._offline_vad_low_max_inactive_prob,
            high_level, self._offline_vad_high_max_inactive_prob, sense_level)

        self.min_active_probability = _interpolate_float(
            low_level, self._offline_vad_low_min_active_prob,
            high_level, self._offline_vad_high_min_active_prob, sense_level)

        self.min_active_frames = _interpolate_int(
            low_level, self._offline_vad_low_min_active_frames,
            high_level, self._offline_vad_high_min_active_frames, sense_level)

        self.min_inactive_frames = _interpolate_int(
            frag_mode, self._offline_vad_frag_min_inactive_frames,
            cont_mode, self._offline_vad_cont_min_inactive_frames, seg_mode)

        self.min_segment_frames = _interpolate_int(
            low_level, self._offline_vad_low_min_segment_frames,
            high_level, self._offline_vad_high_min_segment_frames, sense_level)

        self.margin_frames = _interpolate_int(
            frag_mode, self._offline_vad_frag_margin_frames,
            cont_mode, self._offline_vad_cont_margin_frames, seg_mode)

        self.max_segment_frames = self._offline_vad_max_segment_frames

        self.alpha_power = _interpolate_float(
            frag_mode, self._offline_vad_frag_alpha_power,
            cont_mode, self._offline_vad_cont_alpha_power, seg_mode)

        self.boost_environment_size = _interpolate_int(
            low_level, self._offline_vad_low_boost_environment_size,
            high_level, self._offline_vad_high_boost_environment_size, sense_level)

        self.interval_size = self._offline_vad_estimator_interval_size


# =============================================================================
# VAD Core 
# =============================================================================

class Vad:
    """Implementation of a multi-channel voice activity detection algorithm."""

    _min_frequency = 60
    _band_width = 200
    _floor_power = 2.512e-10

    class Estimator:
        """Auxiliary class for estimating background and signal power."""

        def __init__(self):
            self.active_mean_power_ = 0
            self.active_variance_ = 1
            self.inactive_mean_power_ = 0
            self.inactive_variance_ = 1
            self.num_updates_ = 0
            self.total_alphas_ = 0

        def background_power(self) -> float:
            return self.inactive_mean_power_

        def activity_probability(self, power_val: float) -> float:
            if power_val < self.inactive_mean_power_:
                return 0
            if power_val < self.active_mean_power_:
                return 1

            diff_inactive = power_val - self.inactive_mean_power_
            inactive_prob = exp(-(diff_inactive ** 2 / (2 * self.inactive_variance_)))

            diff_active = power_val - self.active_mean_power_
            active_prob = exp(-(diff_active ** 2 / (2 * self.active_variance_)))

            return 0.5 * (active_prob + (1 - inactive_prob))

        def process(self, arr: np.ndarray):
            _min_variance = 0.5
            _min_average_alpha = 0.5

            num_values = len(arr)
            low_val = np.min(arr)
            high_val = np.max(arr)

            for _ in range(3):
                high_flags = np.less(np.abs(arr - high_val), np.abs(arr - low_val))
                num_high_values = np.sum(high_flags)
                num_low_values = num_values - num_high_values

                low_values = (1 - high_flags) * arr
                low_accm = np.sum(low_values)
                low_sqr_accm = np.sum(low_values ** 2)

                high_values = high_flags * arr
                high_accm = np.sum(high_values)
                high_sqr_accm = np.sum(high_values ** 2)

                if num_low_values > 0:
                    low_val = low_accm / num_low_values
                if num_high_values > 0:
                    high_val = high_accm / num_high_values

            prev_power_diff = self.active_mean_power_ - self.inactive_mean_power_
            curr_power_diff = high_val - low_val
            alpha_power = 1

            if self.num_updates_ > 0 and curr_power_diff < prev_power_diff:
                alpha_power = 10 ** (0.1 * (curr_power_diff - prev_power_diff))

            if (self.num_updates_ == 1) or (
                    self.num_updates_ > 1 and (self.total_alphas_ - 1) / (self.num_updates_ - 1) < _min_average_alpha):
                alpha_power = max(alpha_power, _min_average_alpha)

            self.active_mean_power_ = (1 - alpha_power) * self.active_mean_power_ + alpha_power * high_val

            if num_high_values > 0:
                self.active_variance_ = max(high_sqr_accm / num_high_values - high_val ** 2, _min_variance)
            else:
                self.active_variance_ = (1 - alpha_power) * self.active_variance_ + alpha_power * _min_variance

            self.inactive_mean_power_ = (1 - alpha_power) * self.inactive_mean_power_ + alpha_power * low_val

            if num_low_values > 0:
                self.inactive_variance_ = max(low_sqr_accm / num_low_values - low_val ** 2, _min_variance)
            else:
                self.inactive_variance_ = (1 - alpha_power) * self.inactive_variance_ + alpha_power * _min_variance

            self.num_updates_ += 1
            self.total_alphas_ += alpha_power

    class _CandidateSegment:
        """Auxiliary class: Representation of an initial candidate segment."""

        def __init__(self, start_frame_ind: int, end_frame_ind: int,
                     confidence: float, power: float, snr: float):
            self.orig_start_ind_ = start_frame_ind
            self.orig_end_ind_ = end_frame_ind
            self.num_orig_frames_ = end_frame_ind - start_frame_ind
            self.start_ind_ = start_frame_ind
            self.end_ind_ = end_frame_ind
            self.confidence_ = confidence
            self.power_ = power
            self.snr_ = snr

        def clone(self):
            return Vad._CandidateSegment(
                self.orig_start_ind_, self.orig_end_ind_,
                self.confidence_, self.power_, self.snr_)

        def num_frames(self):
            return self.end_ind_ - self.start_ind_

        def add_margins(self, margin_size: int, max_frame_ind: int):
            self.start_ind_ = max(0, self.start_ind_ - margin_size)
            self.end_ind_ = min(max_frame_ind, self.end_ind_ + margin_size)

        def concat(self, other):
            num_frames_1 = self.num_orig_frames_
            num_frames_2 = other.num_orig_frames_

            self.orig_end_ind_ = other.orig_end_ind_
            self.end_ind_ = other.end_ind_

            beta = num_frames_1 / (num_frames_1 + num_frames_2)
            self.confidence_ = beta * self.confidence_ + (1 - beta) * other.confidence_
            self.power_ = beta * self.power_ + (1 - beta) * other.power_
            self.snr_ = beta * self.snr_ + (1 - beta) * other.snr_

            self.num_orig_frames_ = num_frames_1 + num_frames_2

        def full_vad_segment(self, channel_id: int, frame_duration: float) -> VadSegment:
            start_time = frame_duration * self.start_ind_
            end_time = frame_duration * self.end_ind_
            return VadSegment(channel_id, start_time, end_time,
                              sqrt(self.confidence_), self.power_, self.snr_)

        def partial_vad_segment(self, channel_id: int, frame_duration: float,
                                start_frame_ind: int, end_frame_ind: int) -> VadSegment:
            start_time = frame_duration * start_frame_ind
            end_time = frame_duration * end_frame_ind
            return VadSegment(channel_id, start_time, end_time,
                              sqrt(self.confidence_), self.power_, self.snr_)

    def __init__(self, vad_params: VadParameters):
        """Constructor."""
        self.params_ = vad_params

        self.window_ = sig.windows.hamming(vad_params.window_size)

        fft_size = 1
        while fft_size < vad_params.window_size:
            fft_size *= 2

        num_bins = fft_size + 1
        self.fft_size_ = 2 * fft_size

        self.num_bands_ = int(
            ((0.5 * vad_params.sampling_rate - (Vad._min_frequency + Vad._band_width)) / Vad._band_width) + 0.5)

        max_frequency = 0.5 * vad_params.sampling_rate - Vad._band_width

        self.filters_mat_ = np.ones((self.num_bands_ + 1, num_bins))
        self.filters_mat_[1:, :] = spectral_filter_bank(
            vad_params.sampling_rate, num_bins, self.num_bands_,
            Vad._min_frequency, max_frequency, 'linear', 'triangular')

    def process(self, audio_buffer: np.ndarray) -> list:
        """
        Process the input audio buffer and compute the activity segments.

        Parameters
        ----------
        audio_buffer : np.ndarray
            Shape (num_channels, num_samples) containing normalized audio samples.

        Returns
        -------
        list[VadSegment]
            A sorted list of VAD segments.
        """
        num_channels = audio_buffer.shape[0]

        band_estimators = [Vad.Estimator() for _ in range(self.num_bands_ + 1)]

        num_overlapping_samples = self.params_.window_size - self.params_.window_slide

        chan_signal_powers = None
        chan_active_probs = None
        chan_background_powers = None

        for chan_ind in range(num_channels):
            _, _, spec = sig.spectrogram(
                audio_buffer[chan_ind],
                fs=self.params_.sampling_rate,
                window=self.window_,
                noverlap=num_overlapping_samples,
                nfft=self.fft_size_,
                mode='psd')
            num_frames = spec.shape[1]

            band_powers = np.matmul(self.filters_mat_, spec)
            band_powers_spl = 96 + 10 * np.log10(np.maximum(band_powers, Vad._floor_power))

            if chan_ind == 0:
                chan_signal_powers = np.zeros((num_channels, num_frames))
                chan_active_probs = np.zeros((num_channels, num_frames))
                chan_background_powers = np.zeros((num_channels, num_frames))

            alpha = self.params_.alpha_power

            for frame_ind in range(num_frames):
                if frame_ind % self.params_.interval_size == 0 and frame_ind + 10 < num_frames:
                    end_frame_ind = min(frame_ind + self.params_.interval_size, num_frames)
                    for band_ind in range(self.num_bands_ + 1):
                        band_estimators[band_ind].process(band_powers_spl[band_ind, frame_ind:end_frame_ind])

                if frame_ind > 0:
                    prev_powers = band_powers_spl[:, frame_ind - 1]
                    curr_powers = alpha * prev_powers + (1 - alpha) * band_powers_spl[:, frame_ind]
                else:
                    curr_powers = band_powers_spl[:, frame_ind]

                total_band_probs = 0
                for band_ind, estimator in enumerate(band_estimators):
                    if band_ind == 0:
                        total_power_prob = estimator.activity_probability(curr_powers[0])
                    else:
                        total_band_probs += estimator.activity_probability(curr_powers[band_ind])
                bands_prob = total_band_probs / self.num_bands_

                chan_signal_powers[chan_ind, frame_ind] = curr_powers[0]
                chan_active_probs[chan_ind, frame_ind] = (total_power_prob * bands_prob ** 2) ** 0.3333333
                chan_background_powers[chan_ind, frame_ind] = band_estimators[0].background_power()

        if num_channels > 1:
            max_powers = np.max(chan_signal_powers, axis=0)
            for chan_ind in range(num_channels):
                relative_probs = 10 ** (0.1 * (chan_signal_powers[chan_ind] - max_powers))
                chan_active_probs[chan_ind] *= relative_probs

        vad_segments = []

        for chan_ind in range(num_channels):
            chan_cand_segments = self._create_initial_channel_segments(
                chan_active_probs[chan_ind],
                chan_signal_powers[chan_ind],
                chan_background_powers[chan_ind])

            self._create_final_channel_segments(
                chan_ind + 1, chan_active_probs.shape[1], chan_cand_segments, vad_segments)

        vad_segments.sort()
        return vad_segments

    def _create_initial_channel_segments(self, channel_activity_probs, channel_signal_powers,
                                         channel_background_powers):
        env_size = self.params_.boost_environment_size

        if env_size > 0:
            mean_env = np.ones((2 * env_size + 1)) / (2 * env_size + 1)
            active_probs = np.convolve(
                np.pad(channel_activity_probs, (env_size, env_size), mode='edge'),
                mean_env, mode='valid')
        else:
            active_probs = channel_activity_probs

        start_segment_ind = -1
        in_active_segment = False
        first_pending_frame_ind = -1
        num_pending_frames = 0
        candidate_segments = []

        for frame_ind, curr_prob in enumerate(active_probs):
            if curr_prob >= self.params_.min_active_probability:
                if in_active_segment:
                    first_pending_frame_ind = -1
                    num_pending_frames = 0
                else:
                    if first_pending_frame_ind < 0:
                        first_pending_frame_ind = frame_ind
                    num_pending_frames += 1

                    if num_pending_frames == self.params_.min_active_frames:
                        start_segment_ind = first_pending_frame_ind
                        in_active_segment = True
                        first_pending_frame_ind = -1
                        num_pending_frames = 0

            elif curr_prob <= self.params_.max_inactive_probability:
                if not in_active_segment:
                    first_pending_frame_ind = -1
                    num_pending_frames = 0
                else:
                    if first_pending_frame_ind < 0:
                        first_pending_frame_ind = frame_ind
                    num_pending_frames += 1

                    if num_pending_frames == self.params_.min_inactive_frames:
                        end_segment_ind = first_pending_frame_ind
                        in_active_segment = False
                        first_pending_frame_ind = -1
                        num_pending_frames = 0

                        segment_confidence = np.average(channel_activity_probs[start_segment_ind:end_segment_ind])
                        segment_power = np.average(channel_signal_powers[start_segment_ind:end_segment_ind])
                        segment_snr = segment_power - np.average(
                            channel_background_powers[start_segment_ind:end_segment_ind])

                        candidate_segments.append(Vad._CandidateSegment(
                            start_segment_ind, end_segment_ind,
                            segment_confidence, segment_power, segment_snr))

        if in_active_segment:
            end_segment_ind = len(active_probs)
            segment_confidence = np.average(channel_activity_probs[start_segment_ind:end_segment_ind])
            segment_power = np.average(channel_signal_powers[start_segment_ind:end_segment_ind])
            segment_snr = segment_power - np.average(channel_background_powers[start_segment_ind:end_segment_ind])

            candidate_segments.append(Vad._CandidateSegment(
                start_segment_ind, end_segment_ind,
                segment_confidence, segment_power, segment_snr))

        return candidate_segments

    def _create_final_channel_segments(self, channel_id: int, num_frames: int,
                                        candidate_segments, final_vad_segments):
        frame_duration = self.params_.window_slide / self.params_.sampling_rate
        min_segment_frames_with_margins = self.params_.min_segment_frames + 2 * self.params_.margin_frames
        num_initial_segments = len(candidate_segments)
        curr_segment_ind = 0

        clipped_start_frame_ind = -1
        clipped_end_frame_ind = -1

        while curr_segment_ind < num_initial_segments:
            curr_segment = candidate_segments[curr_segment_ind]
            curr_segment.add_margins(self.params_.margin_frames, num_frames)

            if clipped_start_frame_ind >= 0:
                curr_segment.start_ind_ = clipped_start_frame_ind
                clipped_start_frame_ind = -1

            next_segment_ind = curr_segment_ind + 1
            while next_segment_ind < num_initial_segments:
                next_segment = candidate_segments[next_segment_ind].clone()
                next_segment.add_margins(self.params_.margin_frames, num_frames)

                if next_segment.start_ind_ > curr_segment.end_ind_:
                    break

                if next_segment.end_ind_ - curr_segment.start_ind_ > self.params_.max_segment_frames:
                    clipped_end_frame_ind = (curr_segment.orig_end_ind_ + next_segment.orig_start_ind_) // 2
                    clipped_start_frame_ind = clipped_end_frame_ind
                    break

                curr_segment.concat(next_segment)
                next_segment_ind += 1

            if clipped_end_frame_ind >= 0:
                curr_segment.end_ind_ = clipped_end_frame_ind
                clipped_end_frame_ind = -1

            if curr_segment.num_frames() >= 2 * self.params_.max_segment_frames:
                num_subsegments = int(1 + curr_segment.num_frames() / (1.5 * self.params_.max_segment_frames))
                num_subsegment_frames = int(curr_segment.num_frames() / num_subsegments)

                subsegment_start_frame_ind = curr_segment.start_ind_

                for k in range(num_subsegments):
                    subsegment_end_frame_ind = subsegment_start_frame_ind + num_subsegment_frames
                    if k + 1 == num_subsegments:
                        subsegment_end_frame_ind = curr_segment.end_ind_

                    final_vad_segments.append(curr_segment.partial_vad_segment(
                        channel_id, frame_duration,
                        subsegment_start_frame_ind, subsegment_end_frame_ind))

                    subsegment_start_frame_ind = subsegment_end_frame_ind
            else:
                if curr_segment.num_frames() >= min_segment_frames_with_margins:
                    final_vad_segments.append(curr_segment.full_vad_segment(channel_id, frame_duration))

            curr_segment_ind = next_segment_ind


# =============================================================================
# Main CLI
# =============================================================================

def perform_vad(in_wav: str, sensitivity: int = 3, segmentation: int = 3) -> list:
    """
    Perform VAD on an audio file

    Parameters
    ----------
    in_wav : str
        Path to input WAV file.
    sensitivity : int
        VAD sensitivity level (1-5).
    segmentation : int
        VAD segmentation level (1-5).

    Returns
    -------
    list of tuples
        List of (start_time, end_time) in seconds.
    """
    # Load audio
    audio, sr = sf.read(in_wav, dtype='float32')

    # Ensure 2D shape (channels, samples)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    else:
        # soundfile returns (samples, channels), transpose to (channels, samples)
        audio = audio.T

    # Create VAD parameters and processor
    vad_params = VadParameters(sr, sensitivity_level=sensitivity, segmentation_level=segmentation)
    vad = Vad(vad_params)

    # Process and get segments
    segments = vad.process(audio)

    # Extract start/end times
    results = [(seg.start_time, seg.end_time) for seg in segments]

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Voice Activity Detection')
    parser.add_argument('in_wav', type=str, help='Input WAV file path')
    parser.add_argument('out_txt', type=str, help='Output TXT file path')
    parser.add_argument('--sensitivity', type=int, default=3,
                        help='VAD sensitivity level 1-5 (default: 3)')
    parser.add_argument('--segmentation', type=int, default=3,
                        help='VAD segmentation level 1-5 (default: 3)')

    args = parser.parse_args()

    # Perform VAD
    segments = perform_vad(args.in_wav, args.sensitivity, args.segmentation)

    # Write output
    with open(args.out_txt, 'w') as f:
        for start, end in segments:
            f.write(f'{start:.3f} {end:.3f}\n')

    print(f'VAD complete. Found {len(segments)} segments.')
    print(f'Output written to: {args.out_txt}')


if __name__ == '__main__':
    main()
