import numpy as np
import scipy.signal as sig
from math import exp, sqrt

from k2_scripts.vad_params import VadParameters
from k2_scripts.vad_segment import VadSegment
from k2_scripts.filter_bank import spectral_filter_bank

class Vad:
    """
    Implemantation of a multi-channel voice activity detection algorithm.
    """
    
    __min_frequency = 60
    __band_width = 200

    __floor_power = 2.512e-10
    
    # Auxiliary class for estimating the background power and the signal power in a sequence
    # of given power values.
    class Estimator:
        
        # Constructor.
        def __init__(self):
            self.active_mean_power_ = 0
            self.active_variance_ = 1

            self.inactive_mean_power_ = 0
            self.inactive_variance_ = 1

            self.num_updates_ = 0
            self.total_alphas_ = 0


        # Get the updated estimation for the background power.
        def background_power(self) -> float:
            return self.inactive_mean_power_


        # Get the activity probability given the power value.
        def activity_probability(self, power_val: float) -> float:
            # In case the power is below the mean inactive power, the activity probability is 0.
            if power_val < self.inactive_mean_power_:
                return 0
            
            # In case the power is above the mean active power, the activity probability is 1.
            if power_val < self.active_mean_power_:
                return 1
            
            # In case the power value is in between the pair of mean power values, estimate the activity probability and the
            # inactivity probability, both are modeled using pseudo-Gaussian distributions.
            diff_inactive = power_val - self.inactive_mean_power_
            inactive_prob = exp(-(diff_inactive**2 / (2 * self.inactive_variance_)))
        
            diff_active = power_val - self.active_mean_power_
            active_prob = exp(-(diff_active**2 / (2 * self.active_variance_)))
        
            # Return the mean of the activity probability and the complement of the inactivity probability.
            return (0.5 * (active_prob + (1 - inactive_prob)))


        # Process the given array of power values, and update the power estimations.
        def process(self, arr: np.ndarray):
            __min_variance = 0.5
            __min_average_alpha = 0.5

            # Start by performing 3 rounds of the k-means algorithm with k = 2, in order to subdivide
            # the given array into a subset of low values and a subset of high values.
            num_values = len(arr)
            low_val = np.min(arr)
            high_val = np.max(arr)
            
            for i in range(3):
                high_flags = np.less(np.abs(arr - high_val), np.abs(arr - low_val))
                num_high_values = np.sum(high_flags)
                num_low_values = num_values - num_high_values
                
                low_values = (1 - high_flags) * arr
                low_accm = np.sum(low_values)
                low_sqr_accm = np.sum(low_values**2)
                
                high_values = high_flags * arr
                high_accm = np.sum(high_values)
                high_sqr_accm = np.sum(high_values**2)
                
                if num_low_values > 0:
                    low_val = low_accm / num_low_values
                    
                if num_high_values > 0:
                    high_val = high_accm / num_high_values
                    

            # In order to avoid updating the statistics on long silence periods (or on long periods of loud noise), make sure
            # that the differences between the high and low centroid is of the same magnitude as the current power difference.
            # Otherwise, use a very low update rate alpha.
            prev_power_diff = self.active_mean_power_ - self.inactive_mean_power_
            curr_power_diff = high_val - low_val
            alpha_power = 1

            if (self.num_updates_ > 0) and (curr_power_diff < prev_power_diff):
                alpha_power = 10**(0.1 * (curr_power_diff - prev_power_diff))

            # However, if the thresholds have only been updated once, or in case the average update rate falls below a
            # threshold, make sure that the update rate is at least equal to the threshold. This is done in order to avoid
            # getting stuck of a pair of values that are no longer valid (e.g. when we start with some sequence of zero frames
            # followed by a sequence of some very loud frames).
            if (self.num_updates_ == 1) or (self.num_updates_ > 1 and (self.total_alphas_ - 1) / (self.num_updates_ - 1) < __min_average_alpha):
                alpha_power = max(alpha_power, __min_average_alpha)

            # Compute the average power of the active signal and its variance.
            self.active_mean_power_ = (1 - alpha_power) * self.active_mean_power_ + alpha_power * high_val

            if num_high_values > 0:
                # Update the power variance as well.
                self.active_variance_ = max(high_sqr_accm / num_high_values - high_val**2, __min_variance)
                
            else:
                # Interpolate the current variance with the floor variance value.
                self.active_variance_ = (1 - alpha_power) * self.active_variance_ + alpha_power * __min_variance

            # Compute the average power of the inactive signal (i.e. the background) and its variance.
            self.inactive_mean_power_ = (1 - alpha_power) * self.inactive_mean_power_ + alpha_power * low_val

            if num_low_values > 0:
                self.inactive_variance_ = max(low_sqr_accm / num_low_values - low_val**2, __min_variance)
            
            else:
                # Interpolate the current variance with the floor variance value.
                self.inactive_variance_ = (1 - alpha_power) * self.inactive_variance_ + alpha_power * __min_variance

            # Update the statistics of the number of threshold updates.
            self.num_updates_ += 1
            self.total_alphas_ += alpha_power


    # Auxiliary class: Represnetation of an initial candidate segment.   
    class _CandidateSegment:

        # Constructor.
        def __init__(self, start_frame_ind: int, end_frame_ind: int,
                     confidence: float, power: float, snr: float) :
            self.orig_start_ind_ = start_frame_ind
            self.orig_end_ind_ = end_frame_ind
            self.num_orig_frames_ = end_frame_ind - start_frame_ind
            self.start_ind_ = start_frame_ind
            self.end_ind_ = end_frame_ind
            self.confidence_ = confidence
            self.power_ = power
            self.snr_ = snr


        # Get a clone of the candidate segment.
        def clone(self):
            return Vad._CandidateSegment(self.orig_start_ind_, self.orig_end_ind_,
                                         self.confidence_, self.power_, self.snr_)


        # Get the number of frames in the segment.
        def num_frames(self):
            return (self.end_ind_ - self.start_ind_)

        
        # Add some margin frames to the segment.
        def add_margins(self, margin_size: int, max_frame_ind: int):
            # Subtract the margin frames from the beginning of the segment.
            if (self.start_ind_ > margin_size):
                self.start_ind_ -= margin_size
            else:
                self.start_ind_ = 0

            # Add the margin frames to the end of the segment.
            self.end_ind_ += margin_size

            if (self.end_ind_ > max_frame_ind):
                self.end_ind_ = max_frame_ind
                
                
        # Concatenate the other segment to the current one.
        def concat(self, other):
            # We assume that the other segment's start index is larger than the current segment's end index,
            # so we just have to update the end indices.
            num_frames_1 = self.num_orig_frames_
            num_frames_2 = other.num_orig_frames_

            self.orig_end_ind_ = other.orig_end_ind_
            self.end_ind_ = other.end_ind_

            # Compute the weighted averages of the segment quality measures.
            beta = (num_frames_1) / (num_frames_1 + num_frames_2)
            self.confidence_ = beta * self.confidence_ + (1 - beta) * other.confidence_
            self.power_ = beta * self.power_ + (1 - beta) * other.power_
            self.snr_ = beta * self.snr_ + (1 - beta) * other.snr_

            # Update the number of original frames (not counting any margins).
            self.num_orig_frames_ = num_frames_1 + num_frames_2;


        # Convert the full segment into a VAD segment.
        def full_vad_segment(self, channel_id: int, frame_duration: float) -> VadSegment:
            start_time = frame_duration * self.start_ind_
            end_time = frame_duration * self.end_ind_
            return VadSegment(channel_id, start_time, end_time,
                              sqrt(self.confidence_), self.power_, self.snr_)

        
        # Convert a partial segment into a VAD segment.
        def partial_vad_segment(self, channel_id: int, frame_duration: float,
                                start_frame_ind: int, end_frame_ind: int) -> VadSegment:
            start_time = frame_duration * start_frame_ind
            end_time = frame_duration * end_frame_ind
            return VadSegment(channel_id, start_time, end_time,
                              sqrt(self.confidence_), self.power_, self.snr_)

            
    def __init__(self, vad_params: VadParameters):
        """
        Constructor.

        Parameters
        ----------
        
        vad_params : VadParameters
            The parameters for the voice activity detection algorithm.

        """
        self.params_ = vad_params
        
        # Create the window function for the spectrogram computation.
        self.window_ = sig.windows.hamming(vad_params.window_size)
        
        # Determine the FFT size in use.
        fft_size = 1
        while (fft_size < vad_params.window_size):
            fft_size *= 2
            
        num_bins = fft_size + 1
        self.fft_size_ = 2 * fft_size
        
        self.num_bands_ = int(((0.5 * vad_params.sampling_rate - (Vad.__min_frequency + Vad.__band_width)) / Vad.__band_width) + 0.5)

        max_frequency = 0.5 * vad_params.sampling_rate - Vad.__band_width
        
        self.filters_mat_ = np.ones((self.num_bands_ + 1, num_bins))
        self.filters_mat_[1:, :] = spectral_filter_bank(vad_params.sampling_rate,
                                                        num_bins, self.num_bands_,
                                                        Vad.__min_frequency, max_frequency,
                                                        'linear', 'triangular')
        
    def process(self, audio_buffer: np.ndarray) -> [VadSegment]:
        """
        Process the input audio buffer, and compute the activity segments it contains.

        Parameters
        ----------
        audio_buffer : np.ndarray
            An array whose shape is (num_channels, num_samples) containing the normalized audio samples.

        Returns
        -------
        
        list[VadSegment]
            A sorted list of VAD segments.

        """
        
        # Allocate the per-channel power estimators.        
        num_channels = audio_buffer.shape[0]
        
        # Allocate the power estimators.
        band_estimators = []
        for k in range(self.num_bands_ + 1):
            band_estimators.append(Vad.Estimator())

        num_overlapping_samples = self.params_.window_size - self.params_.window_slide
        
        for chan_ind in range(num_channels):            
            # Compute the power spectrum of the current channel.
            _, _, spec = sig.spectrogram(audio_buffer[chan_ind],
                                         fs=self.params_.sampling_rate,
                                         window=self.window_,
                                         noverlap=num_overlapping_samples,
                                         nfft=self.fft_size_, 
                                         mode='psd')
            num_frames = spec.shape[1]
                        
            band_powers = np.matmul(self.filters_mat_, spec)
            band_powers_spl = 96 + 10 * np.log10(np.maximum(band_powers, Vad.__floor_power))

            # Allocate the channel data, if necessary.
            if (chan_ind == 0):
                chan_signal_powers = np.zeros((num_channels, num_frames))
                chan_active_probs = np.zeros((num_channels, num_frames))
                chan_background_powers = np.zeros((num_channels, num_frames))
                
            # Allocate and fill the channel data.
            alpha = self.params_.alpha_power

            for frame_ind in range(num_frames):
                if frame_ind % self.params_.interval_size == 0 and frame_ind + 10 < num_frames:
                    end_frame_ind = min(frame_ind + self.params_.interval_size, num_frames)
                    for band_ind in range(self.num_bands_ + 1):
                        band_estimators[band_ind].process(band_powers_spl[band_ind, frame_ind: end_frame_ind])

                # Smooth the power values of the current frame.
                if frame_ind > 0:
                    prev_powers = band_powers_spl[:, frame_ind - 1]
                    curr_powers = band_powers_spl[:, frame_ind]
                    
                    curr_powers = alpha * prev_powers + (1 - alpha) * curr_powers
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
                chan_active_probs[chan_ind, frame_ind] = (total_power_prob * bands_prob**2) ** 0.3333333
                chan_background_powers[chan_ind, frame_ind] = band_estimators[0].background_power()

        # If necessary, consider the relative signal powers across channels, and update the acitivity
        # probabilities accordingly.
        if num_channels > 1:
            max_powers = np.max(chan_signal_powers, axis=0)
            
            for chan_ind in range(num_channels):
                relative_probs = 10**(0.1 * (chan_signal_powers[chan_ind] - max_powers))
                chan_active_probs[chan_ind] *= relative_probs
                
        # Go over all channels.
        vad_segments = []            

        for chan_ind in range(num_channels):
            # Compute the initial candidate segments for the current channel.
            chan_cand_segments = self._create_initial_channel_segments(chan_active_probs[chan_ind],
                                                                       chan_signal_powers[chan_ind],
                                                                       chan_background_powers[chan_ind])

            # Convert to final VAD segments.
            self._create_final_channel_segments(chan_ind + 1, num_frames, chan_cand_segments,
                                                vad_segments)
            
        # Sort the list of VAD segments before returning it.
        vad_segments.sort()
        return vad_segments            
    
    
    # Auxiliary function: Compute the initial candidate segments for a specific audio channels,
    # given the activity probabilities, signal powers and background power values for this channel.
    def _create_initial_channel_segments(self, channel_activity_probs, channel_signal_powers, channel_background_powers):
        env_size = self.params_.boost_environment_size
        
        if env_size > 0:
            mean_env = np.ones((2 * env_size + 1)) / (2 * env_size + 1)
            active_probs = np.convolve(np.pad(channel_activity_probs, (env_size, env_size), mode='edge'),
                                         mean_env, mode='valid')
        else:
            active_probs = channel_activity_probs
                
        start_segment_ind = -1
        in_active_segment = False
        first_pending_frame_ind = -1
        num_pending_frames = 0
        candidate_segments = []

        for frame_ind, curr_prob in enumerate(active_probs):
            # Check whether the current frame is active or non-active.
            if (curr_prob >= self.params_.min_active_probability):
                # The current frame is active.
                if in_active_segment:
                    # Reset the number of pending non-active frames.
                    first_pending_frame_ind = -1
                    num_pending_frames = 0;

                else:
                    # Increment the number of pending active frames.
                    if first_pending_frame_ind < 0:
                        first_pending_frame_ind = frame_ind
                    num_pending_frames += 1

                    if (num_pending_frames == self.params_.min_active_frames):
                        # Mark the beginning of an activity segment.
                        start_segment_ind = first_pending_frame_ind
                        in_active_segment = True

                        # Reset the number of pending frames.
                        first_pending_frame_ind = -1
                        num_pending_frames = 0
                        
            elif (curr_prob <= self.params_.max_inactive_probability):
                # The current frame is non-active.
                if not in_active_segment:
                    # Reset the number of pending active frames.
                    first_pending_frame_ind = -1
                    num_pending_frames = 0
    
                else:
                    # Increment the number of pending non-active frames.
                    if first_pending_frame_ind < 0:
                        first_pending_frame_ind = frame_ind
                    num_pending_frames += 1
    
                    if (num_pending_frames == self.params_.min_inactive_frames):
                        # Mark the end of the current activity segment.
                        end_segment_ind = first_pending_frame_ind
                        in_active_segment = False;
    
                        # Reset the number of pending frames.
                        first_pending_frame_ind = -1
                        num_pending_frames = 0
    
                        # Compute the average activity probability, signal power and SNR over the frame range that form
                        # the current activity segment [start_segment_ind, end_segment_ind).    
                        segment_confidence = np.average(channel_activity_probs[start_segment_ind: end_segment_ind])
                        segment_power = np.average(channel_signal_powers[start_segment_ind: end_segment_ind])
                        segment_snr = segment_power - np.average(channel_background_powers[start_segment_ind: end_segment_ind])
    
                        # Create the initial segment.
                        candidate_segments.append(Vad._CandidateSegment(start_segment_ind, end_segment_ind,
                                                                        segment_confidence, segment_power, segment_snr))

        # In case we end up in an activity segment, add this last segment.
        if in_active_segment:
            # Compute the average activity probability, signal power and SNR over the frame range that form the last
            # activity segment [start_segment_ind, num_frames).
            end_segment_ind = len(active_probs)

            segment_confidence = np.average(channel_activity_probs[start_segment_ind: end_segment_ind])
            segment_power = np.average(channel_signal_powers[start_segment_ind: end_segment_ind])
            segment_snr = segment_power - np.average(channel_background_powers[start_segment_ind: end_segment_ind])

            # Create the initial segment.
            candidate_segments.append(Vad._CandidateSegment(start_segment_ind, end_segment_ind,
                                                            segment_confidence, segment_power, segment_snr))
            
        return candidate_segments


    # Auxiliary function: Convert the list of candidate segments of the given channel into a list of VAD segments.
    def _create_final_channel_segments(self, channel_id: int, num_frames: int, candidate_segments,
                                       final_vad_segments):
        # Go over all initial candidate segments.
        frame_duration = self.params_.window_slide / self.params_.sampling_rate
        min_segment_frames_with_margins = self.params_.min_segment_frames + 2 * self.params_.margin_frames
        num_initial_segments = len(candidate_segments)
        curr_segment_ind = 0;

        clipped_start_frame_ind = -1
        clipped_end_frame_ind = -1

        
        while (curr_segment_ind < num_initial_segments):
            # Get the current candidate segment and add the appropriate margins to it.
            curr_segment = candidate_segments[curr_segment_ind]
    
            curr_segment.add_margins(self.params_.margin_frames, num_frames)
    
            # Determine the start frame index for the current segment.
            if (clipped_start_frame_ind >= 0):
                # If we have a stored clipped frame index, use it.
                curr_segment.start_ind_ = clipped_start_frame_ind
                clipped_start_frame_ind = -1
        
            # Check whether it is possible to merge the current segment with the next segments.
            next_segment_ind = curr_segment_ind + 1;
            while (next_segment_ind < num_initial_segments):
                # Get the next candidate segment and add the appropriate margins to it.
                next_segment = candidate_segments[next_segment_ind].clone()
    
                next_segment.add_margins(self.params_.margin_frames, num_frames)
                    
                if (next_segment.start_ind_ > curr_segment.end_ind_):
                    # There is no overlap between the current segment and the next one, even after adding margins,
                    # so we should not try to concatenate these segments any further.
                    break
    
                # Try merging the next segment with the current one: Check that the merged segment does not exceed the
                # maximal segment size.
                if next_segment.end_ind_ - curr_segment.start_ind_ > self.params_.max_segment_frames:
                    # Do not merge the current segment with the next one, and use the midpoint of the gap between the
                    # original segments (before adding any margins) as the split point between them.
                    clipped_end_frame_ind = (curr_segment.orig_end_ind_ + next_segment.orig_start_ind_) // 2
                    clipped_start_frame_ind = clipped_end_frame_ind
                    break

                # If we reached here, we can merge the next segment with the current one.
                curr_segment.concat(next_segment)
    
                # Move to the next segment.
                next_segment_ind += 1
    
            # Determine the end frame index for the current segment.
            if (clipped_end_frame_ind >= 0):
                # If we have a stored clipped frame index, use it.
                curr_segment.end_ind_ = clipped_end_frame_ind
                clipped_end_frame_ind = -1
        
            # Prevent the creation of exceedingly long segments.
            if (curr_segment.num_frames() >= 2 * self.params_.max_segment_frames):
                # In case of a segment that is twice the maximal segment length, split this long segment into several
                # contiguous sub-segments.
                num_subsegments = int(1 + curr_segment.num_frames() / (1.5 * self.params_.max_segment_frames))
                num_subsegment_frames = int(curr_segment.num_frames() / num_subsegments);

                subsegment_start_frame_ind = curr_segment.start_ind_
    
                for k in range(num_subsegments):
                    subsegment_end_frame_ind = subsegment_start_frame_ind + num_subsegment_frames
                    
                    if k + 1 == num_subsegments:
                        subsegment_end_frame_ind = curr_segment.end_ind_

                    final_vad_segments.append(curr_segment.partial_vad_segment(channel_id, frame_duration,
                                                                               subsegment_start_frame_ind, subsegment_end_frame_ind))
    
                    subsegment_start_frame_ind = subsegment_end_frame_ind;

            else:
                # Check that the resulting segment is not too short.
                if (curr_segment.num_frames() >= min_segment_frames_with_margins):
                    # Create the full VAD segment.
                    final_vad_segments.append(curr_segment.full_vad_segment(channel_id, frame_duration))
    
            # Move to the next segment.
            curr_segment_ind = next_segment_ind
