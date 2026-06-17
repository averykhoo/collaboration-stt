
import argparse
from typing import List
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torchaudio

from k2_scripts.gen_utils import *
from zipformer.beam_search import greedy_search_batch
from icefall.icefall.utils import num_tokens
from zipformer.train import add_model_arguments, get_model, get_params
from k2_scripts.symbol_table import SymbolTable
from vad_v2.audio_tool import AudioTool as VadAudioTool

MARGIN=0.4
MODEL_SAMPLING_RATE = 16000.0
CHECKPOINT="asr_model/pretrained.pt"
TOKENS="asr_model/tokens.txt"

class AudioTool:
    """
    Implemantation of K2 stt
    """

    # Constructor.
    def __init__(self):
        self.parser = get_parser()
        self.args = self.parser.parse_args()
        self.params = get_params()
        self.params.method = "greedy_search"
        self.params.update(vars(self.args))
        self.model_sampling_rate = MODEL_SAMPLING_RATE

    def load_model(self,device):
        #token_table = k2.SymbolTable.from_file(self.params.tokens)
        #fid = open("asr_model/token_table.pkl", 'rb')
        #self.token_table = pickle.load(fid)
        #pickle.dump(token_table, fid)
        #fid.close()
        #self.token_table = k2.SymbolTable.from_file(self.params.tokens)
        self.token_table = SymbolTable.from_file(self.params.tokens)
        self.params.blank_id = self.token_table["<blk>"]
        self.params.unk_id = self.token_table["<unk>"]
        self.params.vocab_size = num_tokens(self.token_table) + 1
        self.device = torch.device(device)
        self.model = get_model(self.params)
        self.num_param = sum([p.numel() for p in self.model.parameters()])
        self.checkpoint = torch.load(self.params.checkpoint, map_location="cpu")
        self.model.load_state_dict(self.checkpoint["model"], strict=False)
        self.model.to(device)
        self.model.eval()
        self.vader = VadAudioTool(sum_range=[3,200])
        self.vader.load_model(device)
        

    @torch.no_grad()
    def infer(self):

        r = my_reader(self.params.wav_pathes)
        w = my_writer(self.params.out_predict)
        for line in r:
            line = line.strip()
            if not line:
                continue

            waveform,samplerate = torchaudio.load(line)

            if samplerate != 16000:
                waveform = torchaudio.transforms.Resample(samplerate, 16000)(waveform)
                samplerate = 16000

            assert samplerate == 16000, "ERROR. 16K WAVEFORM IS EXPECTED"
            
            wav_duration = float(waveform.shape[1]) / float(samplerate) - 0.01
            
            # VAD
            start_end_arr = [] # Get initial start end array
            if wav_duration < 1:
                start_end_arr.append([0,wav_duration])
            else:
                vad_segs, vad_pred_sum = self.vader.infer(waveform, samplerate, thresh=5, n_moving=5) # Please use default values thresh=5, n_moving=5
                for k,v in vad_segs[0].items():
                    start_end_arr.append([v[0],v[1]])
            start_end_arr_new = [] # Expand start-end array
            segment_start = -1
            prev_start = -1
            prev_end = -1        
            for se in start_end_arr:
                cur_start = max(se[0] - MARGIN, 0.0)
                cur_end = min(se[1] + MARGIN, wav_duration)
                if prev_start == -1:
                    segment_start = cur_start
                else:
                    if prev_end > cur_start:
                        dummy = 1
                    else:
                        segment_end = prev_end
                        start_end_arr_new.append([segment_start,segment_end])
                        segment_start = cur_start
                prev_start = cur_start
                prev_end = cur_end
            start_end_arr_new.append([segment_start,cur_end])

            # VAD LIMITATION ALGORITHM
            # This implements a Voice Activity Detection (VAD) limitation algorithm
            # using a recursive approach to split long segments.

            def find_split_candidate(candidate_ind, ind_offset, vad_pred_sum, global_th):
                """Recursively find a candidate index where vad_pred_sum is below threshold."""
                candidate_ind += ind_offset
                if vad_pred_sum[candidate_ind] < global_th:
                    return candidate_ind
                ind_offset += 1
                candidate_ind -= ind_offset
                if vad_pred_sum[candidate_ind] < global_th:
                    return candidate_ind
                ind_offset += 1
                return find_split_candidate(candidate_ind, ind_offset, vad_pred_sum, global_th)

            def limit_segments(segments):
                """Recursively split segments that exceed the duration limit."""
                limited = []
                long_segment_found = False

                for se in segments:
                    start_time = se[0]
                    end_time = se[1]
                    cur_dur = end_time - start_time

                    if cur_dur <= self.params.splittime:
                        limited.append([start_time, end_time])
                    else:
                        long_segment_found = True

                        # Calculate energy indices based on start and end times
                        start_energy_ind = int(start_time * MODEL_SAMPLING_RATE / 1024.0)
                        end_energy_ind = int(end_time * MODEL_SAMPLING_RATE / 1024.0)

                        # Find the global minimum and maximum of VAD predictions within the segment
                        global_min = np.min(vad_pred_sum[start_energy_ind:end_energy_ind])
                        global_max = np.max(vad_pred_sum[start_energy_ind:end_energy_ind])

                        # Set a threshold for segment splitting
                        global_th = (global_max - global_min) * 0.05 + global_min

                        # Find the candidate index for splitting
                        candidate_ind = int((end_energy_ind - start_energy_ind) / 2.0 + start_energy_ind)

                        # Adjust the candidate index based on VAD predictions and threshold
                        if vad_pred_sum[candidate_ind] > global_th:
                            candidate_ind = find_split_candidate(candidate_ind, 1, vad_pred_sum, global_th)

                        if (candidate_ind == end_energy_ind) or (candidate_ind == start_energy_ind):
                            print("WARNING IT IS RECOMMENDED TO INCREASE PERCENT OF ENERGY THRESHOLD.")
                            int((end_energy_ind - start_energy_ind) / 2.0 + start_energy_ind)

                        # Calculate the midpoint time based on the adjusted candidate index
                        mid_time = candidate_ind * 1024 / MODEL_SAMPLING_RATE

                        # Add two new segments (before and after the midpoint)
                        limited.append([start_time, mid_time])
                        limited.append([mid_time, end_time])

                # If long segments were found, recurse with the new list
                if long_segment_found:
                    return limit_segments(limited)
                return limited

            start_end_arr_new_limited = limit_segments(start_end_arr_new)

            obs_arr = []
            for se in start_end_arr_new_limited:
                start_time = se[0]
                end_time = se[1]

                seg_wav = waveform[:,int(samplerate*start_time):int(samplerate*end_time)].to(self.device)

                features = torchaudio.compliance.kaldi.fbank(waveform=seg_wav,snip_edges=False,high_freq=-400.0,num_mel_bins=80,energy_floor=1e-10)
                feature_lengths = [len(features)]
                feature_lengths = torch.tensor(feature_lengths, device=self.device)
                features = features.unsqueeze(0)

                #torch.set_num_threads(1)

                encoder_out, encoder_out_lens = self.model.forward_encoder(features, feature_lengths)
                
                hyps = []
                def token_ids_to_words(token_ids: List[int]) -> str:
                    text = ""
                    for i in token_ids:
                        text += self.token_table[i]
                    return text.replace("▁", " ").strip()
                hyp_tokens = greedy_search_batch(model=self.model,encoder_out=encoder_out,encoder_out_lens=encoder_out_lens)
                for hyp in hyp_tokens:
                    hyps.append(token_ids_to_words(hyp))
                for hyp in hyps:
                    write_line(w,hyp)
                    obs_arr.append(hyp)
        w.close()

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--wav_pathes",
        type=str,
        default="",
        help="Path to list of waves. "
    )

    parser.add_argument(
        "--out_predict",
        type=str,
        default="",
        help="Path to predicted output. "
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="asr_model/pretrained.pt",
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="asr_model/tokens.txt",
        help="""Path to tokens.txt.""",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - modified_beam_search
          - fast_beam_search
        """,
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=8,
        help="""Used only when --method is fast_beam_search""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame. Used only when
        --method is greedy_search.
        """,
    )

    parser.add_argument(
        "--splittime",
        type=float,
        default=30.0,
        help="split time",
    )


    add_model_arguments(parser)

    return parser

if __name__ == '__main__':
    
    #torch_wav,sample_rate = torchaudio.load(wav_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_tool = AudioTool()
    asr_tool.load_model(device)
    asr_tool.infer()

