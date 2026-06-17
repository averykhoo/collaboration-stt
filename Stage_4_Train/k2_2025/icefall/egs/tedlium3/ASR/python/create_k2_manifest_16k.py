# # # # # # # # ######################################################################
# # # # # # # # # python python/create_k2_manifest.py tedlium_train.list tedlium train wav stm manifest
import os
import sys
import re
import string
import codecs
import shutil
from gen_utils import *
import numpy as np

def main():
    in_list = sys.argv[1]
    db_name = sys.argv[2]
    db_part = sys.argv[3]
    wav_dir = sys.argv[4]
    stm_dir = sys.argv[5]
    manifest_dir = sys.argv[6]
    language = sys.argv[7]

    out_recordings = manifest_dir + "/" + db_name + "_recordings_" + db_part + ".jsonl"
    out_supervisions = manifest_dir + "/" + db_name + "_supervisions_" + db_part + ".jsonl"

    w_r = my_writer(out_recordings)
    w_s = my_writer(out_supervisions)

    r = my_reader(in_list)

    fs = 16000
    for line in r:
        line = line.strip()
        if not line:
            continue
        cur_wav = wav_dir + "/" + line + ".wav"
        cur_num_samples = get_wav_nos(cur_wav)
        cur_duration = get_wav_duration_16k(cur_wav)
        cur_recording_candidate = "{\"id\": \"" +  line + "\", \"sources\": [{\"type\": \"file\", \"channels\": [0],\"source\": \"" + cur_wav +  "\"}],\"sampling_rate\": 16000, \"num_samples\": " + str(cur_num_samples) + ", \"duration\": " + str(cur_duration) +", \"channel_ids\": [0]}"
        write_line(w_r,cur_recording_candidate)
        cur_stm = stm_dir + "/" + line + ".stm"
        r_stm = my_reader(cur_stm)
        cur_count = 1
        for line_stm in r_stm:
            line_stm = line_stm.strip()
            if not line_stm:
                continue
            a = line_stm.split(" ") # 911Mothers_2010W 1 911Mothers_2010W 16.12 25.02 <NA> the fact that we have what most people consider an unusual friendship <unk> and it is and yet it feels natural to us
            cur_start = a[3]
            cur_end = a[4]
            cur_duration = float2str2(float(cur_end) - float(cur_start))
            cur_trans = " ".join(a[6:])
            cur_trans = cur_trans.replace("\\","").strip()
            #cur_trans = cur_trans.replace("<unk>","").strip()
            cur_trans = continuous_replace("  "," ",cur_trans)
            cur_supervision_candidate = "{\"id\": \"" + line + "-" + str(cur_count) + "\", \"recording_id\": \"" + line + "\", \"start\": " + cur_start + ", \"duration\": " + cur_duration + ", \"channel\": 0, \"text\": \"" + cur_trans + "\", \"language\": \"" + language + "\", \"speaker\": \"" + line + "\"}"
            write_line(w_s,cur_supervision_candidate)
            cur_count += 1
    w_r.close()
    w_s.close()



if __name__ == "__main__":
      main()
# # # # # # # # #
# # # # # # # # #####################################################################
