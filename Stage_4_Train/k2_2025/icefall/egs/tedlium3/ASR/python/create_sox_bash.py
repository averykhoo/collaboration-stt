import os
import sys
import re
import string
import codecs
import shutil
from gen_utils import *
import numpy as np
import matplotlib.pyplot as plt
import pickle

def main():
    in_wav_file = sys.argv[1]
    vad_info = sys.argv[2]
    out_bash = sys.argv[3]
    out_dir = sys.argv[4]
    out_paths_file = sys.argv[5]

    w = my_writer(out_bash)
    r = my_reader(vad_info)
    w_paths = my_writer(out_paths_file)
    for line in r:
        line = line.strip()
        if not line:
            continue 
        a = line.split("\t")
        cur_start = a[0]
        cur_end = a[1]
        cur_dur = float2str2(float(cur_end) - float(cur_start))
        out_wav_file = out_dir + "/" + cur_start + "_" + cur_end + ".wav"
        write_line(w_paths,out_wav_file)
        cur_cmd = "sox " + in_wav_file + " " + out_wav_file + " trim " + cur_start + " " + cur_dur
        write_line(w,cur_cmd)

    w.close()
    w_paths.close()

if __name__ == "__main__":
      main()
