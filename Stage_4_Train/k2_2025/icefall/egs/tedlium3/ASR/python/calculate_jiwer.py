from jiwer import wer
from gen_utils import *

r_ref = my_reader(sys.argv[1])
r_obs = my_reader(sys.argv[2])

ref_list = []
obs_list = []

for line_ref in r_ref:
    line_ref = line_ref.strip()
    if not line_ref:
        continue
    line_ref = continuous_replace("  "," ",line_ref)
    words = line_ref.split()
    ref_list.extend(words)

for line_obs in r_obs:
    line_obs = line_obs.strip()
    if not line_obs:
        continue
    line_obs = continuous_replace("  "," ",line_obs)
    words = line_obs.split()
    obs_list.extend(words)

my_ref = " ".join(ref_list)
my_obs = " ".join(obs_list)

cur_wer = wer(my_ref, my_obs)

#print(my_ref[0:100])
#print(my_obs[0:100])

print("WER: " + str(cur_wer*100) + "%")