import sentencepiece as spm
from gen_utils import *
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

sp = spm.SentencePieceProcessor()
sp.load("data_verbit_fa/lang_bpe_500/bpe.model")

r = my_reader(in_file)
w = my_writer(out_file)

for line in r:
    line = line.strip()
    if not line:
        continue
    y = sp.encode(line)
    result_string = ' '.join(map(str, y))
    write_line(w,result_string)

w.close()


