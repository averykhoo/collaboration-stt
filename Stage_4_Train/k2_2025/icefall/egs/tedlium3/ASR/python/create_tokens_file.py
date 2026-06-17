import sys
from gen_utils import *

vocab_file = sys.argv[1]
tokens_file = sys.argv[2]

cur_count = 0
r = my_reader(vocab_file)
w = my_writer(tokens_file)

for line in r:
	line = line.strip()
	if not line:
		continue
	a = line.split("\t")
	write_line(w, a[0] + " " + str(cur_count))
	cur_count += 1

write_line(w,"#0 " + str(cur_count))
cur_count += 1
write_line(w,"#1 " + str(cur_count))
cur_count += 1
w.close()

