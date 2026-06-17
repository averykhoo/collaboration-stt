#!/bin/bash

python wer.py gt.txt predict.txt allowed_replacements.txt allowed_insertions.txt allowed_deletions.txt out_stat.txt

python wer.py gt_small.txt predict_small.txt allowed_replacements.txt allowed_insertions.txt allowed_deletions.txt out_stat_small.txt



