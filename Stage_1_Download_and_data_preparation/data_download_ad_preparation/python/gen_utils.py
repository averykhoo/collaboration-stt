import operator
import os
import os
import sys
import re
import string
import codecs
import numpy as np
from random import randint
import struct
import random
#from itertools import imap

def normalize_matrix_by_sum_column(mat):
    for i in range(len(mat[:,0])):
        col_sum = mat[:, i].sum()
        if col_sum != 0:
            mat[:, i] = mat[:, i] / col_sum
        else:
            pass

def path_get_base_name(in_path):
    return os.path.basename(in_path)

def path_get_base_name_without_extension(in_path):
    base_name = os.path.basename(in_path)
    my_array = base_name.split('.')
    return  my_array[0]

def my_close(myset):
    for k in myset:
        k.close()

def my_reader(in_path):
    fid = codecs.open(in_path,'r','utf-8-sig')
    return fid

def my_appender(out_path):
    fid = codecs.open(out_path,'a','utf-8')
    return fid

def my_writer(out_path):
    fid = codecs.open(out_path,'w','utf-8')
    return fid

def write_line(fid, str):
    fid.write(str + "\n")

def file_2_list(input_file):
    fid = codecs.open(input_file,'r','utf-8-sig')
    out_list = []
    for line in fid:
        line = line.strip()
        if not line:
            break
        out_list.append(line)
    fid.close()
    return out_list


def file_2_hash(input_file):
    h = {}
    r = my_reader(input_file)
    for line in r:
        line = line.strip()
        if not line:
            continue
        h[line] = 1
    return h

def dir_get_file_list(input_dir, my_extension):
    input_list = os.listdir(input_dir)
    out_list = [i for i in input_list if i.endswith(my_extension)]
    return out_list

def dir_get_file_list_sorted(input_dir, my_extension):
    input_list = os.listdir(input_dir)
    out_list = [i for i in input_list if i.endswith(my_extension)]
    out_list.sort()  # Sort the list in alphabetical order
    return out_list


def dir_get_dir_list(input_dir):
    return os.listdir(input_dir)

def replace_at_start(pattern, replacement, in_str):
    s = re.sub('^' + pattern, replacement, in_str)
    return s

def replace_at_end(pattern, replacement, in_str):
    s = re.sub(pattern + '$', replacement, in_str)
    return s

def str_2_sec(in_str):
    sec = 0.0
    arr =  in_str.split(':')
    if len(arr) == 1:
        sec = float(in_str)
    elif len(arr) == 2:
        sec = float(arr[0])*60 + float(arr[1])
    elif len(arr) == 3:
        sec = float(arr[0])*60*60 + float(arr[1])*60 + float(arr[2])
    return sec

def mono_2_dct(in_mono, out_dct):
    fid_in = codecs.open(in_mono, 'r', 'utf-8-sig')
    fid_out = codecs.open(out_dct, 'w', 'utf-8-sig')
    for line in fid_in:
        line = line.strip()
        if not line:
            break
        write_line(fid_out, line + " " + line)
    fid_in.close()
    fid_out.close()

def to_string(a,n):
    s = "{0:." + str(n) + "f}"
    return s.format(a)

def flat_list(some_list = []):
    elements=[]
    for item in some_list:
        if type(item) == type([]):
            elements += flat_list(item)
        else:
            elements.append(item)
    return elements

def my_round(x):
    return int(x+0.5)

def write_csh_header(w):
    write_line(w,"#!/bin/csh -f")
    write_line(w, "")

def my_write_float_to_file(in_val,out_file_path):
    w = my_writer(out_file_path)
    write_line(w,to_string(in_val,4))
    w.close()

def read_matrix_from_file(in_file):
    r = my_reader(in_file)
    nor = 0
    for line in r:
        line = line.strip()
        if not line:
            continue
        nor += 1
    r.close()

    r = my_reader(in_file)
    header_line = next(r)
    header_line = header_line.strip()
    header_array = header_line.split("\t")
    noc = len(header_array)

    out_mat = [["" for j in range(noc)] for j in range(nor)]
    out_mat[0][:] = header_array

    for i in range(1,nor):
        line = next(r).strip()
        a = line.split("\t")
        out_mat[i][:] = a

    r.close()
    return out_mat

def read_confusion_matrix_from_file(in_file):
    r = my_reader(in_file)
    conf_mat_size = 0
    for line in r:
        line = line.strip()
        if not line:
            continue
        conf_mat_size += 1
    r.close()

    r = my_reader(in_file)
    header_line = next(r)
    header_line = header_line.strip()
    header_array = header_line.split("\t")


    out_mat = [["" for j in range(conf_mat_size)] for j in range(conf_mat_size)]
    out_mat[0][1:] = header_array

    for i in range(1,conf_mat_size):
        line = next(r).strip()
        a = line.split("\t")
        out_mat[i][:] = a

    r.close()
    return out_mat

def write_confusion_matrix(conf_mat_file,value_factor,value_precison,conf_mat_size,id2label,conf_mat):
    w = my_writer(conf_mat_file)
    first_line = "\t"
    for i in range(0, conf_mat_size):
        first_line += id2label[i] + "\t"
    write_line(w, first_line)
    k = 0
    for my_row in conf_mat:
        current_line = id2label[k] + "\t"
        for val in my_row:
            current_line += to_string(val*value_factor,value_precison) + "\t"
        write_line(w, current_line)
        k += 1
    w.close()

def get_wav_duration_16k(in_file):
    return (os.path.getsize(in_file) - 44)/32000.0

def get_wav_duration_8k(in_file):
    return (os.path.getsize(in_file) - 44)/16000.0

def get_wav_nos(in_file):
    return int((os.path.getsize(in_file) - 44) / 2)

# def pearsonr(x, y):
#   # Assume len(x) == len(y)
#   n = len(x)
#   sum_x = float(sum(x))
#   sum_y = float(sum(y))
#   sum_x_sq = sum(map(lambda x: pow(x, 2), x))
#   sum_y_sq = sum(map(lambda x: pow(x, 2), y))
#   psum = sum(imap(lambda x, y: x * y, x, y))
#   num = psum - (sum_x * sum_y/n)
#   den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
#   if den == 0: return 0
#   return num / den


def continuous_replace(src,dest,inString):
    prevString = inString

    while True:
        inString = inString.replace(src,dest)
        if inString == prevString:
            break
        prevString = inString

    return inString

# RETURNS RANDOM INTEGER BETWEEN START AND END INCLUDING START ADN END.
def random_index(start_ind,end_ind):
    return randint(start_ind,end_ind)

def random_float(start_float,end_float):
    return random.uniform(start_float,end_float)

# Compare two lists
def my_list_compare(list1,list2):
    if len(list1) != len(list2):
        return False

    for i in range(0,len(list1)):
        if list1[i] != list2[i]:
            return False

    return True

def write_wav(sampleRate,in_array,out_wav_name):
    import wave, struct, math, random

    wavef = wave.open(out_wav_name, 'w')
    wavef.setnchannels(1)  # mono
    wavef.setsampwidth(2)  # number of bytes per sample
    wavef.setframerate(sampleRate)
    wavef.setnframes(len(in_array))

    max_val = np.max(np.abs(in_array))
    assert max_val <= 1, "ERROR. IT IS ASSUMED THAT MAXIMAL VALUE OF INPUT ARRAY IS LESS THAN 1"

    for i in range(len(in_array)):
        value = int(32767.0 * in_array[i])
        data = struct.pack('<h', value)
        wavef.writeframesraw(data)

    wavef.close()

########################################################################

def write_wav_2(sampleRate,in_array,out_wav_name):
    import scipy.io.wavfile

    in_array = (32767 * in_array).astype('short')

    scipy.io.wavfile.write(out_wav_name, sampleRate, in_array)

########################################################################

########################################################################

def read_wav(wav_path):
    import scipy.io.wavfile

    [fs, data] = scipy.io.wavfile.read(wav_path)

    return [fs,data]

########################################################################

class HTKRead:
    """ Class to load binary HTK file.
        Details on the format can be found online in HTK Book chapter 5.7.1.
        Not everything is implemented 100%, but most features should be supported.
        Not implemented:
            CRC checking - files can have CRC, but it won't be checked for correctness
            VQ - Vector features are not implemented.
    """

    data = None
    nSamples = 0
    nFeatures = 0
    sampPeriod = 0
    sampSize = 0
    paramKind = 0
    basicKind = None
    qualifiers = None

    def load(self, filename):
        """ Loads HTK file.
            After loading the file you can check the following members:
                data (matrix) - data contained in the file
                nSamples (int) - number of frames in the file
                nFeatures (int) - number if features per frame
                sampPeriod (int) - sample period in 100ns units (e.g. fs=16 kHz -> 625)
                basicKind (string) - basic feature kind saved in the file
                qualifiers (string) - feature options present in the file
        """
        with open(filename, "rb") as f:

            header = f.read(12)
            self.nSamples, self.sampPeriod, self.sampSize, self.paramKind = struct.unpack(">iihh", header)
            basicParameter = self.paramKind & 0x3F

            if basicParameter == 0:
                self.basicKind = "WAVEFORM"
            elif basicParameter == 1:
                self.basicKind = "LPC"
            elif basicParameter == 2:
                self.basicKind = "LPREFC"
            elif basicParameter == 3:
                self.basicKind = "LPCEPSTRA"
            elif basicParameter == 4:
                self.basicKind = "LPDELCEP"
            elif basicParameter == 5:
                self.basicKind = "IREFC"
            elif basicParameter == 6:
                self.basicKind = "MFCC"
            elif basicParameter == 7:
                self.basicKind = "FBANK"
            elif basicParameter == 8:
                self.basicKind = "MELSPEC"
            elif basicParameter == 9:
                self.basicKind = "USER"
            elif basicParameter == 10:
                self.basicKind = "DISCRETE"
            elif basicParameter == 11:
                self.basicKind = "PLP"
            else:
                self.basicKind = "ERROR"

            self.qualifiers = []
            if (self.paramKind & 0o100) != 0:
                self.qualifiers.append("E")
            if (self.paramKind & 0o200) != 0:
                self.qualifiers.append("N")
            if (self.paramKind & 0o400) != 0:
                self.qualifiers.append("D")
            if (self.paramKind & 0o1000) != 0:
                self.qualifiers.append("A")
            if (self.paramKind & 0o2000) != 0:
                self.qualifiers.append("C")
            if (self.paramKind & 0o4000) != 0:
                self.qualifiers.append("Z")
            if (self.paramKind & 0o10000) != 0:
                self.qualifiers.append("K")
            if (self.paramKind & 0o20000) != 0:
                self.qualifiers.append("0")
            if (self.paramKind & 0o40000) != 0:
                self.qualifiers.append("V")
            if (self.paramKind & 0o100000) != 0:
                self.qualifiers.append("T")

            if "C" in self.qualifiers or "V" in self.qualifiers or self.basicKind == "IREFC" or self.basicKind == "WAVEFORM":
                self.nFeatures = self.sampSize // 2
            else:
                self.nFeatures = self.sampSize // 4

            if "C" in self.qualifiers:
                self.nSamples -= 4

            if "V" in self.qualifiers:
                raise NotImplementedError("VQ is not implemented")

            self.data = []
            if self.basicKind == "IREFC" or self.basicKind == "WAVEFORM":
                for x in range(self.nSamples):
                    s = f.read(self.sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(">h", s, v * 2)[0] / 32767.0
                        frame.append(val)
                    self.data.append(frame)
            elif "C" in self.qualifiers:

                A = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    A.append(struct.unpack_from(">f", s, x * 4)[0])
                B = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    B.append(struct.unpack_from(">f", s, x * 4)[0])

                for x in range(self.nSamples):
                    s = f.read(self.sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        frame.append((struct.unpack_from(">h", s, v * 2)[0] + B[v]) / A[v])
                    self.data.append(frame)
            else:
                for x in range(self.nSamples):
                    s = f.read(self.sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(">f", s, v * 4)
                        frame.append(val[0])
                    self.data.append(frame)

            if "K" in self.qualifiers:
                print("CRC checking not implememnted...")

#######################################################################

# Class for HTK binary file writing
class HTKWrite:
    def write_data(self,nSamples,samplePeriod,sampleSize,paramKind,np_data,out_path):
        self.nSamples = nSamples
        self.samplePeriod = samplePeriod
        self.sampleSize = sampleSize
        self.paramKind = paramKind

        wb = open(out_path, "wb")
        wb.write(struct.pack(">IIHH",self.nSamples,self.samplePeriod,self.sampleSize,self.paramKind))

        (nor,noc) = np_data.shape
        # for i in range(0,nor):
        #     for j in range(0,noc):
        #         wb.write(np_data[i,j])

        np.array(np_data,'f').byteswap().tofile(wb)

        wb.close()

########################################################################

def tolerant_comparison(a,b,t):
    if np.abs(a-b) < t:
        return True
    else:
        return False

########################################################################

def sort_table(table, cols):
    """ sort a table by multiple columns
        table: a list of lists (or tuple of tuples) where each inner list
               represents a row
        cols:  a list (or tuple) specifying the column numbers to sort by
               e.g. (1,0) would sort by column 1, then by column 0
    """
    for col in reversed(cols):
        table = sorted(table, key=operator.itemgetter(col))
    return table

########################################################################

def sort_table_files(in_table_file,out_table_file, sort_column):

    table_list = []
    r = my_reader(in_table_file)
    for line in r:
        line = line.strip()
        if not line:
            continue
        a = line.split("\t")
        cur_tupple = tuple(a)
        table_list.append(cur_tupple)

    mytable = tuple(table_list)

    w = my_writer(out_table_file)

    for row in sort_table(mytable, (sort_column,)):
        candidate = ""
        for cur_word in row:
            candidate += "\t" + cur_word
        candidate = candidate.strip()
        write_line(w,candidate)

    w.close()

########################################################################

def get_number_of_not_empty_lines(in_path):
    nol = 0
    r = my_reader(in_path)
    for line in r:
        line = line.strip()
        if not line:
            continue
        nol += 1
    return nol

########################################################################

def generate_random_number(length):
    return int(''.join([str(random.randint(0,9)) for _ in range(length)]))

########################################################################

def random_float(start,end):
    return random.uniform(start, end)

########################################################################

def generate_random_number_in_str_format(length):
    my_number = ' '.join([str(random.randint(0,9)) for _ in range(length)])

    zero_str = "zero"
    my_float_rand = random_float(0, 1)
    if my_float_rand > 0.5:
        zero_str = "oh"

    my_number = my_number.replace("0",zero_str).replace("1","one").replace("2","two").replace("3","three").replace("4","four").replace("5","five")\
                .replace("6","six").replace("7","seven").replace("8","eight").replace("9","nine")

    return my_number

########################################################################

def float2str2(in_str):
    return '{:.2f}'.format(in_str)

########################################################################

def float2str3(in_str):
    return '{:.3f}'.format(in_str)

########################################################################

def reversed_string(in_str):
    return in_str[::-1]

########################################################################

def dtw(ref_arr, obs_arr):
    DELETION_COST = 1
    INSERTION_COST = 1
    SUBSTITUTION_COST = 1
    PSI_SUB = 100
    PSI_DEL = 101
    PSI_INS = 102

    N = len(ref_arr)
    M = len(obs_arr)

    scores_matrix = np.zeros((N+1,M+1))
    psi_i = np.zeros((N+1,M+1))
    psi_j  = np.zeros((N+1,M+1))
    psi = np.zeros((N+1,M+1))
    psi[1:N+1,0] = PSI_DEL
    psi[0,1:M+1] = PSI_INS

    # Initialization steps
    scores_matrix[0,0] = 0
    for i in range(1,N+1):
        scores_matrix[i,0] = i * DELETION_COST
    for j in range(1,M+1):
        scores_matrix[0,j] = j * INSERTION_COST

    for j in range(1,M+1):
        for i in range(1,N+1):
            sub_score = scores_matrix[i-1,j-1]
            if ref_arr[i-1] != obs_arr[j-1]:
                sub_score += SUBSTITUTION_COST
            del_score = scores_matrix[i-1,j] + DELETION_COST
            ins_score = scores_matrix[i,j-1] + INSERTION_COST
            if sub_score <= np.min([del_score,ins_score]):
                scores_matrix[i,j] =  sub_score
                psi_i[i,j] = i - 1
                psi_j[i,j] = j - 1
                psi[i,j] = PSI_SUB
            elif del_score <= np.min([sub_score,ins_score]):
                scores_matrix[i, j] = del_score
                psi_i[i,j] = i - 1
                psi_j[i,j] = j
                psi[i,j] = PSI_DEL
            else:
                scores_matrix[i, j] = ins_score
                psi_i[i,j] = i
                psi_j[i,j] = j - 1
                psi[i,j] =  PSI_INS

    cur_i = N
    cur_j = M

    ref_str = ""
    obs_str = ""
    S = 0 # substitutions
    I = 0 # insertions
    D = 0 # deletions
    T = N # total

    ref_out_arr = []
    obs_out_arr = []
    while True:
        if cur_i == 0 and cur_j ==0:
            break
        if psi[cur_i,cur_j] == PSI_SUB:
            if ref_arr[cur_i - 1] != obs_arr[cur_j - 1]:
                S += 1
            ref_out_arr.append(ref_arr[cur_i - 1])
            obs_out_arr.append(obs_arr[cur_j - 1])
            cur_i -= 1
            cur_j -= 1
        elif psi[cur_i,cur_j] == PSI_DEL:
            # rons_out.append((ref_arr[cur_i-1],"-"))
            D += 1
            ref_out_arr.append(ref_arr[cur_i-1])
            obs_out_arr.append("-")
            cur_i -= 1
        else:
            I += 1
            ref_out_arr.append("+")
            obs_out_arr.append(obs_arr[cur_j-1])
            cur_j -= 1

    ref_str = "\t".join(reversed(ref_out_arr))
    obs_str = "\t".join(reversed(obs_out_arr))

    return ref_str, obs_str, S, I, D, T

########################################################################

def insert_space_between_letters(s):
    return " ".join(s)

########################################################################

import operator
def sort_dictionary(d,reverse_order):
    return dict(sorted(d.items(), key=operator.itemgetter(1),reverse=reverse_order))

def sort_dictionary_by_val(d,reverse_order):
    return dict(sorted(d.items(), key=operator.itemgetter(1),reverse=reverse_order))

def sort_dictionary_by_key(d,reverse_order):
    return dict(sorted(d.items(), key=operator.itemgetter(0),reverse=reverse_order))

########################################################################

def my_floor(f):
    return int(f)

########################################################################

def zero_leading_5(i):
    return '{:05d}'.format(i)

########################################################################

def zero_leading_6(i):
    return '{:06d}'.format(i)

########################################################################

def zero_leading_7(i):
    return '{:07d}'.format(i)

########################################################################

def file2hash(in_file,sep):
    out_hash = {}
    r = my_reader(in_file)
    for line in r:
        line = line.strip()
        if not line:
            continue
        a = line.split(sep)
        out_hash[a[0]] = a[1]
    return out_hash

########################################################################

def file2hash_inverse(in_file,sep):
    out_hash = {}
    r = my_reader(in_file)
    for line in r:
        line = line.strip()
        if not line:
            continue
        a = line.split(sep)
        out_hash[a[1]] = a[0]
    return out_hash

########################################################################

def file2hash_int(in_file,sep):
    out_hash = {}
    r = my_reader(in_file)
    for line in r:
        line = line.strip()
        if not line:
            continue
        a = line.split(sep)
        out_hash[a[0]] = int(a[1])
    return out_hash

########################################################################

def std_hash_2_count_update(in_hash,in_key):
    if in_key not in in_hash:
        in_hash[in_key] = 1
    else:
        in_hash[in_key] += 1

########################################################################

def get_filename_without_extension(file_path):
    # Get the base name of the file (including the extension)
    base_name = os.path.basename(file_path)

    # Split the base name into the file name and extension
    file_name, file_extension = os.path.splitext(base_name)

    return file_name

########################################################################
