# # # # # # # # # # # # ######################################################################
# This script extracts all transcriptions from STM files and writes them
# into a single text file (one transcription per line).
#
# Example usage:
#   python python/create_trans_all.py ${inter_path}/stm_1 ${inter_path}/trans_all.txt

import sys
import codecs
import os

def write_line(fid, str):
    fid.write(str + "\n")

def my_reader(in_path):
    fid = codecs.open(in_path,'r','utf-8-sig')
    return fid

def my_writer(out_path):
    # Open an output file for writing text in UTF-8 encoding
    # Returns a writable file handle
    fid = codecs.open(out_path,'w','utf-8')
    return fid

def dir_get_file_list(input_dir, my_extension):
    input_list = os.listdir(input_dir)
    out_list = [i for i in input_list if i.endswith(my_extension)]
    return out_list

def main():
    # Input directory containing STM files
    in_stm_dir = sys.argv[1]

    # Output file where all transcriptions will be concatenated
    out_trans_all = sys.argv[2]

    # Open output file for writing
    w = my_writer(out_trans_all)

    # Get list of STM files from the input directory
    # Assumes dir_get_file_list returns filenames (not full paths)
    stm_files = dir_get_file_list(in_stm_dir,".stm")

    # Iterate over each STM file
    for stm_file in stm_files:
        # Construct full path to STM file
        in_stm_file  = in_stm_dir + "/" + stm_file

        # Open STM file for reading
        r = my_reader(in_stm_file)

        # Read STM file line by line
        for line in r:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Split STM line into tokens by space
            a = line.split(" ")

            # STM format:
            #   tokens 0–5 : metadata
            #   token 6+   : transcription text
            trans = " ".join(a[6:])

            # Write transcription text to output file
            write_line(w,trans)

    # Close output file after processing all STM files
    w.close()


if __name__ == "__main__":
      # Entry point of the script
      main()

# # # # # # # # # # # # #
# # # # # # # # # # # # #####################################################################
