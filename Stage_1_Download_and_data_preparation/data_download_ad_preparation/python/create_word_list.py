import sys

# Expect exactly two command-line arguments:
#   1) input text file
#   2) output file for unique words
if len(sys.argv) != 3:
    print("Usage: python extract_unique_words.py <input_file> <output_file>")
    sys.exit(1)

# Path to the input text file
input_file = sys.argv[1]

# Path to the output file where unique words will be written
output_file = sys.argv[2]

# Use a set to store unique words (automatically removes duplicates)
unique_words = set()

# Read the input file line by line and extract words
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        # Split each line into words using whitespace
        words = line.split()

        # Add words to the set (duplicates are ignored)
        unique_words.update(words)

# Convert the set to a list and sort words alphabetically
unique_words = sorted(unique_words)

# Write the sorted unique words to the output file
# Each word is written on a separate line
with open(output_file, 'w', encoding='utf-8') as out_file:
    out_file.write("\n".join(unique_words))

# Print confirmation message
print(f"Unique words extracted and saved in {output_file} in alphabetical order.")
