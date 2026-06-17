import sys

# Expect exactly two command-line arguments:
#   1) input text file (list of words, one per line)
#   2) output file (symbol-to-ID mapping)
if len(sys.argv) != 3:
    print("Usage: python transform_text_file.py <input_file> <output_file>")
    sys.exit(1)

# Input file containing words (one word per line)
input_file = sys.argv[1]

# Output file where symbol IDs will be written
output_file = sys.argv[2]

# Dictionary used to track which words already exist
# This is mainly used to check if <unk> appears in the input
h = {}

# Open input file for reading and output file for writing
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    # Write mandatory special symbols with fixed IDs
    # <eps> is usually required by FST-based decoders
    outfile.write("<eps> 0\n")

    # Silence symbol commonly used in ASR pipelines
    outfile.write("!SIL 1\n")

    # Counter for assigning incremental IDs to words
    id_counter = 2

    # Process each word from the input file
    for line in infile:
        # Strip whitespace to get the word
        word = line.strip()

        # Record the word as seen
        h[word] = 1

        # Write word and its assigned ID
        outfile.write(f"{word} {id_counter}\n")

        # Increment ID counter for next word
        id_counter += 1

    # Add special symbols depending on whether <unk> already exists
    if "<unk>" not in h:
        # Add unknown-word symbol if it was not present in input
        outfile.write(f"<unk> {id_counter}\n")

        # Add disambiguation and sentence boundary symbols
        outfile.write(f"#0 {id_counter + 1}\n")
        outfile.write(f"<s> {id_counter + 2}\n")
        outfile.write(f"</s> {id_counter + 3}\n")
    else:
        # If <unk> already exists, skip adding it again
        outfile.write(f"#0 {id_counter}\n")
        outfile.write(f"<s> {id_counter + 1}\n")
        outfile.write(f"</s> {id_counter + 2}\n")

# Print confirmation message
print(f"Output saved in {output_file}")
