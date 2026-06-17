import sys

def extract_lines(input_file, output_file, delution_factor):
    try:
        with open(input_file, 'r', encoding='utf-8') as input_f:
            lines = input_f.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as output_f:
            for i, line in enumerate(lines, start=1):
                if i % delution_factor == 0:
                    output_f.write(line)

        print(f"Every {delution_factor}th line extracted from '{input_file}' and saved to '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_lines.py <delution_factor> <input_file> <output_file>")
        sys.exit(1)

    delution_factor = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    extract_lines(input_file, output_file, delution_factor)
