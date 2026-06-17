import os
import sys

def main():
    # Check if both arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python create_paths.py <input_file> <output_txt>")
        sys.exit(1)

    # Get arguments from command line
    input_path = sys.argv[1]
    output_filename = sys.argv[2]

    try:
        # Resolve the input to an absolute path
        # This handles relative paths like 'input/fleurs.wav' automatically
        full_path = os.path.abspath(input_path)

        # Write the full path to the specified text file
        with open(output_filename, 'w') as f:
            f.write(full_path)
            
        print(f"Success: Full path written to {output_filename}")
        print(f"Path: {full_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()