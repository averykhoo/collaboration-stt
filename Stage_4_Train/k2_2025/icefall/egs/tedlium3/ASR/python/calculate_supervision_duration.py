import sys
import json

# Check if the input file is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python calculate_duration.py <input_file.json>")
    sys.exit(1)

# Get the input file from the command-line argument
input_file = sys.argv[1]

try:
    total_duration_seconds = 0  # Initialize total duration to 0 seconds

    # Read data from the input file with UTF-8 encoding, line by line
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)  # Parse each line as JSON
                duration = data.get("duration", 0)  # Extract the "duration" field
                total_duration_seconds += duration
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON format in line: {line}")

    # Convert total duration to hours
    total_duration_hours = total_duration_seconds / 3600  # 3600 seconds in an hour

    print(f"Total duration of the dataset in hours: {total_duration_hours:.2f} hours")
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found.")
