import os
import sys
from collections import defaultdict

def count_data_points(directory, filter_string=None):
    data_points_per_model = defaultdict(int)
    unparseable_per_model = defaultdict(int)

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filter_string is not None and filter_string not in filename:
            continue

        # Check if it's a file and not a directory
        if os.path.isfile(file_path) and 'no_abst' in filename and 'norm' in filename:
            with open(file_path, 'r') as file:
                model_name = filename.split('-q')[0].split('_')[1]
                for line in file:
                    # Splitting each line into grade and status
                    parts = line.strip().split()

                    # Ensure the line has two parts: grade and status
                    if len(parts) == 2 and parts[0] != 'grade':
                        data_points_per_model[model_name] += 1

                        # Check if the status is 'Unparseable'
                        if parts[0] == 'Unparseable':
                            unparseable_per_model[model_name] += 1

    return data_points_per_model, unparseable_per_model

# Get directory path from command line argument, and possibly a string to filter the filenames
if len(sys.argv) not in [2, 3]:
    print("Usage: python count_unparseable.py <directory_path> [filter string]")
    sys.exit(1)
directory_path = sys.argv[1]
filter_string = sys.argv[2] if len(sys.argv) == 3 else None
data_points, unparseable = count_data_points(directory_path, filter_string)
print(f"Total: {sum(data_points.values())} data points, {sum(unparseable.values())} unparseable, {100* round(sum(unparseable.values())/sum(data_points.values()),4)} percentage unparseable")

# Print the number of data points and unparseable data points for each model, sorted by model name
for model, count in sorted(data_points.items()):
    print(f"{model}: {count} data points, {unparseable[model]} unparseable, {100* round(unparseable[model]/count,4)} percentage unparseable")
