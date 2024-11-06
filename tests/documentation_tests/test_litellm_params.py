import os
import re

# Define the directory to search through
directory = "../../docs/my-website/docs"  # Change this to the correct path

# Regular expression to match metadata={} with keys, ignoring inline comments
metadata_pattern = re.compile(r"metadata\s*=\s*\{\s*([^}]+?)\s*\}")

# Dictionary to collect metadata keys and their occurrences
metadata_keys = {}

# Walk through all files in the directory
for root, dirs, files in os.walk(directory):
    print(f"root: {root}, dirs: {dirs}, files: {files}")
    for file in files:
        if file.endswith(".md"):
            file_path = os.path.join(root, file)

            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

                # Search for metadata={...} patterns
                matches = metadata_pattern.findall(content)

                for match in matches:
                    # Split by commas to get individual key-value pairs
                    key_value_pairs = [
                        pair.split("#")[0].strip() for pair in match.split(",")
                    ]

                    for pair in key_value_pairs:
                        if ":" in pair:
                            key = pair.split(":")[0].strip()  # Extract the key

                            if key not in metadata_keys:
                                metadata_keys[key] = []
                            if file not in metadata_keys[key]:
                                metadata_keys[key].append(file)

# Display all metadata keys found and where they were found
for key, files in metadata_keys.items():
    print(f"Metadata Key: '{key}' found in files: {', '.join(files)}")
