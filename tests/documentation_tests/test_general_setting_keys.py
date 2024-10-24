import os
import re

# Define the base directory for the litellm repository and documentation path
repo_base = "./litellm"  # Change this to your actual path


# Regular expressions to capture the keys used in general_settings.get() and general_settings[]
get_pattern = re.compile(
    r'general_settings\.get\(\s*[\'"]([^\'"]+)[\'"](,?\s*[^)]*)?\)'
)
bracket_pattern = re.compile(r'general_settings\[\s*[\'"]([^\'"]+)[\'"]\s*\]')

# Set to store unique keys from the code
general_settings_keys = set()

# Walk through all files in the litellm repo to find references of general_settings
for root, dirs, files in os.walk(repo_base):
    for file in files:
        if file.endswith(".py"):  # Only process Python files
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Find all keys using general_settings.get()
                get_matches = get_pattern.findall(content)
                general_settings_keys.update(
                    match[0] for match in get_matches
                )  # Extract only the key part

                # Find all keys using general_settings[]
                bracket_matches = bracket_pattern.findall(content)
                general_settings_keys.update(bracket_matches)

# Parse the documentation to extract documented keys
repo_base = "./"
print(os.listdir(repo_base))
docs_path = "./docs/my-website/docs/proxy/configs.md"  # Path to the documentation
documented_keys = set()
try:
    with open(docs_path, "r", encoding="utf-8") as docs_file:
        content = docs_file.read()

        # Find the section titled "general_settings - Reference"
        general_settings_section = re.search(
            r"### general_settings - Reference(.*?)###", content, re.DOTALL
        )
        if general_settings_section:
            # Extract the table rows, which contain the documented keys
            table_content = general_settings_section.group(1)
            doc_key_pattern = re.compile(
                r"\|\s*([^\|]+?)\s*\|"
            )  # Capture the key from each row of the table
            documented_keys.update(doc_key_pattern.findall(table_content))
except Exception as e:
    raise Exception(
        f"Error reading documentation: {e}, \n repo base - {os.listdir(repo_base)}"
    )

# Compare and find undocumented keys
undocumented_keys = general_settings_keys - documented_keys

# Print results
print("Keys expected in 'general_settings' (found in code):")
for key in sorted(general_settings_keys):
    print(key)

if undocumented_keys:
    raise Exception(
        f"\nKeys not documented in 'general_settings - Reference': {undocumented_keys}"
    )
else:
    print(
        "\nAll keys are documented in 'general_settings - Reference'. - {}".format(
            general_settings_keys
        )
    )
