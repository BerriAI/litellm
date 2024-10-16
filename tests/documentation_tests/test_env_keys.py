import os
import re

# Define the base directory for the litellm repository and documentation path
repo_base = "./litellm"  # Change this to your actual path

# Regular expressions to capture the keys used in os.getenv() and litellm.get_secret()
getenv_pattern = re.compile(r'os\.getenv\(\s*[\'"]([^\'"]+)[\'"]\s*(?:,\s*[^)]*)?\)')
get_secret_pattern = re.compile(
    r'litellm\.get_secret\(\s*[\'"]([^\'"]+)[\'"]\s*(?:,\s*[^)]*|,\s*default_value=[^)]*)?\)'
)
get_secret_str_pattern = re.compile(
    r'litellm\.get_secret_str\(\s*[\'"]([^\'"]+)[\'"]\s*(?:,\s*[^)]*|,\s*default_value=[^)]*)?\)'
)

# Set to store unique keys from the code
env_keys = set()

# Walk through all files in the litellm repo to find references of os.getenv() and litellm.get_secret()
for root, dirs, files in os.walk(repo_base):
    for file in files:
        if file.endswith(".py"):  # Only process Python files
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

                # Find all keys using os.getenv()
                getenv_matches = getenv_pattern.findall(content)
                env_keys.update(
                    match for match in getenv_matches
                )  # Extract only the key part

                # Find all keys using litellm.get_secret()
                get_secret_matches = get_secret_pattern.findall(content)
                env_keys.update(match for match in get_secret_matches)

                # Find all keys using litellm.get_secret_str()
                get_secret_str_matches = get_secret_str_pattern.findall(content)
                env_keys.update(match for match in get_secret_str_matches)

# Print the unique keys found
print(env_keys)


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
            r"### environment variables - Reference(.*?)###", content, re.DOTALL
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
undocumented_keys = env_keys - documented_keys

# Print results
print("Keys expected in 'environment settings' (found in code):")
for key in sorted(env_keys):
    print(key)

if undocumented_keys:
    raise Exception(
        f"\nKeys not documented in 'environment settings - Reference': {undocumented_keys}"
    )
else:
    print(
        "\nAll keys are documented in 'environment settings - Reference'. - {}".format(
            env_keys
        )
    )
