import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import re

# Backup the original sys.path
original_sys_path = sys.path.copy()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm

public_exceptions = litellm.LITELLM_EXCEPTION_TYPES
# Regular expression to extract the error name
error_name_pattern = re.compile(r"\.exceptions\.([A-Za-z]+Error)")

# Extract error names from each item
error_names = {
    error_name_pattern.search(str(item)).group(1)
    for item in public_exceptions
    if error_name_pattern.search(str(item))
}


# sys.path = original_sys_path


# Parse the documentation to extract documented keys
# repo_base = "./"
repo_base = "../../"
print(os.listdir(repo_base))
docs_path = f"{repo_base}/docs/my-website/docs/exception_mapping.md"  # Path to the documentation
documented_keys = set()
try:
    with open(docs_path, "r", encoding="utf-8") as docs_file:
        content = docs_file.read()

        exceptions_section = re.search(
            r"## LiteLLM Exceptions(.*?)\n##", content, re.DOTALL
        )
        if exceptions_section:
            # Step 2: Extract the table content
            table_content = exceptions_section.group(1)

            # Step 3: Create a pattern to capture the Error Types from each row
            error_type_pattern = re.compile(r"\|\s*[^|]+\s*\|\s*([^\|]+?)\s*\|")

            # Extract the error types
            exceptions = error_type_pattern.findall(table_content)
            print(f"exceptions: {exceptions}")

            # Remove extra spaces if any
            exceptions = [exception.strip() for exception in exceptions]

            print(exceptions)
            documented_keys.update(exceptions)

except Exception as e:
    raise Exception(
        f"Error reading documentation: {e}, \n repo base - {os.listdir(repo_base)}"
    )

print(documented_keys)
print(public_exceptions)
print(error_names)

# Compare and find undocumented keys
undocumented_keys = error_names - documented_keys

if undocumented_keys:
    raise Exception(
        f"\nKeys not documented in 'LiteLLM Exceptions': {undocumented_keys}"
    )
else:
    print("\nAll keys are documented in 'LiteLLM Exceptions'. - {}".format(error_names))
