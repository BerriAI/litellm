import os
import re
import inspect
import sys
from pathlib import Path
from typing import Type

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


def get_init_params(cls: Type) -> list[str]:
    """
    Retrieve all parameters supported by the `__init__` method of a given class.

    Args:
        cls: The class to inspect.

    Returns:
        A list of parameter names.
    """
    if not hasattr(cls, "__init__"):
        raise ValueError(
            f"The provided class {cls.__name__} does not have an __init__ method."
        )

    init_method = cls.__init__
    argspec = inspect.getfullargspec(init_method)

    # The first argument is usually 'self', so we exclude it
    return argspec.args[1:]  # Exclude 'self'


router_init_params = set(get_init_params(litellm.router.Router))
print(router_init_params)
router_init_params.remove("model_list")

# Parse the documentation to extract documented keys
repo_base = Path(__file__).resolve().parents[2]
print(os.listdir(repo_base))
docs_path = repo_base / "docs" / "my-website" / "docs" / "proxy" / "config_settings.md"
documented_keys = set()
try:
    with open(docs_path, "r", encoding="utf-8") as docs_file:
        content = docs_file.read()

        # Find the section titled "general_settings - Reference"
        general_settings_section = re.search(
            r"### router_settings - Reference(.*?)###", content, re.DOTALL
        )
        if general_settings_section:
            # Extract the table rows, which contain the documented keys
            table_content = general_settings_section.group(1)
            for line in table_content.splitlines():
                stripped = line.strip()
                if not stripped.startswith("|"):
                    continue
                columns = [column.strip() for column in stripped.strip("|").split("|")]
                if len(columns) < 2:
                    continue
                if columns[0] == "Name" or set(columns[0]) == {"-"}:
                    continue
                documented_keys.add(columns[0])
except Exception as e:
    raise Exception(
        f"Error reading documentation: {e}, \n repo base - {os.listdir(repo_base)}"
    )


# Compare and find undocumented keys
undocumented_keys = router_init_params - documented_keys

# Print results
print("Keys expected in 'router settings' (found in code):")
for key in sorted(router_init_params):
    print(key)

if undocumented_keys:
    raise Exception(
        f"\nKeys not documented in 'router settings - Reference': {undocumented_keys}"
    )
else:
    print(
        "\nAll keys are documented in 'router settings - Reference'. - {}".format(
            router_init_params
        )
    )
