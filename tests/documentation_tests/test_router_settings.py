import os
import re
import inspect
from typing import Type

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
        raise ValueError(f"The provided class {cls.__name__} does not have an __init__ method.")

    init_method = cls.__init__
    argspec = inspect.getfullargspec(init_method)

    # The first argument is usually 'self', so we exclude it
    return argspec.args[1:]  # Exclude 'self'


def documented_table_keys(table_content: str) -> set[str]:
    """Return the first cell of each markdown table row.

    A ``| a | b |`` search consumes the delimiter, so a 4-column reference table
    (odd pipe count) hides every other row's setting name.
    """
    keys: set[str] = set()
    for line in table_content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        first_cell = stripped.strip("|").split("|", 1)[0].strip()
        if not first_cell or set(first_cell) <= {"-", ":"} or first_cell == "Name":
            continue
        keys.add(first_cell)
    return keys


router_init_params = set(get_init_params(litellm.router.Router))
print(router_init_params)
router_init_params.remove("model_list")

# Parse the documentation to extract documented keys
_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_test_dir, "..", ".."))
print(os.listdir(_repo_root))
docs_path = os.path.join(_repo_root, "docs", "my-website", "docs", "proxy", "config_settings.md")
documented_keys: set[str] = set()
try:
    with open(docs_path, "r", encoding="utf-8") as docs_file:
        content = docs_file.read()

        # Find the section titled "router_settings - Reference"
        general_settings_section = re.search(r"### router_settings - Reference(.*?)###", content, re.DOTALL)
        if general_settings_section:
            documented_keys.update(documented_table_keys(general_settings_section.group(1)))
except Exception as e:
    raise Exception(f"Error reading documentation: {e}, \n repo base - {os.listdir(_repo_root)}")


# Compare and find undocumented keys
undocumented_keys = router_init_params - documented_keys

# Print results
print("Keys expected in 'router settings' (found in code):")
for key in sorted(router_init_params):
    print(key)

if undocumented_keys:
    raise Exception(f"\nKeys not documented in 'router settings - Reference': {undocumented_keys}")
else:
    print("\nAll keys are documented in 'router settings - Reference'. - {}".format(router_init_params))
