import ast
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm

SEARCH_PROVIDERS = [
    "tavily",
    "dataforseo",
    "google_pse",
    "parallel_ai",
    "exa_ai",
    "brave",
    "firecrawl",
    "searxng",
    "linkup",
]

ALLOWED_FILES_IN_LLMS_FOLDER = [
    "__init__",
    "base",
    "base_llm",
    "custom_httpx",
    "custom_llm",
    "deprecated_providers",
    "pass_through"
] + SEARCH_PROVIDERS


def get_unique_names_from_llms_dir(base_dir: str):
    """
    Returns a set of unique file and folder names from the root level of litellm/llms directory,
    excluding file extensions and __init__.py
    """
    unique_names = set()

    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist.")
        return unique_names

    # Get only root level items
    items = os.listdir(base_dir)

    for item in items:
        item_path = os.path.join(base_dir, item)

        if os.path.isdir(item_path):
            if item != "__pycache__":
                unique_names.add(item)
        elif item.endswith(".py") and item != "__init__.py":
            name_without_ext = os.path.splitext(item)[0]
            unique_names.add(name_without_ext)

    return unique_names


def run_lint_check(unique_names):
    _all_litellm_providers = [str(provider.value) for provider in litellm.LlmProviders]
    violations = []
    for name in unique_names:
        if (
            name.lower() not in _all_litellm_providers
            and name not in ALLOWED_FILES_IN_LLMS_FOLDER
        ):
            violations.append(name)

    if len(violations) > 0:
        raise ValueError(
            f"There are {len(violations)} violations in the llms folder. \n\n {violations}. \n\n Valid providers: {_all_litellm_providers}"
        )


def main():
    llms_dir = "./litellm/llms/"  # Update this path if needed
    # llms_dir = "../../litellm/llms/"  # LOCAL TESTING

    unique_names = get_unique_names_from_llms_dir(llms_dir)
    print("Unique names in llms directory:", sorted(list(unique_names)))
    run_lint_check(unique_names)


if __name__ == "__main__":
    main()
