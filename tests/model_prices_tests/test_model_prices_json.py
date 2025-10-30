#!/usr/bin/env python3

"""
Script to lint and format "model_prices_and_context_window.json" files.

Models are sorted alphabetically. Small floats are formatted in e-06
notation to make prices easier to interpret ('per million') and to
enforce consistency across models.

usage:
    - Lint: `make lint-model-prices`
    - Format: `make format-model-prices`

options:
  -h, --help  show the help message and exit
  --format    Format non-compliant files.
"""

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

FILES_TO_PROCESS: List[Path] = [
    Path("./model_prices_and_context_window.json"),
    Path("./litellm/model_prices_and_context_window_backup.json"),
]

# Floats below this value are formatted in e-06 notation
TINY_FLOAT_THRESHOLD = 0.001


def _detect_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    seen_keys = set()
    output_dict = {}

    for key, value in pairs:
        if key in seen_keys:
            raise ValueError(f"Duplicate key found: '{key}'")

        seen_keys.add(key)
        output_dict[key] = value

    return output_dict


# Internal placeholders used to bypass standard JSON float formatting
_PLACEHOLDER_PREFIX = "_TINYFLOAT_PLACEHOLDER_START_"
_PLACEHOLDER_SUFFIX = "_TINYFLOAT_PLACEHOLDER_END_"


class CustomJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that replaces placeholder strings with unquoted numbers.

    This allows for specific float formatting not supported by the standard
    JSON encoder (e.g., 1.5e-06) to make it obvious what the cost of a model
    is 'per million'.
    """

    def encode(self, o: Any) -> str:
        s = super().encode(o)
        # Replace quoted placeholders with the unquoted number
        s = s.replace(f'"{_PLACEHOLDER_PREFIX}', "")
        s = s.replace(f'{_PLACEHOLDER_SUFFIX}"', "")
        return s


JSONData = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def _format_numbers(item: JSONData) -> JSONData:
    """
    Apply custom number formatting rules.
    """
    # Format objects recursively
    if isinstance(item, dict):
        return {k: _format_numbers(v) for k, v in item.items()}

    if isinstance(item, list):
        return [_format_numbers(x) for x in item]

    if isinstance(item, float):
        # Format whole numbers (1.0) as ints (1)
        if item.is_integer():
            return int(item)

        # Format small floats as e-06 for 'per million'-style pricing
        # e.g., 0.0000015 -> "1.5e-06" (to represent $1.50 per million)
        if 0 < item <= TINY_FLOAT_THRESHOLD:
            formatted_float_str = f"{item * 1e6:g}e-06"
            return f"{_PLACEHOLDER_PREFIX}{formatted_float_str}{_PLACEHOLDER_SUFFIX}"

        return item

    # Other types
    return item


def format_price_file(file_path: Path) -> str:
    """
    Sort keys alphatebically and format numbers, but preserve 'sample_spec'.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=_detect_duplicates)

    # Pop sample spec
    try:
        sample_spec = data.pop("sample_spec")
    except KeyError:
        raise ValueError("Missing 'sample_spec' key")

    processed_models = {}

    for model_name in sorted(data.keys()):
        # Format numbers within each model
        model_data = _format_numbers(data[model_name])

        # Sort attributes within each model
        if isinstance(model_data, dict):
            processed_models[model_name] = {
                key: model_data[key] for key in sorted(model_data.keys())
            }
        else:
            raise Exception(f"Invalid model: {model_name}")

    # Assemble the data with 'sample_spec' at the top
    final_data = {"sample_spec": sample_spec, **processed_models}
    output = json.dumps(final_data, indent=4, cls=CustomJSONEncoder, ensure_ascii=False)

    return output + "\n"


def _shout(message: str, file=sys.stdout):
    print("\n")
    print("=" * 64)
    print(message, file=file)
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="A linter and formatter for 'model_prices_and_context_window' JSON files."
    )
    parser.add_argument(
        "--format",
        action="store_true",
        help="Format non-compliant files.",
    )
    args = parser.parse_args()

    updated_files: List[Path] = []
    non_compliant_files: List[Path] = []
    files_with_errors: List[Path] = []

    for file_path in FILES_TO_PROCESS:
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            original_content = file_path.read_text(encoding="utf-8")
            formatted_content = format_price_file(file_path)

        except Exception as e:
            print(f"\n❌ Error processing '{file_path}':\n    - {e}", file=sys.stderr)
            files_with_errors.append(file_path)
            continue

        if original_content != formatted_content:
            if args.format:
                file_path.write_text(formatted_content, encoding="utf-8")
                updated_files.append(file_path)
            else:
                print(f"\n❌ File is not compliant: '{file_path}'", file=sys.stderr)
                non_compliant_files.append(file_path)

                # Generate and print a diff to show the required changes
                diff = difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    formatted_content.splitlines(keepends=True),
                    fromfile=f"{file_path} (original)",
                    tofile=f"{file_path} (formatted)",
                )
                sys.stderr.writelines(diff)

    if args.format:
        if updated_files:
            _shout(
                f"✅ Formatted {len(updated_files)} file(s):\n"
                + "\n".join(f"    - {p}" for p in updated_files)
            )

        if files_with_errors:
            _shout(
                f"❌ {len(files_with_errors)} file(s) coudln't be fixed:\n"
                + "\n".join(f"    - {p}" for p in files_with_errors),
                file=sys.stderr,
            )
    else:
        if non_compliant_files or files_with_errors:
            count = len(non_compliant_files) + len(files_with_errors)
            _shout(
                f"❌ Found {count} non-compliant file(s).\n   Fix them using: `make format-model-prices`.",
                file=sys.stderr,
            )

    if files_with_errors or non_compliant_files:
        sys.exit(1)

    print("\n✅ All good!")
    sys.exit(0)


if __name__ == "__main__":
    main()
