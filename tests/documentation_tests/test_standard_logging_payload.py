import os
import re
import sys

from typing import get_type_hints

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.types.utils import StandardLoggingPayload


def get_all_fields(type_dict, prefix=""):
    """Recursively get all fields from TypedDict and its nested types"""
    fields = set()

    # Get type hints for the TypedDict
    hints = get_type_hints(type_dict)

    for field_name, field_type in hints.items():
        full_field_name = f"{prefix}{field_name}" if prefix else field_name
        fields.add(full_field_name)

        # Check if the field type is another TypedDict we should process
        if hasattr(field_type, "__annotations__"):
            nested_fields = get_all_fields(field_type)
            fields.update(nested_fields)
    return fields


def test_standard_logging_payload_documentation():
    # Get all fields from StandardLoggingPayload and its nested types
    all_fields = get_all_fields(StandardLoggingPayload)

    print("All fields in StandardLoggingPayload: ")
    for _field in all_fields:
        print(_field)

    # Read the documentation
    docs_path = "../../docs/my-website/docs/proxy/logging_spec.md"

    try:
        with open(docs_path, "r", encoding="utf-8") as docs_file:
            content = docs_file.read()

            # Extract documented fields from the table
            doc_field_pattern = re.compile(r"\|\s*`([^`]+?)`\s*\|")
            documented_fields = set(doc_field_pattern.findall(content))

            # Clean up documented fields (remove whitespace)
            documented_fields = {field.strip() for field in documented_fields}

            # Clean up documented fields (remove whitespace)
            documented_fields = {field.strip() for field in documented_fields}
            print("\n\nDocumented fields: ")
            for _field in documented_fields:
                print(_field)

            # Compare and find undocumented fields
            undocumented_fields = all_fields - documented_fields

            print("\n\nUndocumented fields: ")
            for _field in undocumented_fields:
                print(_field)

            if undocumented_fields:
                raise Exception(
                    f"\nFields not documented in 'StandardLoggingPayload': {undocumented_fields}"
                )

            print(
                f"All {len(all_fields)} fields are documented in 'StandardLoggingPayload'"
            )

    except FileNotFoundError:
        raise Exception(
            f"Documentation file not found at {docs_path}. Please ensure the documentation exists."
        )
