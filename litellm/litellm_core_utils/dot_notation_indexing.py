"""
Path-based navigation utilities for nested dictionaries.

This module provides utilities for reading and deleting values in nested
dictionaries using dot notation and JSONPath array syntax.

Uses jsonpath-ng library for standard JSONPath parsing and navigation.

Supported syntax:
- "field" - top-level field
- "parent.child" - nested field
- "array[*]" - all array elements (wildcard)
- "array[0]" - specific array element (index)
- "array[*].field" - field in all array elements

Examples:
    >>> data = {"tools": [{"name": "t1", "input_examples": ["ex"]}]}
    >>> delete_nested_value(data, "tools[*].input_examples")
    {"tools": [{"name": "t1"}]}

Used by JWT Auth to get the user role from the token, and by
additional_drop_params to remove nested fields from optional parameters.
"""

from typing import Any, Dict, Optional, TypeVar

T = TypeVar("T")


def get_nested_value(
    data: Dict[str, Any], key_path: str, default: Optional[T] = None
) -> Optional[T]:
    """
    Retrieves a value from a nested dictionary using dot notation.

    Args:
        data: The dictionary to search in
        key_path: The path to the value using dot notation (e.g., "a.b.c")
        default: The default value to return if the path is not found

    Returns:
        The value at the specified path, or the default value if not found

    Example:
        >>> data = {"a": {"b": {"c": "value"}}}
        >>> get_nested_value(data, "a.b.c")
        'value'
        >>> get_nested_value(data, "a.b.d", "default")
        'default'
    """
    if not key_path:
        return default

    # Remove metadata. prefix if it exists
    key_path = (
        key_path.replace("metadata.", "", 1)
        if key_path.startswith("metadata.")
        else key_path
    )

    # Split the key path into parts
    parts = key_path.split(".")

    # Traverse through the dictionary
    current: Any = data
    for part in parts:
        try:
            current = current[part]
        except (KeyError, TypeError):
            return default

    # If default is None, we can return any type
    if default is None:
        return current

    # Otherwise, ensure the type matches the default
    return current if isinstance(current, type(default)) else default


def delete_nested_value(
    data: Dict[str, Any],
    path: str,
    depth: int = 0,
    max_depth: int = 20,
) -> Dict[str, Any]:
    """
    Delete a field from nested data using JSONPath notation.

    Uses jsonpath-ng library for standard JSONPath parsing.

    Supports:
    - "field" - top-level field
    - "parent.child" - nested field
    - "array[*]" - all array elements
    - "array[0]" - specific array element
    - "array[*].field" - field in all array elements

    Args:
        data: Dictionary to modify (creates deep copy)
        path: JSONPath-like path string
        depth: Current recursion depth (kept for API compatibility)
        max_depth: Maximum recursion depth (kept for API compatibility)

    Returns:
        New dictionary with field removed at path

    Example:
        >>> data = {"tools": [{"name": "t1", "input_examples": ["ex"]}]}
        >>> delete_nested_value(data, "tools[*].input_examples")
        {"tools": [{"name": "t1"}]}
    """
    import copy

    from jsonpath_ng import parse

    result = copy.deepcopy(data)

    # Add $ prefix required by jsonpath-ng
    if not path.startswith("$"):
        path = f"$.{path}"

    try:
        expr = parse(path)
        matches = expr.find(result)

        # Process matches in reverse to handle array deletions correctly
        for match in reversed(matches):
            parent = match.context.value if match.context else result

            if isinstance(parent, list):
                if hasattr(match.path, "index"):
                    idx = match.path.index
                    if 0 <= idx < len(parent):
                        parent.pop(idx)
            elif isinstance(parent, dict):
                if hasattr(match.path, "fields") and match.path.fields:
                    field = match.path.fields[0]
                    parent.pop(field, None)
    except Exception:
        # Invalid path or parsing error - silently skip
        pass

    return result


def is_nested_path(path: str) -> bool:
    """
    Check if path requires nested handling.

    Returns True if path contains '.' or '[' (array notation).
    """
    return "." in path or "[" in path
