"""
Path-based navigation utilities for nested dictionaries.

This module provides utilities for reading and deleting values in nested
dictionaries using dot notation and JSONPath-like array syntax.

Custom implementation with zero external dependencies.

Supported syntax:
- "field" - top-level field
- "parent.child" - nested field
- "parent\\.with\\.dots.child" - keys containing dots (escape with backslash)
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

from typing import Any, Dict, List, Optional, TypeVar, Union

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
        >>> data = {"kubernetes.io": {"namespace": "default"}}
        >>> get_nested_value(data, "kubernetes\\.io.namespace")
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

    # Split the key path into parts, respecting escaped dots (\.)
    # Use a temporary placeholder, split on unescaped dots, then restore
    placeholder = "\x00"
    parts = key_path.replace("\\.", placeholder).split(".")
    parts = [p.replace(placeholder, ".") for p in parts]

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


def _parse_path_segments(path: str) -> list:
    """
    Parse a JSONPath-like string into segments using regex.

    Handles:
    - Dot notation: "a.b.c" → ["a", "b", "c"]
    - Array wildcards: "a[*].b" → ["a", "[*]", "b"]
    - Array indices: "a[0].b" → ["a", "[0]", "b"]

    Args:
        path: JSONPath-like path string

    Returns:
        List of path segments

    Example:
        >>> _parse_path_segments("tools[*].arr[0].field")
        ["tools", "[*]", "arr", "[0]", "field"]
    """
    import re

    # Match field names OR bracket expressions
    # Pattern: field_name (anything except . or [) | [anything_in_brackets]
    pattern = r'[^\.\[]+|\[[^\]]*\]'
    segments = re.findall(pattern, path)
    return segments


def _delete_nested_value_custom(
    data: Union[Dict[str, Any], List[Any]],
    segments: list,
    segment_index: int = 0,
) -> None:
    """
    Recursively delete a field from nested data using parsed segments.

    Modifies data in-place (caller must deep copy first).

    Args:
        data: Dictionary or list to modify
        segments: Parsed path segments
        segment_index: Current position in segments list
    """
    if segment_index >= len(segments):
        return

    segment = segments[segment_index]
    is_last = segment_index == len(segments) - 1

    # Handle array wildcard: [*]
    if segment == "[*]":
        if isinstance(data, list):
            for item in data:
                if is_last:
                    # Can't delete array elements themselves, skip
                    pass
                else:
                    # Only recurse if item is a dict or list (nested structure)
                    if isinstance(item, (dict, list)):
                        _delete_nested_value_custom(item, segments, segment_index + 1)
        return

    # Handle array index: [0], [1], [2], etc.
    if segment.startswith("[") and segment.endswith("]"):
        try:
            index = int(segment[1:-1])
            if isinstance(data, list) and 0 <= index < len(data):
                if is_last:
                    # Can't delete array elements themselves, skip
                    pass
                else:
                    # Only recurse if element is a dict or list (nested structure)
                    element = data[index]
                    if isinstance(element, (dict, list)):
                        _delete_nested_value_custom(element, segments, segment_index + 1)
        except (ValueError, IndexError):
            # Invalid index, skip
            pass
        return

    # Handle regular field navigation
    if isinstance(data, dict):
        if is_last:
            # Delete the field
            data.pop(segment, None)
        else:
            # Navigate deeper
            if segment in data:
                next_segment = segments[segment_index + 1] if segment_index + 1 < len(segments) else None

                # If next segment is array notation, current field should be list
                if next_segment and (next_segment.startswith("[")):
                    if isinstance(data[segment], list):
                        _delete_nested_value_custom(data[segment], segments, segment_index + 1)
                # Otherwise navigate into dict
                elif isinstance(data[segment], dict):
                    _delete_nested_value_custom(data[segment], segments, segment_index + 1)


def delete_nested_value(
    data: Dict[str, Any],
    path: str,
    depth: int = 0,
    max_depth: int = 20,
) -> Dict[str, Any]:
    """
    Delete a field from nested data using JSONPath notation.

    Custom implementation - no external dependencies.

    Supports:
    - "field" - top-level field
    - "parent.child" - nested field
    - "array[*]" - all array elements (wildcard)
    - "array[0]" - specific array element (index)
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

    result = copy.deepcopy(data)

    try:
        # Parse path into segments
        segments = _parse_path_segments(path)

        if not segments:
            return result

        # Delete using custom recursive implementation
        _delete_nested_value_custom(result, segments, 0)

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
