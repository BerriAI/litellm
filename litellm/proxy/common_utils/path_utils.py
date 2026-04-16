"""
Safe filesystem path construction for user-controlled inputs.

Use safe_join() instead of os.path.join() whenever a path component
comes from user input (request parameters, uploaded filenames, etc.)
to prevent directory traversal attacks.
"""

import os
from pathlib import Path


def safe_join(base_dir: str, *parts: str) -> str:
    """
    Join path components and verify the result stays within base_dir.

    Resolves symlinks and '..' sequences, then checks the final path
    is a descendant of base_dir. Raises ValueError if traversal is
    detected.

    Args:
        base_dir: The trusted base directory.
        *parts: User-controlled path components to append.

    Returns:
        The resolved absolute path as a string.

    Raises:
        ValueError: If the resolved path escapes base_dir.
    """
    base = os.path.realpath(base_dir)
    resolved = os.path.realpath(os.path.join(base, *parts))
    if not (resolved.startswith(base + os.sep) or resolved == base):
        raise ValueError(f"Path escapes base directory")
    return resolved


def safe_filename(filename: str) -> str:
    """
    Extract a safe filename from a user-supplied path.

    Strips all directory components, returning only the final name.
    Use this for uploaded file names before writing to disk.

    Args:
        filename: User-supplied filename (may contain path separators).

    Returns:
        The basename only, with no directory components.

    Raises:
        ValueError: If the resulting filename is empty.
    """
    name = Path(filename).name
    if not name:
        raise ValueError("Empty filename")
    return name
