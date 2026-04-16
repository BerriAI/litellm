"""
Safe filesystem path construction for user-controlled inputs.

Use safe_join() instead of os.path.join() whenever a path component
comes from user input (request parameters, uploaded filenames, etc.)
to prevent directory traversal attacks.
"""

import os


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
    for part in parts:
        if "\x00" in part:
            raise ValueError("Path contains null byte")
    base = os.path.realpath(base_dir)
    resolved = os.path.realpath(os.path.join(base, *parts))
    if not (resolved.startswith(base + os.sep) or resolved == base):
        raise ValueError(f"Path {resolved!r} escapes base directory {base!r}")
    return resolved


def safe_filename(filename: str) -> str:
    """
    Extract a safe filename from a user-supplied path.

    Strips all directory components (both Unix and Windows separators),
    returning only the final name. Use this for uploaded file names
    before writing to disk.

    Args:
        filename: User-supplied filename (may contain path separators).

    Returns:
        The basename only, with no directory components.

    Raises:
        ValueError: If the resulting filename is empty or contains null bytes.
    """
    if "\x00" in filename:
        raise ValueError("Filename contains null byte")
    # Normalize backslash separators for cross-platform safety
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in (".", ".."):
        raise ValueError("Empty or unsafe filename")
    return name
