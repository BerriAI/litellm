import mimetypes
from typing import Optional, Tuple


def normalize_logo_content_type(content_type: str) -> str:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized == "image/svg":
        return "image/svg+xml"
    if normalized == "image/jpg":
        return "image/jpeg"
    return normalized


def infer_logo_content_type(
    path: str, default: str = "image/jpeg"
) -> Tuple[str, Optional[str]]:
    """
    Infer logo content type + optional content encoding from a path/URL.

    Note: Python's built-in `mimetypes` can be inconsistent across OSes for SVG,
    so handle it explicitly.
    """
    lower_path = (path or "").lower()
    if lower_path.endswith((".svg", ".svgz")):
        return "image/svg+xml", "gzip" if lower_path.endswith(".svgz") else None

    media_type, _encoding = mimetypes.guess_type(path)
    return normalize_logo_content_type(media_type or default), None


def cache_extension_for_logo(
    content_type: str, content_encoding: Optional[str] = None
) -> str:
    normalized = normalize_logo_content_type(content_type)
    if normalized == "image/svg+xml":
        if (content_encoding or "").lower() == "gzip":
            return ".svgz"
        return ".svg"
    return mimetypes.guess_extension(normalized) or ".img"
