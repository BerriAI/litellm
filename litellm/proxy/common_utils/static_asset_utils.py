"""Helpers for unauthenticated logo / favicon endpoints."""

import os
from typing import Optional, Tuple

from litellm._logging import verbose_proxy_logger

LOCAL_IMAGE_HEADER_BYTES = 512


def detect_local_image_media_type(header: bytes) -> Optional[str]:
    """Return a browser image media type for supported local image signatures."""
    if header[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[0:4] == b"GIF8" and header[5:6] == b"a":
        return "image/gif"
    if header[0:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if header[0:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header[0:4] in (b"\x00\x00\x01\x00", b"\x00\x00\x02\x00"):
        return "image/x-icon"
    return None


def resolve_validated_local_image_path(candidate: str) -> Optional[Tuple[str, str]]:
    """Resolve ``candidate`` only when it is an existing supported image file."""
    if not candidate:
        return None
    try:
        resolved = os.path.realpath(os.path.expanduser(candidate))
    except (OSError, ValueError):
        return None
    if not os.path.isfile(resolved):
        return None

    try:
        with open(resolved, "rb") as f:
            header = f.read(LOCAL_IMAGE_HEADER_BYTES)
    except OSError as exc:
        verbose_proxy_logger.debug("Could not read local asset %r: %s", candidate, exc)
        return None

    media_type = detect_local_image_media_type(header)
    if media_type is None:
        verbose_proxy_logger.warning(
            "Local asset %r is not a supported image file; falling back to default.",
            candidate,
        )
        return None

    return resolved, media_type
