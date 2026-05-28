"""Helpers for unauthenticated logo / favicon endpoints."""

import os
from typing import Optional, Tuple

from litellm._logging import verbose_proxy_logger

LOCAL_IMAGE_HEADER_BYTES = 512


_SVG_TAG_BOUNDARY = (b" ", b"\t", b"\r", b"\n", b">", b"/")


def _is_svg_header(header: bytes) -> bool:
    """Detect SVG by checking for an ``<svg`` root tag in the leading bytes.

    SVGs may begin with an optional UTF-8 BOM, leading whitespace, an XML
    declaration (``<?xml … ?>``), an SVG-specific ``<!DOCTYPE svg …>``,
    or an XML comment prologue before the root ``<svg>`` element. We accept
    those prologues but only return True when an ``<svg`` *tag* actually
    appears in the leading window and no ``<html`` / ``<!doctype html`` marker
    precedes it, so HTML pages (even those that embed an inline SVG) and
    unrelated XML payloads — including elements whose names merely *start*
    with ``svg`` (e.g. ``<svgIcon>``) — are never misclassified as SVG.
    """
    if not header:
        return False
    # Tolerate the optional UTF-8 BOM.
    if header.startswith(b"\xef\xbb\xbf"):
        header = header[3:]
    stripped = header.lstrip()
    if not stripped:
        return False
    # Bare ``<svg`` root tag — require a tag-boundary character after ``svg``
    # so element names that happen to start with ``svg`` (e.g. ``<svgIcon>``)
    # are not misclassified as the SVG root.
    if stripped[:4] == b"<svg" and stripped[4:5] in _SVG_TAG_BOUNDARY:
        return True
    lowered = header.lower()
    # Never accept HTML pages, even if they embed an inline ``<svg>`` element.
    if b"<html" in lowered or b"<!doctype html" in lowered:
        return False
    if (
        stripped[:5].lower() == b"<?xml"
        or stripped[:13].lower() == b"<!doctype svg"
        or stripped[:4] == b"<!--"
    ):
        idx = 0
        while True:
            idx = lowered.find(b"<svg", idx)
            if idx == -1:
                return False
            if lowered[idx + 4 : idx + 5] in _SVG_TAG_BOUNDARY:
                return True
            idx += 4
    return False


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
    if _is_svg_header(header):
        return "image/svg+xml"
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
