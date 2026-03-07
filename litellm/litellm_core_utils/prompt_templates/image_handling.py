"""
Helper functions to handle images passed in messages
"""

import base64
from typing import Optional

from httpx import Response

import litellm
from litellm import verbose_logger
from litellm.caching.caching import InMemoryCache
from litellm.constants import MAX_IMAGE_URL_DOWNLOAD_SIZE_MB

MAX_IMGS_IN_MEMORY = 10

in_memory_cache = InMemoryCache(max_size_in_memory=MAX_IMGS_IN_MEMORY)


# Supported image media types for Anthropic/Vertex AI
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Mapping from file extensions to media types
EXTENSION_TO_MEDIA_TYPE = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def _infer_media_type_from_url(url: str) -> str:
    """
    Infer media type from URL extension.
    Handles signed URLs by stripping query parameters before extracting extension.
    """
    # Strip query parameters for signed URLs (e.g., ?x-oss-signature=...)
    url_without_query = url.split("?")[0]
    # Only take the last path segment to avoid matching dots in the path
    last_segment = url_without_query.rstrip("/").split("/")[-1]
    # Check if the segment contains a dot (has an extension)
    extension = last_segment.rsplit(".", 1)[-1].lower() if "." in last_segment else ""
    media_type = EXTENSION_TO_MEDIA_TYPE.get(extension)
    if media_type is None:
        raise Exception(
            f"Error: Unsupported image format. Could not infer media type from URL '{url}'. "
            f"Supported types = {list(SUPPORTED_IMAGE_TYPES)}"
        )
    return media_type


def _get_valid_media_type(content_type: Optional[str], url: str) -> str:
    """
    Get a valid media type from Content-Type header, falling back to URL extension.

    - Strips Content-Type parameters (e.g., 'image/png; charset=utf-8' -> 'image/png')
    - Validates against supported types
    - Falls back to URL extension inference if Content-Type is missing or invalid
    """
    if content_type is not None:
        # Strip parameters (e.g., "image/png; charset=utf-8" -> "image/png")
        media_type = content_type.split(";")[0].strip()
        if media_type in SUPPORTED_IMAGE_TYPES:
            return media_type
        # Content-Type is invalid (e.g., application/octet-stream, application/x-www-form-urlencoded)
        # Fall through to URL extension inference

    # Fallback to URL extension
    return _infer_media_type_from_url(url)


def _process_image_response(response: Response, url: str) -> str:
    if response.status_code != 200:
        raise litellm.ImageFetchError(
            f"Error: Unable to fetch image from URL. Status code: {response.status_code}, url={url}"
        )

    # Check size before downloading if Content-Length header is present
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > MAX_IMAGE_URL_DOWNLOAD_SIZE_MB:
            raise litellm.ImageFetchError(
                f"Error: Image size ({size_mb:.2f}MB) exceeds maximum allowed size ({MAX_IMAGE_URL_DOWNLOAD_SIZE_MB}MB). url={url}"
            )

    # Stream download with size checking to prevent downloading huge files
    max_bytes = int(MAX_IMAGE_URL_DOWNLOAD_SIZE_MB * 1024 * 1024)
    image_bytes = bytearray()
    bytes_downloaded = 0

    for chunk in response.iter_bytes(chunk_size=8192):
        bytes_downloaded += len(chunk)
        if bytes_downloaded > max_bytes:
            size_mb = bytes_downloaded / (1024 * 1024)
            raise litellm.ImageFetchError(
                f"Error: Image size ({size_mb:.2f}MB) exceeds maximum allowed size ({MAX_IMAGE_URL_DOWNLOAD_SIZE_MB}MB). url={url}"
            )
        image_bytes.extend(chunk)

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    content_type = response.headers.get("Content-Type")
    media_type = _get_valid_media_type(content_type, url)

    result = f"data:{media_type};base64,{base64_image}"
    in_memory_cache.set_cache(url, result)
    return result


async def async_convert_url_to_base64(url: str) -> str:
    # If MAX_IMAGE_URL_DOWNLOAD_SIZE_MB is 0, block all image downloads
    if MAX_IMAGE_URL_DOWNLOAD_SIZE_MB == 0:
        raise litellm.ImageFetchError(
            f"Error: Image URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0). url={url}"
        )

    cached_result = in_memory_cache.get_cache(url)
    if cached_result:
        return cached_result

    client = litellm.module_level_aclient
    for _ in range(3):
        try:
            response = await client.get(url, follow_redirects=True)
            return _process_image_response(response, url)
        except litellm.ImageFetchError:
            raise
        except Exception:
            pass
    raise litellm.ImageFetchError(
        f"Error: Unable to fetch image from URL after 3 attempts. url={url}"
    )


def convert_url_to_base64(url: str) -> str:
    # If MAX_IMAGE_URL_DOWNLOAD_SIZE_MB is 0, block all image downloads
    if MAX_IMAGE_URL_DOWNLOAD_SIZE_MB == 0:
        raise litellm.ImageFetchError(
            f"Error: Image URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0). url={url}"
        )

    cached_result = in_memory_cache.get_cache(url)
    if cached_result:
        return cached_result

    client = litellm.module_level_client
    for _ in range(3):
        try:
            response = client.get(url, follow_redirects=True)
            return _process_image_response(response, url)
        except litellm.ImageFetchError:
            raise
        except Exception as e:
            verbose_logger.exception(e)
            pass
    raise litellm.ImageFetchError(
        f"Error: Unable to fetch image from URL after 3 attempts. url={url}",
    )
