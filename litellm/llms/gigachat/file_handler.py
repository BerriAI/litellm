"""
GigaChat File Handler

Handles file uploads to GigaChat API for image processing.
GigaChat requires files to be uploaded first, then referenced by file_id.
"""

import base64
import hashlib
import re
import uuid
from typing import Final

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import LlmProviders

from .authenticator import get_access_token, get_access_token_async

# GigaChat API endpoint
GIGACHAT_BASE_URL: Final = "https://gigachat.devices.sberbank.ru/api/v1"

# Simple in-memory cache for file IDs
_file_cache: Final[dict[str, str]] = {}


def _get_url_hash(url: str) -> str:
    """Generate hash for URL to use as cache key."""
    return hashlib.sha256(url.encode()).hexdigest()


def _parse_data_url(data_url: str) -> tuple[bytes, str, str] | None:
    """
    Parse data URL (base64 image).

    Returns:
        Tuple of (content_bytes, content_type, extension) or None
    """
    match: Final = re.match(r"data:([^;]+);base64,(.+)", data_url)
    if not match:
        return None

    content_type: Final = match.group(1)
    base64_data: Final = match.group(2)
    content_bytes: Final = base64.b64decode(base64_data)
    ext: Final = content_type.split("/")[-1].split(";")[0] or "jpg"

    return content_bytes, content_type, ext


def _download_image_sync(url: str) -> tuple[bytes, str, str]:
    """Download image from URL synchronously."""
    client: Final = _get_httpx_client(params={"ssl_verify": False})
    response: Final = client.get(url)
    response.raise_for_status()

    content_type: Final = response.headers.get("content-type", "image/jpeg")
    ext: Final = content_type.split("/")[-1].split(";")[0] or "jpg"

    return response.content, content_type, ext


async def _download_image_async(url: str) -> tuple[bytes, str, str]:
    """Download image from URL asynchronously."""
    client: Final = get_async_httpx_client(
        llm_provider=LlmProviders.GIGACHAT,
        params={"ssl_verify": False},
    )
    response: Final = await client.get(url)
    response.raise_for_status()

    content_type: Final = response.headers.get("content-type", "image/jpeg")
    ext: Final = content_type.split("/")[-1].split(";")[0] or "jpg"

    return response.content, content_type, ext


def upload_file_sync(
    image_url: str,
    credentials: str | None = None,
    api_base: str | None = None,
) -> str | None:
    """
    Upload file to GigaChat and return file_id (sync).

    Args:
        image_url: URL or base64 data URL of the image
        credentials: GigaChat credentials for auth
        api_base: Optional custom API base URL

    Returns:
        file_id string or None if upload failed
    """
    url_hash: Final = _get_url_hash(image_url)

    # Check cache
    if url_hash in _file_cache:
        verbose_logger.debug("Image found in cache: %s...", url_hash[:16])
        return _file_cache[url_hash]

    try:
        # Get image data
        parsed: Final = _parse_data_url(image_url)
        if parsed:
            content_bytes, content_type, ext = parsed
            verbose_logger.debug("Decoded base64 image")
        else:
            verbose_logger.debug("Downloading image from URL: %s...", image_url[:80])
            content_bytes, content_type, ext = _download_image_sync(image_url)

        filename: Final = f"{uuid.uuid4()}.{ext}"

        # Get access token
        access_token: Final = get_access_token(credentials)

        # Upload to GigaChat
        base_url: Final = api_base or GIGACHAT_BASE_URL
        upload_url: Final = f"{base_url}/files"

        client: Final = _get_httpx_client(params={"ssl_verify": False})
        response: Final = client.post(
            upload_url,
            headers={"Authorization": f"Bearer {access_token}"},
            files={"file": (filename, content_bytes, content_type)},
            data={"purpose": "general"},
            timeout=60,
        )
        response.raise_for_status()
        result: Final = response.json()

        file_id: Final = result.get("id")
        if file_id:
            _file_cache[url_hash] = file_id
            verbose_logger.debug("File uploaded successfully, file_id: %s", file_id)

        return file_id

    except Exception as e:
        verbose_logger.error("Error uploading file to GigaChat: %s", e)
        return None


async def upload_file_async(
    image_url: str,
    credentials: str | None = None,
    api_base: str | None = None,
) -> str | None:
    """
    Upload file to GigaChat and return file_id (async).

    Args:
        image_url: URL or base64 data URL of the image
        credentials: GigaChat credentials for auth
        api_base: Optional custom API base URL

    Returns:
        file_id string or None if upload failed
    """
    url_hash: Final = _get_url_hash(image_url)

    # Check cache
    if url_hash in _file_cache:
        verbose_logger.debug("Image found in cache: %s...", url_hash[:16])
        return _file_cache[url_hash]

    try:
        # Get image data
        parsed: Final = _parse_data_url(image_url)
        if parsed:
            content_bytes, content_type, ext = parsed
            verbose_logger.debug("Decoded base64 image")
        else:
            verbose_logger.debug("Downloading image from URL: %s...", image_url[:80])
            content_bytes, content_type, ext = await _download_image_async(image_url)

        filename: Final = f"{uuid.uuid4()}.{ext}"

        # Get access token
        access_token: Final = await get_access_token_async(credentials)

        # Upload to GigaChat
        base_url: Final = api_base or GIGACHAT_BASE_URL
        upload_url: Final = f"{base_url}/files"

        client: Final = get_async_httpx_client(
            llm_provider=LlmProviders.GIGACHAT,
            params={"ssl_verify": False},
        )
        response: Final = await client.post(
            upload_url,
            headers={"Authorization": f"Bearer {access_token}"},
            files={"file": (filename, content_bytes, content_type)},
            data={"purpose": "general"},
            timeout=60,
        )
        response.raise_for_status()
        result: Final = response.json()

        file_id: Final = result.get("id")
        if file_id:
            _file_cache[url_hash] = file_id
            verbose_logger.debug("File uploaded successfully, file_id: %s", file_id)

        return file_id

    except Exception as e:
        verbose_logger.error("Error uploading file to GigaChat: %s", e)
        return None
