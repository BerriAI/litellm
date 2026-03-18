"""
GigaChat File Handler

Handles file uploads to GigaChat API for image processing.
GigaChat requires files to be uploaded first, then referenced by file_id.
"""

import base64
import hashlib
import re
import uuid
from typing import Dict, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import LlmProviders

from .authenticator import get_access_token, get_access_token_async

# GigaChat API endpoint
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"

# Simple in-memory cache for file IDs
_file_cache: Dict[str, str] = {}


def _get_url_hash(url: str) -> str:
    """Generate hash for URL to use as cache key."""
    return hashlib.sha256(url.encode()).hexdigest()


def _parse_data_url(data_url: str) -> Optional[Tuple[bytes, str, str]]:
    """
    Parse data URL (base64 image).

    Returns:
        Tuple of (content_bytes, content_type, extension) or None
    """
    match = re.match(r"data:([^;]+);base64,(.+)", data_url)
    if not match:
        return None

    content_type = match.group(1)
    base64_data = match.group(2)
    content_bytes = base64.b64decode(base64_data)
    ext = content_type.split("/")[-1].split(";")[0] or "jpg"

    return content_bytes, content_type, ext


def _download_image_sync(url: str) -> Tuple[bytes, str, str]:
    """Download image from URL synchronously."""
    client = _get_httpx_client(params={"ssl_verify": False})
    response = client.get(url)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "image/jpeg")
    ext = content_type.split("/")[-1].split(";")[0] or "jpg"

    return response.content, content_type, ext


async def _download_image_async(url: str) -> Tuple[bytes, str, str]:
    """Download image from URL asynchronously."""
    client = get_async_httpx_client(
        llm_provider=LlmProviders.GIGACHAT,
        params={"ssl_verify": False},
    )
    response = await client.get(url)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "image/jpeg")
    ext = content_type.split("/")[-1].split(";")[0] or "jpg"

    return response.content, content_type, ext


def upload_file_sync(
    image_url: str,
    credentials: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[str]:
    """
    Upload file to GigaChat and return file_id (sync).

    Args:
        image_url: URL or base64 data URL of the image
        credentials: GigaChat credentials for auth
        api_base: Optional custom API base URL

    Returns:
        file_id string or None if upload failed
    """
    url_hash = _get_url_hash(image_url)

    # Check cache
    if url_hash in _file_cache:
        verbose_logger.debug(f"Image found in cache: {url_hash[:16]}...")
        return _file_cache[url_hash]

    try:
        # Get image data
        parsed = _parse_data_url(image_url)
        if parsed:
            content_bytes, content_type, ext = parsed
            verbose_logger.debug("Decoded base64 image")
        else:
            verbose_logger.debug(f"Downloading image from URL: {image_url[:80]}...")
            content_bytes, content_type, ext = _download_image_sync(image_url)

        filename = f"{uuid.uuid4()}.{ext}"

        # Get access token
        access_token = get_access_token(credentials)

        # Upload to GigaChat
        base_url = api_base or GIGACHAT_BASE_URL
        upload_url = f"{base_url}/files"

        client = _get_httpx_client(params={"ssl_verify": False})
        response = client.post(
            upload_url,
            headers={"Authorization": f"Bearer {access_token}"},
            files={"file": (filename, content_bytes, content_type)},
            data={"purpose": "general"},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        file_id = result.get("id")
        if file_id:
            _file_cache[url_hash] = file_id
            verbose_logger.debug(f"File uploaded successfully, file_id: {file_id}")

        return file_id

    except Exception as e:
        verbose_logger.error(f"Error uploading file to GigaChat: {e}")
        return None


async def upload_file_async(
    image_url: str,
    credentials: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[str]:
    """
    Upload file to GigaChat and return file_id (async).

    Args:
        image_url: URL or base64 data URL of the image
        credentials: GigaChat credentials for auth
        api_base: Optional custom API base URL

    Returns:
        file_id string or None if upload failed
    """
    url_hash = _get_url_hash(image_url)

    # Check cache
    if url_hash in _file_cache:
        verbose_logger.debug(f"Image found in cache: {url_hash[:16]}...")
        return _file_cache[url_hash]

    try:
        # Get image data
        parsed = _parse_data_url(image_url)
        if parsed:
            content_bytes, content_type, ext = parsed
            verbose_logger.debug("Decoded base64 image")
        else:
            verbose_logger.debug(f"Downloading image from URL: {image_url[:80]}...")
            content_bytes, content_type, ext = await _download_image_async(image_url)

        filename = f"{uuid.uuid4()}.{ext}"

        # Get access token
        access_token = await get_access_token_async(credentials)

        # Upload to GigaChat
        base_url = api_base or GIGACHAT_BASE_URL
        upload_url = f"{base_url}/files"

        client = get_async_httpx_client(
            llm_provider=LlmProviders.GIGACHAT,
            params={"ssl_verify": False},
        )
        response = await client.post(
            upload_url,
            headers={"Authorization": f"Bearer {access_token}"},
            files={"file": (filename, content_bytes, content_type)},
            data={"purpose": "general"},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        file_id = result.get("id")
        if file_id:
            _file_cache[url_hash] = file_id
            verbose_logger.debug(f"File uploaded successfully, file_id: {file_id}")

        return file_id

    except Exception as e:
        verbose_logger.error(f"Error uploading file to GigaChat: {e}")
        return None
