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

import httpx

from litellm._logging import verbose_logger

from .authenticator import get_access_token, get_access_token_async
from .common_utils import GIGACHAT_BASE_URL, GigaChatError


class GigaChatFileHandler:
    """
    Handles file uploads to GigaChat API.

    GigaChat requires images to be uploaded to /files endpoint first,
    then referenced by file_id in message attachments.
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}  # url_hash -> file_id

    def _get_url_hash(self, url: str) -> str:
        """Generate hash for URL to use as cache key."""
        return hashlib.sha256(url.encode()).hexdigest()

    def _parse_data_url(self, data_url: str) -> Optional[Tuple[bytes, str, str]]:
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

    async def _download_image_async(self, url: str) -> Tuple[bytes, str, str]:
        """Download image from URL asynchronously."""
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(url, timeout=30, follow_redirects=True)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "image/jpeg")
        ext = content_type.split("/")[-1].split(";")[0] or "jpg"

        return response.content, content_type, ext

    def _download_image_sync(self, url: str) -> Tuple[bytes, str, str]:
        """Download image from URL synchronously."""
        with httpx.Client(verify=False) as client:
            response = client.get(url, timeout=30, follow_redirects=True)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "image/jpeg")
        ext = content_type.split("/")[-1].split(";")[0] or "jpg"

        return response.content, content_type, ext

    async def upload_file_async(
        self,
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
        url_hash = self._get_url_hash(image_url)

        # Check cache
        if url_hash in self._cache:
            verbose_logger.debug(f"Image found in cache: {url_hash[:16]}...")
            return self._cache[url_hash]

        try:
            # Get image data
            parsed = self._parse_data_url(image_url)
            if parsed:
                content_bytes, content_type, ext = parsed
                verbose_logger.debug("Decoded base64 image")
            else:
                verbose_logger.debug(f"Downloading image from URL: {image_url[:80]}...")
                content_bytes, content_type, ext = await self._download_image_async(image_url)

            filename = f"{uuid.uuid4()}.{ext}"

            # Get access token
            access_token = await get_access_token_async(credentials)

            # Upload to GigaChat
            base_url = api_base or GIGACHAT_BASE_URL
            upload_url = f"{base_url}/files"

            async with httpx.AsyncClient(verify=False) as client:
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
                self._cache[url_hash] = file_id
                verbose_logger.debug(f"File uploaded successfully, file_id: {file_id}")

            return file_id

        except Exception as e:
            verbose_logger.error(f"Error uploading file to GigaChat: {e}")
            return None

    def upload_file_sync(
        self,
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
        url_hash = self._get_url_hash(image_url)

        # Check cache
        if url_hash in self._cache:
            verbose_logger.debug(f"Image found in cache: {url_hash[:16]}...")
            return self._cache[url_hash]

        try:
            # Get image data
            parsed = self._parse_data_url(image_url)
            if parsed:
                content_bytes, content_type, ext = parsed
                verbose_logger.debug("Decoded base64 image")
            else:
                verbose_logger.debug(f"Downloading image from URL: {image_url[:80]}...")
                content_bytes, content_type, ext = self._download_image_sync(image_url)

            filename = f"{uuid.uuid4()}.{ext}"

            # Get access token
            access_token = get_access_token(credentials)

            # Upload to GigaChat
            base_url = api_base or GIGACHAT_BASE_URL
            upload_url = f"{base_url}/files"

            with httpx.Client(verify=False) as client:
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
                self._cache[url_hash] = file_id
                verbose_logger.debug(f"File uploaded successfully, file_id: {file_id}")

            return file_id

        except Exception as e:
            verbose_logger.error(f"Error uploading file to GigaChat: {e}")
            return None


# Singleton instance
_file_handler: Optional[GigaChatFileHandler] = None


def get_file_handler() -> GigaChatFileHandler:
    """Get singleton file handler instance."""
    global _file_handler
    if _file_handler is None:
        _file_handler = GigaChatFileHandler()
    return _file_handler
