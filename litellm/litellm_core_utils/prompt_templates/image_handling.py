"""
Helper functions to handle images passed in messages
"""

import base64

from httpx import Response

import litellm
from litellm import verbose_logger
from litellm.caching.caching import InMemoryCache
from litellm.constants import MAX_IMAGE_URL_DOWNLOAD_SIZE_MB

MAX_IMGS_IN_MEMORY = 10

in_memory_cache = InMemoryCache(max_size_in_memory=MAX_IMGS_IN_MEMORY)


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

    image_type = response.headers.get("Content-Type")
    if image_type is None:
        img_type = url.split(".")[-1].lower()
        _img_type = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }.get(img_type)
        if _img_type is None:
            raise Exception(
                f"Error: Unsupported image format. Format={_img_type}. Supported types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']"
            )
        img_type = _img_type
    else:
        img_type = image_type

    result = f"data:{img_type};base64,{base64_image}"
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
