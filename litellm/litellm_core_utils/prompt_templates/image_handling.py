"""
Helper functions to handle images passed in messages
"""

import base64

from httpx import Response

import litellm
from litellm import verbose_logger
from litellm.caching.caching import InMemoryCache

MAX_IMGS_IN_MEMORY = 10

in_memory_cache = InMemoryCache(max_size_in_memory=MAX_IMGS_IN_MEMORY)


def _process_image_response(response: Response, url: str) -> str:
    if response.status_code != 200:
        raise litellm.ImageFetchError(
            f"Error: Unable to fetch image from URL. Status code: {response.status_code}, url={url}"
        )

    image_bytes = response.content
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

async def handle_string_url_to_base64(source: str) -> dict:
    """
    Handle string URL source and convert to base64 format (sync version).
    
    Args:
        source: String URL starting with http:// or https://
        
    Returns:
        dict: Base64 formatted source with type, media_type, and data
    """
    base64_data = await async_convert_url_to_base64(source)
    
    # Extract media type from the base64 data URL
    if base64_data.startswith("data:"):
        media_type = base64_data.split(";")[0].split(":")[1]
        base64_content = base64_data.split(",")[1]
    else:
        media_type = "image/jpeg"  # Default fallback
        base64_content = base64_data
    
    return {
        "type": "base64",
        "media_type": media_type,
        "data": base64_content
    }


async def handle_url_object_to_base64(source: dict) -> dict:
    """
    Handle URL object source and convert to base64 format (sync version).
    
    Args:
        source: Dictionary with type="url" and url field
        
    Returns:
        dict: Base64 formatted source with type, media_type, and data
    """
    url = source.get("url", "")
    base64_data = await async_convert_url_to_base64(url)
    
    # Extract media type from the base64 data URL
    if base64_data.startswith("data:"):
        media_type = base64_data.split(";")[0].split(":")[1]
        base64_content = base64_data.split(",")[1]
    else:
        media_type = "image/jpeg"  # Default fallback
        base64_content = base64_data
    
    return {
        "type": "base64",
        "media_type": media_type,
        "data": base64_content
    }


async def async_anthropic_provider_process_image_content(
    messages: list,
) -> list:
    """
    Process image content in messages, converting URLs to base64 format (async version).
    
    Args:
        messages: List of message dictionaries

    Returns:
        list: Processed messages with converted image sources
    """
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), list):
            for content in message["content"]:
                if content.get("type") == "image":
                    source = content.get("source", {})
                    
                    # Check if source has a URL that needs conversion
                    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
                        try:
                            content["source"] = await handle_string_url_to_base64(source)
                        except Exception:
                            # If conversion fails, keep original content
                            pass
                    elif isinstance(source, dict) and source.get("type") == "url":
                        # Handle URL object format
                        url = source.get("url", "")
                        if url.startswith("http://") or url.startswith("https://"):
                            try:
                                content["source"] = await handle_url_object_to_base64(source)
                            except Exception:
                                # If conversion fails, keep original content
                                pass
    return messages
