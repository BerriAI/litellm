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


def _process_media_response(response: Response, url: str) -> str:
    if response.status_code != 200:
        raise Exception(
            f"Error: Unable to fetch media from URL. Status code: {response.status_code}, url={url}"
        )

    media_bytes = response.content
    base64_data = base64.b64encode(media_bytes).decode("utf-8")

    media_tyte = response.headers.get("Content-Type")
    if media_tyte is None:
        med_type = url.split(".")[-1].lower()
        _med_type = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }.get(med_type)
        if _med_type is None:
            raise Exception(
                f"Error: Unsupported media format. Format={_med_type}. Supported types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']"
            )
        med_type = _med_type
    else:
        med_type = media_tyte

    result = f"data:{med_type};base64,{base64_data}"
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
            return _process_media_response(response, url)
        except Exception:
            pass
    raise Exception(
        f"Error: Unable to fetch media from URL after 3 attempts. url={url}"
    )


def convert_url_to_base64(url: str) -> str:
    cached_result = in_memory_cache.get_cache(url)
    if cached_result:
        return cached_result

    client = litellm.module_level_client
    for _ in range(3):
        try:
            response = client.get(url, follow_redirects=True)
            return _process_media_response(response, url)
        except Exception as e:
            verbose_logger.exception(e)
            # print(e)
            pass
    raise Exception(
        f"Error: Unable to fetch media from URL after 3 attempts. url={url}"
    )
