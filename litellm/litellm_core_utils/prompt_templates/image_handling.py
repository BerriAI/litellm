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
        raise Exception(
            f"Error: Unable to fetch image from URL. Status code: {response.status_code}, url={url}"
        )

    image_bytes = response.content
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    img_type_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    img_type = response.headers.get("Content-Type")
    if img_type not in img_type_map:
        _img_type = url.split('?')[0].split(".")[-1].lower()
        img_type = img_type_map.get(_img_type)
        if img_type is None:
            raise Exception(
                f"Error: Unsupported image format. Format={_img_type}. Supported types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']"
            )

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
        except Exception:
            pass
    raise Exception(
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
        except Exception as e:
            verbose_logger.exception(e)
            # print(e)
            pass
    raise Exception(
        f"Error: Unable to fetch image from URL after 3 attempts. url={url}"
    )
