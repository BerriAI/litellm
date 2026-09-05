"""
Helper functions to handle images passed in messages
"""

import asyncio
import base64
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from httpx import Response

import litellm
from litellm import verbose_logger
from litellm.caching.caching import InMemoryCache
from litellm.constants import MAX_IMAGE_URL_DOWNLOAD_SIZE_MB
from litellm.litellm_core_utils.url_utils import async_safe_get, safe_get
from litellm.types.llms.openai import AllMessageValues

MAX_IMGS_IN_MEMORY: Final = 10
MAX_CONCURRENT_REMOTE_MEDIA_FETCHES: Final = 20

in_memory_cache: Final = InMemoryCache(max_size_in_memory=MAX_IMGS_IN_MEMORY)


def _process_image_response(response: Response, url: str) -> str:
    if response.status_code != 200:
        raise litellm.ImageFetchError(
            f"Error: Unable to fetch image from URL. Status code: {response.status_code}, url={url}"
        )

    # Check size before downloading if Content-Length header is present
    content_length: Final = response.headers.get("Content-Length")
    if content_length is not None:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > MAX_IMAGE_URL_DOWNLOAD_SIZE_MB:
            raise litellm.ImageFetchError(
                f"Error: Image size ({size_mb:.2f}MB) exceeds maximum allowed size ({MAX_IMAGE_URL_DOWNLOAD_SIZE_MB}MB). url={url}"
            )

    # Stream download with size checking to prevent downloading huge files
    max_bytes: Final = int(MAX_IMAGE_URL_DOWNLOAD_SIZE_MB * 1024 * 1024)
    image_bytes: Final = bytearray()
    bytes_downloaded = 0

    for chunk in response.iter_bytes(chunk_size=8192):
        bytes_downloaded += len(chunk)
        if bytes_downloaded > max_bytes:
            size_mb = bytes_downloaded / (1024 * 1024)
            raise litellm.ImageFetchError(
                f"Error: Image size ({size_mb:.2f}MB) exceeds maximum allowed size ({MAX_IMAGE_URL_DOWNLOAD_SIZE_MB}MB). url={url}"
            )
        image_bytes.extend(chunk)

    base64_image: Final = base64.b64encode(image_bytes).decode("utf-8")

    image_type: Final = response.headers.get("Content-Type")
    if image_type is None:
        img_type = url.split(".")[-1].lower()
        _img_type: Final = {
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

    result: Final = f"data:{img_type};base64,{base64_image}"
    in_memory_cache.set_cache(url, result)
    return result


async def async_convert_url_to_base64(url: str) -> str:
    if url.startswith("data:") and ";base64," in url:
        return url

    # If MAX_IMAGE_URL_DOWNLOAD_SIZE_MB is 0, block all image downloads
    if MAX_IMAGE_URL_DOWNLOAD_SIZE_MB == 0:
        raise litellm.ImageFetchError(
            f"Error: Image URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0). url={url}"
        )

    cached_result: Final = in_memory_cache.get_cache(url)
    if cached_result:
        return cached_result

    client: Final = litellm.module_level_aclient
    for _ in range(3):
        try:
            response = await async_safe_get(client, url)
            return _process_image_response(response, url)
        except litellm.ImageFetchError:
            raise
        except Exception:
            pass
    raise litellm.ImageFetchError(f"Error: Unable to fetch image from URL after 3 attempts. url={url}")


def convert_url_to_base64(url: str) -> str:
    if url.startswith("data:") and ";base64," in url:
        return url

    # If MAX_IMAGE_URL_DOWNLOAD_SIZE_MB is 0, block all image downloads
    if MAX_IMAGE_URL_DOWNLOAD_SIZE_MB == 0:
        raise litellm.ImageFetchError(
            f"Error: Image URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0). url={url}"
        )

    cached_result: Final = in_memory_cache.get_cache(url)
    if cached_result:
        return cached_result

    client: Final = litellm.module_level_client
    for _ in range(3):
        try:
            response = safe_get(client, url)
            return _process_image_response(response, url)
        except litellm.ImageFetchError:
            raise
        except Exception as e:
            verbose_logger.exception(e)
    raise litellm.ImageFetchError(
        f"Error: Unable to fetch image from URL after 3 attempts. url={url}",
    )


_REMOTE_URL_PREFIXES: Final = ("http://", "https://")


@dataclass(frozen=True, slots=True)
class _RemoteImage:
    part: Mapping[str, object]
    image_url: Mapping[str, object] | None
    url: str


@dataclass(frozen=True, slots=True)
class _RemoteFile:
    part: Mapping[str, object]
    file: Mapping[str, object]
    url: str


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None  # pyright: ignore[reportUnknownVariableType]  # fields are parsed one by one


def _remote_url(candidate: object) -> str | None:
    return candidate if isinstance(candidate, str) and candidate.startswith(_REMOTE_URL_PREFIXES) else None


def _parse_remote_part(part: object) -> _RemoteImage | _RemoteFile | None:
    fields: Final = _as_mapping(part)
    if fields is None:
        return None
    if fields.get("type") == "image_url":
        image_url: Final = fields.get("image_url")
        image_url_fields: Final = _as_mapping(image_url)
        url: Final = _remote_url(image_url_fields.get("url") if image_url_fields is not None else image_url)
        return _RemoteImage(fields, image_url_fields, url) if url is not None else None
    file: Final = _as_mapping(fields.get("file")) if fields.get("type") == "file" else None
    file_url: Final = _remote_url(file.get("file_id")) if file is not None else None
    return _RemoteFile(fields, file, file_url) if file is not None and file_url is not None else None


_PDF_FORMAT: Final = MappingProxyType({"format": "application/pdf"})


def _inferred_format(file: Mapping[str, object], url: str) -> Mapping[str, str]:
    return _PDF_FORMAT if "format" not in file and url.lower().endswith(".pdf") else MappingProxyType({})


def _inlined_image_url(image_url: Mapping[str, object] | None, data_url: str) -> Mapping[str, object] | str:
    return {**image_url, "url": data_url} if image_url is not None else data_url  # mutable-ok: json-serialized part


def _inlined_file(file: Mapping[str, object], url: str, data_url: str) -> Mapping[str, object]:
    kept: Final = {k: v for k, v in file.items() if k != "file_id"}  # mutable-ok: json-serialized message part
    return {**kept, **_inferred_format(file, url), "file_data": data_url}  # mutable-ok: json-serialized part


def _inline(remote: _RemoteImage | _RemoteFile, data_url: str) -> Mapping[str, object]:
    match remote:
        case _RemoteImage(part, image_url, _):
            return {**part, "image_url": _inlined_image_url(image_url, data_url)}  # mutable-ok: json-serialized part
        case _RemoteFile(part, file, url):
            return {**part, "file": _inlined_file(file, url, data_url)}  # mutable-ok: json-serialized message part


def _content_parts(message: Mapping[str, object]) -> tuple[object, ...]:
    content: Final = message.get("content")
    return tuple(content) if isinstance(content, list) else ()  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # parts are parsed one by one


def _inline_part(part: object, data_urls: Mapping[str, str]) -> object:
    remote: Final = _parse_remote_part(part)
    data_url: Final = data_urls.get(remote.url) if remote is not None else None
    return _inline(remote, data_url) if remote is not None and data_url is not None else part


def _inline_message(message: AllMessageValues, data_urls: Mapping[str, str]) -> AllMessageValues:
    parts: Final = _content_parts(message)
    if not parts:
        return message
    inlined_parts: Final = [  # mutable-ok: content must stay a list for the transforms' isinstance checks
        _inline_part(part, data_urls) for part in parts
    ]
    inlined_message: Final = {**message, "content": inlined_parts}  # mutable-ok: json-serialized message
    return inlined_message  # pyright: ignore[reportReturnType]  # the same message with its remote parts inlined


async def _fetch_data_url(url: str, in_flight: asyncio.Semaphore) -> str:
    async with in_flight:
        return await async_convert_url_to_base64(url)


async def async_inline_remote_media(
    messages: list[AllMessageValues],  # mutable-ok: every transform_request takes list[AllMessageValues]
    skip_url_prefixes: tuple[str, ...] = (),
) -> list[AllMessageValues]:  # mutable-ok: every transform_request takes list[AllMessageValues]
    remote_urls: Final = tuple(
        dict.fromkeys(
            remote.url
            for message in messages
            for part in _content_parts(message)
            if (remote := _parse_remote_part(part)) is not None and not remote.url.startswith(skip_url_prefixes)
        )
    )
    if not remote_urls:
        return messages
    in_flight: Final = asyncio.Semaphore(MAX_CONCURRENT_REMOTE_MEDIA_FETCHES)
    data_urls: Final = await asyncio.gather(*(_fetch_data_url(url, in_flight) for url in remote_urls))
    inlined: Final = MappingProxyType(dict(zip(remote_urls, data_urls, strict=True)))
    return [_inline_message(message, inlined) for message in messages]  # mutable-ok: transform_request takes a list
