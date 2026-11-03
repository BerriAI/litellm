"""
Helper functions to handle images passed in messages
"""

import asyncio
import base64
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import accumulate, chain, takewhile, tee
from types import MappingProxyType
from typing import Final, TypeVar
from urllib.parse import urlparse

import httpx
from httpx import USE_CLIENT_DEFAULT, Response

import litellm
from litellm import verbose_logger
from litellm.caching.caching import InMemoryCache
from litellm.constants import MAX_IMAGE_URL_DOWNLOAD_SIZE_MB, NON_LLM_CONNECTION_TIMEOUT
from litellm.litellm_core_utils.prompt_templates.common_utils import infer_content_type_from_url_and_content
from litellm.litellm_core_utils.url_utils import SSRFError, async_safe_get, safe_get
from litellm.types.llms.openai import AllMessageValues

MAX_IMGS_IN_MEMORY: Final = 10
MAX_CONCURRENT_REMOTE_MEDIA_FETCHES: Final = 20
_Fetched = TypeVar("_Fetched")
_FETCH_ATTEMPTS: Final = 3

in_memory_cache: Final = InMemoryCache(max_size_in_memory=MAX_IMGS_IN_MEMORY)
pdf_probe_verdicts: Final = InMemoryCache()


def _header_value(response: Response, name: str) -> str | None:
    try:
        return response.headers[name]
    except KeyError:
        return None


def _process_image_response(response: Response, url: str) -> str:
    if response.status_code != 200:
        raise litellm.ImageFetchError(
            f"Error: Unable to fetch image from URL. Status code: {response.status_code}, url={url}"
        )

    # Check size before downloading if Content-Length header is present
    content_length: Final = _header_value(response, "Content-Length")
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

    try:
        img_type: Final = infer_content_type_from_url_and_content(
            url=url,
            content=bytes(image_bytes),
            current_content_type=_header_value(response, "Content-Type"),
        )
    except ValueError as e:
        raise litellm.ImageFetchError(
            f"Error: Unable to determine image content type from the server's headers, the URL, or the image bytes. url={url}"
        ) from e

    result: Final = f"data:{img_type};base64,{base64_image}"
    in_memory_cache.set_cache(url, result)
    return result


def _attempt_timeouts(timeout: float | httpx.Timeout | None) -> Iterable[float | httpx.Timeout | None]:
    if timeout is None or isinstance(timeout, httpx.Timeout):
        return (timeout,) * _FETCH_ATTEMPTS
    deadline: Final = time.monotonic() + timeout
    return (remaining for _ in range(_FETCH_ATTEMPTS) if (remaining := deadline - time.monotonic()) > 0)


def _rejected_image_fetch(url: str, verdict: SSRFError) -> "litellm.ImageFetchError":
    verbose_logger.warning("Image fetch of %s rejected before any request went out: %s", url, verdict)
    return litellm.ImageFetchError(
        "Error: Unable to fetch image from URL. The proxy could not resolve this host or its URL policy rejected it; "
        f"an admin can check the proxy log and `user_url_allowed_hosts` in litellm_settings. url={url}"
    )


async def async_convert_url_to_base64(url: str, timeout: float | httpx.Timeout | None = None) -> str:
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
    for attempt_timeout in _attempt_timeouts(timeout):
        try:
            response = await async_safe_get(client, url, timeout=attempt_timeout)
            return _process_image_response(response, url)
        except litellm.ImageFetchError:
            raise
        except SSRFError as e:
            raise _rejected_image_fetch(url, e) from e
        except Exception:
            pass
    raise litellm.ImageFetchError(f"Error: Unable to fetch image from URL after 3 attempts. url={url}")


def convert_url_to_base64(url: str, timeout: float | httpx.Timeout | None = None) -> str:
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
    for attempt_timeout in _attempt_timeouts(timeout):
        try:
            response = safe_get(client, url, timeout=attempt_timeout)
            return _process_image_response(response, url)
        except litellm.ImageFetchError:
            raise
        except SSRFError as e:
            raise _rejected_image_fetch(url, e) from e
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


_ANTHROPIC_MEDIA_BLOCK_TYPES: Final = frozenset({"document", "image"})


@dataclass(frozen=True, slots=True)
class _RemoteSource:
    part: Mapping[str, object]
    source: Mapping[str, object]
    url: str


@dataclass(frozen=True, slots=True)
class RemoteMedia:
    url: str
    fields: Mapping[str, object]
    part_type: str


_NO_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})


def inline_every_remote_url(_media: RemoteMedia) -> bool:
    return True


def inline_remote_image_urls(media: RemoteMedia) -> bool:
    return media.part_type == "image_url"


_PDF_MEDIA_TYPES: Final = frozenset({"application/pdf", "application/x-pdf"})
_GENERIC_MEDIA_TYPES: Final = frozenset({"application/octet-stream", "binary/octet-stream"})
_PDF_SIGNATURE: Final = b"%PDF-"
_PDF_SIGNATURE_WINDOW: Final = 1024
_PDF_PROBE_HEADERS: Final = MappingProxyType({"Range": f"bytes=0-{_PDF_SIGNATURE_WINDOW - 1}"})
_PDF_PROBE_STATUSES: Final = frozenset({200, 206})
_PDF_PROBE_FAILURES: Final = (httpx.HTTPError, httpx.InvalidURL, ValueError)
_ASYNC_PDF_PROBE_FAILURES: Final = (*_PDF_PROBE_FAILURES, asyncio.TimeoutError)
_PDF_PROBE_VERDICT_TTL: Final = 600
_PDF_PROBE_FAILURE_TTL: Final = 60
_DECODED_BODY_HEADERS: Final = frozenset({"content-encoding", "content-length", "transfer-encoding"})


def _bare_media_type(media_type: str) -> str:
    return media_type.split(";", 1)[0].strip().lower()


def is_pdf_media_type(media_type: str) -> bool:
    return _bare_media_type(media_type) in _PDF_MEDIA_TYPES


def is_unformatted_pdf_link(url: str, image_format: str | None) -> bool:
    lowered: Final = url.lower()
    return not image_format and (urlparse(lowered).path.endswith(".pdf") or lowered.endswith(".pdf"))


def is_unformatted_pdf_link_image(media: RemoteMedia) -> bool:
    image_format: Final = media.fields.get("format")
    return inline_remote_image_urls(media) and is_unformatted_pdf_link(
        media.url, image_format if isinstance(image_format, str) else None
    )


def _serves_pdf(response: Response) -> bool:
    if response.status_code not in _PDF_PROBE_STATUSES:
        return False
    served_type: Final = _bare_media_type(_header_value(response, "Content-Type") or "")
    if served_type and served_type not in _GENERIC_MEDIA_TYPES:
        return served_type in _PDF_MEDIA_TYPES
    return _PDF_SIGNATURE in response.content[:_PDF_SIGNATURE_WINDOW]


def _signature_window_chunks(chunks: Iterable[bytes], deadline: float) -> Iterator[bytes]:
    reading, counting = tee(chunks)
    read_so_far: Final = chain((0,), accumulate(map(len, counting)))
    still_reading: Final = takewhile(
        lambda total: total < _PDF_SIGNATURE_WINDOW and time.monotonic() < deadline, read_so_far
    )
    return (chunk for _, chunk in zip(still_reading, reading))


def _with_body_prefix(response: Response, prefix: bytes) -> Response:
    return Response(
        response.status_code,
        headers=[(name, value) for name, value in response.headers.multi_items() if name not in _DECODED_BODY_HEADERS],
        content=prefix,
        request=response.request,
    )


@dataclass(frozen=True, slots=True)
class _BodyPrefixReader:
    client: httpx.Client
    deadline: float

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        follow_redirects: bool = False,
        timeout: float | httpx.Timeout | None = None,
    ) -> Response:
        with self.client.stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=follow_redirects,
            timeout=USE_CLIENT_DEFAULT if timeout is None else timeout,
        ) as response:
            prefix: Final = b"".join(_signature_window_chunks(response.iter_bytes(), self.deadline))
            return _with_body_prefix(response, prefix[:_PDF_SIGNATURE_WINDOW])


@dataclass(frozen=True, slots=True)
class _AsyncBodyPrefixReader:
    client: httpx.AsyncClient

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        follow_redirects: bool = False,
        timeout: float | httpx.Timeout | None = None,
        max_response_bytes: int | None = None,
    ) -> Response:
        async with self.client.stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=follow_redirects,
            timeout=USE_CLIENT_DEFAULT if timeout is None else timeout,
        ) as response:
            return _with_body_prefix(response, await anext(response.aiter_bytes(max_response_bytes), b""))


def _cached_pdf_verdict(url: str) -> bool | None:
    cached: Final = pdf_probe_verdicts.get_cache(url)
    return cached if isinstance(cached, bool) else None


def _remember_pdf_verdict(url: str, verdict: bool, ttl: int) -> bool:
    pdf_probe_verdicts.set_cache(url, verdict, ttl=ttl)
    return verdict


def _remember_probe_response(url: str, response: Response) -> bool:
    ttl: Final = _PDF_PROBE_VERDICT_TTL if response.status_code in _PDF_PROBE_STATUSES else _PDF_PROBE_FAILURE_TTL
    return _remember_pdf_verdict(url, _serves_pdf(response), ttl)


def _failed_pdf_probe(url: str, error: Exception) -> bool:
    verbose_logger.debug("PDF probe of %s failed, sending it as an image link: %s", url, error)
    return _remember_pdf_verdict(url, False, _PDF_PROBE_FAILURE_TTL)


def serves_pdf(url: str, timeout: float = NON_LLM_CONNECTION_TIMEOUT) -> bool:
    if MAX_IMAGE_URL_DOWNLOAD_SIZE_MB == 0:
        return False
    cached: Final = _cached_pdf_verdict(url)
    if cached is not None:
        return cached
    try:
        response: Final = safe_get(
            _BodyPrefixReader(litellm.module_level_client.client, deadline=time.monotonic() + timeout),
            url,
            headers=dict(_PDF_PROBE_HEADERS),
            timeout=timeout,
        )
    except _PDF_PROBE_FAILURES as e:
        return _failed_pdf_probe(url, e)
    return _remember_probe_response(url, response)


async def async_serves_pdf(url: str, timeout: float = NON_LLM_CONNECTION_TIMEOUT) -> bool:
    if MAX_IMAGE_URL_DOWNLOAD_SIZE_MB == 0:
        return False
    cached: Final = _cached_pdf_verdict(url)
    if cached is not None:
        return cached
    try:
        response: Final = await asyncio.wait_for(
            async_safe_get(
                _AsyncBodyPrefixReader(litellm.module_level_aclient.client),
                url,
                headers=dict(_PDF_PROBE_HEADERS),
                timeout=timeout,
                max_response_bytes=_PDF_SIGNATURE_WINDOW,
            ),
            timeout,
        )
    except _ASYNC_PDF_PROBE_FAILURES as e:
        return _failed_pdf_probe(url, e)
    return _remember_probe_response(url, response)


def _parse_remote_image(fields: Mapping[str, object]) -> _RemoteImage | None:
    if fields.get("type") != "image_url":
        return None
    image_url: Final = fields.get("image_url")
    image_url_fields: Final = _as_mapping(image_url)
    url: Final = _remote_url(image_url_fields.get("url") if image_url_fields is not None else image_url)
    return _RemoteImage(fields, image_url_fields, url) if url is not None else None


def _parse_remote_file(fields: Mapping[str, object]) -> _RemoteFile | None:
    file: Final = _as_mapping(fields.get("file")) if fields.get("type") == "file" else None
    url: Final = _remote_url(file.get("file_id")) if file is not None else None
    return _RemoteFile(fields, file, url) if file is not None and url is not None else None


def _parse_remote_source(fields: Mapping[str, object]) -> _RemoteSource | None:
    source: Final = _as_mapping(fields.get("source")) if fields.get("type") in _ANTHROPIC_MEDIA_BLOCK_TYPES else None
    url: Final = _remote_url(source.get("url")) if source is not None and source.get("type") == "url" else None
    return _RemoteSource(fields, source, url) if source is not None and url is not None else None


def _parse_remote_part(part: object) -> _RemoteImage | _RemoteFile | _RemoteSource | None:
    fields: Final = _as_mapping(part)
    if fields is None:
        return None
    return _parse_remote_image(fields) or _parse_remote_file(fields) or _parse_remote_source(fields)


def _remote_media(remote: _RemoteImage | _RemoteFile | _RemoteSource) -> RemoteMedia:
    match remote:
        case _RemoteImage(_, image_url, url):
            return RemoteMedia(url, image_url if image_url is not None else _NO_FIELDS, "image_url")
        case _RemoteFile(_, file, url):
            return RemoteMedia(url, file, "file")
        case _RemoteSource(part, source, url):
            return RemoteMedia(url, source, str(part.get("type")))


_PDF_FORMAT: Final = MappingProxyType({"format": "application/pdf"})


def _inferred_format(file: Mapping[str, object], url: str) -> Mapping[str, str]:
    return _PDF_FORMAT if "format" not in file and url.lower().endswith(".pdf") else MappingProxyType({})


def _inlined_image_url(image_url: Mapping[str, object] | None, data_url: str) -> Mapping[str, object] | str:
    return {**image_url, "url": data_url} if image_url is not None else data_url


def _inlined_file(file: Mapping[str, object], url: str, data_url: str) -> Mapping[str, object]:
    kept: Final = {k: v for k, v in file.items() if k != "file_id"}
    return {**kept, **_inferred_format(file, url), "file_data": data_url}


def _base64_source(url: str, data_url: str) -> Mapping[str, str]:
    fetched_media_type, data = data_url.removeprefix("data:").split(";base64,", 1)
    media_type: Final = "application/pdf" if url.lower().endswith(".pdf") else fetched_media_type
    return {"type": "base64", "media_type": media_type, "data": data}


def _inline(remote: _RemoteImage | _RemoteFile | _RemoteSource, data_url: str) -> Mapping[str, object]:
    match remote:
        case _RemoteImage(part, image_url, _):
            return {**part, "image_url": _inlined_image_url(image_url, data_url)}
        case _RemoteFile(part, file, url):
            return {**part, "file": _inlined_file(file, url, data_url)}
        case _RemoteSource(part, _, url):
            return {**part, "source": _base64_source(url, data_url)}


def _content_parts(message: Mapping[str, object]) -> tuple[object, ...]:
    content: Final = message.get("content")
    return tuple(content) if isinstance(content, list) else ()  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # parts are parsed one by one


def _inline_part(part: object, data_urls: Mapping[str, str], should_inline: Callable[[RemoteMedia], bool]) -> object:
    remote: Final = _parse_remote_part(part)
    if remote is None or not should_inline(_remote_media(remote)):
        return part
    data_url: Final = data_urls.get(remote.url)
    return _inline(remote, data_url) if data_url is not None else part


def _with_parts(message: AllMessageValues, rewrite: Callable[[object], object]) -> AllMessageValues:
    parts: Final = _content_parts(message)
    if not parts:
        return message
    rewritten: Final = {**message, "content": [rewrite(part) for part in parts]}
    return rewritten  # pyright: ignore[reportReturnType]  # the same message with some of its parts rewritten


def _inline_message(
    message: AllMessageValues, data_urls: Mapping[str, str], should_inline: Callable[[RemoteMedia], bool]
) -> AllMessageValues:
    return _with_parts(message, partial(_inline_part, data_urls=data_urls, should_inline=should_inline))


def _pdf_formatted_part(part: object, pdf_links: frozenset[str], should_probe: Callable[[RemoteMedia], bool]) -> object:
    remote: Final = _parse_remote_part(part)
    match remote:
        case _RemoteImage(fields, image_url, url) if url in pdf_links and should_probe(_remote_media(remote)):
            image_url_fields: Final = image_url if image_url is not None else _NO_FIELDS
            return {**fields, "image_url": {**image_url_fields, "url": url, "format": "application/pdf"}}
        case _:
            return part


def _with_pdf_formats(
    messages: Iterable[AllMessageValues],
    probed: Iterable[tuple[str, bool]],
    should_probe: Callable[[RemoteMedia], bool],
) -> tuple[AllMessageValues, ...]:
    pdf_links: Final = frozenset(url for url, is_pdf in probed if is_pdf)
    if not pdf_links:
        return tuple(messages)
    rewrite: Final = partial(_pdf_formatted_part, pdf_links=pdf_links, should_probe=should_probe)
    return tuple(_with_parts(message, rewrite) for message in messages)


async def _fetch_data_url(url: str, in_flight: asyncio.Semaphore, timeout: float | httpx.Timeout | None) -> str:
    async with in_flight:
        return await async_convert_url_to_base64(url, timeout=timeout)


async def _serves_pdf_within(url: str, in_flight: asyncio.Semaphore) -> bool:
    async with in_flight:
        return await async_serves_pdf(url)


async def _gather_or_cancel_all(fetches: tuple["asyncio.Task[_Fetched]", ...]) -> tuple[_Fetched, ...]:
    try:
        return tuple(await asyncio.gather(*fetches))
    except BaseException:
        for fetch in fetches:
            fetch.cancel()
        await asyncio.gather(*fetches, return_exceptions=True)
        raise


async def async_convert_urls_to_base64(
    remote_urls: tuple[str, ...], timeout: float | httpx.Timeout | None = None
) -> tuple[str, ...]:
    in_flight: Final = asyncio.Semaphore(MAX_CONCURRENT_REMOTE_MEDIA_FETCHES)
    return await _gather_or_cancel_all(
        tuple(asyncio.create_task(_fetch_data_url(url, in_flight, timeout)) for url in remote_urls)
    )


def _remote_urls_to_inline(
    messages: Iterable[AllMessageValues], should_inline: Callable[[RemoteMedia], bool]
) -> tuple[str, ...]:
    parts: Final = chain.from_iterable(_content_parts(message) for message in messages)
    remotes: Final = (remote for part in parts if (remote := _parse_remote_part(part)) is not None)
    return tuple(dict.fromkeys(remote.url for remote in remotes if should_inline(_remote_media(remote))))


def inline_remote_media(
    messages: list[AllMessageValues],  # mutable-ok: every transform_request takes list[AllMessageValues]
    should_inline: Callable[[RemoteMedia], bool] = inline_every_remote_url,
) -> list[AllMessageValues]:  # mutable-ok: every transform_request takes list[AllMessageValues]
    remote_urls: Final = _remote_urls_to_inline(messages, should_inline)
    if not remote_urls:
        return messages
    data_urls: Final = MappingProxyType({url: convert_url_to_base64(url) for url in remote_urls})
    return [_inline_message(message, data_urls, should_inline) for message in messages]


async def async_inline_remote_media(
    messages: list[AllMessageValues],  # mutable-ok: every transform_request takes list[AllMessageValues]
    should_inline: Callable[[RemoteMedia], bool] = inline_every_remote_url,
) -> list[AllMessageValues]:  # mutable-ok: every transform_request takes list[AllMessageValues]
    remote_urls: Final = _remote_urls_to_inline(messages, should_inline)
    if not remote_urls:
        return messages
    data_urls: Final = await async_convert_urls_to_base64(remote_urls)
    inlined: Final = MappingProxyType(dict(zip(remote_urls, data_urls, strict=True)))
    return [_inline_message(message, inlined, should_inline) for message in messages]


def format_pdf_links(
    messages: Sequence[AllMessageValues], should_probe: Callable[[RemoteMedia], bool]
) -> tuple[AllMessageValues, ...]:
    links: Final = _remote_urls_to_inline(messages, should_probe)
    if not links:
        return tuple(messages)
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REMOTE_MEDIA_FETCHES, len(links))) as probes:
        verdicts: Final = tuple(probes.map(serves_pdf, links))
    return _with_pdf_formats(messages, zip(links, verdicts, strict=True), should_probe)


async def async_format_pdf_links(
    messages: Sequence[AllMessageValues], should_probe: Callable[[RemoteMedia], bool]
) -> tuple[AllMessageValues, ...]:
    links: Final = _remote_urls_to_inline(messages, should_probe)
    if not links:
        return tuple(messages)
    in_flight: Final = asyncio.Semaphore(MAX_CONCURRENT_REMOTE_MEDIA_FETCHES)
    verdicts: Final = await _gather_or_cancel_all(
        tuple(asyncio.create_task(_serves_pdf_within(url, in_flight)) for url in links)
    )
    return _with_pdf_formats(messages, zip(links, verdicts, strict=True), should_probe)
