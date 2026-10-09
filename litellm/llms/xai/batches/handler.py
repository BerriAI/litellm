import asyncio
import threading
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Mapping
from functools import partial
from itertools import chain
from typing import Final, TypeAlias

import httpx
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.openai import CreateBatchRequest, HttpxBinaryResponseContent, OpenAIFileObject
from litellm.types.utils import LiteLLMBatch, LlmProviders

from .transformation import (
    XAI_RESULTS_PAGE_SIZE,
    OpenAIBatchListResponse,
    XAIBatch,
    XAIBatchList,
    XAIBatchResult,
    XAIBatchResultsPage,
    get_xai_auth_headers,
    raise_for_xai_status,
    results_to_openai_jsonl,
    to_create_batch_body,
    to_litellm_batch,
    to_openai_batch_list,
    xai_batch_output_file_id,
    xai_batch_results_file_object,
    xai_batches_error,
    xai_batches_url,
)

_JSONL_CONTENT_TYPE: Final = ("content-type", "application/jsonl")
_RESULTS_SIZE_CACHE_SIZE: Final = 1024
_NO_HEADERS: Final = httpx.Headers()
_ResultsKey: TypeAlias = tuple[str, str]
_DownloadKey: TypeAlias = tuple[_ResultsKey, str]


class _PageParams(TypedDict):
    limit: ReadOnly[int]
    pagination_token: NotRequired[ReadOnly[str]]


def _results_params(after: str | None, limit: int | None) -> dict[str, object]:  # mutable-ok: httpx params
    if after is None:
        return dict(_PageParams(limit=limit or XAI_RESULTS_PAGE_SIZE))
    return dict(_PageParams(limit=limit or XAI_RESULTS_PAGE_SIZE, pagination_token=after))


def _flatten(pages: list[XAIBatchResultsPage]) -> tuple[XAIBatchResult, ...]:
    return tuple(chain.from_iterable(page.results for page in pages))


def _jsonl_response(url: str, results: tuple[XAIBatchResult, ...]) -> HttpxBinaryResponseContent:
    return HttpxBinaryResponseContent(
        response=httpx.Response(
            status_code=200,
            content=results_to_openai_jsonl(results),
            headers=(_JSONL_CONTENT_TYPE,),
            request=httpx.Request(method="GET", url=url),
        )
    )


def _results_key(url: str, headers: Mapping[str, str]) -> _ResultsKey:
    return url, headers["Authorization"]


def _request_total(batch: LiteLLMBatch) -> int | None:
    return batch.request_counts.total if batch.request_counts is not None else None


def _complete_size(batch: LiteLLMBatch, size: int, count: int) -> int:
    total: Final = _request_total(batch)
    if count != total:
        raise xai_batches_error(
            f"xAI batch {batch.id} results are not complete yet ({count} of {total} results)", 409, _NO_HEADERS
        )
    return size


class _ResultsSizes:
    """Byte size and result count of each batch's latest results download, per results URL and API key."""

    def __init__(self) -> None:
        self._capacity: Final = _RESULTS_SIZE_CACHE_SIZE
        self._lock: Final = threading.Lock()
        self._sizes: Final[OrderedDict[_ResultsKey, tuple[int, int]]] = OrderedDict()  # mutable-ok: LRU state
        self._downloads: Final[dict[_DownloadKey, asyncio.Task[tuple[int, int]]]] = {}  # mutable-ok: in flight

    def record(self, key: _ResultsKey, size: int, count: int) -> None:
        with self._lock:
            self._sizes[key] = (size, count)
            self._sizes.move_to_end(key)
            while len(self._sizes) > self._capacity:
                self._sizes.popitem(last=False)

    def complete_size(self, key: _ResultsKey, total: int | None) -> int | None:
        with self._lock:
            observed: Final = self._sizes.get(key)
            if observed is None or observed[1] != total:
                return None
            self._sizes.move_to_end(key)
            return observed[0]

    def shared_download(
        self, key: _DownloadKey, start: Callable[[], Coroutine[None, None, tuple[int, int]]]
    ) -> asyncio.Task[tuple[int, int]]:
        loop: Final = asyncio.get_running_loop()
        with self._lock:
            running: Final = self._downloads.get(key)
            if running is not None and running.get_loop() is loop:
                return running
            for stale in [stale for stale, task in self._downloads.items() if task.get_loop().is_closed()]:
                del self._downloads[stale]
            download: Final = loop.create_task(start())
            self._downloads[key] = download
        download.add_done_callback(partial(self._finished, key))
        return download

    def _finished(self, key: _DownloadKey, download: asyncio.Task[tuple[int, int]]) -> None:
        with self._lock:
            if self._downloads.get(key) is download:
                del self._downloads[key]
        if not download.cancelled():
            download.exception()


class XAIBatchesHandler:
    def __init__(self, sync_client: HTTPHandler | None = None, async_client: AsyncHTTPHandler | None = None) -> None:
        self._sync_client = sync_client
        self._async_client = async_client
        self._results_sizes: Final = _ResultsSizes()

    def _sync(self, timeout: float | httpx.Timeout) -> HTTPHandler:
        return self._sync_client or HTTPHandler(timeout=timeout)

    def _async(self, timeout: float | httpx.Timeout) -> AsyncHTTPHandler:
        return self._async_client or get_async_httpx_client(
            llm_provider=LlmProviders.XAI,
            params={"timeout": timeout},
        )

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
    ) -> LiteLLMBatch | Coroutine[None, None, LiteLLMBatch]:
        url: Final = xai_batches_url(api_base)
        headers: Final = get_xai_auth_headers(api_key=api_key)
        body: Final = dict(to_create_batch_body(create_batch_data))
        endpoint: Final = create_batch_data.get("endpoint") or "/v1/chat/completions"
        if _is_async:

            async def _acreate() -> LiteLLMBatch:
                response: Final = await self._async(timeout).post(url, json=body, headers=headers, timeout=timeout)
                return to_litellm_batch(XAIBatch.model_validate(raise_for_xai_status(response).json()), endpoint)

            return _acreate()
        response: Final = self._sync(timeout).post(url, json=body, headers=headers, timeout=timeout)
        return to_litellm_batch(XAIBatch.model_validate(raise_for_xai_status(response).json()), endpoint)

    def retrieve_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
    ) -> LiteLLMBatch | Coroutine[None, None, LiteLLMBatch]:
        url: Final = xai_batches_url(api_base, batch_id)
        headers: Final = get_xai_auth_headers(api_key=api_key)
        if _is_async:
            return self._aget_batch(url, headers, timeout)
        return self._get_batch(url, headers, timeout)

    async def _aget_batch(self, url: str, headers: Mapping[str, str], timeout: float | httpx.Timeout) -> LiteLLMBatch:
        response: Final = await self._async(timeout).get(url, headers=dict(headers), timeout=timeout)
        return to_litellm_batch(XAIBatch.model_validate(raise_for_xai_status(response).json()))

    def _get_batch(self, url: str, headers: Mapping[str, str], timeout: float | httpx.Timeout) -> LiteLLMBatch:
        response: Final = self._sync(timeout).get(url, headers=dict(headers), timeout=timeout)
        return to_litellm_batch(XAIBatch.model_validate(raise_for_xai_status(response).json()))

    def cancel_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
    ) -> LiteLLMBatch | Coroutine[None, None, LiteLLMBatch]:
        url: Final = xai_batches_url(api_base, batch_id, suffix=":cancel")
        headers: Final = get_xai_auth_headers(api_key=api_key)
        if _is_async:

            async def _acancel() -> LiteLLMBatch:
                response: Final = await self._async(timeout).post(url, headers=headers, timeout=timeout)
                return to_litellm_batch(XAIBatch.model_validate(raise_for_xai_status(response).json()))

            return _acancel()
        response: Final = self._sync(timeout).post(url, headers=headers, timeout=timeout)
        return to_litellm_batch(XAIBatch.model_validate(raise_for_xai_status(response).json()))

    def list_batches(
        self,
        _is_async: bool,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        after: str | None = None,
        limit: int | None = None,
    ) -> OpenAIBatchListResponse | Coroutine[None, None, OpenAIBatchListResponse]:
        url: Final = xai_batches_url(api_base)
        headers: Final = get_xai_auth_headers(api_key=api_key)
        params: Final = _results_params(after, limit)
        if _is_async:

            async def _alist() -> OpenAIBatchListResponse:
                response: Final = await self._async(timeout).get(url, params=params, headers=headers, timeout=timeout)
                return to_openai_batch_list(XAIBatchList.model_validate(raise_for_xai_status(response).json()))

            return _alist()
        response: Final = self._sync(timeout).get(url, params=params, headers=headers, timeout=timeout)
        return to_openai_batch_list(XAIBatchList.model_validate(raise_for_xai_status(response).json()))

    def _observed_results(
        self, url: str, headers: Mapping[str, str], results: tuple[XAIBatchResult, ...]
    ) -> HttpxBinaryResponseContent:
        content: Final = _jsonl_response(url, results)
        self._results_sizes.record(_results_key(url, headers), len(content.content), len(results))
        return content

    def batch_results_content(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
    ) -> HttpxBinaryResponseContent | Coroutine[None, None, HttpxBinaryResponseContent]:
        url: Final = xai_batches_url(api_base, batch_id, suffix="/results")
        headers: Final = get_xai_auth_headers(api_key=api_key)
        if _is_async:
            return self._aget_results(url, headers, timeout)
        return self._get_results(url, headers, timeout)

    async def _aget_results(
        self, url: str, headers: Mapping[str, str], timeout: float | httpx.Timeout
    ) -> HttpxBinaryResponseContent:
        client: Final = self._async(timeout)

        async def _page(after: str | None) -> XAIBatchResultsPage:
            response: Final = await client.get(
                url, params=_results_params(after, None), headers=dict(headers), timeout=timeout
            )
            return XAIBatchResultsPage.model_validate(raise_for_xai_status(response).json())

        pages = [await _page(None)]  # rebind-ok: grows one results page at a time
        while pages[-1].pagination_token and pages[-1].results:
            pages.append(await _page(pages[-1].pagination_token))
        return self._observed_results(url, headers, _flatten(pages))

    def _get_results(
        self, url: str, headers: Mapping[str, str], timeout: float | httpx.Timeout
    ) -> HttpxBinaryResponseContent:
        client: Final = self._sync(timeout)

        def _page(after: str | None) -> XAIBatchResultsPage:
            response: Final = client.get(
                url, params=_results_params(after, None), headers=dict(headers), timeout=timeout
            )
            return XAIBatchResultsPage.model_validate(raise_for_xai_status(response).json())

        pages = [_page(None)]
        while pages[-1].pagination_token and pages[-1].results:
            pages.append(_page(pages[-1].pagination_token))
        return self._observed_results(url, headers, _flatten(pages))

    def batch_results_file(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
    ) -> OpenAIFileObject | Coroutine[None, None, OpenAIFileObject]:
        """Describe a completed batch's results as a file, sized from a complete earlier download or a fresh one."""
        batch_url: Final = xai_batches_url(api_base, batch_id)
        results_url: Final = xai_batches_url(api_base, batch_id, suffix="/results")
        headers: Final = get_xai_auth_headers(api_key=api_key)
        key: Final = _results_key(results_url, headers)
        if not _is_async:
            batch: Final = self._get_batch(batch_url, headers, timeout)
            output_file_id: Final = xai_batch_output_file_id(batch)
            known_size: Final = self._results_sizes.complete_size(key, _request_total(batch))
            if known_size is not None:
                return xai_batch_results_file_object(output_file_id, batch.created_at, known_size)
            content: Final = self._get_results(results_url, headers, timeout).content
            return xai_batch_results_file_object(
                output_file_id, batch.created_at, _complete_size(batch, len(content), len(content.splitlines()))
            )

        async def _measure() -> tuple[int, int]:
            downloaded: Final = (await self._aget_results(results_url, headers, timeout)).content
            return len(downloaded), len(downloaded.splitlines())

        async def _afile() -> OpenAIFileObject:
            completed: Final = await self._aget_batch(batch_url, headers, timeout)
            completed_file_id: Final = xai_batch_output_file_id(completed)
            already_measured: Final = self._results_sizes.complete_size(key, _request_total(completed))
            if already_measured is not None:
                return xai_batch_results_file_object(completed_file_id, completed.created_at, already_measured)
            size, count = await asyncio.shield(self._results_sizes.shared_download((key, repr(timeout)), _measure))
            return xai_batch_results_file_object(
                completed_file_id, completed.created_at, _complete_size(completed, size, count)
            )

        return _afile()
