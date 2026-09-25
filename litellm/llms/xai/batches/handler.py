from collections.abc import Coroutine
from itertools import chain
from typing import Final

import httpx
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.openai import CreateBatchRequest, HttpxBinaryResponseContent
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
    xai_batches_url,
)

_JSONL_CONTENT_TYPE: Final = ("content-type", "application/jsonl")


class _PageParams(TypedDict):
    limit: ReadOnly[int]
    pagination_token: NotRequired[ReadOnly[str]]


def _results_params(after: str | None, limit: int | None) -> dict[str, object]:  # mutable-ok: httpx params
    if after is None:
        return dict(_PageParams(limit=limit or XAI_RESULTS_PAGE_SIZE))  # mutable-ok: httpx params
    return dict(_PageParams(limit=limit or XAI_RESULTS_PAGE_SIZE, pagination_token=after))  # mutable-ok: httpx params


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


class XAIBatchesHandler:
    def __init__(self, sync_client: HTTPHandler | None = None, async_client: AsyncHTTPHandler | None = None) -> None:
        self._sync_client = sync_client
        self._async_client = async_client

    def _sync(self, timeout: float | httpx.Timeout) -> HTTPHandler:
        return self._sync_client or HTTPHandler(timeout=timeout)

    def _async(self, timeout: float | httpx.Timeout) -> AsyncHTTPHandler:
        return self._async_client or get_async_httpx_client(
            llm_provider=LlmProviders.XAI,
            params={"timeout": timeout},  # mutable-ok: get_async_httpx_client takes a dict
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
        body: Final = dict(to_create_batch_body(create_batch_data))  # mutable-ok: httpx json body
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

            async def _aretrieve() -> LiteLLMBatch:
                response: Final = await self._async(timeout).get(url, headers=headers, timeout=timeout)
                return to_litellm_batch(XAIBatch.model_validate(raise_for_xai_status(response).json()))

            return _aretrieve()
        response: Final = self._sync(timeout).get(url, headers=headers, timeout=timeout)
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

            async def _aresults() -> HttpxBinaryResponseContent:
                client: Final = self._async(timeout)

                async def _page(after: str | None) -> XAIBatchResultsPage:
                    response: Final = await client.get(
                        url, params=_results_params(after, None), headers=headers, timeout=timeout
                    )
                    return XAIBatchResultsPage.model_validate(raise_for_xai_status(response).json())

                pages = [await _page(None)]  # mutable-ok: page walk terminates on the cursor, not on a fixed count
                while pages[-1].pagination_token and pages[-1].results:
                    pages.append(await _page(pages[-1].pagination_token))
                return _jsonl_response(url, _flatten(pages))

            return _aresults()
        client: Final = self._sync(timeout)

        def _page(after: str | None) -> XAIBatchResultsPage:
            response: Final = client.get(url, params=_results_params(after, None), headers=headers, timeout=timeout)
            return XAIBatchResultsPage.model_validate(raise_for_xai_status(response).json())

        pages = [_page(None)]  # mutable-ok: page walk terminates on the cursor, not on a fixed count
        while pages[-1].pagination_token and pages[-1].results:
            pages.append(_page(pages[-1].pagination_token))
        return _jsonl_response(url, _flatten(pages))
