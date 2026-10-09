import asyncio
import json
from collections.abc import Coroutine
from typing import Final

import httpx
import pytest
import respx

import litellm
import litellm.files.main as files_main
from litellm.llms.xai.batches import handler as xai_batches_handler
from litellm.llms.xai.batches.handler import XAIBatchesHandler
from litellm.llms.xai.batches.transformation import XAIBatchesError
from litellm.types.llms.openai import OpenAIFileObject
from litellm.types.utils import LiteLLMBatch

API_BASE: Final = "https://api.x.ai"
KEY: Final = "xai-test-key"


@pytest.fixture(autouse=True)
def _httpx_transport_so_respx_can_intercept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@pytest.fixture(autouse=True)
def results_handler(monkeypatch: pytest.MonkeyPatch) -> XAIBatchesHandler:
    handler: Final = XAIBatchesHandler()
    monkeypatch.setattr(files_main, "xai_batch_results_instance", handler)
    return handler


_XAI_BATCH: Final = {
    "batch_id": "batch_1",
    "name": "litellm-batch",
    "create_time": "2026-09-23",
    "expire_time": "2026-10-23",
    "cancel_time": None,
    "cancel_by_xai_message": None,
    "state": {"num_requests": 2, "num_pending": 0, "num_success": 2, "num_error": 0, "num_cancelled": 0},
    "input_file_id": "file_1",
}


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_create_batch_posts_input_file_id_with_bearer_auth(sync_mode: bool) -> None:
    route: Final = respx.post(f"{API_BASE}/v1/batches").respond(200, json=_XAI_BATCH)

    kwargs: Final = {
        "completion_window": "24h",
        "endpoint": "/v1/embeddings",
        "input_file_id": "file_1",
        "custom_llm_provider": "xai",
        "api_key": KEY,
        "api_base": API_BASE,
    }
    batch: Final = litellm.create_batch(**kwargs) if sync_mode else await litellm.acreate_batch(**kwargs)

    assert isinstance(batch, LiteLLMBatch)
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {"name": "litellm-batch", "input_file_id": "file_1"}
    assert (batch.id, batch.endpoint, batch.status, batch.output_file_id) == (
        "batch_1",
        "/v1/embeddings",
        "completed",
        "batch_1",
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/completions",
        "/v1/responses",
        "/v1/ocr",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/videos/generations",
        "/v1/videos",
        "/v1/videos/edits",
        "/v1/videos/extensions",
    ],
)
@respx.mock
async def test_create_batch_keeps_image_and_video_endpoints_on_the_batch(endpoint: str) -> None:
    respx.post(f"{API_BASE}/v1/batches").respond(200, json=_XAI_BATCH)

    batch: Final = await litellm.acreate_batch(
        completion_window="24h",
        endpoint=endpoint,
        input_file_id="file_1",
        custom_llm_provider="xai",
        api_key=KEY,
        api_base=API_BASE,
    )

    assert isinstance(batch, LiteLLMBatch)
    assert batch.endpoint == endpoint
    assert json.loads(respx.calls.last.request.content) == {"name": "litellm-batch", "input_file_id": "file_1"}


@respx.mock
async def test_retrieve_after_a_non_chat_create_reports_chat() -> None:
    respx.post(f"{API_BASE}/v1/batches").respond(200, json=_XAI_BATCH)
    respx.get(f"{API_BASE}/v1/batches/batch_1").respond(200, json=_XAI_BATCH)

    created: Final = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/embeddings",
        input_file_id="file_1",
        custom_llm_provider="xai",
        api_key=KEY,
        api_base=API_BASE,
    )
    retrieved: Final = await litellm.aretrieve_batch(
        batch_id="batch_1", custom_llm_provider="xai", api_key=KEY, api_base=API_BASE
    )

    assert isinstance(created, LiteLLMBatch) and isinstance(retrieved, LiteLLMBatch)
    assert (created.endpoint, retrieved.endpoint) == ("/v1/embeddings", "/v1/chat/completions")


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_retrieve_batch_reads_native_batch_route(sync_mode: bool) -> None:
    respx.get(f"{API_BASE}/v1/batches/batch_1").respond(
        200, json={**_XAI_BATCH, "state": {"num_requests": 2, "num_pending": 2}}
    )

    kwargs: Final = {"batch_id": "batch_1", "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}
    batch: Final = litellm.retrieve_batch(**kwargs) if sync_mode else await litellm.aretrieve_batch(**kwargs)

    assert isinstance(batch, LiteLLMBatch)
    assert (batch.status, batch.output_file_id, batch.input_file_id, batch.endpoint) == (
        "in_progress",
        None,
        "file_1",
        "/v1/chat/completions",
    )


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_cancel_batch_uses_colon_cancel_route(sync_mode: bool) -> None:
    route: Final = respx.post(f"{API_BASE}/v1/batches/batch_1:cancel").respond(
        200, json={**_XAI_BATCH, "cancel_time": "2026-09-23", "state": {}}
    )

    kwargs: Final = {"batch_id": "batch_1", "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}
    batch: Final = litellm.cancel_batch(**kwargs) if sync_mode else await litellm.acancel_batch(**kwargs)

    assert route.called
    assert isinstance(batch, LiteLLMBatch)
    assert (batch.status, batch.endpoint) == ("cancelled", "/v1/chat/completions")


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_list_batches_forwards_cursor_and_returns_openai_list(sync_mode: bool) -> None:
    route: Final = respx.get(f"{API_BASE}/v1/batches").respond(
        200, json={"batches": [_XAI_BATCH], "pagination_token": "next"}
    )

    kwargs: Final = {"custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE, "after": "cur", "limit": 5}
    listed: Final = litellm.list_batches(**kwargs) if sync_mode else await litellm.alist_batches(**kwargs)

    assert dict(route.calls.last.request.url.params) == {"limit": "5", "pagination_token": "cur"}
    assert listed.object == "list"
    assert [(b.id, b.endpoint) for b in listed.data] == [("batch_1", "/v1/chat/completions")]
    assert (listed.has_more, listed.next_page_token) == (True, "next")


@respx.mock
async def test_list_batches_treats_empty_pagination_token_as_last_page() -> None:
    respx.get(f"{API_BASE}/v1/batches").respond(200, json={"batches": [_XAI_BATCH], "pagination_token": ""})

    listed: Final = await litellm.alist_batches(custom_llm_provider="xai", api_key=KEY, api_base=API_BASE)

    assert (listed.has_more, listed.next_page_token) == (False, None)
    assert [batch.endpoint for batch in listed.data] == ["/v1/chat/completions"]


@respx.mock
async def test_file_content_stops_paging_on_empty_pagination_token() -> None:
    route: Final = respx.get(f"{API_BASE}/v1/batches/batch_1/results").respond(
        200,
        json={
            "results": [{"batch_request_id": "r1", "batch_result": {"error": {"code": 3, "message": "boom"}}}],
            "pagination_token": "",
        },
    )

    content: Final = await litellm.afile_content(
        file_id="batch_1", custom_llm_provider="xai", api_key=KEY, api_base=API_BASE
    )

    assert route.call_count == 1
    assert len(content.content.decode().splitlines()) == 1


@pytest.mark.parametrize("operation", ["create", "retrieve", "cancel", "list", "file_content"])
@respx.mock
async def test_batch_calls_fall_back_to_litellm_xai_key(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "xai_key", "configured-xai-key")
    monkeypatch.setattr(litellm, "api_key", "generic-key-must-not-be-used")
    routes: Final = {
        "create": respx.post(f"{API_BASE}/v1/batches").respond(200, json=_XAI_BATCH),
        "retrieve": respx.get(f"{API_BASE}/v1/batches/batch_1").respond(200, json=_XAI_BATCH),
        "cancel": respx.post(f"{API_BASE}/v1/batches/batch_1:cancel").respond(200, json=_XAI_BATCH),
        "list": respx.get(f"{API_BASE}/v1/batches").respond(
            200, json={"batches": [_XAI_BATCH], "pagination_token": None}
        ),
        "file_content": respx.get(f"{API_BASE}/v1/batches/batch_1/results").respond(
            200, json={"results": [], "pagination_token": None}
        ),
    }

    if operation == "create":
        await litellm.acreate_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="file_1",
            custom_llm_provider="xai",
            api_base=API_BASE,
        )
    elif operation == "retrieve":
        await litellm.aretrieve_batch(batch_id="batch_1", custom_llm_provider="xai", api_base=API_BASE)
    elif operation == "cancel":
        await litellm.acancel_batch(batch_id="batch_1", custom_llm_provider="xai", api_base=API_BASE)
    elif operation == "list":
        await litellm.alist_batches(custom_llm_provider="xai", api_base=API_BASE)
    else:
        await litellm.afile_content(file_id="batch_1", custom_llm_provider="xai", api_base=API_BASE)

    assert routes[operation].calls.last.request.headers["authorization"] == "Bearer configured-xai-key"


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_file_content_of_a_batch_id_walks_every_results_page(sync_mode: bool) -> None:
    def _page(request: httpx.Request) -> httpx.Response:
        token: Final = request.url.params.get("pagination_token")
        if token is None:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "batch_request_id": "r1",
                            "batch_result": {"response": {"chat_get_completion": {"id": "c1", "choices": []}}},
                        }
                    ],
                    "pagination_token": "r1",
                },
            )
        assert token == "r1"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"batch_request_id": "r2", "batch_result": {"error": {"code": 3, "message": "boom"}}},
                ],
                "pagination_token": None,
            },
        )

    route: Final = respx.get(f"{API_BASE}/v1/batches/batch_1/results").mock(side_effect=_page)

    kwargs: Final = {"file_id": "batch_1", "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}
    content: Final = litellm.file_content(**kwargs) if sync_mode else await litellm.afile_content(**kwargs)

    assert route.call_count == 2
    assert [dict(c.request.url.params) for c in route.calls] == [
        {"limit": "1000"},
        {"limit": "1000", "pagination_token": "r1"},
    ]
    assert [json.loads(line) for line in content.content.decode().splitlines()] == [
        {
            "id": "batch_req_r1",
            "custom_id": "r1",
            "response": {"status_code": 200, "request_id": "c1", "body": {"id": "c1", "choices": []}},
            "error": None,
        },
        {"id": "batch_req_r2", "custom_id": "r2", "response": None, "error": {"code": "3", "message": "boom"}},
    ]


def _two_results_pages(request: httpx.Request) -> httpx.Response:
    if request.url.params.get("pagination_token") is None:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "batch_request_id": "r1",
                        "batch_result": {"response": {"chat_get_completion": {"id": "c1", "choices": []}}},
                    }
                ],
                "pagination_token": "r1",
            },
        )
    return httpx.Response(
        200,
        json={
            "results": [{"batch_request_id": "r2", "batch_result": {"error": {"code": 3, "message": "boom"}}}],
            "pagination_token": None,
        },
    )


def _completed_batch(batch_id: str) -> respx.Route:
    return respx.get(f"{API_BASE}/v1/batches/{batch_id}").respond(200, json={**_XAI_BATCH, "batch_id": batch_id})


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_file_retrieve_of_a_batch_id_reports_the_results_file_without_a_files_lookup(sync_mode: bool) -> None:
    batch_id: Final = f"batch_retrieve_{sync_mode}"
    batch: Final = _completed_batch(batch_id)
    results: Final = respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_two_results_pages)
    files_lookup: Final = respx.get(f"{API_BASE}/v1/files/{batch_id}").respond(404, json={"code": "not-found"})

    kwargs: Final = {"file_id": batch_id, "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}
    file_object: Final = litellm.file_retrieve(**kwargs) if sync_mode else await litellm.afile_retrieve(**kwargs)
    content: Final = litellm.file_content(**kwargs)

    assert (files_lookup.call_count, batch.call_count, results.call_count) == (0, 1, 4)
    assert (
        file_object.id,
        file_object.purpose,
        file_object.filename,
        file_object.bytes,
        file_object.status,
        file_object.created_at,
    ) == (
        batch_id,
        "batch_output",
        f"{batch_id}_results.jsonl",
        len(content.content),
        "processed",
        1790121600,
    )
    assert [json.loads(line)["custom_id"] for line in content.content.decode().splitlines()] == ["r1", "r2"]


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_file_retrieve_of_an_unfinished_batch_is_a_404_without_downloading_results(sync_mode: bool) -> None:
    batch_id: Final = f"batch_unfinished_{sync_mode}"
    respx.get(f"{API_BASE}/v1/batches/{batch_id}").respond(
        200,
        json={**_XAI_BATCH, "batch_id": batch_id, "state": {**_XAI_BATCH["state"], "num_pending": 1}},
    )
    results: Final = respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_two_results_pages)

    kwargs: Final = {"file_id": batch_id, "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}
    with pytest.raises(XAIBatchesError, match="has no results file until it completes") as raised:
        litellm.file_retrieve(**kwargs) if sync_mode else await litellm.afile_retrieve(**kwargs)

    assert (raised.value.status_code, results.call_count) == (404, 0)


@respx.mock
async def test_async_file_retrieve_of_a_batch_id_downloads_the_results_once() -> None:
    batch_id: Final = "batch_concurrent"
    batch: Final = _completed_batch(batch_id)
    results: Final = respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_two_results_pages)

    kwargs: Final = {"file_id": batch_id, "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}
    concurrent: Final = await asyncio.gather(*(litellm.afile_retrieve(**kwargs) for _ in range(5)))
    later: Final = await litellm.afile_retrieve(**kwargs)

    assert (batch.call_count, results.call_count) == (6, 2)
    assert {(file_object.filename, file_object.bytes) for file_object in (*concurrent, later)} == {
        (f"{batch_id}_results.jsonl", concurrent[0].bytes)
    }
    assert concurrent[0].bytes > 0


@respx.mock
async def test_async_file_retrieve_keeps_downloading_after_its_caller_gives_up() -> None:
    batch_id: Final = "batch_abandoned"
    _completed_batch(batch_id)
    first_page_requested: Final = asyncio.Event()
    release_first_page: Final = asyncio.Event()

    page_requests: Final[list[httpx.URL]] = []

    async def _held_pages(request: httpx.Request) -> httpx.Response:
        page_requests.append(request.url)
        first_page_requested.set()
        await release_first_page.wait()
        return _two_results_pages(request)

    respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_held_pages)
    kwargs: Final = {"file_id": batch_id, "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}

    abandoned: Final = asyncio.ensure_future(litellm.afile_retrieve(**kwargs))
    await first_page_requested.wait()
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    release_first_page.set()
    retried: Final = await litellm.afile_retrieve(**kwargs)

    assert (len(page_requests), retried.filename) == (2, f"{batch_id}_results.jsonl")


@respx.mock
def test_a_download_left_pending_by_a_closed_event_loop_is_dropped_and_restarted(
    results_handler: XAIBatchesHandler,
) -> None:
    first_page_requested: Final = asyncio.Event()
    abandoned_loop_open: Final = [True]

    async def _pages(request: httpx.Request) -> httpx.Response:
        if abandoned_loop_open[0]:
            first_page_requested.set()
            await asyncio.Event().wait()
        return _two_results_pages(request)

    for batch_id in ("batch_closed_loop", "batch_after_closed_loop"):
        _completed_batch(batch_id)
        respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_pages)

    def _retrieve(batch_id: str) -> Coroutine[None, None, OpenAIFileObject]:
        return litellm.afile_retrieve(file_id=batch_id, custom_llm_provider="xai", api_key=KEY, api_base=API_BASE)

    async def _abandon() -> None:
        caller: Final = asyncio.ensure_future(_retrieve("batch_closed_loop"))
        await first_page_requested.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller

    abandoned_loop: Final = asyncio.new_event_loop()
    abandoned_loop.run_until_complete(_abandon())
    abandoned_loop.close()
    abandoned_loop_open[0] = False
    fresh_loop: Final = asyncio.new_event_loop()
    try:
        other: Final = fresh_loop.run_until_complete(_retrieve("batch_after_closed_loop"))
        pending_after_other: Final = tuple(key[0] for key in results_handler._results_sizes._downloads)
        restarted: Final = fresh_loop.run_until_complete(_retrieve("batch_closed_loop"))
    finally:
        fresh_loop.close()

    assert pending_after_other == ()
    assert (other.filename, restarted.filename, restarted.bytes) == (
        "batch_after_closed_loop_results.jsonl",
        "batch_closed_loop_results.jsonl",
        other.bytes,
    )


@respx.mock
def test_a_download_running_on_another_event_loop_is_not_awaited_across_loops() -> None:
    batch_id: Final = "batch_two_loops"
    _completed_batch(batch_id)
    first_page_requested: Final = asyncio.Event()
    first_loop_waiting: Final = [True]

    async def _pages(request: httpx.Request) -> httpx.Response:
        if first_loop_waiting[0]:
            first_loop_waiting[0] = False
            first_page_requested.set()
            await asyncio.Event().wait()
        return _two_results_pages(request)

    respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_pages)

    def _retrieve() -> Coroutine[None, None, OpenAIFileObject]:
        return litellm.afile_retrieve(file_id=batch_id, custom_llm_provider="xai", api_key=KEY, api_base=API_BASE)

    async def _start_and_leave_pending() -> asyncio.Future[OpenAIFileObject]:
        caller: Final = asyncio.ensure_future(_retrieve())
        await first_page_requested.wait()
        return caller

    first_loop: Final = asyncio.new_event_loop()
    second_loop: Final = asyncio.new_event_loop()
    try:
        still_waiting: Final = first_loop.run_until_complete(_start_and_leave_pending())
        described: Final = second_loop.run_until_complete(_retrieve())
        still_waiting.cancel()
    finally:
        second_loop.close()
        first_loop.close()

    assert (described.filename, described.bytes > 0) == (f"{batch_id}_results.jsonl", True)


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_file_retrieve_after_a_complete_content_download_reuses_its_size(sync_mode: bool) -> None:
    batch_id: Final = f"batch_content_first_{sync_mode}"
    batch: Final = _completed_batch(batch_id)
    results: Final = respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_two_results_pages)

    kwargs: Final = {"file_id": batch_id, "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}
    content: Final = litellm.file_content(**kwargs) if sync_mode else await litellm.afile_content(**kwargs)
    file_object: Final = litellm.file_retrieve(**kwargs) if sync_mode else await litellm.afile_retrieve(**kwargs)

    assert (batch.call_count, results.call_count, file_object.bytes) == (1, 2, len(content.content))


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_file_retrieve_after_a_partial_content_download_downloads_the_results_again(sync_mode: bool) -> None:
    batch_id: Final = f"batch_partial_first_{sync_mode}"
    _completed_batch(batch_id)
    partial_served: Final = [False]

    def _partial_then_complete(request: httpx.Request) -> httpx.Response:
        if partial_served[0]:
            return _two_results_pages(request)
        partial_served[0] = True
        return httpx.Response(
            200,
            json={
                "results": [{"batch_request_id": "r1", "batch_result": {"error": {"code": 3, "message": "boom"}}}],
                "pagination_token": None,
            },
        )

    results: Final = respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_partial_then_complete)

    kwargs: Final = {"file_id": batch_id, "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}
    partial: Final = litellm.file_content(**kwargs) if sync_mode else await litellm.afile_content(**kwargs)
    file_object: Final = litellm.file_retrieve(**kwargs) if sync_mode else await litellm.afile_retrieve(**kwargs)

    assert results.call_count == 3
    assert file_object.bytes > len(partial.content)


@respx.mock
async def test_async_file_retrieve_forgets_the_oldest_batch_beyond_the_cache_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xai_batches_handler, "_RESULTS_SIZE_CACHE_SIZE", 1)
    monkeypatch.setattr(files_main, "xai_batch_results_instance", XAIBatchesHandler())
    results: Final = {
        batch_id: respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_two_results_pages)
        for batch_id in ("batch_evicted", "batch_newer")
    }
    for batch_id in results:
        _completed_batch(batch_id)

    for batch_id in ("batch_evicted", "batch_newer", "batch_evicted", "batch_evicted"):
        await litellm.afile_retrieve(file_id=batch_id, custom_llm_provider="xai", api_key=KEY, api_base=API_BASE)

    assert (results["batch_evicted"].call_count, results["batch_newer"].call_count) == (4, 2)


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_file_retrieve_measures_again_when_the_batch_gains_requests(sync_mode: bool) -> None:
    batch_id: Final = f"batch_grown_{sync_mode}"
    grown: Final = [False]

    def _state(_request: httpx.Request) -> httpx.Response:
        state: Final = {**_XAI_BATCH["state"], "num_requests": 3, "num_success": 3} if grown[0] else _XAI_BATCH["state"]
        return httpx.Response(200, json={**_XAI_BATCH, "batch_id": batch_id, "state": state})

    def _pages(request: httpx.Request) -> httpx.Response:
        if not grown[0]:
            return _two_results_pages(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"batch_request_id": f"r{index}", "batch_result": {"error": {"code": 3, "message": "boom"}}}
                    for index in range(3)
                ],
                "pagination_token": None,
            },
        )

    respx.get(f"{API_BASE}/v1/batches/{batch_id}").mock(side_effect=_state)
    respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_pages)
    kwargs: Final = {"file_id": batch_id, "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}

    before: Final = litellm.file_retrieve(**kwargs) if sync_mode else await litellm.afile_retrieve(**kwargs)
    grown[0] = True
    after: Final = litellm.file_retrieve(**kwargs) if sync_mode else await litellm.afile_retrieve(**kwargs)
    content: Final = litellm.file_content(**kwargs)

    assert (after.bytes, after.bytes != before.bytes) == (len(content.content), True)


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_file_retrieve_of_a_download_missing_results_is_a_retryable_error(sync_mode: bool) -> None:
    batch_id: Final = f"batch_short_{sync_mode}"
    _completed_batch(batch_id)
    complete: Final = [False]

    def _pages(request: httpx.Request) -> httpx.Response:
        if complete[0]:
            return _two_results_pages(request)
        return httpx.Response(
            200,
            json={
                "results": [{"batch_request_id": "r1", "batch_result": {"error": {"code": 3, "message": "boom"}}}],
                "pagination_token": None,
            },
        )

    results: Final = respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_pages)
    kwargs: Final = {"file_id": batch_id, "custom_llm_provider": "xai", "api_key": KEY, "api_base": API_BASE}

    with pytest.raises(XAIBatchesError, match="1 of 2 results") as short:
        litellm.file_retrieve(**kwargs) if sync_mode else await litellm.afile_retrieve(**kwargs)
    complete[0] = True
    full: Final = litellm.file_retrieve(**kwargs) if sync_mode else await litellm.afile_retrieve(**kwargs)
    again: Final = litellm.file_retrieve(**kwargs) if sync_mode else await litellm.afile_retrieve(**kwargs)
    content: Final = litellm.file_content(**kwargs)

    assert (short.value.status_code, results.call_count) == (409, 5)
    assert (full.bytes, again.bytes) == (len(content.content), len(content.content))


@respx.mock
async def test_async_file_retrieve_keeps_a_recently_read_batch_over_an_older_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xai_batches_handler, "_RESULTS_SIZE_CACHE_SIZE", 2)
    monkeypatch.setattr(files_main, "xai_batch_results_instance", XAIBatchesHandler())
    results: Final = {
        batch_id: respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_two_results_pages)
        for batch_id in ("batch_reread", "batch_idle", "batch_latest")
    }
    for batch_id in results:
        _completed_batch(batch_id)

    for batch_id in ("batch_reread", "batch_idle", "batch_reread", "batch_latest", "batch_reread"):
        await litellm.afile_retrieve(file_id=batch_id, custom_llm_provider="xai", api_key=KEY, api_base=API_BASE)

    assert {batch_id: route.call_count for batch_id, route in results.items()} == {
        "batch_reread": 2,
        "batch_idle": 2,
        "batch_latest": 2,
    }


@respx.mock
async def test_async_file_retrieve_does_not_share_a_download_across_timeouts() -> None:
    batch_id: Final = "batch_two_timeouts"
    _completed_batch(batch_id)
    results: Final = respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_two_results_pages)

    await asyncio.gather(
        *(
            litellm.afile_retrieve(
                file_id=batch_id, custom_llm_provider="xai", api_key=KEY, api_base=API_BASE, timeout=timeout
            )
            for timeout in (5.0, 5.0, 60.0)
        )
    )

    assert results.call_count == 4


@respx.mock
async def test_async_file_retrieve_does_not_share_results_across_api_keys() -> None:
    batch_id: Final = "batch_two_keys"
    _completed_batch(batch_id)
    results: Final = respx.get(f"{API_BASE}/v1/batches/{batch_id}/results").mock(side_effect=_two_results_pages)

    for key in (KEY, "xai-other-key"):
        await litellm.afile_retrieve(file_id=batch_id, custom_llm_provider="xai", api_key=key, api_base=API_BASE)

    assert [call.request.headers["authorization"] for call in results.calls] == [
        f"Bearer {KEY}",
        f"Bearer {KEY}",
        "Bearer xai-other-key",
        "Bearer xai-other-key",
    ]


@respx.mock
async def test_file_content_unwraps_image_and_video_result_bodies() -> None:
    respx.get(f"{API_BASE}/v1/batches/batch_1/results").respond(
        200,
        json={
            "results": [
                {
                    "batch_request_id": "img",
                    "batch_result": {
                        "response": {"image_generation": {"data": [{"url": "https://cdn.example/img.png"}]}}
                    },
                },
                {
                    "batch_request_id": "vid",
                    "batch_result": {
                        "response": {"video_generation": {"id": "vid_1", "url": "https://cdn.example/clip.mp4"}}
                    },
                },
            ],
            "pagination_token": None,
        },
    )

    content: Final = await litellm.afile_content(
        file_id="batch_1", custom_llm_provider="xai", api_key=KEY, api_base=API_BASE
    )

    assert [json.loads(line)["response"]["body"] for line in content.content.decode().splitlines()] == [
        {"data": [{"url": "https://cdn.example/img.png"}]},
        {"id": "vid_1", "url": "https://cdn.example/clip.mp4"},
    ]


@respx.mock
async def test_missing_xai_key_is_a_401_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", "generic-key-must-not-be-used")
    route: Final = respx.post(f"{API_BASE}/v1/batches").respond(200, json=_XAI_BATCH)

    with pytest.raises(XAIBatchesError) as exc:
        await litellm.acreate_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="file_1",
            custom_llm_provider="xai",
            api_base=API_BASE,
        )

    assert exc.value.status_code == 401
    assert route.called is False


@respx.mock
async def test_upstream_error_surfaces_status_code_and_body() -> None:
    respx.get(f"{API_BASE}/v1/batches/batch_missing").respond(404, json={"code": "404", "error": "not found"})

    with pytest.raises(XAIBatchesError) as exc:
        await litellm.aretrieve_batch(
            batch_id="batch_missing", custom_llm_provider="xai", api_key=KEY, api_base=API_BASE
        )

    assert exc.value.status_code == 404
    assert "not found" in exc.value.message
