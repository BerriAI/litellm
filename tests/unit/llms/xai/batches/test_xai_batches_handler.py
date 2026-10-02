import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.xai.batches.transformation import XAIBatchesError
from litellm.types.utils import LiteLLMBatch

API_BASE: Final = "https://api.x.ai"
KEY: Final = "xai-test-key"


@pytest.fixture(autouse=True)
def _httpx_transport_so_respx_can_intercept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


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
