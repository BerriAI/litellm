import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

from litellm.completion_extras.litellm_interactions_transformation import (
    InteractionsToCompletionBridgeHandler,
)
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.interactions.transformation import VertexAIInteractionsConfig
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import CallTypes, Choices, ModelResponse


_OBJ_MAP: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def _logged_original_response(log_obj: Logging) -> str:
    details: Final = _OBJ_MAP.validate_python(log_obj.model_call_details)  # pyright: ignore[reportAny]  # Logging.model_call_details is untyped dict[str, Any]
    return str(details.get("original_response", ""))


def _logging_stub(model: str, *, stream: bool = False) -> Logging:
    return Logging(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        stream=stream,
        call_type=CallTypes.completion.value,
        start_time=datetime(2026, 10, 22, tzinfo=UTC),
        litellm_call_id="call-handler-test",
        function_id="fn-handler-test",
    )


def test_gemini_sync_completion_sends_interactions_payload_and_headers() -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: transport request spy

    def transport_fn(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json={
                "id": "v1_sync_1",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "PONG"}]}],
                "usage": {"total_input_tokens": 5, "total_output_tokens": 2, "total_tokens": 7},
            },
        )

    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=httpx.MockTransport(transport_fn))

    bridge: Final = InteractionsToCompletionBridgeHandler()
    result: Final = bridge.completion(
        model="gemini-3-flash-preview",
        messages=[
            {"role": "system", "content": "Reply uppercase."},
            {"role": "user", "content": "ping"},
        ],
        model_response=ModelResponse(id="fallback"),
        logging_obj=_logging_stub("gemini-3-flash-preview"),
        optional_params={"temperature": 0.2},
        litellm_params={},
        api_key="test-gemini-key",
        api_base="https://generativelanguage.googleapis.com",
        client=sync_client,
        custom_llm_provider="gemini",
        stream=False,
    )

    assert isinstance(result, ModelResponse)
    choice: Final = result.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "PONG"
    assert len(seen) == 1
    assert str(seen[0].url) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert seen[0].headers["x-goog-api-key"] == "test-gemini-key"
    assert seen[0].headers["Api-Revision"] == "2026-05-20"
    assert json.loads(seen[0].content) == {
        "model": "gemini-3-flash-preview",
        "input": [{"type": "user_input", "content": [{"type": "text", "text": "ping"}]}],
        "system_instruction": "Reply uppercase.",
        "generation_config": {"temperature": 0.2},
    }


@pytest.mark.asyncio
async def test_vertex_async_completion_uses_global_v1beta1_and_oauth_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-east5")
    seen: Final[list[httpx.Request]] = []  # mutable-ok: transport request spy

    def minter(_creds: VERTEX_CREDENTIALS_TYPES | None, project_id: str | None) -> tuple[str, str]:
        return "vertex-oauth-token", project_id or "default-proj"

    def transport_fn(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json={
                "id": "v1_vertex_1",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "VERTEX_PONG"}]}],
                "usage": {"total_input_tokens": 6, "total_output_tokens": 3, "total_tokens": 9},
            },
        )

    async_client: Final = AsyncHTTPHandler()
    async_client.client = httpx.AsyncClient(transport=httpx.MockTransport(transport_fn))

    bridge: Final = InteractionsToCompletionBridgeHandler(
        provider_config=VertexAIInteractionsConfig(mint_access_token=minter)
    )
    result: Final = await bridge.acompletion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "ping"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=_logging_stub("gemini-3-flash-preview"),
        optional_params={},
        litellm_params={"vertex_project": "demo-proj"},
        client=async_client,
        custom_llm_provider="vertex_ai",
        stream=False,
    )

    assert isinstance(result, ModelResponse)
    choice: Final = result.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "VERTEX_PONG"
    assert len(seen) == 1
    assert (
        str(seen[0].url) == "https://aiplatform.googleapis.com/v1beta1/projects/demo-proj/locations/global/interactions"
    )
    assert seen[0].headers["authorization"] == "Bearer vertex-oauth-token"


@pytest.mark.asyncio
async def test_streaming_error_reads_unconsumed_response_body_into_provider_exception() -> None:
    def failing_stream_transport(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            stream=httpx.ByteStream(b'{"error":{"message":"Quota exceeded for interactions"}}'),
        )

    async_client: Final = AsyncHTTPHandler()
    async_client.client = httpx.AsyncClient(transport=httpx.MockTransport(failing_stream_transport))
    log_obj: Final = _logging_stub("gemini-3-flash-preview", stream=True)

    bridge: Final = InteractionsToCompletionBridgeHandler()
    with pytest.raises(RateLimitError) as exc_info:
        await bridge.acompletion(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "ping"}],
            model_response=ModelResponse(id="fallback"),
            logging_obj=log_obj,
            optional_params={"stream": True},
            litellm_params={},
            api_key="test-gemini-key",
            client=async_client,
            custom_llm_provider="gemini",
            stream=True,
        )

    assert exc_info.value.status_code == 429
    assert getattr(exc_info.value, "model", None) == "gemini-3-flash-preview"
    assert getattr(exc_info.value, "llm_provider", None) == "gemini"
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
    assert "Quota exceeded for interactions" in str(exc_info.value)
    assert "Quota exceeded for interactions" in _logged_original_response(log_obj)


def test_gemini_sync_streaming_yields_custom_stream_wrapper() -> None:
    sse_payload: Final = (
        b'data: {"event_type": "interaction.created", "interaction": {"id": "v1_stream_1"}}\n\n'
        b'data: {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "STREAM_OK"}}\n\n'
        b'data: {"event_type": "interaction.completed", "interaction": {"id": "v1_stream_1", "status": "completed", "usage": {"total_input_tokens": 4, "total_output_tokens": 2, "total_tokens": 6}}}\n\n'
        b"data: [DONE]\n\n"
    )

    def stream_transport(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(sse_payload))

    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=httpx.MockTransport(stream_transport))
    log_obj: Final = _logging_stub("gemini-3-flash-preview", stream=True)

    bridge: Final = InteractionsToCompletionBridgeHandler()
    stream_resp: Final = bridge.completion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "ping"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=log_obj,
        optional_params={"stream": True},
        litellm_params={},
        api_key="test-gemini-key",
        client=sync_client,
        custom_llm_provider="gemini",
        stream=True,
    )

    assert isinstance(stream_resp, CustomStreamWrapper)
    assert _logged_original_response(log_obj) == "first stream response received"
    emitted: Final = "".join(chunk.choices[0].delta.content or "" for chunk in stream_resp)
    assert emitted == "STREAM_OK"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "grpc_status", "expected_cls"),
    (
        (400, "INVALID_ARGUMENT", BadRequestError),
        (401, "UNAUTHENTICATED", AuthenticationError),
        (403, "PERMISSION_DENIED", PermissionDeniedError),
        (404, "NOT_FOUND", NotFoundError),
        (429, "RESOURCE_EXHAUSTED", RateLimitError),
        (500, "INTERNAL", InternalServerError),
    ),
)
@pytest.mark.parametrize("provider", ("gemini", "vertex_ai"))
async def test_http_status_errors_map_realistic_google_envelopes_sync_and_async(
    status_code: int,
    grpc_status: str,
    expected_cls: type[Exception],
    provider: str,
) -> None:
    def minter(_creds: VERTEX_CREDENTIALS_TYPES | None, project_id: str | None) -> tuple[str, str]:
        return "vertex-tok", project_id or "proj-1"

    def failing_transport(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={
                "error": {
                    "code": status_code,
                    "message": f"HTTP {status_code} failure ({grpc_status})",
                    "status": grpc_status,
                }
            },
        )

    bridge: Final = InteractionsToCompletionBridgeHandler(
        provider_config=VertexAIInteractionsConfig(mint_access_token=minter) if provider == "vertex_ai" else None
    )
    llm_params: Final[Mapping[str, object]] = {"vertex_project": "proj-1"} if provider == "vertex_ai" else {}

    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=httpx.MockTransport(failing_transport))
    sync_log: Final = _logging_stub("gemini-3-flash-preview")
    with pytest.raises(expected_cls) as sync_exc:
        bridge.completion(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "ping"}],
            model_response=ModelResponse(id="fallback"),
            logging_obj=sync_log,
            optional_params={},
            litellm_params=llm_params,
            api_key="test-gemini-key",
            client=sync_client,
            custom_llm_provider=provider,
            stream=False,
        )
    assert getattr(sync_exc.value, "model", None) == "gemini-3-flash-preview"
    assert getattr(sync_exc.value, "llm_provider", None) == provider
    assert isinstance(sync_exc.value.__cause__, httpx.HTTPStatusError)
    assert f"HTTP {status_code} failure" in _logged_original_response(sync_log)

    async_client: Final = AsyncHTTPHandler()
    async_client.client = httpx.AsyncClient(transport=httpx.MockTransport(failing_transport))
    async_log: Final = _logging_stub("gemini-3-flash-preview")
    with pytest.raises(expected_cls) as async_exc:
        await bridge.acompletion(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "ping"}],
            model_response=ModelResponse(id="fallback"),
            logging_obj=async_log,
            optional_params={},
            litellm_params=llm_params,
            api_key="test-gemini-key",
            client=async_client,
            custom_llm_provider=provider,
            stream=False,
        )
    assert getattr(async_exc.value, "model", None) == "gemini-3-flash-preview"
    assert getattr(async_exc.value, "llm_provider", None) == provider
    assert isinstance(async_exc.value.__cause__, httpx.HTTPStatusError)
    assert f"HTTP {status_code} failure" in _logged_original_response(async_log)


def test_truncated_streaming_error_body_read_error_is_suppressed_cleanly() -> None:
    class _BrokenErrorStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            raise httpx.ReadError("connection reset reading error body")

    def broken_403_transport(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, stream=_BrokenErrorStream())

    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=httpx.MockTransport(broken_403_transport))
    bridge: Final = InteractionsToCompletionBridgeHandler()

    with pytest.raises(PermissionDeniedError) as exc_info:
        bridge.completion(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "ping"}],
            model_response=ModelResponse(id="fallback"),
            logging_obj=_logging_stub("gemini-3-flash-preview", stream=True),
            api_key="test-gemini-key",
            client=sync_client,
            custom_llm_provider="gemini",
            stream=True,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_connect_error_and_non_raising_4xx_response_map_cleanly_sync_and_async() -> None:
    def connect_fail(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("DNS lookup failed", request=req)

    def raw_400(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Invalid generation_config"}})

    sync_conn_client: Final = HTTPHandler()
    sync_conn_client.client = httpx.Client(transport=httpx.MockTransport(connect_fail))
    sync_conn_log: Final = _logging_stub("gemini-3-flash-preview")
    bridge: Final = InteractionsToCompletionBridgeHandler()

    with pytest.raises(APIConnectionError, match="DNS lookup failed") as sync_conn_exc:
        bridge.completion(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "ping"}],
            model_response=ModelResponse(id="fallback"),
            logging_obj=sync_conn_log,
            api_key="test-gemini-key",
            client=sync_conn_client,
            custom_llm_provider="gemini",
        )
    assert getattr(sync_conn_exc.value, "model", None) == "gemini-3-flash-preview"
    assert getattr(sync_conn_exc.value, "llm_provider", None) == "gemini"
    assert isinstance(sync_conn_exc.value.__cause__, httpx.ConnectError)
    assert "DNS lookup failed" in _logged_original_response(sync_conn_log)

    async_conn_log: Final = _logging_stub("gemini-3-flash-preview")
    async with httpx.AsyncClient(transport=httpx.MockTransport(connect_fail)) as async_conn_client:
        with pytest.raises(APIConnectionError, match="DNS lookup failed") as async_conn_exc:
            await bridge.acompletion(
                model="gemini-3-flash-preview",
                messages=[{"role": "user", "content": "ping"}],
                model_response=ModelResponse(id="fallback"),
                logging_obj=async_conn_log,
                api_key="test-gemini-key",
                client=async_conn_client,
                custom_llm_provider="gemini",
            )
    assert getattr(async_conn_exc.value, "model", None) == "gemini-3-flash-preview"
    assert getattr(async_conn_exc.value, "llm_provider", None) == "gemini"
    assert isinstance(async_conn_exc.value.__cause__, httpx.ConnectError)
    assert "DNS lookup failed" in _logged_original_response(async_conn_log)

    sync_raw_log: Final = _logging_stub("gemini-3-flash-preview")
    with (
        httpx.Client(transport=httpx.MockTransport(raw_400)) as non_raising_sync,
        pytest.raises(BadRequestError, match="Invalid generation_config") as sync_400_exc,
    ):
        bridge.completion(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "ping"}],
            model_response=ModelResponse(id="fallback"),
            logging_obj=sync_raw_log,
            api_key="test-gemini-key",
            client=non_raising_sync,
            custom_llm_provider="gemini",
        )
    assert getattr(sync_400_exc.value, "model", None) == "gemini-3-flash-preview"
    assert getattr(sync_400_exc.value, "llm_provider", None) == "gemini"
    assert "Invalid generation_config" in _logged_original_response(sync_raw_log)

    async_raw_log: Final = _logging_stub("gemini-3-flash-preview")
    async with httpx.AsyncClient(transport=httpx.MockTransport(raw_400)) as non_raising_async:
        with pytest.raises(BadRequestError, match="Invalid generation_config") as async_400_exc:
            await bridge.acompletion(
                model="gemini-3-flash-preview",
                messages=[{"role": "user", "content": "ping"}],
                model_response=ModelResponse(id="fallback"),
                logging_obj=async_raw_log,
                api_key="test-gemini-key",
                client=non_raising_async,
                custom_llm_provider="gemini",
            )
    assert getattr(async_400_exc.value, "model", None) == "gemini-3-flash-preview"
    assert getattr(async_400_exc.value, "llm_provider", None) == "gemini"
    assert "Invalid generation_config" in _logged_original_response(async_raw_log)


_RECITATION_BLOCK: Final = {
    "error": {
        "message": "Request blocked due to copyright/recitation content. Please modify your input and retry.",
        "code": "recitation",
    }
}


@pytest.mark.asyncio
async def test_answer_blocked_for_recitation_comes_back_as_a_content_filter_finish_sync_and_async() -> None:
    def blocked(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=_RECITATION_BLOCK, headers={"x-goog-request-id": "blocked-1"})

    bridge: Final = InteractionsToCompletionBridgeHandler()
    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=httpx.MockTransport(blocked))
    async_client: Final = AsyncHTTPHandler()
    async_client.client = httpx.AsyncClient(transport=httpx.MockTransport(blocked))
    sync_log: Final = _logging_stub("gemini-3-flash-preview")
    async_log: Final = _logging_stub("gemini-3-flash-preview")
    sync_answer: Final = bridge.completion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "recite it"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=sync_log,
        api_key="test-gemini-key",
        client=sync_client,
        custom_llm_provider="gemini",
    )
    async_answer: Final = await bridge.acompletion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "recite it"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=async_log,
        api_key="test-gemini-key",
        client=async_client,
        custom_llm_provider="gemini",
    )

    for answer, log in ((sync_answer, sync_log), (async_answer, async_log)):
        assert isinstance(answer, ModelResponse)
        assert answer.choices[0].finish_reason == "content_filter"
        assert answer.choices[0].message.content is None
        assert answer._hidden_params["headers"]["x-goog-request-id"] == "blocked-1"
        assert "recitation" in str(log.model_call_details["original_response"])


_SAFETY_BLOCK: Final = {"error": {"message": "Request blocked due to safety violations.", "code": "safety"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_body", [_RECITATION_BLOCK, _SAFETY_BLOCK], ids=["recitation", "safety"])
async def test_stream_refused_before_it_opens_ends_with_a_content_filter_finish_sync_and_async(
    blocked_body: Mapping[str, object],
) -> None:
    def blocked(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=blocked_body)

    bridge: Final = InteractionsToCompletionBridgeHandler()
    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=httpx.MockTransport(blocked))
    async_client: Final = AsyncHTTPHandler()
    async_client.client = httpx.AsyncClient(transport=httpx.MockTransport(blocked))
    sync_stream: Final = bridge.completion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "say it"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=_logging_stub("gemini-3-flash-preview"),
        api_key="test-gemini-key",
        client=sync_client,
        custom_llm_provider="gemini",
        optional_params={"stream": True},
    )
    async_stream: Final = await bridge.acompletion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "say it"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=_logging_stub("gemini-3-flash-preview"),
        api_key="test-gemini-key",
        client=async_client,
        custom_llm_provider="gemini",
        optional_params={"stream": True},
    )

    assert isinstance(sync_stream, CustomStreamWrapper)
    assert isinstance(async_stream, CustomStreamWrapper)
    sync_chunks: Final = tuple(sync_stream)
    async_chunks: Final = tuple([chunk async for chunk in async_stream])
    for chunks in (sync_chunks, async_chunks):
        assert [c.choices[0].finish_reason for c in chunks if c.choices[0].finish_reason] == ["content_filter"]
        assert "".join(c.choices[0].delta.content or "" for c in chunks) == ""


@pytest.mark.asyncio
async def test_answer_blocked_for_safety_comes_back_as_a_content_filter_finish_sync_and_async() -> None:
    def blocked(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=_SAFETY_BLOCK)

    bridge: Final = InteractionsToCompletionBridgeHandler()
    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=httpx.MockTransport(blocked))
    async_client: Final = AsyncHTTPHandler()
    async_client.client = httpx.AsyncClient(transport=httpx.MockTransport(blocked))
    sync_answer: Final = bridge.completion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "say it"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=_logging_stub("gemini-3-flash-preview"),
        api_key="test-gemini-key",
        client=sync_client,
        custom_llm_provider="gemini",
    )
    async_answer: Final = await bridge.acompletion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "say it"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=_logging_stub("gemini-3-flash-preview"),
        api_key="test-gemini-key",
        client=async_client,
        custom_llm_provider="gemini",
    )

    for answer in (sync_answer, async_answer):
        assert isinstance(answer, ModelResponse)
        assert answer.choices[0].finish_reason == "content_filter"
        assert answer.choices[0].message.content is None


def _function_call_transport(seen: list[httpx.Request]) -> httpx.MockTransport:  # mutable-ok: transport request spy
    def respond(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json={
                "id": "v1_legacy_fn",
                "status": "requires_action",
                "steps": [
                    {"type": "function_call", "id": "call_1", "name": "get_weather", "arguments": {"city": "SF"}}
                ],
                "usage": {"total_input_tokens": 5, "total_output_tokens": 2, "total_tokens": 7},
            },
        )

    return httpx.MockTransport(respond)


def test_legacy_functions_request_gets_a_function_call_answer_and_replays_function_call_history() -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: transport request spy
    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=_function_call_transport(seen))

    result: Final = InteractionsToCompletionBridgeHandler().completion(
        model="gemini-3-flash-preview",
        messages=[
            {"role": "user", "content": "weather in NYC then SF?"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            },
            {"role": "function", "name": "get_weather", "content": "rainy"},
        ],
        model_response=ModelResponse(id="fallback"),
        logging_obj=_logging_stub("gemini-3-flash-preview"),
        optional_params={
            "functions": [{"name": "get_weather", "parameters": {"type": "object", "properties": {}}}],
        },
        litellm_params={},
        api_key="test-gemini-key",
        api_base="https://generativelanguage.googleapis.com",
        client=sync_client,
        custom_llm_provider="gemini",
        stream=False,
    )

    assert isinstance(result, ModelResponse)
    choice: Final = result.choices[0]
    assert isinstance(choice, Choices)
    assert choice.finish_reason == "function_call"
    assert choice.message.tool_calls is None
    assert choice.message.function_call is not None
    assert choice.message.function_call.name == "get_weather"
    assert json.loads(choice.message.function_call.arguments) == {"city": "SF"}
    sent_steps: Final = _OBJ_MAP.validate_python(json.loads(seen[0].content))["input"]
    assert isinstance(sent_steps, list)
    assert [step for step in sent_steps if step["type"] in ("function_call", "function_result")] == [  # pyright: ignore[reportUnknownVariableType]  # parsed JSON steps
        {"type": "function_call", "id": "call_get_weather_1", "name": "get_weather", "arguments": {"city": "NYC"}},
        {
            "type": "function_result",
            "call_id": "call_get_weather_1",
            "name": "get_weather",
            "result": [{"type": "text", "text": "rainy"}],
        },
    ]


def test_tools_request_still_gets_tool_calls() -> None:
    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=_function_call_transport([]))

    result: Final = InteractionsToCompletionBridgeHandler().completion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "weather in SF?"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=_logging_stub("gemini-3-flash-preview"),
        optional_params={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
                }
            ],
        },
        litellm_params={},
        api_key="test-gemini-key",
        api_base="https://generativelanguage.googleapis.com",
        client=sync_client,
        custom_llm_provider="gemini",
        stream=False,
    )

    assert isinstance(result, ModelResponse)
    choice: Final = result.choices[0]
    assert isinstance(choice, Choices)
    assert choice.finish_reason == "tool_calls"
    assert choice.message.function_call is None
    assert [call.function.name for call in choice.message.tool_calls or []] == ["get_weather"]


def test_continued_answer_is_logged_as_the_joined_response_not_the_first_hop() -> None:
    hops: Final = iter(
        (
            {
                "id": "v1_hop_1",
                "status": "requires_continuation",
                "continuation_token": "tok_1",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "first "}]}],
                "usage": {"total_input_tokens": 5, "total_output_tokens": 2, "total_tokens": 7},
            },
            {
                "id": "v1_hop_2",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "second"}]}],
                "usage": {"total_input_tokens": 7, "total_output_tokens": 3, "total_tokens": 10},
            },
        )
    )
    sync_client: Final = HTTPHandler()
    sync_client.client = httpx.Client(transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=next(hops))))
    log_obj: Final = _logging_stub("gemini-3-flash-preview")

    result: Final = InteractionsToCompletionBridgeHandler().completion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "ping"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=log_obj,
        optional_params={"max_tokens": 1000},
        litellm_params={},
        api_key="test-gemini-key",
        api_base="https://generativelanguage.googleapis.com",
        client=sync_client,
        custom_llm_provider="gemini",
        stream=False,
    )

    assert isinstance(result, ModelResponse)
    choice: Final = result.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "first second"
    logged: Final = _logged_original_response(log_obj)
    assert "first " in logged
    assert "second" in logged
