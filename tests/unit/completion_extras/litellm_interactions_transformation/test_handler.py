import json
from datetime import UTC, datetime
from typing import Final

import httpx
import pytest

from litellm.completion_extras.litellm_interactions_transformation import (
    InteractionsToCompletionBridgeHandler,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.interactions.transformation import VertexAIInteractionsConfig
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import CallTypes, Choices, ModelResponse


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

    bridge: Final = InteractionsToCompletionBridgeHandler()
    with pytest.raises(BaseLLMException) as exc_info:
        await bridge.acompletion(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "ping"}],
            model_response=ModelResponse(id="fallback"),
            logging_obj=_logging_stub("gemini-3-flash-preview", stream=True),
            optional_params={"stream": True},
            litellm_params={},
            api_key="test-gemini-key",
            client=async_client,
            custom_llm_provider="gemini",
            stream=True,
        )

    assert exc_info.value.status_code == 429
    assert "Quota exceeded for interactions" in str(exc_info.value)


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

    bridge: Final = InteractionsToCompletionBridgeHandler()
    stream_resp: Final = bridge.completion(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "ping"}],
        model_response=ModelResponse(id="fallback"),
        logging_obj=_logging_stub("gemini-3-flash-preview", stream=True),
        optional_params={"stream": True},
        litellm_params={},
        api_key="test-gemini-key",
        client=sync_client,
        custom_llm_provider="gemini",
        stream=True,
    )

    assert isinstance(stream_resp, CustomStreamWrapper)
    emitted: Final = "".join(chunk.choices[0].delta.content or "" for chunk in stream_resp)
    assert emitted == "STREAM_OK"
