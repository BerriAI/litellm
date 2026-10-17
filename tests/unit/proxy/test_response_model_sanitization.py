import asyncio
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient


import litellm
from tests._master_key import MASTER_KEY

pytestmark = pytest.mark.flaky(condition=False)


def _initialize_proxy_with_config(config: dict, tmp_path) -> TestClient:
    """
    Initialize the proxy server with a temporary config file and return a TestClient.

    IMPORTANT: proxy_server.initialize() mutates module-level globals. We must call
    cleanup_router_config_variables() before initializing to prevent cross-test bleed.
    """
    from litellm.proxy.proxy_server import (
        app,
        cleanup_router_config_variables,
        initialize,
    )

    cleanup_router_config_variables()

    config_fp = tmp_path / "proxy_config.yaml"
    config_fp.write_text(yaml.safe_dump(config))

    asyncio.run(initialize(config=str(config_fp), debug=True))
    return TestClient(app)


def _make_minimal_chat_completion_response(model: str) -> litellm.ModelResponse:
    response = litellm.ModelResponse()
    response.model = model
    response.choices[0].message.content = "hello"  # type: ignore[union-attr]
    response.choices[0].finish_reason = "stop"  # type: ignore[union-attr]
    return response


def _make_model_response_stream_chunk(model: str) -> litellm.ModelResponseStream:
    """
    Create a minimal OpenAI-compatible chat.completion.chunk object.
    """
    chunk_dict = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hi"},
                "finish_reason": None,
            }
        ],
    }
    return litellm.ModelResponseStream(**chunk_dict)


def _decode_sse_chunk(chunk) -> str:
    return chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk


def _make_response_created_event(model: str | None):
    from litellm.types.llms.openai import (
        ResponseCreatedEvent,
        ResponsesAPIResponse,
        ResponsesAPIStreamEvents,
    )

    return ResponseCreatedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_CREATED,
        response=ResponsesAPIResponse(id="resp_test", created_at=0, output=[], model=model),
    )


def test_restamp_streaming_chunk_skips_matching_model():
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    chunk = _make_model_response_stream_chunk("client-model")

    result, model_mismatch_logged = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert result is chunk
    assert result.model == "client-model"
    assert model_mismatch_logged is False


def test_fast_serialize_simple_streaming_chunk_matches_model_dump_json():
    from litellm.proxy.proxy_server import _serialize_streaming_chunk

    chunk = _make_model_response_stream_chunk("client-model")

    assert json.loads(_serialize_streaming_chunk(chunk)) == json.loads(
        chunk.model_dump_json(exclude_none=True, exclude_unset=True)
    )


def test_fast_serialize_returns_none_when_model_field_is_missing():
    """
    The fast path must mirror ``model_dump_json(exclude_none=True)``: when
    ``chunk.model`` is ``None`` the slow path omits the field entirely.
    Emitting ``"model": null`` would diverge and trip strict OpenAI-
    compatible clients that reject ``null`` for optional string fields.
    Falling back to ``None`` lets the canonical serializer handle the edge.
    """
    from litellm.proxy.proxy_server import (
        _fast_serialize_simple_model_response_stream,
        _serialize_streaming_chunk,
    )

    chunk = _make_model_response_stream_chunk("client-model")
    chunk.model = None  # type: ignore[assignment]

    assert _fast_serialize_simple_model_response_stream(chunk) is None

    # Going through the public ``_serialize_streaming_chunk`` should still
    # produce a serialized result via the slow-path fallback, and it must
    # not contain ``"model": null``.
    serialized = _serialize_streaming_chunk(chunk)
    payload_str = serialized.decode("utf-8") if isinstance(serialized, bytes) else serialized
    assert '"model": null' not in payload_str
    assert '"model":null' not in payload_str
    assert json.loads(payload_str) == json.loads(chunk.model_dump_json(exclude_none=True, exclude_unset=True))


def test_proxy_chat_completion_does_not_return_provider_prefixed_model(tmp_path, monkeypatch):
    """
    Regression test:

    - Client asks for `model="vllm-model"` (no provider prefix)
    - Internal provider path uses `hosted_vllm/...`
    - Proxy should not leak `hosted_vllm/` in the client-facing `model` field.
    """
    client_model = "vllm-model"
    internal_model = f"hosted_vllm/{client_model}"

    client = _initialize_proxy_with_config(
        config={
            "general_settings": {"master_key": MASTER_KEY},
            "model_list": [
                {
                    "model_name": client_model,
                    "litellm_params": {"model": internal_model},
                }
            ],
        },
        tmp_path=tmp_path,
    )

    # Patch router call to avoid making any real network request.
    from litellm.proxy import proxy_server

    monkeypatch.setattr(
        proxy_server.llm_router,  # type: ignore[arg-type]
        "acompletion",
        AsyncMock(return_value=_make_minimal_chat_completion_response(model=internal_model)),
    )

    # Also no-op proxy logging hooks to keep this test focused and deterministic.
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "during_call_hook", AsyncMock(return_value=None))
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "update_request_status",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "post_call_success_hook",
        AsyncMock(side_effect=lambda **kwargs: kwargs["response"]),
    )

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={"model": client_model, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == client_model
    assert not body["model"].startswith("hosted_vllm/")


@pytest.mark.asyncio
async def test_proxy_streaming_chunks_do_not_return_provider_prefixed_model(
    monkeypatch,
):
    """
    Regression test for streaming:

    Even if a streaming chunk contains `model="hosted_vllm/<...>"`, the proxy SSE layer
    should not leak the provider prefix to the client.
    """
    client_model = "vllm-model"
    internal_model = f"hosted_vllm/{client_model}"

    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    # Patch proxy_logging_obj hooks so async_data_generator yields exactly our chunk.
    async def _iterator_hook(
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator,
        request_data: dict,
    ):
        yield _make_model_response_stream_chunk(model=internal_model)

    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_iterator_hook",
        _iterator_hook,
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_hook",
        AsyncMock(side_effect=lambda **kwargs: kwargs["response"]),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "has_streaming_callbacks",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "needs_iterator_wrap",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "needs_per_chunk_streaming_hook",
        MagicMock(return_value=True),
    )

    user_api_key_dict = UserAPIKeyAuth(api_key=MASTER_KEY)

    gen = proxy_server.async_data_generator(
        response=MagicMock(),
        user_api_key_dict=user_api_key_dict,
        request_data={"model": client_model},
    )

    chunks = []
    async for item in gen:
        chunks.append(item)

    # First chunk is expected to be JSON, last chunk is [DONE]
    assert len(chunks) >= 2
    first = _decode_sse_chunk(chunks[0])
    assert first.startswith("data: ")

    payload = json.loads(first[len("data: ") :].strip())
    assert payload["model"] == client_model
    assert not payload["model"].startswith("hosted_vllm/")


@pytest.mark.asyncio
async def test_proxy_streaming_chunks_use_client_requested_model_before_alias_mapping(
    monkeypatch,
):
    """
    Regression test for alias mapping on streaming:

    - `common_processing_pre_call_logic` can rewrite `request_data["model"]` via model_alias_map / key-specific aliases.
    - Non-streaming responses are restamped using the original client-requested model (captured before the rewrite).
    - Streaming chunks must do the same to avoid mismatched `model` values between streaming and non-streaming.
    """
    client_model_alias = "alias-model"
    canonical_model = "vllm-model"
    internal_model = f"hosted_vllm/{canonical_model}"

    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    async def _iterator_hook(
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator,
        request_data: dict,
    ):
        yield _make_model_response_stream_chunk(model=internal_model)

    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_iterator_hook",
        _iterator_hook,
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_hook",
        AsyncMock(side_effect=lambda **kwargs: kwargs["response"]),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "has_streaming_callbacks",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "needs_iterator_wrap",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "needs_per_chunk_streaming_hook",
        MagicMock(return_value=True),
    )

    user_api_key_dict = UserAPIKeyAuth(api_key=MASTER_KEY)

    gen = proxy_server.async_data_generator(
        response=MagicMock(),
        user_api_key_dict=user_api_key_dict,
        request_data={
            "model": canonical_model,
            "_litellm_client_requested_model": client_model_alias,
        },
    )

    chunks = []
    async for item in gen:
        chunks.append(item)

    assert len(chunks) >= 2
    first = _decode_sse_chunk(chunks[0])
    assert first.startswith("data: ")

    payload = json.loads(first[len("data: ") :].strip())
    assert payload["model"] == client_model_alias
    assert not payload["model"].startswith("hosted_vllm/")


@pytest.mark.asyncio
async def test_proxy_streaming_azure_model_router_preserves_actual_model(monkeypatch):
    """
    Regression test for Azure Model Router streaming:

    When the client requests azure_ai/model_router, the streaming chunks should
    preserve the actual model used (e.g., azure_ai/gpt-5-nano-2025-08-07) from
    the downstream response, NOT override to the router model.
    """
    router_model = "azure_ai/model_router"
    actual_model_used = "azure_ai/gpt-5-nano-2025-08-07"

    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    async def _iterator_hook(
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator,
        request_data: dict,
    ):
        yield _make_model_response_stream_chunk(model=actual_model_used)

    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_iterator_hook",
        _iterator_hook,
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_hook",
        AsyncMock(side_effect=lambda **kwargs: kwargs["response"]),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "has_streaming_callbacks",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "needs_iterator_wrap",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "needs_per_chunk_streaming_hook",
        MagicMock(return_value=True),
    )

    user_api_key_dict = UserAPIKeyAuth(api_key=MASTER_KEY)

    gen = proxy_server.async_data_generator(
        response=MagicMock(),
        user_api_key_dict=user_api_key_dict,
        request_data={
            "model": router_model,
            "_litellm_client_requested_model": router_model,
        },
    )

    chunks = []
    async for item in gen:
        chunks.append(item)

    assert len(chunks) >= 2
    first = _decode_sse_chunk(chunks[0])
    assert first.startswith("data: ")

    payload = json.loads(first[len("data: ") :].strip())
    # Azure Model Router: preserve actual model used, not the router model
    assert payload["model"] == actual_model_used
    assert payload["model"] != router_model


@pytest.mark.asyncio
async def test_proxy_streaming_fastest_response_preserves_winning_model(monkeypatch):
    """
    Regression test for fastest_response streaming:

    When the client sends a comma-separated model list with fastest_response=True,
    the streaming chunks should preserve the winning model's name from the
    downstream response, NOT override to the comma-separated list.
    """
    comma_separated_models = "openai/gpt-4o,gemini/gemini-2.5-flash"
    winning_model = "gemini-2.5-flash"

    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    async def _iterator_hook(
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator,
        request_data: dict,
    ):
        yield _make_model_response_stream_chunk(model=winning_model)

    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_iterator_hook",
        _iterator_hook,
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_hook",
        AsyncMock(side_effect=lambda **kwargs: kwargs["response"]),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "has_streaming_callbacks",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "needs_iterator_wrap",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "needs_per_chunk_streaming_hook",
        MagicMock(return_value=True),
    )

    user_api_key_dict = UserAPIKeyAuth(api_key=MASTER_KEY)

    gen = proxy_server.async_data_generator(
        response=MagicMock(),
        user_api_key_dict=user_api_key_dict,
        request_data={
            "model": comma_separated_models,
            "_litellm_client_requested_model": comma_separated_models,
            "fastest_response": True,
        },
    )

    chunks = []
    async for item in gen:
        chunks.append(item)

    assert len(chunks) >= 2
    first = _decode_sse_chunk(chunks[0])
    assert first.startswith("data: ")

    payload = json.loads(first[len("data: ") :].strip())
    assert payload["model"] == winning_model
    assert payload["model"] != comma_separated_models


def test_restamp_streaming_chunk_rewrites_nested_response_model():
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    chunk = _make_response_created_event("hosted_vllm/some-model")
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert result.response.model == "client-model"
    assert chunk.response.model == "hosted_vllm/some-model"


def _make_response_completed_event(model: str):
    from litellm.types.llms.openai import (
        ResponseCompletedEvent,
        ResponsesAPIResponse,
        ResponsesAPIStreamEvents,
    )

    return ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=ResponsesAPIResponse(id="resp_test", created_at=0, output=[], model=model),
    )


def test_restamp_streaming_chunk_rewrites_nested_model_on_completed_event():
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    chunk = _make_response_completed_event("hosted_vllm/some-model")
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert result.response.model == "client-model"
    assert chunk.response.model == "hosted_vllm/some-model", "the priced original must not change"


@pytest.mark.parametrize("event_type", ["response.in_progress", "response.failed", "response.incomplete"])
def test_restamp_streaming_chunk_rewrites_nested_model_on_other_lifecycle_events(event_type):
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    payload = {
        "type": event_type,
        "sequence_number": 1,
        "response": {"id": "resp_test", "object": "response", "created_at": 0, "output": [], "model": "gpt-backend"},
    }
    chunk = OpenAIResponsesAPIConfig.get_event_model_class(event_type).model_validate(payload)
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert result.response.model == "client-model"
    assert chunk.response.model == "gpt-backend", "the priced original must not change"


def test_restamp_streaming_chunk_leaves_nested_none_model_untouched():
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    chunk = _make_response_created_event(None)
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert result.response.model is None


@pytest.mark.parametrize("event_type", ["response.output_text.delta", "response.custom_event"])
def test_restamp_streaming_chunk_does_not_add_nested_response_to_other_events(event_type):
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    chunk = OpenAIResponsesAPIConfig.get_event_model_class(event_type).model_validate(
        {
            "type": event_type,
            "sequence_number": 3,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hi",
        }
    )
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert "response" not in result.model_dump()
    assert result.delta == "hi"


def test_restamp_streaming_chunk_keeps_nested_model_when_raw_name_requested():
    from litellm.constants import RETURN_RAW_MODEL_NAME_METADATA_KEY
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    chunk = _make_response_created_event("hosted_vllm/some-model")
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"metadata": {RETURN_RAW_MODEL_NAME_METADATA_KEY: True}},
        model_mismatch_logged=False,
    )

    assert result.response.model == "hosted_vllm/some-model"


def test_restamp_streaming_chunk_uses_fallback_model_for_nested_response():
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    chunk = _make_response_created_event("hosted_vllm/some-model")
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
        fallback_was_attempted=True,
        fallback_model_from_metadata="fallback-public",
    )

    assert result.response.model == "fallback-public"


@pytest.mark.asyncio
async def test_proxy_responses_stream_reports_requested_model_in_nested_response(monkeypatch):
    """
    End-to-end regression for /v1/responses streaming:

    The nested `response.model` on lifecycle events reaching the wire must report the
    client-requested model, not the provider-prefixed deployment name.
    """
    client_model = "vllm-model"
    internal_model = f"hosted_vllm/{client_model}"

    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    async def _iterator_hook(
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator,
        request_data: dict,
    ):
        yield _make_response_created_event(model=internal_model)

    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_iterator_hook",
        _iterator_hook,
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "async_post_call_streaming_hook",
        AsyncMock(side_effect=lambda **kwargs: kwargs["response"]),
    )
    for name in ("has_streaming_callbacks", "needs_iterator_wrap", "needs_per_chunk_streaming_hook"):
        monkeypatch.setattr(proxy_server.proxy_logging_obj, name, MagicMock(return_value=True))

    gen = proxy_server.async_data_generator(
        response=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(api_key=MASTER_KEY),
        request_data={"model": client_model},
        responses_stream_errors=True,
    )

    payloads = []
    async for item in gen:
        text = _decode_sse_chunk(item)
        for line in text.splitlines():
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                payloads.append(json.loads(line[len("data: ") :].strip()))

    created = next(p for p in payloads if p.get("type") == "response.created")
    assert created["response"]["model"] == client_model
    assert not created["response"]["model"].startswith("hosted_vllm/")


def test_restamp_streaming_chunk_rewrites_nested_model_on_queued_event():
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    queued_payload = {
        "type": "response.queued",
        "sequence_number": 0,
        "response": {"id": "resp_test", "object": "response", "status": "queued", "model": "gpt-backend-2026-01-01"},
    }
    chunk = OpenAIResponsesAPIConfig.get_event_model_class("response.queued").model_validate(queued_payload)
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert result.model_dump()["response"] == {**queued_payload["response"], "model": "client-model"}
    assert chunk.model_dump()["response"]["model"] == "gpt-backend-2026-01-01", "the priced original must not change"


def test_restamp_streaming_chunk_rewrites_nested_model_when_lifecycle_event_kept_raw_response():
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    partial_response = {"id": "resp_test", "object": "response", "status": "in_progress", "model": "gpt-backend-2026-01-01"}
    chunk = OpenAIResponsesAPIConfig().transform_streaming_response(
        model="gpt-backend",
        parsed_chunk={"type": "response.created", "sequence_number": 0, "response": partial_response},
        logging_obj=MagicMock(),
    )
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert result.response == {**partial_response, "model": "client-model"}
    assert chunk.response["model"] == "gpt-backend-2026-01-01", "the priced original must not change"


def test_restamp_streaming_chunk_leaves_generic_event_without_nested_response_untouched():
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    shell_payload = {"type": "response.shell_call.in_progress", "sequence_number": 3, "item_id": "sh_1"}
    chunk = OpenAIResponsesAPIConfig.get_event_model_class(shell_payload["type"]).model_validate(shell_payload)
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert "response" not in result.model_dump()


@pytest.mark.parametrize("stream_started", [False, True])
def test_restamp_streaming_chunk_rewrites_nested_model_in_guardrail_block_frames(stream_started):
    from litellm.exceptions import ModifyResponseException
    from litellm.llms.openai.responses.guardrail_translation.handler import OpenAIResponsesHandler
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    sent_before_block = (
        [{"type": "response.created", "response": {"id": "resp_live", "model": "gpt-backend-2026-01-01"}}]
        if stream_started
        else []
    )
    block = ModifyResponseException(
        message="Blocked by policy.",
        model="gpt-backend-2026-01-01",
        request_data={},
        original_response=sent_before_block,
    )
    frames = OpenAIResponsesHandler().build_block_sse_chunks(
        block, stream_started=stream_started, responses_so_far=sent_before_block
    )
    client_frames = [
        _restamp_streaming_chunk_model(
            chunk=frame,
            requested_model_from_client="client-model",
            request_data={"litellm_call_id": "test-call-id"},
            model_mismatch_logged=False,
        )[0]
        for frame in frames
    ]
    events = [json.loads(frame.decode().removeprefix("data: ")) for frame in client_frames]

    assert all(frame.endswith(b"\n\n") for frame in client_frames)
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Blocked by policy."
    assert {event["response"]["model"] for event in events if "response" in event} == {"client-model"}
    assert [frame for frame, event in zip(client_frames, events) if "response" not in event] == [
        frame for frame in frames if b'"response"' not in frame
    ]


def test_restamp_streaming_chunk_rewrites_nested_model_on_dict_lifecycle_events():
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    chunk = {"type": "response.completed", "response": {"id": "resp_test", "model": "gpt-backend-2026-01-01"}}
    result, _ = _restamp_streaming_chunk_model(
        chunk=chunk,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    assert result["response"] == {"id": "resp_test", "model": "client-model"}
    assert chunk["response"]["model"] == "gpt-backend-2026-01-01", "the priced original must not change"


_BACKEND_COMPLETED_EVENT = b'{"type":"response.completed","response":{"id":"resp_1","model":"gpt-backend-2026-01-01"}}'


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(b"event: response.completed\ndata: " + _BACKEND_COMPLETED_EVENT + b"\n\n", id="event_line_first"),
        pytest.param(b"data:" + _BACKEND_COMPLETED_EVENT + b"\n\n", id="no_space_after_data"),
        pytest.param(
            b"data: " + _BACKEND_COMPLETED_EVENT + b"\n\ndata: " + _BACKEND_COMPLETED_EVENT + b"\r\n\r\n",
            id="coalesced_frames",
        ),
    ],
)
def test_restamp_streaming_chunk_rewrites_nested_model_on_every_sse_data_line(frame):
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    result, _ = _restamp_streaming_chunk_model(
        chunk=frame,
        requested_model_from_client="client-model",
        request_data={"litellm_call_id": "test-call-id"},
        model_mismatch_logged=False,
    )

    def is_data_line(line: bytes) -> bool:
        return line.startswith(b"data:")

    result_lines = result.splitlines(keepends=True)
    frame_lines = frame.splitlines(keepends=True)
    data_events = [json.loads(line.removeprefix(b"data:")) for line in result_lines if is_data_line(line)]
    assert len(data_events) == len([line for line in frame_lines if is_data_line(line)])
    assert {event["response"]["model"] for event in data_events} == {"client-model"}
    assert [line for line in result_lines if not is_data_line(line)] == [
        line for line in frame_lines if not is_data_line(line)
    ]
    assert [line[len(line.rstrip(b"\r\n")) :] for line in result_lines] == [
        line[len(line.rstrip(b"\r\n")) :] for line in frame_lines
    ]


_SKIPPED_RESTAMPS = {
    "azure_model_router": ("azure_ai/model_router", {}),
    "fastest_response": ("openai/gpt-4o,gemini/gemini-2.5-flash", {"fastest_response": True}),
}


@pytest.mark.parametrize("skip", _SKIPPED_RESTAMPS.keys())
def test_restamp_streaming_chunk_keeps_nested_model_where_the_top_level_model_is_kept(skip):
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    requested_model, request_data = _SKIPPED_RESTAMPS[skip]
    frame = b"data: " + _BACKEND_COMPLETED_EVENT + b"\n\n"
    results = [
        _restamp_streaming_chunk_model(
            chunk=chunk,
            requested_model_from_client=requested_model,
            request_data=request_data,
            model_mismatch_logged=False,
        )[0]
        for chunk in (_make_response_created_event("gpt-backend-2026-01-01"), frame)
    ]

    assert results[0].response.model == "gpt-backend-2026-01-01"
    assert results[1] == frame


def test_restamp_streaming_chunk_leaves_raw_sse_passthrough_frames_untouched():
    from litellm.proxy.proxy_server import _restamp_streaming_chunk_model

    frame = b"data: " + _BACKEND_COMPLETED_EVENT + b"\n\n"
    result, _ = _restamp_streaming_chunk_model(
        chunk=frame,
        requested_model_from_client="client-model",
        request_data={"_litellm_raw_sse_stream": True},
        model_mismatch_logged=False,
    )

    assert result == frame
