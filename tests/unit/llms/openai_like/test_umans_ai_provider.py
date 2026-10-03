import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_umans_ai_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UMANS_AI_API_KEY", "umans-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="umans-ai/umans-deepseek-v4-flash-0731",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "umans-deepseek-v4-flash-0731"
    assert provider == "umans-ai"
    assert api_key == "umans-test-key"
    assert api_base == "https://api.code.umans.ai/v1"


def test_umans_ai_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UMANS_AI_API_KEY", "umans-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="umans-ai/umans-deepseek-v4-flash-0731",
        custom_llm_provider=None,
        api_base="https://umans.internal.example/v1",
        api_key="umans-explicit-key",
    )

    assert provider == "umans-ai"
    assert api_key == "umans-explicit-key"
    assert api_base == "https://umans.internal.example/v1"


def test_umans_ai_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    umans = next(provider for provider in providers if provider["litellm_provider"] == "umans-ai")

    assert umans["provider"] == "UMANS_AI"
    assert umans["provider_display_name"] == "Umans AI"
    assert umans["default_model_placeholder"] == "umans-ai/umans-deepseek-v4-flash-0731"
    assert {field["key"]: field["required"] for field in umans["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_umans_ai_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["umans-ai"]["endpoints"] == {
        "chat_completions": True,
        "messages": True,
        "responses": True,
        "embeddings": False,
        "image_generations": False,
        "audio_transcriptions": False,
        "audio_speech": False,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
        "interactions": False,
    }


def test_umans_ai_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.code.umans.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_umans",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "umans-deepseek-v4-flash-0731",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Umans AI"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="umans-ai/umans-deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="umans-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.code.umans.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer umans-test-key"
    assert body["model"] == "umans-deepseek-v4-flash-0731"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Umans AI"


def test_umans_ai_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.code.umans.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_umans",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "umans-deepseek-v4-flash-0731",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_umans",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Umans AI", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="umans-ai/umans-deepseek-v4-flash-0731",
            input="Say hello",
            api_key="umans-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.code.umans.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer umans-test-key"
    assert body["model"] == "umans-deepseek-v4-flash-0731"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Umans AI"


@pytest.mark.asyncio
async def test_umans_ai_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.code.umans.ai/v1/messages").respond(
            200,
            json={
                "id": "msg_umans",
                "type": "message",
                "role": "assistant",
                "model": "umans-deepseek-v4-flash-0731",
                "content": [{"type": "text", "text": "Hello from Umans AI"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="umans-ai/umans-deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="umans-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.code.umans.ai/v1/messages"
    assert request.headers["authorization"] == "Bearer umans-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "umans-deepseek-v4-flash-0731"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Umans AI"


UMANS_PRICING: Final = (
    # model, then USD per token for input, output and cache read (app.umans.ai/pricing)
    ("umans-ai/umans-deepseek-v4-flash-0731", 1.4e-07, 2.8e-07, 2.8e-08),
    ("umans-ai/umans-deepseek-v4.1-flash", 1.5e-07, 6e-07, 2.8e-08),
    ("umans-ai/umans-glm-5.3", 1.4e-06, 4.4e-06, 2.6e-07),
    ("umans-ai/umans-glm-5.3-flash", 1.5e-07, 5e-07, 3e-08),
    ("umans-ai/umans-kimi-k3", 3e-06, 1.5e-05, 3e-07),
    ("umans-ai/umans-flash", 1.5e-07, 1e-06, 5e-08),
    ("umans-ai/umans-coder", 1.5e-07, 5e-07, 3e-08),
)


def _load_cost_map(filename: str = "model_prices_and_context_window.json") -> dict:
    with open(Path(__file__).parents[4] / filename) as f:
        return json.load(f)


@pytest.mark.parametrize(("model", "input_cost", "output_cost", "cache_read_cost"), UMANS_PRICING)
def test_umans_ai_models_are_priced(model: str, input_cost: float, output_cost: float, cache_read_cost: float):
    entry: Final = _load_cost_map()[model]

    assert entry["litellm_provider"] == "umans-ai"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == input_cost
    assert entry["output_cost_per_token"] == output_cost
    assert entry["cache_read_input_token_cost"] == cache_read_cost
    assert _load_cost_map("litellm/model_prices_and_context_window_backup.json")[model] == entry


def test_umans_ai_usage_is_billed():
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="umans-ai/umans-deepseek-v4-flash-0731",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    assert prompt_cost == pytest.approx(0.14)
    assert completion_cost == pytest.approx(0.28)


def _sse(*events: dict) -> bytes:
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n"


def _typed_sse(*events: dict) -> bytes:
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def test_umans_ai_chat_completion_streaming_request():
    chunk: Final = {
        "id": "chatcmpl_umans",
        "object": "chat.completion.chunk",
        "created": 1_789_550_000,
        "model": "umans-deepseek-v4-flash-0731",
    }
    stream_body: Final = _sse(
        {
            **chunk,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello from "}, "finish_reason": None}],
        },
        {**chunk, "choices": [{"index": 0, "delta": {"content": "Umans AI"}, "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.code.umans.ai/v1/chat/completions").respond(
            200, content=stream_body, headers={"content-type": "text/event-stream"}
        )
        chunks: Final = list(
            litellm.completion(
                model="umans-ai/umans-deepseek-v4-flash-0731",
                messages=[{"role": "user", "content": "Say hello"}],
                api_key="umans-test-key",
                stream=True,
            )
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert str(route.calls.last.request.url) == "https://api.code.umans.ai/v1/chat/completions"
    assert body["stream"] is True
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello from Umans AI"


def test_umans_ai_responses_streaming_request():
    response: Final = {
        "id": "resp_umans",
        "object": "response",
        "created_at": 1_789_550_000,
        "model": "umans-deepseek-v4-flash-0731",
        "status": "in_progress",
        "output": [],
    }
    message: Final = {
        "id": "msg_umans",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Hello from Umans AI", "annotations": []}],
    }
    stream_body: Final = _typed_sse(
        {"type": "response.created", "sequence_number": 0, "response": response},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_umans",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello from Umans AI",
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": {
                **response,
                "status": "completed",
                "output": [message],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        },
    )
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.code.umans.ai/v1/responses").respond(
            200, content=stream_body, headers={"content-type": "text/event-stream"}
        )
        events: Final = list(
            litellm.responses(
                model="umans-ai/umans-deepseek-v4-flash-0731",
                input="Say hello",
                api_key="umans-test-key",
                stream=True,
            )
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert str(route.calls.last.request.url) == "https://api.code.umans.ai/v1/responses"
    assert body["stream"] is True
    assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == (
        "Hello from Umans AI"
    )
    assert events[-1].type == "response.completed"


@pytest.mark.asyncio
async def test_umans_ai_anthropic_messages_streaming_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    stream_body: Final = _typed_sse(
        {
            "type": "message_start",
            "message": {
                "id": "msg_umans",
                "type": "message",
                "role": "assistant",
                "model": "umans-deepseek-v4-flash-0731",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello from Umans AI"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    )
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.code.umans.ai/v1/messages").respond(
            200, content=stream_body, headers={"content-type": "text/event-stream"}
        )
        stream = await litellm.anthropic.messages.acreate(
            model="umans-ai/umans-deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="umans-test-key",
            stream=True,
        )
        streamed: Final = b"".join([chunk async for chunk in stream]).decode()

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert str(route.calls.last.request.url) == "https://api.code.umans.ai/v1/messages"
    assert body["stream"] is True
    assert "event: message_start" in streamed
    assert "Hello from Umans AI" in streamed
    assert "event: message_stop" in streamed
