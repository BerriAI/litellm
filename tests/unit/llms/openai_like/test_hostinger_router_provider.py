import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_hostinger_router_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hostinger-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="hostinger_router/claude-sonnet-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-5"
    assert provider == "hostinger_router"
    assert api_key == "hostinger-test-key"
    assert api_base == "https://router.hostinger.com/v1"


def test_hostinger_router_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hostinger-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="hostinger_router/claude-sonnet-5",
        custom_llm_provider=None,
        api_base="https://router.internal.example/v1",
        api_key="hostinger-explicit-key",
    )

    assert provider == "hostinger_router"
    assert api_key == "hostinger-explicit-key"
    assert api_base == "https://router.internal.example/v1"


HOSTINGER_ROUTER_MODELS = tuple(sorted(name for name in litellm.model_cost if name.startswith("hostinger_router/")))
HOSTINGER_ROUTER_CHAT_MODELS = tuple(
    name for name in HOSTINGER_ROUTER_MODELS if litellm.get_model_info(name)["mode"] == "chat"
)


@pytest.mark.parametrize("model", HOSTINGER_ROUTER_CHAT_MODELS)
def test_hostinger_router_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="hostinger_router",
    )
    model_info = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)
    assert model_info["output_cost_per_token"] > 0
    assert model_info["max_tokens"] == model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "hostinger_router"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_response_schema"] is True
    if model_info.get("supports_prompt_caching"):
        assert 0 < model_info["cache_read_input_token_cost"] < model_info["input_cost_per_token"]
    if "supports_vision" in model_info:
        assert litellm.supports_vision(model) is model_info["supports_vision"]
    else:
        assert litellm.supports_vision(model) in (True, False, None)


def test_hostinger_router_backup_registry_mirrors_cost_map():
    package_root = Path(litellm.__file__).parent
    cost_map = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    hostinger_entries = {name: entry for name, entry in cost_map.items() if name.startswith("hostinger_router/")}

    assert tuple(sorted(hostinger_entries)) == HOSTINGER_ROUTER_MODELS
    assert hostinger_entries
    assert hostinger_entries == {name: backup[name] for name in hostinger_entries}


def test_hostinger_router_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    hostinger = next(provider for provider in providers if provider["litellm_provider"] == "hostinger_router")

    assert hostinger["provider"] == "HOSTINGER_ROUTER"
    assert hostinger["provider_display_name"] == "Hostinger Router"
    assert hostinger["default_model_placeholder"] == "hostinger_router/claude-sonnet-5"
    assert {field["key"]: field["required"] for field in hostinger["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_hostinger_router_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["hostinger_router"]["endpoints"]["chat_completions"] is True
    assert providers["hostinger_router"]["endpoints"]["messages"] is True
    assert providers["hostinger_router"]["endpoints"]["responses"] is True
    assert providers["hostinger_router"]["endpoints"]["embeddings"] is False
    assert providers["hostinger_router"]["endpoints"]["image_generations"] is False
    assert providers["hostinger_router"]["endpoints"]["audio_transcriptions"] is False
    assert providers["hostinger_router"]["endpoints"]["batches"] is False


def test_hostinger_router_chat_completions_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_hostinger",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "claude-sonnet-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Hostinger Router"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="hostinger_router/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="hostinger-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer hostinger-test-key"
    assert body["model"] == "claude-sonnet-5"
    assert response.choices[0].message.content == "Hello from Hostinger Router"


def test_hostinger_router_chat_completions_streaming_request():
    chunks = [
        'data: {"id":"chatcmpl_hostinger","object":"chat.completion.chunk","created":1789550000,"model":"claude-sonnet-5","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl_hostinger","object":"chat.completion.chunk","created":1789550000,"model":"claude-sonnet-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n',
        "data: [DONE]\n\n",
    ]
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/chat/completions").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(chunks).encode(),
        )
        response: Final = litellm.completion(
            model="hostinger_router/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
            api_key="hostinger-test-key",
        )
        collected = [chunk for chunk in response]

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert body["stream"] is True
    assert any(chunk.choices and chunk.choices[0].delta.content == "Hi" for chunk in collected)


def test_hostinger_router_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/responses").respond(
            200,
            json={
                "id": "resp_hostinger",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_hostinger",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Hostinger Router", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="hostinger_router/gpt-5.2",
            input="Say hello",
            api_key="hostinger-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://router.hostinger.com/v1/responses"
    assert request.headers["authorization"] == "Bearer hostinger-test-key"
    assert body["model"] == "gpt-5.2"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Hostinger Router"


@pytest.mark.asyncio
async def test_hostinger_router_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/messages").respond(
            200,
            json={
                "id": "msg_hostinger",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": "Hello from Hostinger Router"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="hostinger_router/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="hostinger-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://router.hostinger.com/v1/messages"
    assert request.headers["authorization"] == "Bearer hostinger-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Hostinger Router"
