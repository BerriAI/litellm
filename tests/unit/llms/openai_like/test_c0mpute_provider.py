import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://c0mpute.com/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_c0mpute",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "qwen3:4b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from c0mpute"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_STREAM_CHUNK: Final = {
    "id": "chatcmpl_c0mpute",
    "object": "chat.completion.chunk",
    "created": 1_790_000_000,
    "model": "qwen3:4b",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello from c0mpute"}, "finish_reason": None}],
}
_STREAM_DONE_CHUNK: Final = {
    **_STREAM_CHUNK,
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
}
_STREAM_BODY: Final = (
    ': status {"state":"queued","eta_s":3}\n\n'
    f"data: {json.dumps(_STREAM_CHUNK)}\n\n"
    ': status {"state":"running"}\n\n'
    f"data: {json.dumps(_STREAM_DONE_CHUNK)}\n\n"
    "data: [DONE]\n\n"
)


def test_c0mpute_is_a_registered_provider() -> None:
    assert litellm.LlmProviders.C0MPUTE.value == "c0mpute"
    assert "c0mpute" in litellm.provider_list
    assert "c0mpute" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_c0mpute_provider_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("C0MPUTE_API_KEY", "c0k_test")
    monkeypatch.delenv("C0MPUTE_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="c0mpute/auto",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "auto"
    assert provider == "c0mpute"
    assert api_key == "c0k_test"
    assert api_base == _DEFAULT_API_BASE


def test_c0mpute_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("C0MPUTE_API_KEY", "c0k_env")
    monkeypatch.setenv("C0MPUTE_API_BASE", "https://c0mpute.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="c0mpute/auto:fast",
        custom_llm_provider=None,
        api_base="https://c0mpute.internal.example/v1",
        api_key="c0k_explicit",
    )

    assert provider == "c0mpute"
    assert api_key == "c0k_explicit"
    assert api_base == "https://c0mpute.internal.example/v1"


def test_c0mpute_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("C0MPUTE_API_KEY", "c0k_env")
    monkeypatch.setenv("C0MPUTE_API_BASE", "https://c0mpute.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="c0mpute/auto",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "c0mpute"
    assert api_base == "https://c0mpute.env.example/v1"


def test_c0mpute_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("C0MPUTE_API_KEY", "c0k_env")

    model, provider, api_key, api_base = get_llm_provider(
        model="qwen3:4b",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "qwen3:4b"
    assert provider == "c0mpute"
    assert api_key == "c0k_env"
    assert api_base == _DEFAULT_API_BASE


def test_c0mpute_chat_completion_sends_string_content_and_max_tokens() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="c0mpute/auto",
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "Say"}, {"type": "text", "text": "hello"}]},
            ],
            max_completion_tokens=64,
            api_key="c0k_test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer c0k_test"
    assert body["model"] == "auto"
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Say\nhello"},
    ]
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body
    assert response.choices[0].message.content == "Hello from c0mpute"


def test_c0mpute_stream_skips_sse_status_comments() -> None:
    with respx.mock() as upstream:
        upstream.post(_CHAT_COMPLETIONS_URL).respond(
            200, content=_STREAM_BODY.encode(), headers={"content-type": "text/event-stream"}
        )
        chunks: Final = tuple(
            litellm.completion(
                model="c0mpute/auto",
                messages=[{"role": "user", "content": "Say hello"}],
                stream=True,
                api_key="c0k_test",
            )
        )

    text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert text == "Hello from c0mpute"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_c0mpute_responses_request_is_bridged_to_chat_completions() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="c0mpute/auto",
            input="Say hello",
            api_key="c0k_test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "auto"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from c0mpute"


def test_c0mpute_is_available_in_add_model_form() -> None:
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    c0mpute: Final = next(provider for provider in providers if provider["litellm_provider"] == "c0mpute")

    assert c0mpute["default_model_placeholder"].startswith("c0mpute/")
    assert {field["key"]: field["required"] for field in c0mpute["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
