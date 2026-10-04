"""Tests for the ClinePass provider.

The point of the end-to-end tests here is that they drive ``litellm.completion()``
with a mocked transport rather than calling the transforms directly -- a unit test
that calls ``transform_response()`` itself proves the function is correct but not
that anything invokes it, which is exactly how the envelope unwrap was previously
shipped as dead code.
"""

import json
from typing import Final
from unittest.mock import patch

import httpx
import pytest
from starlette.websockets import WebSocket

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.clinepass.chat.transformation import (
    ClinePassConfig,
    _apply_model_prefix,
    _unwrap_response_envelope,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.responses.main import _aresponses_websocket
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

API_KEY = "sk-clinepass-test-not-real"

ENVELOPED_COMPLETION = {
    "success": True,
    "data": {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "clinepass/deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "pong",
                    "reasoning": "the user asked for pong",
                },
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
}


def _response(payload: dict, url: str = "https://api.cline.bot/api/v1/chat/completions") -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("POST", url))


@pytest.fixture(autouse=True)
def _clinepass_env(monkeypatch):
    monkeypatch.setenv("CLINEPASS_API_KEY", API_KEY)
    monkeypatch.delenv("CLINEPASS_API_BASE", raising=False)


@pytest.fixture
def unrelated_credentials(monkeypatch):
    for env_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(env_name, f"sk-unrelated-{env_name}")
    for attribute in ("api_key", "openai_key", "anthropic_key"):
        monkeypatch.setattr(litellm, attribute, f"sk-unrelated-{attribute}")


@pytest.mark.parametrize("credential_source", ["missing", "environment", "explicit"])
@pytest.mark.parametrize("custom_llm_provider", [None, "clinepass"])
def test_chat_sends_only_clinepass_credentials(
    monkeypatch, unrelated_credentials, credential_source, custom_llm_provider
):
    if credential_source == "missing":
        monkeypatch.delenv("CLINEPASS_API_KEY", raising=False)
    explicit_key: Final = "cp-request-key" if credential_source == "explicit" else None
    captured = {}

    def fake_post(self, url, *args, **kwargs):
        captured["url"] = str(url)
        captured["headers"] = httpx.Headers(kwargs["headers"])
        return _response(ENVELOPED_COMPLETION)

    with patch.object(HTTPHandler, "post", fake_post):
        response = litellm.completion(
            model="deepseek-v4-flash" if custom_llm_provider else "clinepass/deepseek-v4-flash",
            custom_llm_provider=custom_llm_provider,
            api_key=explicit_key,
            messages=[{"role": "user", "content": "ping"}],
        )

    expected_key: Final = explicit_key or (API_KEY if credential_source == "environment" else None)
    assert captured["url"] == "https://api.cline.bot/api/v1/chat/completions"
    assert captured["headers"].get("authorization") == (f"Bearer {expected_key}" if expected_key else None)
    assert response.choices[0].message.content == "pong"


@pytest.mark.parametrize("credential_source", ["missing", "environment", "explicit"])
@pytest.mark.asyncio
async def test_async_chat_sends_only_clinepass_credentials(monkeypatch, unrelated_credentials, credential_source):
    if credential_source == "missing":
        monkeypatch.delenv("CLINEPASS_API_KEY", raising=False)
    explicit_key: Final = "cp-request-key" if credential_source == "explicit" else None
    captured = {}

    async def fake_post(self, url, *args, **kwargs):
        captured["url"] = str(url)
        captured["headers"] = httpx.Headers(kwargs["headers"])
        return _response(ENVELOPED_COMPLETION)

    with patch.object(AsyncHTTPHandler, "post", fake_post):
        response = await litellm.acompletion(
            model="clinepass/deepseek-v4-flash",
            api_key=explicit_key,
            messages=[{"role": "user", "content": "ping"}],
        )

    expected_key: Final = explicit_key or (API_KEY if credential_source == "environment" else None)
    assert captured["url"] == "https://api.cline.bot/api/v1/chat/completions"
    assert captured["headers"].get("authorization") == (f"Bearer {expected_key}" if expected_key else None)
    assert response.choices[0].message.content == "pong"


@pytest.mark.parametrize("credential_source", ["missing", "environment", "explicit"])
@pytest.mark.parametrize("connection_provider", ["clinepass", "mistral"])
@pytest.mark.parametrize("changed_model", [None, "openai/gpt-4o", "clinepass/unauthorized-model"])
@pytest.mark.asyncio
async def test_managed_responses_websocket_sends_only_clinepass_credentials(
    monkeypatch, unrelated_credentials, credential_source, connection_provider, changed_model
):
    if credential_source == "missing":
        monkeypatch.delenv("CLINEPASS_API_KEY", raising=False)
    connection_key: Final = (
        "sk-unrelated-mistral"
        if connection_provider == "mistral"
        else "cp-request-key"
        if credential_source == "explicit"
        else None
    )
    sent = []
    received = []
    foreign_requests = []
    lifecycle = iter(
        (
            {"type": "websocket.connect"},
            *(
                (
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({"type": "response.create", "model": changed_model, "input": "hi"}),
                    },
                )
                if changed_model is not None
                else ()
            ),
            {"type": "websocket.disconnect", "code": 1000},
        )
    )

    async def block_foreign_request(self, request, *args, **kwargs):
        foreign_requests.append(str(request.url))
        raise AssertionError("Unexpected provider HTTP request")

    monkeypatch.setattr(httpx.AsyncClient, "send", block_foreign_request)

    async def receive():
        return next(lifecycle)

    async def send(message):
        if message["type"] == "websocket.send":
            received.append(json.loads(message["text"]))

    websocket: Final = WebSocket(
        scope={"type": "websocket", "path": "/v1/responses", "headers": [], "query_string": b""},
        receive=receive,
        send=send,
    )
    await websocket.accept()
    chunk: Final = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "cline-pass/deepseek-v4-flash",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    }

    async def fake_post(self, url, *args, **kwargs):
        sent.append((str(url), httpx.Headers(kwargs["headers"]).get("authorization")))
        return httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", str(url)),
        )

    with patch.object(AsyncHTTPHandler, "post", fake_post):
        result = await _aresponses_websocket.__wrapped__(
            model=f"{connection_provider}/deepseek-v4-flash",
            websocket=websocket,
            api_key=connection_key,
            user_api_key_dict=(
                UserAPIKeyAuth(models=[f"{connection_provider}/deepseek-v4-flash"])
                if changed_model is not None
                else None
            ),
            first_message=json.dumps(
                {"type": "response.create", "model": "clinepass/deepseek-v4-flash", "input": "ping"}
            ),
            litellm_logging_obj=Logging(
                model=f"{connection_provider}/deepseek-v4-flash",
                messages=[],
                stream=True,
                call_type="aresponses",
                start_time=0,
                litellm_call_id="cp-ws-test",
                function_id="cp-ws-test",
            ),
        )

    expected_key: Final = (
        connection_key
        if connection_provider == "clinepass" and credential_source == "explicit"
        else API_KEY
        if credential_source != "missing"
        else None
    )
    assert sent == (
        []
        if changed_model is not None and connection_provider == "mistral"
        else [("https://api.cline.bot/api/v1/chat/completions", f"Bearer {expected_key}" if expected_key else None)]
    )
    assert result is None
    errors: Final = [event["error"] for event in received if event["type"] == "error"]
    assert errors == (
        [
            {
                "type": "invalid_request_error",
                "message": "Changing models requires a new authorized WebSocket connection",
            }
        ]
        * (2 if connection_provider == "mistral" else 1)
        if changed_model is not None
        else []
    )
    assert foreign_requests == []
    assert ("response.completed" in [event["type"] for event in received]) == bool(sent)


# --------------------------------------------------------------------------
# Registration / routing
# --------------------------------------------------------------------------


def test_get_llm_provider_resolves_clinepass():
    model, provider, api_key, api_base = litellm.get_llm_provider(model="clinepass/deepseek-v4-flash")
    assert model == "deepseek-v4-flash"
    assert provider == "clinepass"
    assert api_key == API_KEY
    assert api_base == "https://api.cline.bot/api/v1"


def test_provider_config_manager_returns_clinepass_config():
    config = ProviderConfigManager.get_provider_chat_config(model="deepseek-v4-flash", provider=LlmProviders.CLINEPASS)
    assert isinstance(config, ClinePassConfig)


def test_clinepass_is_not_a_json_configured_provider_via_behaviour():
    """ClinePass needs a response transform, which the JSON provider system's
    OpenAI-SDK dispatch path never invokes. We assert it is not on that path
    by verifying the envelope unwrap actually triggers."""
    captured = {}

    def fake_post(self, url, *args, **kwargs):
        captured["body"] = json.loads(kwargs["data"])
        return _response(ENVELOPED_COMPLETION)

    with patch.object(HTTPHandler, "post", fake_post):
        response = litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
        )

    # The unwrap works, meaning we didn't drift into the JSON provider registry
    # which would have bypassed our custom transform.
    assert response.choices[0].message.content == "pong"


def test_clinepass_is_not_in_openai_compatible_providers():
    """The cheap structural guard for the credential leak. Keep it.

    The behavioural test below proves the *consequence*; this proves the
    *cause*, in one line and with no mocking that could itself be wrong. Both
    are wanted: a mocked behavioural test can drift into passing for the wrong
    reason, while this cannot.

    The membership is not routing-inert, which is what made it dangerous. The
    list is also read by the speech branch in `main.py` -- which sends to the
    provider's own `api_base` while taking the key from `OPENAI_API_KEY`, so a
    `litellm.speech(model="clinepass/...")` call shipped the caller's OpenAI
    credential to the Cline host for an endpoint ClinePass does not implement --
    by `OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS`, which is derived from it, by
    image generation in `litellm/images/main.py`, and by
    `_add_provider_specific_params`, which nests unknown kwargs under
    `extra_body` for listed providers. `BaseLLMHTTPHandler` merges that back into
    the request body, so the wire body is the same either way: membership is
    pinned structurally here and at the params level in
    `test_unknown_kwargs_are_flattened_not_wrapped_in_extra_body`, not by the
    request body.

    Exception mapping is preserved by registering ClinePass explicitly beside
    `mistral` in `exception_mapping_utils.py`; see
    `test_upstream_401_maps_to_authentication_error`.
    """
    assert "clinepass" not in litellm.openai_compatible_providers
    assert "clinepass" not in litellm.constants.OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS


def test_clinepass_is_not_in_openai_compatible_providers_via_behaviour():
    """ClinePass must NOT be in `openai_compatible_providers`.
    The drift guard here is the transcription half: a listed provider is picked up
    by the OpenAI transcription branch instead of raising as unmapped.
    The body assertions only pin the contract that an unknown kwarg reaches the
    JSON body flat; they hold for listed providers too, so they do not detect
    drift (the request body is identical either way)."""
    captured = {}

    def fake_post(self, url, *args, **kwargs):
        captured["body"] = json.loads(kwargs["data"])
        return _response(ENVELOPED_COMPLETION)

    with patch.object(HTTPHandler, "post", fake_post):
        litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
            custom_vendor_flag=True,  # unknown kwarg
        )

    assert "extra_body" not in captured["body"]
    assert captured["body"].get("custom_vendor_flag") is True

    # Audio transcription should outright fail as unmapped, confirming it
    # isn't implicitly picked up by OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS
    with pytest.raises(ValueError, match="Unmapped provider"):
        litellm.transcription(
            model="clinepass/deepseek-v4-flash",
            file=b"fake audio data",
        )


def test_api_base_env_override(monkeypatch):
    monkeypatch.setenv("CLINEPASS_API_BASE", "https://proxy.internal/api/v1")
    _, _, _, api_base = litellm.get_llm_provider(model="clinepass/deepseek-v4-flash")
    assert api_base == "https://proxy.internal/api/v1"


@pytest.mark.parametrize(
    "api_base,expected",
    [
        (None, "https://api.cline.bot/api/v1/chat/completions"),
        ("https://api.cline.bot/api/v1", "https://api.cline.bot/api/v1/chat/completions"),
        ("https://api.cline.bot/api/v1/", "https://api.cline.bot/api/v1/chat/completions"),
        (
            "https://api.cline.bot/api/v1/chat/completions",
            "https://api.cline.bot/api/v1/chat/completions",
        ),
    ],
)
def test_get_complete_url(api_base, expected):
    url = ClinePassConfig().get_complete_url(
        api_base=api_base, api_key=API_KEY, model="deepseek-v4-flash", optional_params={}, litellm_params={}
    )
    assert url == expected


# --------------------------------------------------------------------------
# Model prefix
# --------------------------------------------------------------------------


def test_model_prefix_restored_on_bare_id():
    """The restored qualifier is the catalog namespace ``cline-pass/`` (hyphenated),
    NOT LiteLLM's own ``clinepass/`` routing prefix.

    The API validates only the *shape* of a model id, so a wrong namespace still
    returns HTTP 200 -- but it does not always resolve to the same underlying
    model, which makes a wrong value silent rather than harmless.
    """
    assert _apply_model_prefix({"model": "deepseek-v4-flash"})["model"] == "cline-pass/deepseek-v4-flash"


def test_model_prefix_left_alone_when_qualifier_present():
    """`clinepass/openrouter/foo` arrives here as `openrouter/foo` and must pass through."""
    assert _apply_model_prefix({"model": "openrouter/foo"})["model"] == "openrouter/foo"


def test_model_prefix_ignores_missing_model():
    assert _apply_model_prefix({}) == {}


# --------------------------------------------------------------------------
# Response envelope
# --------------------------------------------------------------------------


def test_unwrap_envelope_extracts_inner_completion():
    unwrapped = _unwrap_response_envelope(_response(ENVELOPED_COMPLETION))
    assert unwrapped.json() == ENVELOPED_COMPLETION["data"]


def test_unwrap_envelope_content_length_describes_the_new_body():
    """The original content-length describes the enveloped bytes and must not be
    carried over; httpx recomputes a correct one for the rewritten body."""
    raw = _response(ENVELOPED_COMPLETION)
    unwrapped = _unwrap_response_envelope(raw)
    assert unwrapped.headers["content-length"] != raw.headers["content-length"]
    assert int(unwrapped.headers["content-length"]) == len(unwrapped.content)


def test_unwrap_envelope_passes_through_openai_shaped_body():
    payload = ENVELOPED_COMPLETION["data"]
    assert _unwrap_response_envelope(_response(payload)).json() == payload


def test_unwrap_envelope_passes_through_error_nested_under_same_key():
    """An error under `data` has no `choices` and must not be mistaken for a completion."""
    payload = {"success": False, "data": {"message": "bad model"}}
    assert _unwrap_response_envelope(_response(payload)).json() == payload


def test_unwrap_envelope_passes_through_non_json_body():
    raw = httpx.Response(
        200, content=b"not json", request=httpx.Request("POST", "https://api.cline.bot/api/v1/chat/completions")
    )
    assert _unwrap_response_envelope(raw) is raw


# --------------------------------------------------------------------------
# Parameter mapping
# --------------------------------------------------------------------------


def test_max_completion_tokens_mapped_to_max_tokens():
    mapped = ClinePassConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 4000},
        optional_params={},
        model="deepseek-v4-flash",
        drop_params=False,
    )
    assert mapped == {"max_tokens": 4000}


# --------------------------------------------------------------------------
# End-to-end through litellm.completion() -- these are the load-bearing ones
# --------------------------------------------------------------------------


def test_completion_unwraps_envelope_and_prefixes_model():
    captured = {}

    def fake_post(self, url, *args, **kwargs):
        captured["url"] = str(url)
        captured["body"] = json.loads(kwargs["data"])
        return _response(ENVELOPED_COMPLETION)

    with patch.object(HTTPHandler, "post", fake_post):
        response = litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=4000,
        )

    assert captured["url"] == "https://api.cline.bot/api/v1/chat/completions"
    assert captured["body"]["model"] == "cline-pass/deepseek-v4-flash"
    assert response.choices[0].message.content == "pong"
    assert response.choices[0].message.reasoning_content == "the user asked for pong"


@pytest.mark.asyncio
async def test_acompletion_unwraps_envelope_and_prefixes_model():
    captured = {}

    async def fake_post(self, url, *args, **kwargs):
        captured["url"] = str(url)
        captured["body"] = json.loads(kwargs["data"])
        return _response(ENVELOPED_COMPLETION)

    with patch.object(AsyncHTTPHandler, "post", fake_post):
        response = await litellm.acompletion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=4000,
        )

    assert captured["url"] == "https://api.cline.bot/api/v1/chat/completions"
    assert captured["body"]["model"] == "cline-pass/deepseek-v4-flash"
    assert response.choices[0].message.content == "pong"


@pytest.mark.parametrize("credential_source", ["missing", "environment", "explicit"])
def test_completion_streaming_is_not_unwrapped(monkeypatch, unrelated_credentials, credential_source):
    """ClinePass does NOT wrap SSE chunks -- they are already OpenAI-shaped."""
    if credential_source == "missing":
        monkeypatch.delenv("CLINEPASS_API_KEY", raising=False)
    explicit_key: Final = "cp-stream-key" if credential_source == "explicit" else None
    captured = {}
    chunks = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "clinepass/deepseek-v4-flash",
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        for piece in ["one ", "two ", "three"]
    ]
    chunks.append(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "clinepass/deepseek-v4-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def fake_post(self, url, *args, **kwargs):
        captured["authorization"] = httpx.Headers(kwargs["headers"]).get("authorization")
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", str(url)),
        )

    with patch.object(HTTPHandler, "post", fake_post):
        stream = litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "count"}],
            api_key=explicit_key,
            max_tokens=4000,
            stream=True,
        )
        text = ""
        finish_reason = None
        for c in stream:
            if c.choices:
                if c.choices[0].delta.content:
                    text += c.choices[0].delta.content
                if c.choices[0].finish_reason:
                    finish_reason = c.choices[0].finish_reason

    assert text == "one two three"
    assert finish_reason == "stop"
    expected_key: Final = explicit_key or (API_KEY if credential_source == "environment" else None)
    assert captured["authorization"] == (f"Bearer {expected_key}" if expected_key else None)


@pytest.mark.parametrize("credential_source", ["missing", "environment", "explicit"])
@pytest.mark.asyncio
async def test_acompletion_streaming_is_not_unwrapped(monkeypatch, unrelated_credentials, credential_source):
    if credential_source == "missing":
        monkeypatch.delenv("CLINEPASS_API_KEY", raising=False)
    explicit_key: Final = "cp-stream-key" if credential_source == "explicit" else None
    captured = {}
    chunks = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "clinepass/deepseek-v4-flash",
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        for piece in ["one ", "two ", "three"]
    ]
    chunks.append(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "clinepass/deepseek-v4-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    async def fake_post(self, url, *args, **kwargs):
        captured["authorization"] = httpx.Headers(kwargs["headers"]).get("authorization")
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", str(url)),
        )

    with patch.object(AsyncHTTPHandler, "post", fake_post):
        stream = await litellm.acompletion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "count"}],
            api_key=explicit_key,
            max_tokens=4000,
            stream=True,
        )
        text = ""
        finish_reason = None
        async for c in stream:
            if c.choices:
                if c.choices[0].delta.content:
                    text += c.choices[0].delta.content
                if c.choices[0].finish_reason:
                    finish_reason = c.choices[0].finish_reason

    assert text == "one two three"
    assert finish_reason == "stop"
    expected_key: Final = explicit_key or (API_KEY if credential_source == "environment" else None)
    assert captured["authorization"] == (f"Bearer {expected_key}" if expected_key else None)


def test_completion_streaming_tool_call_reassembly():
    """Tool calls split across chunks must be correctly passed through by the OpenAI-compatible stream processor."""
    chunks = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "clinepass/deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "clinepass/deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "clinepass/deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ation": "NYC"}'}}]},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "clinepass/deepseek-v4-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def fake_post(self, url, *args, **kwargs):
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", str(url)),
        )

    with patch.object(HTTPHandler, "post", fake_post):
        stream = litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "weather"}],
            stream=True,
        )

        args_text = ""
        for c in stream:
            if c.choices and c.choices[0].delta.tool_calls:
                tc = c.choices[0].delta.tool_calls[0]
                if tc.function and tc.function.arguments:
                    args_text += tc.function.arguments

    assert args_text == '{"location": "NYC"}'


def test_completion_preserves_usage():
    def fake_post(self, url, *args, **kwargs):
        return _response(ENVELOPED_COMPLETION)

    with patch.object(HTTPHandler, "post", fake_post):
        response = litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
        )

    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 7


def test_completion_sends_authorization_header():
    captured_headers = {}

    def fake_post(self, url, *args, **kwargs):
        captured_headers.update(kwargs.get("headers", {}))
        return _response(ENVELOPED_COMPLETION)

    with patch.object(HTTPHandler, "post", fake_post):
        litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
        )

    assert captured_headers.get("Authorization") == f"Bearer {API_KEY}"


def test_completion_explicit_api_key_precedence():
    captured_headers = {}

    def fake_post(self, url, *args, **kwargs):
        captured_headers.update(kwargs.get("headers", {}))
        return _response(ENVELOPED_COMPLETION)

    with patch.object(HTTPHandler, "post", fake_post):
        litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
            api_key="sk-clinepass-explicit-key",
        )

    assert captured_headers.get("Authorization") == "Bearer sk-clinepass-explicit-key"


def test_upstream_429_maps_to_rate_limit_error():
    from litellm.exceptions import RateLimitError

    def fake_post(self, url, *args, **kwargs):
        raise httpx.HTTPStatusError(
            "Too Many Requests",
            request=httpx.Request("POST", str(url)),
            response=httpx.Response(
                429,
                json={"error": "Rate limit exceeded"},
                request=httpx.Request("POST", str(url)),
            ),
        )

    with patch.object(HTTPHandler, "post", fake_post), pytest.raises(RateLimitError) as excinfo:
        litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert excinfo.value.status_code == 429


def test_upstream_401_maps_to_authentication_error():
    """ClinePass answers a bad key with HTTP 401; that must not be flattened
    into a generic APIConnectionError."""
    from litellm.exceptions import AuthenticationError

    def fake_post(self, url, *args, **kwargs):
        raise httpx.HTTPStatusError(
            "Unauthorized",
            request=httpx.Request("POST", str(url)),
            response=httpx.Response(
                401,
                json={"error": "Unauthorized"},
                request=httpx.Request("POST", str(url)),
            ),
        )

    with patch.object(HTTPHandler, "post", fake_post), pytest.raises(AuthenticationError) as excinfo:
        litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )

    assert excinfo.value.status_code == 401


# --------------------------------------------------------------------------
# Truncation reporting
#
# ClinePass was once observed returning finish_reason "stop" on a completion cut
# off by max_tokens. Re-probing the live API on 2026-08-22 could not reproduce
# it, so the provider now reports the upstream finish reason unmodified to avoid
# false positives on natural completions that land exactly on the cap.
# --------------------------------------------------------------------------


def _truncated_envelope(completion_tokens: int, finish_reason: str = "stop") -> dict:
    payload = json.loads(json.dumps(ENVELOPED_COMPLETION))
    payload["data"]["choices"][0]["finish_reason"] = finish_reason
    payload["data"]["usage"]["completion_tokens"] = completion_tokens
    return payload


def _complete(payload: dict, **kwargs):
    def fake_post(self, url, *args, **post_kwargs):
        return _response(payload)

    with patch.object(HTTPHandler, "post", fake_post):
        return litellm.completion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
            **kwargs,
        )


def test_completion_at_the_cap_preserves_upstream_stop():
    response = _complete(_truncated_envelope(4000), max_tokens=4000)
    assert response.choices[0].finish_reason == "stop"


def test_completion_over_the_cap_preserves_upstream_stop():
    response = _complete(_truncated_envelope(4001), max_tokens=4000)
    assert response.choices[0].finish_reason == "stop"


def test_completion_below_the_cap_keeps_stop():
    response = _complete(_truncated_envelope(3999), max_tokens=4000)
    assert response.choices[0].finish_reason == "stop"


def test_upstream_length_is_left_alone():
    response = _complete(_truncated_envelope(4000, finish_reason="length"), max_tokens=4000)
    assert response.choices[0].finish_reason == "length"


def test_no_max_tokens_means_no_rewrite():
    response = _complete(_truncated_envelope(4000))
    assert response.choices[0].finish_reason == "stop"


def test_max_completion_tokens_param_preserves_upstream_stop():
    """``max_completion_tokens`` is mapped to ``max_tokens`` before ``request_data`` is built."""
    response = _complete(_truncated_envelope(4000), max_completion_tokens=4000)
    assert response.choices[0].finish_reason == "stop"


# --------------------------------------------------------------------------
# Model catalog
# --------------------------------------------------------------------------


def test_get_models_returns_empty_without_calling_the_api():
    """ClinePass has no /models endpoint (404), and the inherited OpenAI
    implementation would ask for it at the wrong path. It must not make the
    request at all."""

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("get_models() must not perform an HTTP request")

    with patch.object(litellm.module_level_client, "get", explode):
        assert ClinePassConfig().get_models(api_key=API_KEY) == []


# --------------------------------------------------------------------------
# httpx internals
# --------------------------------------------------------------------------


def test_unwrap_envelope_survives_a_response_with_no_request_attached():
    """`httpx.Response.request` RAISES RuntimeError rather than returning None
    when no request is attached, so the unwrap must ask for it defensively."""
    raw = httpx.Response(200, json=ENVELOPED_COMPLETION)
    unwrapped = _unwrap_response_envelope(raw)
    assert unwrapped.json() == ENVELOPED_COMPLETION["data"]


@pytest.mark.asyncio
async def test_acompletion_uses_sync_transform_request_via_behaviour():
    """BaseLLMHTTPHandler builds the body with the sync transform_request on both
    paths, so an async override would be dead code -- the shape of bug this
    provider already shipped once. We assert this by verifying `acompletion`
    invokes the sync transform (which we mock here to prove it runs)."""
    captured = {}

    # We patch the sync transform_request to prove it is the one called
    # during the async flow.
    original_transform = ClinePassConfig().transform_request

    def mock_transform_request(*args, **kwargs):
        captured["called"] = True
        return original_transform(*args, **kwargs)

    async def fake_post(self, url, *args, **kwargs):
        return _response(ENVELOPED_COMPLETION)

    with (
        patch.object(ClinePassConfig, "transform_request", side_effect=mock_transform_request),
        patch.object(AsyncHTTPHandler, "post", fake_post),
    ):
        await litellm.acompletion(
            model="clinepass/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
        )

    assert captured.get("called") is True
