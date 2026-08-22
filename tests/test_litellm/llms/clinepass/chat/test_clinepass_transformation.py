"""Tests for the ClinePass provider.

The point of the end-to-end tests here is that they drive ``litellm.completion()``
with a mocked transport rather than calling the transforms directly -- a unit test
that calls ``transform_response()`` itself proves the function is correct but not
that anything invokes it, which is exactly how the envelope unwrap was previously
shipped as dead code.
"""

import json
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.clinepass.chat.transformation import (
    ClinePassConfig,
    _apply_model_prefix,
    _unwrap_response_envelope,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
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


def test_clinepass_is_not_a_json_configured_provider():
    """ClinePass needs a response transform, which the JSON provider system's
    OpenAI-SDK dispatch path never invokes. Guard against it drifting back."""
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert not JSONProviderRegistry.exists("clinepass")


def test_clinepass_stays_in_openai_compatible_providers():
    """Membership drives `_map_openai_exception`, so dropping it silently
    downgrades a 401 to APIConnectionError. The explicit dispatch branch in
    main.py precedes the openai_compatible_providers catch-all, so being listed
    here does NOT route ClinePass to the OpenAI SDK path."""
    assert "clinepass" in litellm.openai_compatible_providers


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
    assert _apply_model_prefix({"model": "deepseek-v4-flash"})["model"] == "clinepass/deepseek-v4-flash"


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
    assert captured["body"]["model"] == "clinepass/deepseek-v4-flash"
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
    assert captured["body"]["model"] == "clinepass/deepseek-v4-flash"
    assert response.choices[0].message.content == "pong"


def test_completion_streaming_is_not_unwrapped():
    """ClinePass does NOT wrap SSE chunks -- they are already OpenAI-shaped."""
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
            messages=[{"role": "user", "content": "count"}],
            max_tokens=4000,
            stream=True,
        )
        text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)

    assert text == "one two three"


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

    with patch.object(HTTPHandler, "post", fake_post):
        with pytest.raises(AuthenticationError) as excinfo:
            litellm.completion(
                model="clinepass/deepseek-v4-flash",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )

    assert excinfo.value.status_code == 401
