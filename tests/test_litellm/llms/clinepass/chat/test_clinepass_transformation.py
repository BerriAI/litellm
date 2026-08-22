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
    _correct_truncated_finish_reason,
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


# --------------------------------------------------------------------------
# Truncation reporting
#
# ClinePass was once observed returning finish_reason "stop" on a completion cut
# off by max_tokens. Re-probing the live API on 2026-08-22 could not reproduce
# it (see _correct_truncated_finish_reason's docstring), so the correction is a
# conservative safety net: single-choice only, upstream "stop" only, and only
# when usage shows the cap was actually reached.
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


def test_completion_at_the_cap_is_reported_as_length_not_stop():
    response = _complete(_truncated_envelope(4000), max_tokens=4000)
    assert response.choices[0].finish_reason == "length"


def test_completion_over_the_cap_is_reported_as_length():
    response = _complete(_truncated_envelope(4001), max_tokens=4000)
    assert response.choices[0].finish_reason == "length"


def test_completion_below_the_cap_keeps_stop():
    response = _complete(_truncated_envelope(3999), max_tokens=4000)
    assert response.choices[0].finish_reason == "stop"


def test_upstream_length_is_left_alone():
    response = _complete(_truncated_envelope(4000, finish_reason="length"), max_tokens=4000)
    assert response.choices[0].finish_reason == "length"


def test_no_max_tokens_means_no_rewrite():
    response = _complete(_truncated_envelope(4000))
    assert response.choices[0].finish_reason == "stop"


def test_max_completion_tokens_also_detects_truncation():
    response = _complete(_truncated_envelope(4000), max_completion_tokens=4000)
    assert response.choices[0].finish_reason == "length"


@pytest.mark.parametrize("bad", [None, "4000", 0, -1, True, False])
def test_unusable_cap_is_ignored(bad):
    """Non-numeric, zero, negative and bool caps carry no truncation signal.

    ``bool`` matters because it subclasses ``int``: ``True`` would otherwise be
    read as a cap of 1 and relabel every response as truncated.
    """

    class _Choice:
        finish_reason = "stop"

    class _Usage:
        completion_tokens = 9999

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    result = _correct_truncated_finish_reason(_Response(), {"max_tokens": bad})
    assert result.choices[0].finish_reason == "stop"


def test_missing_usage_is_ignored():
    class _Choice:
        finish_reason = "stop"

    class _Response:
        choices = [_Choice()]
        usage = None

    result = _correct_truncated_finish_reason(_Response(), {"max_tokens": 4000})
    assert result.choices[0].finish_reason == "stop"


def _stub_response(finish_reasons, completion_tokens):
    """Minimal ModelResponse-shaped stub for the truncation helper."""

    class _Choice:
        def __init__(self, reason):
            self.finish_reason = reason

    class _Usage:
        pass

    usage = _Usage()
    usage.completion_tokens = completion_tokens

    class _Response:
        pass

    response = _Response()
    response.choices = [_Choice(r) for r in finish_reasons]
    response.usage = usage
    return response


def test_multi_choice_response_is_never_rewritten():
    """`usage.completion_tokens` is an aggregate across choices while `max_tokens`
    is per choice, so the aggregate cannot say WHICH choice was truncated.

    Two naturally-finished 60-token choices under a cap of 100 aggregate to 120,
    which would otherwise relabel both as `length`.
    """
    response = _stub_response(["stop", "stop"], completion_tokens=120)
    result = _correct_truncated_finish_reason(response, {"max_tokens": 100})
    assert [c.finish_reason for c in result.choices] == ["stop", "stop"]


def test_single_choice_at_the_cap_is_still_rewritten():
    """The n>1 guard must not disable the correction for the normal n=1 case."""
    response = _stub_response(["stop"], completion_tokens=100)
    result = _correct_truncated_finish_reason(response, {"max_tokens": 100})
    assert result.choices[0].finish_reason == "length"


def test_float_usage_and_cap_are_honoured():
    """A gateway that reports usage as JSON floats must still be understood."""
    response = _stub_response(["stop"], completion_tokens=4000.0)
    result = _correct_truncated_finish_reason(response, {"max_tokens": 4000.0})
    assert result.choices[0].finish_reason == "length"


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


def test_clinepass_config_has_no_async_transform_request_override():
    """BaseLLMHTTPHandler builds the body with the sync transform_request on both
    paths, so an async override would be dead code -- the shape of bug this
    provider already shipped once."""
    assert "async_transform_request" not in ClinePassConfig.__dict__
