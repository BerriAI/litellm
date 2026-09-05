"""Tests for `BedrockConverseLLM.completion`'s Rust chat completions hook.

The native callables are dependency-injected, so these run without the compiled
extension, and AWS credential resolution is stubbed so nothing reaches STS.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from botocore.credentials import Credentials
from litellm.llms.bedrock.chat.converse_handler import BedrockConverseLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.rust_bridge import chat_completions as bridge
from litellm.types.utils import ModelResponse

RUST_RESPONSE = {
    "created": 1_700_000_000,
    "model": "anthropic.claude-sonnet-4-5-v1:0",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello from rust"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "text_tokens": 11,
        },
    },
}

RESOLVED_CREDENTIALS = Credentials(
    access_key="AKIARESOLVED",
    secret_key="resolved-secret",
    token="resolved-token",
)


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch):
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    bridge.set_rust_chat_completions(
        chat_completions=None, achat_completions=None, decline=None
    )
    yield
    bridge.set_rust_chat_completions(
        chat_completions=None, achat_completions=None, decline=None
    )


def _inject(*, decline_reason=None, error: Exception | None = None):
    seen: dict[str, list[dict]] = {"gate": [], "call": []}

    def gate(**kwargs):
        seen["gate"].append(kwargs)
        return decline_reason

    def native(**kwargs):
        seen["call"].append(kwargs)
        if error is not None:
            raise error
        return dict(RUST_RESPONSE)

    bridge.set_rust_chat_completions(decline=gate, chat_completions=native)
    return seen


def _completion_kwargs(**overrides):
    kwargs = {
        "model": "bedrock/us-east-1/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
        "api_base": None,
        "custom_prompt_dict": {},
        "model_response": ModelResponse(),
        "encoding": None,
        "logging_obj": MagicMock(),
        "optional_params": {"maxTokens": 16},
        "acompletion": False,
        "timeout": 30.0,
        "litellm_params": {"rust": True},
        "extra_headers": None,
        "client": None,
        "api_key": None,
    }
    kwargs.update(overrides)
    return kwargs


def _run(**overrides):
    with patch.object(
        BedrockConverseLLM, "get_credentials", return_value=RESOLVED_CREDENTIALS
    ):
        return BedrockConverseLLM().completion(**_completion_kwargs(**overrides))


def _recording_logging_obj():
    """A logging object that keeps each hook's payload in a real list, so a test
    can assert which path logged and what it carried."""
    calls = {"pre_call": [], "post_call": []}
    logging_obj = MagicMock()
    logging_obj.pre_call.side_effect = lambda **kwargs: calls["pre_call"].append(kwargs)
    logging_obj.post_call.side_effect = lambda **kwargs: calls["post_call"].append(kwargs)
    return logging_obj, calls


def test_rust_true_serves_the_call_and_stamps_the_header():
    seen = _inject()
    response = _run()

    assert response.choices[0].message.content == "hello from rust"
    assert response._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}
    assert len(seen["call"]) == 1


def test_the_core_receives_the_credentials_this_handler_already_resolved():
    """Both paths must sign as the same principal, so the resolved credentials
    are handed down rather than re-derived from ambient AWS state."""
    seen = _inject()
    _run()

    params = seen["call"][0]["optional_params"]
    assert params["aws_access_key_id"] == "AKIARESOLVED"
    assert params["aws_secret_access_key"] == "resolved-secret"
    assert params["aws_session_token"] == "resolved-token"
    assert params["aws_region_name"] == "us-east-1"


def test_the_core_receives_the_converse_url_this_handler_already_built():
    seen = _inject()
    _run()

    assert seen["call"][0]["api_base"].endswith(
        "/model/anthropic.claude-sonnet-4-5-v1%3A0/converse"
    )
    assert "bedrock-runtime.us-east-1.amazonaws.com" in seen["call"][0]["api_base"]


def test_the_core_receives_the_untranslated_openai_messages():
    seen = _inject()
    _run(
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert seen["call"][0]["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_without_the_opt_in_the_core_is_never_consulted():
    seen = _inject()
    try:
        _run(litellm_params={})
    except Exception:
        # The Python path goes on to make an HTTP call; not reaching the gate
        # is the assertion, so a failure past this point is expected.
        pass
    assert seen["gate"] == []
    assert seen["call"] == []


def test_streaming_stays_on_the_python_path():
    seen = _inject()
    try:
        _run(optional_params={"maxTokens": 16, "stream": True})
    except Exception:
        pass
    assert seen["gate"] == []


def test_a_declined_request_never_reaches_the_native_call():
    seen = _inject(decline_reason="unrecognized request parameter")
    try:
        _run()
    except Exception:
        pass
    assert len(seen["gate"]) == 1
    assert seen["call"] == []


def test_pre_call_logging_fires_exactly_once_on_the_rust_path():
    _inject()
    logging_obj = MagicMock()
    _run(logging_obj=logging_obj)
    assert logging_obj.pre_call.call_count == 1


@pytest.mark.asyncio
async def test_the_async_path_falls_back_when_the_core_declines(monkeypatch):
    class _Declined(Exception):
        pass

    class _FakeNative:
        RustBridgeDeclined = _Declined
        RustUpstreamError = type("_Upstream", (Exception,), {})

    monkeypatch.setattr(bridge, "get_native_bridge", lambda: _FakeNative())

    async def declining_native(**_kwargs):
        raise _Declined("blank message text")

    bridge.set_rust_chat_completions(
        decline=lambda **_kwargs: None, achat_completions=declining_native
    )

    sentinel = object()

    async def python_path(**_kwargs):
        return sentinel

    with (
        patch.object(
            BedrockConverseLLM, "get_credentials", return_value=RESOLVED_CREDENTIALS
        ),
        patch.object(
            BedrockConverseLLM, "async_completion", side_effect=python_path
        ) as python_call,
    ):
        result = await BedrockConverseLLM().completion(
            **_completion_kwargs(acompletion=True)
        )

    assert result is sentinel
    assert python_call.called, "a failing rust call must re-enter the python path"


@pytest.mark.asyncio
async def test_the_async_path_serves_the_rust_response_without_the_fallback():
    async def native(**_kwargs):
        return dict(RUST_RESPONSE)

    bridge.set_rust_chat_completions(
        decline=lambda **_kwargs: None, achat_completions=native
    )

    with (
        patch.object(
            BedrockConverseLLM, "get_credentials", return_value=RESOLVED_CREDENTIALS
        ),
        patch.object(BedrockConverseLLM, "async_completion") as python_call,
    ):
        result = await BedrockConverseLLM().completion(
            **_completion_kwargs(acompletion=True)
        )

    assert result.choices[0].message.content == "hello from rust"
    assert result._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}
    assert not python_call.called


@pytest.mark.asyncio
async def test_pre_call_logging_fires_once_even_when_the_rust_path_declines():
    """One request, one pre_call. Without the suppression the Python fallback
    logs a second one and non-idempotent callbacks run twice."""

    class _Declined(Exception):
        pass

    class _FakeNative:
        RustBridgeDeclined = _Declined
        RustUpstreamError = type("_Upstream", (Exception,), {})

    async def declining_native(**_kwargs):
        raise _Declined("blank message text")

    logging_obj = MagicMock()
    served = []

    async def python_path(**kwargs):
        served.append(kwargs)
        return ModelResponse()

    with (
        patch.object(bridge, "get_native_bridge", lambda: _FakeNative()),
        patch.object(
            BedrockConverseLLM, "get_credentials", return_value=RESOLVED_CREDENTIALS
        ),
        patch.object(
            BedrockConverseLLM, "async_completion", side_effect=python_path
        ),
    ):
        bridge.set_rust_chat_completions(
            decline=lambda **_kwargs: None, achat_completions=declining_native
        )
        await BedrockConverseLLM().completion(
            **_completion_kwargs(acompletion=True, logging_obj=logging_obj)
        )

    assert logging_obj.pre_call.call_count == 1
    assert served and served[0]["skip_pre_call_logging"] is True


CONVERSE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
}


async def _drive_async_completion(*, skip_pre_call_logging: bool, logging_obj):
    """Run the real `async_completion` with a stubbed transport."""
    import httpx as _httpx

    client = MagicMock()

    async def post(**_kwargs):
        return _httpx.Response(
            200,
            json=CONVERSE_RESPONSE,
            request=_httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
        )

    client.post = post
    client.__class__ = AsyncHTTPHandler

    return await BedrockConverseLLM().async_completion(
        model="anthropic.claude-sonnet-4-5-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://bedrock-runtime.us-west-2.amazonaws.com/model/m/converse",
        model_response=ModelResponse(),
        timeout=30.0,
        encoding=None,
        logging_obj=logging_obj,
        stream=None,
        optional_params={"maxTokens": 16},
        litellm_params={"aws_region_name": "us-west-2"},
        credentials=RESOLVED_CREDENTIALS,
        headers={},
        client=client,
        skip_pre_call_logging=skip_pre_call_logging,
    )


@pytest.mark.asyncio
async def test_async_completion_honors_the_pre_call_suppression():
    logging_obj = MagicMock()
    await _drive_async_completion(skip_pre_call_logging=True, logging_obj=logging_obj)
    assert logging_obj.pre_call.call_count == 0


@pytest.mark.asyncio
async def test_async_completion_logs_pre_call_by_default():
    """The suppression must be opt-in, so every existing caller keeps its log."""
    logging_obj = MagicMock()
    await _drive_async_completion(skip_pre_call_logging=False, logging_obj=logging_obj)
    assert logging_obj.pre_call.call_count == 1


def _sync_client_returning_converse_response():
    client = MagicMock()
    client.post = lambda **_kwargs: httpx.Response(
        200,
        json=CONVERSE_RESPONSE,
        request=httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
    )
    client.__class__ = HTTPHandler
    return client


def test_pre_call_logging_fires_once_when_the_sync_rust_path_declines():
    """One request, one pre_call, on the synchronous path too.

    The gate accepts and logs, then the native call declines before the
    provider is reached, so execution continues into the Python path below.
    That is the same attempt continuing; without the suppression it logs a
    second pre_call and non-idempotent callbacks run twice for one request.
    """

    class _Declined(Exception):
        pass

    class _FakeNative:
        RustBridgeDeclined = _Declined
        RustUpstreamError = type("_Upstream", (Exception,), {})

    def declining_native(**_kwargs):
        raise _Declined("blank message text")

    logging_obj = MagicMock()

    with patch.object(bridge, "get_native_bridge", lambda: _FakeNative()):
        bridge.set_rust_chat_completions(
            decline=lambda **_kwargs: None, chat_completions=declining_native
        )
        response = _run(
            logging_obj=logging_obj,
            client=_sync_client_returning_converse_response(),
        )

    assert response.choices[0].message.content == "hi"
    assert logging_obj.pre_call.call_count == 1


def test_the_sync_python_path_still_logs_pre_call_without_the_opt_in():
    """The suppression must not swallow the log on a request the gate declined,
    so a deployment with no `rust` flag keeps exactly the log it always had."""
    logging_obj = MagicMock()
    response = _run(
        logging_obj=logging_obj,
        litellm_params={},
        client=_sync_client_returning_converse_response(),
    )

    assert response.choices[0].message.content == "hi"
    assert logging_obj.pre_call.call_count == 1


def test_post_call_logging_fires_on_the_sync_rust_path():
    """The Rust core owns the provider call, so the Converse transform that
    normally raises `post_call` never runs. Without the bridge hook every
    post_call callback goes silent and `original_response` stays unset."""
    import json

    _inject()
    logging_obj = MagicMock()
    _run(logging_obj=logging_obj)

    assert logging_obj.post_call.call_count == 1
    logged = logging_obj.post_call.call_args.kwargs["original_response"]
    assert json.loads(logged)["choices"][0]["message"]["content"] == "hello from rust"


@pytest.mark.asyncio
async def test_post_call_logging_fires_on_the_async_rust_path():
    """The asynchronous path runs through the same hook, so the two paths
    cannot drift apart the way the pre_call suppression once did."""
    import json

    async def native(**_kwargs):
        return dict(RUST_RESPONSE)

    bridge.set_rust_chat_completions(
        decline=lambda **_kwargs: None, achat_completions=native
    )
    logging_obj = MagicMock()

    with patch.object(
        BedrockConverseLLM, "get_credentials", return_value=RESOLVED_CREDENTIALS
    ):
        await BedrockConverseLLM().completion(
            **_completion_kwargs(acompletion=True, logging_obj=logging_obj)
        )

    assert logging_obj.post_call.call_count == 1
    logged = logging_obj.post_call.call_args.kwargs["original_response"]
    assert json.loads(logged)["choices"][0]["message"]["content"] == "hello from rust"


def test_post_call_is_not_logged_twice_when_the_sync_rust_call_declines():
    """A decline never reached the provider, so the Python path serves the
    request and owns the only post_call. Firing the hook there too would double
    every post_call callback for one request."""

    class _Declined(Exception):
        pass

    class _FakeNative:
        RustBridgeDeclined = _Declined
        RustUpstreamError = type("_Upstream", (Exception,), {})

    def declining_native(**_kwargs):
        raise _Declined("blank message text")

    logging_obj, calls = _recording_logging_obj()

    with patch.object(bridge, "get_native_bridge", lambda: _FakeNative()):
        bridge.set_rust_chat_completions(
            decline=lambda **_kwargs: None, chat_completions=declining_native
        )
        response = _run(
            logging_obj=logging_obj,
            client=_sync_client_returning_converse_response(),
        )

    assert response.choices[0].message.content == "hi"
    assert len(calls["post_call"]) == 1
    assert "hi" in calls["post_call"][0]["original_response"]
