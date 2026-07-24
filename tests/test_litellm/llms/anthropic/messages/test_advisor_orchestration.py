"""
Tests for advisor orchestration on non-Anthropic providers.

Tests:
1. can_handle edge cases
2. Anthropic native: interceptor does NOT trigger (routing confirmed)
3. Orchestration loop logic (mocked backend): single advisor call, multi-turn, max_uses cap
4. strip_advisor_blocks_from_messages with replace_with_text=True
"""

from typing import Dict
from unittest.mock import AsyncMock, patch

import pytest

ADVISOR_TOOL = {
    "type": "advisor_20260301",
    "name": "advisor",
    "model": "claude-opus-4-6",
}

MESSAGES = [
    {
        "role": "user",
        "content": "Write a Python function that checks if a number is prime.",
    }
]


def _make_text_response(text: str, model: str = "openai/gpt-4o-mini") -> Dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _make_advisor_tool_use_response(
    question: str = "How should I approach this?",
    tool_id: str = "toolu_advisor_01",
    model: str = "openai/gpt-4o-mini",
) -> Dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "advisor",
                "input": {"question": question},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 15},
    }


# ---------------------------------------------------------------------------
# 1. can_handle edge cases
# ---------------------------------------------------------------------------


def test_can_handle_edge_cases():
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    h = AdvisorOrchestrationHandler()

    assert h.can_handle([ADVISOR_TOOL], "openai")
    assert h.can_handle([ADVISOR_TOOL], "bedrock")
    assert h.can_handle([ADVISOR_TOOL], "gemini")
    assert not h.can_handle([ADVISOR_TOOL], "anthropic")
    assert not h.can_handle([], "openai")
    assert not h.can_handle(None, "openai")
    assert not h.can_handle([{"type": "function", "name": "bash"}], "openai")
    # provider=None: unknown → should intercept (treat as non-native)
    assert h.can_handle([ADVISOR_TOOL], None)


# ---------------------------------------------------------------------------
# 2. Anthropic native: interceptor must NOT trigger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_native_interceptor_skipped():
    """
    For provider=anthropic, can_handle() must return False.
    The interceptor must never call handle().
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    h = AdvisorOrchestrationHandler()
    assert not h.can_handle(
        [ADVISOR_TOOL], "anthropic"
    ), "Interceptor must NOT trigger for anthropic provider"


# ---------------------------------------------------------------------------
# 3. Orchestration loop: no advisor call needed (executor returns text directly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_no_advisor_call():
    """Executor returns text on first try — no advisor call, loop exits immediately."""
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
        _call_messages_handler,
    )

    final_text = "def is_prime(n): return n > 1 and all(n % i for i in range(2, n))"
    executor_response = _make_text_response(final_text)

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        new_callable=AsyncMock,
        return_value=executor_response,
    ) as mock_call:
        h = AdvisorOrchestrationHandler()
        result = await h.handle(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
        )

    # Only one call (executor), no advisor call
    assert mock_call.call_count == 1
    content = result.get("content", [])
    texts = [b for b in content if b.get("type") == "text"]
    assert len(texts) == 1
    assert final_text in texts[0]["text"]


# ---------------------------------------------------------------------------
# 4. Orchestration loop: one advisor call then final text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_one_advisor_call():
    """
    Executor calls advisor once → advisor responds → executor produces final text.
    Total calls: 3 (executor, advisor, executor-final).
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    advisor_tool_use_resp = _make_advisor_tool_use_response(
        question="Should I use a sieve or trial division?",
        tool_id="toolu_01",
    )
    advisor_advice_resp = _make_text_response(
        "Use trial division for simplicity — only check up to sqrt(n).",
        model="claude-opus-4-6",
    )
    final_resp = _make_text_response(
        "def is_prime(n):\n    import math\n    if n < 2: return False\n    for i in range(2, int(math.sqrt(n))+1):\n        if n % i == 0: return False\n    return True"
    )

    call_count = 0

    async def mock_call(model, messages, tools, stream, max_tokens, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return advisor_tool_use_resp  # executor: calls advisor
        if call_count == 2:
            return advisor_advice_resp  # advisor: returns advice
        return final_resp  # executor: final answer

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_call,
    ):
        h = AdvisorOrchestrationHandler()
        result = await h.handle(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
        )

    assert call_count == 3
    content = result.get("content", [])
    texts = [b for b in content if b.get("type") == "text"]
    assert len(texts) == 1
    assert "is_prime" in texts[0]["text"]

    # No advisor tool_use blocks in final response
    advisor_uses = [
        b for b in content if b.get("type") == "tool_use" and b.get("name") == "advisor"
    ]
    assert len(advisor_uses) == 0


# ---------------------------------------------------------------------------
# 5. max_uses cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_max_uses_raises():
    """Loop exceeding max_uses must raise AdvisorMaxIterationsError."""
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorMaxIterationsError,
        AdvisorOrchestrationHandler,
    )

    advisor_tool_with_max = {**ADVISOR_TOOL, "max_uses": 2}
    # Always return an advisor tool_use → loop never terminates naturally
    advisor_tool_use_resp = _make_advisor_tool_use_response()
    advisor_advice_resp = _make_text_response("Here is my advice.")

    call_count = 0

    async def mock_call(model, messages, tools, stream, max_tokens, **kwargs):
        nonlocal call_count
        call_count += 1
        # Executor calls always return advisor tool_use; advisor always returns text
        if tools is None:
            return advisor_advice_resp
        return advisor_tool_use_resp

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_call,
    ):
        h = AdvisorOrchestrationHandler()
        with pytest.raises(AdvisorMaxIterationsError):
            await h.handle(
                model="openai/gpt-4o-mini",
                messages=MESSAGES,
                tools=[advisor_tool_with_max],
                stream=False,
                max_tokens=512,
                custom_llm_provider="openai",
            )


# ---------------------------------------------------------------------------
# 6. Streaming: final response wrapped in FakeAnthropicMessagesStreamIterator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_streaming_wraps_response():
    """stream=True: final response must be wrapped in FakeAnthropicMessagesStreamIterator."""
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    executor_response = _make_text_response("Hello, world!")

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        new_callable=AsyncMock,
        return_value=executor_response,
    ):
        h = AdvisorOrchestrationHandler()
        result = await h.handle(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=True,
            max_tokens=512,
            custom_llm_provider="openai",
        )

    assert isinstance(result, FakeAnthropicMessagesStreamIterator)

    chunks = []
    async for chunk in result:
        chunks.append(chunk)

    assert len(chunks) > 0
    first = chunks[0].decode() if isinstance(chunks[0], bytes) else str(chunks[0])
    assert "message_start" in first


# ---------------------------------------------------------------------------
# 7. Multi-turn: prior advisor blocks replaced with text in history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prior_advisor_blocks_replaced_in_history():
    """
    History containing server_tool_use + advisor_tool_result blocks gets
    collapsed to <advisor_feedback> text before forwarding to the executor.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    messages_with_history = [
        *MESSAGES,
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtool_01",
                    "name": "advisor",
                    "input": {},
                },
                {
                    "type": "advisor_tool_result",
                    "tool_use_id": "srvtool_01",
                    "content": "Use trial division up to sqrt(n).",
                },
                {"type": "text", "text": "I will now write the function."},
            ],
        },
        {"role": "user", "content": "Actually make it more efficient."},
    ]

    captured_messages = []

    async def mock_call(model, messages, tools, stream, max_tokens, **kwargs):
        captured_messages.extend(messages)
        return _make_text_response("Here is the efficient version.")

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_call,
    ):
        h = AdvisorOrchestrationHandler()
        await h.handle(
            model="openai/gpt-4o-mini",
            messages=messages_with_history,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
        )

    # Find the assistant message in forwarded history
    assistant_msgs = [m for m in captured_messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) >= 1
    content = assistant_msgs[0].get("content", [])
    types = [b.get("type") for b in content if isinstance(b, dict)]

    # server_tool_use and advisor_tool_result must be gone
    assert "server_tool_use" not in types
    assert "advisor_tool_result" not in types

    # Text block with advisor feedback must be present
    text_blocks = [b for b in content if b.get("type") == "text"]
    feedback_blocks = [
        b for b in text_blocks if "advisor_feedback" in b.get("text", "")
    ]
    assert len(feedback_blocks) >= 1
    assert "trial division" in feedback_blocks[0]["text"]


# ---------------------------------------------------------------------------
# 8. Advisor tool is translated to a regular tool for the executor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advisor_tool_translated_for_executor():
    """
    The executor must receive a regular tool definition (not advisor_20260301 type).
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    captured_tools = []

    async def mock_call(model, messages, tools, stream, max_tokens, **kwargs):
        if tools:
            captured_tools.extend(tools)
        return _make_text_response("Done.")

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_call,
    ):
        h = AdvisorOrchestrationHandler()
        await h.handle(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
        )

    assert len(captured_tools) > 0
    advisor_tool = next(t for t in captured_tools if t.get("name") == "advisor")
    # Must NOT have the advisor_20260301 type (provider won't understand it)
    assert advisor_tool.get("type") != "advisor_20260301"
    # Must have a description and input_schema
    assert "description" in advisor_tool
    assert "input_schema" in advisor_tool


# ---------------------------------------------------------------------------
# 9. max_uses=0 means zero advisor calls allowed — first call raises immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_uses_zero_raises_on_first_advisor_call():
    """max_uses=0 must cause AdvisorMaxIterationsError on the first advisor call."""
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorMaxIterationsError,
        AdvisorOrchestrationHandler,
    )

    advisor_tool_with_zero = {**ADVISOR_TOOL, "max_uses": 0}
    advisor_tool_use_resp = _make_advisor_tool_use_response()

    async def mock_call(model, messages, tools, stream, max_tokens, **kwargs):
        return advisor_tool_use_resp  # executor always tries to call advisor

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_call,
    ):
        h = AdvisorOrchestrationHandler()
        with pytest.raises(AdvisorMaxIterationsError):
            await h.handle(
                model="openai/gpt-4o-mini",
                messages=MESSAGES,
                tools=[advisor_tool_with_zero],
                stream=False,
                max_tokens=512,
                custom_llm_provider="openai",
            )


# ---------------------------------------------------------------------------
# 10. Missing model in advisor tool definition raises ValueError from handle()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_advisor_model_raises_value_error():
    """handle() must raise ValueError when the advisor tool has no model field."""
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    advisor_tool_no_model = {"type": "advisor_20260301", "name": "advisor"}

    h = AdvisorOrchestrationHandler()
    with pytest.raises(ValueError, match="model"):
        await h.handle(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[advisor_tool_no_model],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
        )


# ---------------------------------------------------------------------------
# 11. max_uses not set → falls back to ADVISOR_MAX_USES default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_uses_none_falls_back_to_default():
    """When max_uses is absent, the handler uses ADVISOR_MAX_USES from constants."""
    import litellm.constants as _c
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorMaxIterationsError,
        AdvisorOrchestrationHandler,
    )

    advisor_tool_use_resp = _make_advisor_tool_use_response()
    advisor_advice_resp = _make_text_response("Here is advice.")

    async def mock_call(model, messages, tools, stream, max_tokens, **kwargs):
        if tools is None:
            return advisor_advice_resp
        return advisor_tool_use_resp

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_call,
    ):
        h = AdvisorOrchestrationHandler()
        with pytest.raises(AdvisorMaxIterationsError) as exc_info:
            await h.handle(
                model="openai/gpt-4o-mini",
                messages=MESSAGES,
                tools=[ADVISOR_TOOL],  # no max_uses — should use default
                stream=False,
                max_tokens=512,
                custom_llm_provider="openai",
            )

    assert str(_c.ADVISOR_MAX_USES) in str(exc_info.value)


# ---------------------------------------------------------------------------
# 12. Defense-in-depth: client-supplied advisor api_base/api_key are dropped
#     unless the proxy admin opted into clientside credentials
# ---------------------------------------------------------------------------


ADVISOR_TOOL_WITH_CREDS = {
    "type": "advisor_20260301",
    "name": "advisor",
    "model": "claude-opus-4-6",
    "api_base": "https://other.example",
    "api_key": "sk-other",
}


async def _run_advisor_and_capture_subcall_kwargs():
    """Run one advisor turn and return the kwargs of the advisor sub-call."""
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    advisor_tool_use_resp = _make_advisor_tool_use_response(tool_id="toolu_01")
    advisor_advice_resp = _make_text_response("advice", model="claude-opus-4-6")
    final_resp = _make_text_response("final answer")

    captured = {}
    call_count = 0

    async def mock_call(model, messages, tools, stream, max_tokens, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return advisor_tool_use_resp
        if call_count == 2:
            # The advisor sub-call — capture its routing kwargs.
            captured["api_key"] = kwargs.get("api_key")
            captured["api_base"] = kwargs.get("api_base")
            return advisor_advice_resp
        return final_resp

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
            side_effect=mock_call,
        ),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.validate_url",
        ),
    ):
        h = AdvisorOrchestrationHandler()
        await h.handle(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL_WITH_CREDS],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
        )
    return captured


@pytest.mark.asyncio
async def test_advisor_creds_dropped_when_proxy_opt_in_disabled():
    """On the proxy without opt-in, the caller's advisor api_base/api_key must
    NOT reach the sub-call (would redirect it / leak the server key)."""
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
        return_value=False,
    ):
        captured = await _run_advisor_and_capture_subcall_kwargs()
    assert captured["api_key"] is None
    assert captured["api_base"] is None


@pytest.mark.asyncio
async def test_advisor_creds_honored_when_proxy_opt_in_enabled():
    """With the admin opt-in, the documented clientside routing still works."""
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
        return_value=True,
    ):
        captured = await _run_advisor_and_capture_subcall_kwargs()
    assert captured["api_key"] == "sk-other"
    assert captured["api_base"] == "https://other.example"


# ---------------------------------------------------------------------------
# 13. The proxy gate itself: _allow_client_side_advisor_credentials() and the
#     full handle() driven by the real proxy general_settings flag.
# ---------------------------------------------------------------------------


def _fake_proxy_server(general_settings: Dict):
    """A stand-in litellm.proxy.proxy_server module exposing general_settings.

    The real proxy_server pulls in heavy optional deps that may be absent in a
    unit-test environment, so the gate's
    ``from litellm.proxy.proxy_server import general_settings`` is satisfied by
    injecting this lightweight module into sys.modules.
    """
    import types

    module = types.ModuleType("litellm.proxy.proxy_server")
    module.general_settings = general_settings  # type: ignore[attr-defined]
    return module


def test_allow_client_side_advisor_credentials_reads_proxy_flag():
    """The gate mirrors the proxy's allow_client_side_credentials opt-in."""
    import sys

    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _allow_client_side_advisor_credentials,
    )

    cases = (
        ({"allow_client_side_credentials": True}, True),
        ({"allow_client_side_credentials": False}, False),
        # Flag absent entirely -> default deny on the proxy.
        ({}, False),
    )
    for settings, expected in cases:
        with patch.dict(
            sys.modules,
            {"litellm.proxy.proxy_server": _fake_proxy_server(settings)},
        ):
            assert _allow_client_side_advisor_credentials() is expected


def test_allow_client_side_advisor_credentials_defaults_true_outside_proxy():
    """Outside the proxy (proxy_server import unavailable), there is no admin
    boundary, so the gate permits client-supplied routing."""
    import builtins
    import sys

    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _allow_client_side_advisor_credentials,
    )

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "litellm.proxy.proxy_server":
            raise ImportError("proxy server unavailable")
        return real_import(name, *args, **kwargs)

    with patch.dict(sys.modules):
        sys.modules.pop("litellm.proxy.proxy_server", None)
        with patch.object(builtins, "__import__", _blocked_import):
            assert _allow_client_side_advisor_credentials() is True


def test_advisor_gate_propagates_non_import_errors():
    """Non-ImportError failures during the proxy module probe must not
    default permissive. If the proxy is partially loaded and raises
    RuntimeError, the gate should surface that rather than silently
    returning True."""
    import sys

    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors import (
        advisor,
    )

    original = sys.modules.get("litellm.proxy.proxy_server")

    class _Broken:
        def __getattr__(self, _name):
            raise RuntimeError("partial proxy boot")

    sys.modules["litellm.proxy.proxy_server"] = _Broken()
    try:
        with pytest.raises(RuntimeError, match="partial proxy boot"):
            advisor._allow_client_side_advisor_credentials()
    finally:
        if original is None:
            sys.modules.pop("litellm.proxy.proxy_server", None)
        else:
            sys.modules["litellm.proxy.proxy_server"] = original


@pytest.mark.asyncio
async def test_advisor_ignores_tool_credentials_when_clientside_disabled():
    """Driven by the real proxy flag (not a patched gate): with
    allow_client_side_credentials False, the tool-supplied api_base/api_key must
    not reach the advisor sub-call."""
    import sys

    with patch.dict(
        sys.modules,
        {
            "litellm.proxy.proxy_server": _fake_proxy_server(
                {"allow_client_side_credentials": False}
            )
        },
    ):
        captured = await _run_advisor_and_capture_subcall_kwargs()
    assert captured["api_key"] is None
    assert captured["api_base"] is None


@pytest.mark.asyncio
async def test_advisor_uses_tool_credentials_when_clientside_enabled():
    """Driven by the real proxy flag: with allow_client_side_credentials True,
    the tool-supplied api_base/api_key flow through to the advisor sub-call."""
    import sys

    with patch.dict(
        sys.modules,
        {
            "litellm.proxy.proxy_server": _fake_proxy_server(
                {"allow_client_side_credentials": True}
            )
        },
    ):
        captured = await _run_advisor_and_capture_subcall_kwargs()
    assert captured["api_key"] == "sk-other"
    assert captured["api_base"] == "https://other.example"


# ---------------------------------------------------------------------------
# 14. _resolve_advisor_credentials: api_base is only honored alongside a
#     caller-supplied api_key, and is SSRF-validated before use.
# ---------------------------------------------------------------------------


def test_resolve_advisor_credentials_returns_none_when_gate_closed():
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
        return_value=False,
    ):
        result = _resolve_advisor_credentials(ADVISOR_TOOL_WITH_CREDS)
    assert result == (None, None)


def test_resolve_advisor_credentials_allows_api_key_without_api_base():
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    tool = {**ADVISOR_TOOL, "api_key": "sk-other"}
    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
            return_value=True,
        ),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.validate_url",
            side_effect=AssertionError("validate_url must not run without an api_base"),
        ),
    ):
        result = _resolve_advisor_credentials(tool)
    assert result == ("sk-other", None)


def test_resolve_advisor_credentials_rejects_api_base_without_api_key():
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    tool = {**ADVISOR_TOOL, "api_base": "https://other.example"}
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
        return_value=True,
    ):
        with pytest.raises(ValueError, match="api_base"):
            _resolve_advisor_credentials(tool)


def test_resolve_advisor_credentials_validates_api_base_before_use():
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
            return_value=True,
        ),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.validate_url"
        ) as mock_validate,
    ):
        result = _resolve_advisor_credentials(ADVISOR_TOOL_WITH_CREDS)
    mock_validate.assert_called_once_with("https://other.example")
    assert result == ("sk-other", "https://other.example")


def test_resolve_advisor_credentials_propagates_ssrf_error():
    from litellm.litellm_core_utils.url_utils import SSRFError
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
            return_value=True,
        ),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.validate_url",
            side_effect=SSRFError("URL targets a blocked address"),
        ),
    ):
        with pytest.raises(SSRFError):
            _resolve_advisor_credentials(ADVISOR_TOOL_WITH_CREDS)


def test_resolve_advisor_credentials_skips_validation_when_url_validation_disabled():
    import litellm

    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
            return_value=True,
        ),
        patch.object(litellm, "user_url_validation", False),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.validate_url",
            side_effect=AssertionError("validate_url must not run when user_url_validation is disabled"),
        ),
    ):
        result = _resolve_advisor_credentials(ADVISOR_TOOL_WITH_CREDS)
    assert result == ("sk-other", "https://other.example")


def test_resolve_advisor_credentials_blocks_real_cloud_metadata_address():
    """End-to-end (no mocked validate_url): a caller can't redirect the
    advisor sub-call to the cloud-metadata address even with an api_key."""
    from litellm.litellm_core_utils.url_utils import SSRFError
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    tool = {
        **ADVISOR_TOOL,
        "api_key": "sk-other",
        "api_base": "https://169.254.169.254/latest/meta-data/",
    }
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
        return_value=True,
    ):
        with pytest.raises(SSRFError):
            _resolve_advisor_credentials(tool)


def test_resolve_advisor_credentials_rejects_non_https_api_base():
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    tool = {**ADVISOR_TOOL, "api_key": "sk-other", "api_base": "http://8.8.8.8"}
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
        return_value=True,
    ):
        with pytest.raises(ValueError, match="https"):
            _resolve_advisor_credentials(tool)


def test_resolve_advisor_credentials_rejects_api_base_when_ssl_verify_disabled():
    import litellm

    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    tool = {**ADVISOR_TOOL, "api_key": "sk-other", "api_base": "https://8.8.8.8"}
    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
            return_value=True,
        ),
        patch.object(litellm, "ssl_verify", False),
    ):
        with pytest.raises(ValueError, match="ssl_verify"):
            _resolve_advisor_credentials(tool)


def test_resolve_advisor_credentials_allows_real_public_ip_address():
    """End-to-end (no mocked validate_url): a globally-routable literal IP
    api_base is honored when paired with an api_key."""
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _resolve_advisor_credentials,
    )

    tool = {**ADVISOR_TOOL, "api_key": "sk-other", "api_base": "https://8.8.8.8"}
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._allow_client_side_advisor_credentials",
        return_value=True,
    ):
        result = _resolve_advisor_credentials(tool)
    assert result == ("sk-other", "https://8.8.8.8")
