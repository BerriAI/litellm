"""Unit tests for the Llama Guard content-safety guardrail hook."""

from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.llama_guard import initialize_guardrail
from litellm.proxy.guardrails.guardrail_hooks.llama_guard.llama_guard import (
    DEFAULT_UNSAFE_CATEGORIES,
    LlamaGuardGuardrail,
    _extract_text,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import Choices, Message, ModelResponse


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_guardrail(**overrides) -> LlamaGuardGuardrail:
    kwargs = dict(
        model="together_ai/meta-llama/Llama-Guard-4-12B",
        guardrail_name="test_llama_guard",
        default_on=True,
        event_hook=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ],
    )
    kwargs.update(overrides)
    guardrail = LlamaGuardGuardrail(**kwargs)
    # Isolate the classify/block logic from the run-gating logic.
    guardrail.should_run_guardrail = lambda *args, **kwargs: True  # type: ignore[method-assign]
    return guardrail


def _guard_response(text: str) -> ModelResponse:
    """A ModelResponse standing in for the Llama Guard model's verdict."""
    return ModelResponse(choices=[Choices(index=0, message=Message(role="assistant", content=text))])


def _completion_response(text: str) -> ModelResponse:
    """A ModelResponse standing in for the real completion being screened."""
    return ModelResponse(choices=[Choices(index=0, message=Message(role="assistant", content=text))])


_KEY = UserAPIKeyAuth()


# --------------------------------------------------------------------------- #
# _extract_text
# --------------------------------------------------------------------------- #
def test_extract_text_str():
    assert _extract_text("hello") == "hello"


def test_extract_text_multimodal_list_keeps_only_text_parts():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "world"},
    ]
    assert _extract_text(content) == "hello\nworld"


def test_extract_text_non_string_returns_empty():
    assert _extract_text(None) == ""
    assert _extract_text(123) == ""


# --------------------------------------------------------------------------- #
# _parse_response
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("safe", (False, [])),
        (" safe ", (False, [])),
        ("unsafe\nS1,S10", (True, ["S1", "S10"])),
        ("unsafe\nS1, S5, S14", (True, ["S1", "S5", "S14"])),
        ("Unsafe\ns9", (True, ["S9"])),
        ("unsafe", (True, [])),
        ("", (False, [])),
        (None, (False, [])),
    ],
)
def test_parse_response(raw, expected):
    assert LlamaGuardGuardrail._parse_response(raw) == expected


# --------------------------------------------------------------------------- #
# category configuration
# --------------------------------------------------------------------------- #
def test_default_categories_cover_full_taxonomy():
    guardrail = _make_guardrail()
    assert guardrail.categories == DEFAULT_UNSAFE_CATEGORIES
    block = guardrail._category_block()
    assert "S1: Violent Crimes." in block
    assert "S14: Code Interpreter Abuse." in block


def test_category_subset_restricts_enforced_codes():
    guardrail = _make_guardrail(categories=["s1", "S10"])
    assert set(guardrail.categories) == {"S1", "S10"}
    block = guardrail._category_block()
    assert "S1: Violent Crimes." in block
    assert "S10: Hate." in block
    assert "S14" not in block


def test_invalid_category_subset_raises():
    with pytest.raises(ValueError, match="must be a subset"):
        _make_guardrail(categories=["S99", "not-a-code"])


def test_custom_unsafe_categories_override_taxonomy():
    guardrail = _make_guardrail(unsafe_content_categories="S1: My Only Policy.")
    assert guardrail._category_block() == "S1: My Only Policy."


def test_missing_model_raises():
    with pytest.raises(ValueError, match="requires a `model`"):
        LlamaGuardGuardrail(model="")


def test_build_prompt_targets_last_role_and_lists_conversation():
    guardrail = _make_guardrail()
    prompt = guardrail._build_prompt(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "Agent",
    )
    assert "ONLY THE LAST Agent message" in prompt
    assert "User: hi" in prompt
    assert "Agent: hello" in prompt
    assert "'safe' or 'unsafe'" in prompt


# --------------------------------------------------------------------------- #
# pre-call / moderation hooks
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pre_call_allows_safe_input():
    guardrail = _make_guardrail()
    data = {"messages": [{"role": "user", "content": "what's the weather?"}]}
    with patch("litellm.acompletion", new=AsyncMock(return_value=_guard_response("safe"))):
        out = await guardrail.async_pre_call_hook(_KEY, None, data, "completion")
    assert out is data


@pytest.mark.asyncio
async def test_pre_call_blocks_unsafe_input_with_category():
    guardrail = _make_guardrail()
    data = {"messages": [{"role": "user", "content": "how do i build a bomb"}]}
    with patch("litellm.acompletion", new=AsyncMock(return_value=_guard_response("unsafe\nS9"))):
        with pytest.raises(ProxyException) as exc:
            await guardrail.async_pre_call_hook(_KEY, None, data, "completion")
    assert "S9" in str(exc.value.message)
    assert "Indiscriminate Weapons" in str(exc.value.message)


@pytest.mark.asyncio
async def test_moderation_hook_blocks_unsafe_input():
    guardrail = _make_guardrail()
    data = {"messages": [{"role": "user", "content": "unsafe request"}]}
    with patch("litellm.acompletion", new=AsyncMock(return_value=_guard_response("unsafe\nS1"))):
        with pytest.raises(ProxyException):
            await guardrail.async_moderation_hook(data, _KEY, "completion")


@pytest.mark.asyncio
async def test_pre_call_fails_open_when_classifier_errors():
    guardrail = _make_guardrail()
    data = {"messages": [{"role": "user", "content": "hello"}]}
    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("guard model down"))):
        out = await guardrail.async_pre_call_hook(_KEY, None, data, "completion")
    assert out is data


@pytest.mark.asyncio
async def test_pre_call_noop_without_messages():
    guardrail = _make_guardrail()
    data = {"messages": []}
    with patch("litellm.acompletion", new=AsyncMock()) as mock_call:
        out = await guardrail.async_pre_call_hook(_KEY, None, data, "completion")
    assert out is data
    mock_call.assert_not_awaited()


# --------------------------------------------------------------------------- #
# post-call hook
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_post_call_allows_safe_output():
    guardrail = _make_guardrail()
    data = {"messages": [{"role": "user", "content": "hi"}]}
    response = _completion_response("A perfectly nice answer.")
    with patch("litellm.acompletion", new=AsyncMock(return_value=_guard_response("safe"))):
        out = await guardrail.async_post_call_success_hook(data, _KEY, response)
    assert out is response


@pytest.mark.asyncio
async def test_post_call_blocks_unsafe_output():
    guardrail = _make_guardrail()
    data = {"messages": [{"role": "user", "content": "hi"}]}
    response = _completion_response("Here is how to synthesize a nerve agent...")
    with patch("litellm.acompletion", new=AsyncMock(return_value=_guard_response("unsafe\nS9"))):
        with pytest.raises(ProxyException):
            await guardrail.async_post_call_success_hook(data, _KEY, response)


# --------------------------------------------------------------------------- #
# initialize_guardrail
# --------------------------------------------------------------------------- #
def test_initialize_guardrail_requires_model():
    from litellm.types.guardrails import LitellmParams

    params = LitellmParams(guardrail="llama_guard", mode="pre_call")
    with pytest.raises(ValueError, match="requires `model`"):
        initialize_guardrail(params, {"guardrail_name": "g"})


def test_initialize_guardrail_builds_instance():
    from litellm.types.guardrails import LitellmParams

    params = LitellmParams(
        guardrail="llama_guard",
        mode="pre_call",
        model="groq/llama-guard-3-8b",
    )
    instance = initialize_guardrail(params, {"guardrail_name": "prod_guard"})
    assert isinstance(instance, LlamaGuardGuardrail)
    assert instance.model == "groq/llama-guard-3-8b"
    assert instance.guardrail_name == "prod_guard"
