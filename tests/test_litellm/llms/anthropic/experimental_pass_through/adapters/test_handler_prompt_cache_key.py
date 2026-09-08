import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def _prepare(model: str, extra_kwargs: dict[str, object], thinking: dict[str, object] | None = None):
    completion_kwargs, _ = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model=model,
        metadata={"user_id": "session-abc"},
        thinking=thinking,
        extra_kwargs=extra_kwargs,
    )
    return completion_kwargs


def test_prepare_completion_kwargs_derives_prompt_cache_key_for_openai_provider():
    completion_kwargs = _prepare("openai/gpt-5.6-luna", {"custom_llm_provider": "openai"})
    assert completion_kwargs["user"] == "session-abc"
    assert completion_kwargs["prompt_cache_key"] == "session-abc"


def test_prepare_completion_kwargs_prefers_explicit_prompt_cache_key_over_derived():
    completion_kwargs = _prepare(
        "openai/gpt-5.6-luna",
        {"custom_llm_provider": "openai", "prompt_cache_key": "explicit-key"},
    )
    assert completion_kwargs["user"] == "session-abc"
    assert completion_kwargs["prompt_cache_key"] == "explicit-key"


@pytest.mark.parametrize(
    "model, extra_kwargs",
    [
        ("gemini/gemini-2.5-pro", {"custom_llm_provider": "gemini"}),
        ("openai/gpt-5.6-luna", {}),
    ],
)
def test_prepare_completion_kwargs_skips_prompt_cache_key_without_provider_support(
    model: str, extra_kwargs: dict[str, object]
):
    completion_kwargs = _prepare(model, extra_kwargs)
    assert completion_kwargs["user"] == "session-abc"
    assert "prompt_cache_key" not in completion_kwargs


def test_prepare_completion_kwargs_skips_prompt_cache_key_for_chained_litellm_proxy():
    completion_kwargs = _prepare("litellm_proxy/xai", {"custom_llm_provider": "litellm_proxy"})
    assert completion_kwargs["user"] == "session-abc"
    assert "prompt_cache_key" not in completion_kwargs


def test_prepare_completion_kwargs_keeps_prompt_cache_key_through_responses_reroute():
    completion_kwargs = _prepare(
        "openai/gpt-5.6-luna",
        {"custom_llm_provider": "openai"},
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    assert completion_kwargs["model"] == "openai/responses/gpt-5.6-luna"
    assert completion_kwargs["prompt_cache_key"] == "session-abc"
