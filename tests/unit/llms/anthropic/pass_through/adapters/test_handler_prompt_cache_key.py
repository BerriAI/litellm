import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.llms.anthropic.pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)

MESSAGES = [{"role": "user", "content": "hello"}]
CLAUDE_CODE_USER_ID = json.dumps({"device_id": "d" * 64, "account_uuid": "", "session_id": "session-abc"})


def _prepare(model: str, extra_kwargs: dict[str, object], thinking: dict[str, object] | None = None):
    completion_kwargs, _ = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model=model,
        metadata={"user_id": CLAUDE_CODE_USER_ID},
        thinking=thinking,
        extra_kwargs=extra_kwargs,
    )
    return completion_kwargs


def test_prepare_completion_kwargs_derives_prompt_cache_key_for_openai_provider():
    completion_kwargs = _prepare("openai/gpt-5.6-luna", {"custom_llm_provider": "openai"})
    assert completion_kwargs["user"] == CLAUDE_CODE_USER_ID
    assert completion_kwargs["prompt_cache_key"] == "session-abc"


def test_prepare_completion_kwargs_prefers_explicit_prompt_cache_key_over_derived():
    completion_kwargs = _prepare(
        "openai/gpt-5.6-luna",
        {"custom_llm_provider": "openai", "prompt_cache_key": "explicit-key"},
    )
    assert completion_kwargs["user"] == CLAUDE_CODE_USER_ID
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
    assert completion_kwargs["user"] == CLAUDE_CODE_USER_ID
    assert "prompt_cache_key" not in completion_kwargs


def test_prepare_completion_kwargs_skips_prompt_cache_key_for_chained_litellm_proxy():
    completion_kwargs = _prepare("litellm_proxy/xai", {"custom_llm_provider": "litellm_proxy"})
    assert completion_kwargs["user"] == CLAUDE_CODE_USER_ID
    assert "prompt_cache_key" not in completion_kwargs


def test_prepare_completion_kwargs_keeps_prompt_cache_key_through_responses_reroute():
    completion_kwargs = _prepare(
        "openai/gpt-5.6-luna",
        {"custom_llm_provider": "openai"},
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    assert completion_kwargs["model"] == "openai/responses/gpt-5.6-luna"
    assert completion_kwargs["prompt_cache_key"] == "session-abc"


@pytest.mark.parametrize(
    ("api_base", "azure_api_base_env", "derives_key"),
    [
        (None, None, True),
        ("https://api.mistral.ai/v1", None, True),
        ("https://my-resource.openai.azure.com/openai/v1", None, False),
        (None, "https://my-resource.services.ai.azure.com/openai", False),
    ],
)
def test_prepare_completion_kwargs_derives_prompt_cache_key_only_for_a_mistral_hosted_endpoint(
    monkeypatch: pytest.MonkeyPatch, api_base: str | None, azure_api_base_env: str | None, derives_key: bool
):
    if azure_api_base_env is None:
        monkeypatch.delenv("MISTRAL_AZURE_API_BASE", raising=False)
    else:
        monkeypatch.setenv("MISTRAL_AZURE_API_BASE", azure_api_base_env)

    completion_kwargs = _prepare(
        "mistral/mistral-small-latest", {"custom_llm_provider": "mistral", "api_base": api_base}
    )

    assert completion_kwargs["user"] == CLAUDE_CODE_USER_ID
    assert ("prompt_cache_key" in completion_kwargs) is derives_key


@pytest.mark.parametrize(
    ("codestral_api_base_env", "derives_key"),
    [(None, True), ("https://my-resource.services.ai.azure.com/openai/v1", False)],
)
def test_prepare_completion_kwargs_derives_prompt_cache_key_for_codestral_only_on_its_mistral_host(
    monkeypatch: pytest.MonkeyPatch, codestral_api_base_env: str | None, derives_key: bool
):
    monkeypatch.delenv("MISTRAL_AZURE_API_BASE", raising=False)
    if codestral_api_base_env is None:
        monkeypatch.delenv("CODESTRAL_API_BASE", raising=False)
    else:
        monkeypatch.setenv("CODESTRAL_API_BASE", codestral_api_base_env)

    completion_kwargs = _prepare("codestral/codestral-latest", {"custom_llm_provider": "codestral"})

    assert ("prompt_cache_key" in completion_kwargs) is derives_key
