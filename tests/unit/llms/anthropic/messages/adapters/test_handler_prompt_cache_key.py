import json
import os
import sys
from collections.abc import Iterator
from typing import Final, List, cast
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

import litellm
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.llms.anthropic.messages.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)
from litellm.types.llms.openai import AllMessageValues

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
) -> None:
    if azure_api_base_env is None:
        monkeypatch.delenv("MISTRAL_AZURE_API_BASE", raising=False)
    else:
        monkeypatch.setenv("MISTRAL_AZURE_API_BASE", azure_api_base_env)

    completion_kwargs: Final = _prepare(
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
) -> None:
    monkeypatch.delenv("MISTRAL_AZURE_API_BASE", raising=False)
    if codestral_api_base_env is None:
        monkeypatch.delenv("CODESTRAL_API_BASE", raising=False)
    else:
        monkeypatch.setenv("CODESTRAL_API_BASE", codestral_api_base_env)

    completion_kwargs: Final = _prepare("codestral/codestral-latest", {"custom_llm_provider": "codestral"})

    assert ("prompt_cache_key" in completion_kwargs) is derives_key


def test_prepare_completion_kwargs_derives_no_prompt_cache_key_mistral_cannot_take_from_a_non_ascii_session() -> None:
    non_ascii_user_id: Final = json.dumps({"device_id": "d" * 64, "account_uuid": "", "session_id": "sitzung-ü"})

    mistral_kwargs: Final = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="mistral/mistral-small-latest",
        metadata={"user_id": non_ascii_user_id},
        thinking=None,
        extra_kwargs={"custom_llm_provider": "mistral"},
    )[0]
    openai_kwargs: Final = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        metadata={"user_id": non_ascii_user_id},
        thinking=None,
        extra_kwargs={"custom_llm_provider": "openai"},
    )[0]

    assert "prompt_cache_key" not in mistral_kwargs
    assert openai_kwargs["prompt_cache_key"] == "sitzung-ü"


_EPHEMERAL: Final = {"type": "ephemeral"}
_MANTLE_GPT_ROW: Final = {
    "litellm_provider": "bedrock_mantle",
    "mode": "responses",
    "supports_prompt_cache_breakpoint": True,
}
_CLAUDE_CODE_MESSAGES: Final = (
    {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": _EPHEMERAL}]},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "checking", "cache_control": _EPHEMERAL},
            {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {}, "cache_control": _EPHEMERAL},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file body", "cache_control": _EPHEMERAL}],
    },
)
_SYSTEM: Final = [{"type": "text", "text": "Stable prefix", "cache_control": _EPHEMERAL}]
_TOOLS: Final = [{"name": "read", "description": "Read a file", "input_schema": {"type": "object"}, "cache_control": _EPHEMERAL}]


def _texts_with(key: str, value: object) -> Iterator[object]:
    if isinstance(value, dict):
        if key in value:
            yield value.get("text", value.get("name"))
        for nested in value.values():
            yield from _texts_with(key, nested)
    if isinstance(value, list):
        for item in value:
            yield from _texts_with(key, item)


def _mantle_wire(model_info: dict[str, object] | None) -> tuple[dict[str, object], dict[str, object]]:
    completion_kwargs, _ = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=64,
        messages=json.loads(json.dumps(_CLAUDE_CODE_MESSAGES)),
        model="bedrock_mantle/openai.gpt-5.6-sol",
        system=json.loads(json.dumps(_SYSTEM)),
        tools=json.loads(json.dumps(_TOOLS)),
        extra_kwargs={"custom_llm_provider": "bedrock_mantle", "model_info": model_info},
    )
    wire: Final = LiteLLMResponsesTransformationHandler().transform_request(
        model="openai.gpt-5.6-sol",
        messages=cast(List[AllMessageValues], completion_kwargs["messages"]),  # cast-ok: chat messages from the adapter
        optional_params={"tools": completion_kwargs["tools"]},
        litellm_params={"custom_llm_provider": "bedrock_mantle", "model_info": model_info},
        headers={},
        litellm_logging_obj=Mock(),
    )
    return dict(completion_kwargs), wire


def test_anthropic_messages_to_a_breakpoint_reading_target_carry_marks_where_the_openai_dialect_reads_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(litellm.model_cost, "bedrock_mantle/openai.gpt-5.6-sol", _MANTLE_GPT_ROW)

    completion_kwargs, wire = _mantle_wire(None)

    assert list(_texts_with("cache_control", completion_kwargs["tools"])) == []
    assert list(_texts_with("cache_control", cast(list[object], completion_kwargs["messages"])[2])) == []
    assert sorted(map(str, _texts_with("prompt_cache_breakpoint", wire["input"]))) == ["Stable prefix", "file body", "hi"]
    assert list(_texts_with("cache_control", wire)) == []


def test_anthropic_messages_to_a_deployment_that_opts_out_of_breakpoints_carry_no_marks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(litellm.model_cost, "bedrock_mantle/openai.gpt-5.6-sol", _MANTLE_GPT_ROW)

    completion_kwargs, wire = _mantle_wire({"supports_prompt_cache_breakpoint": False})

    assert list(_texts_with("cache_control", completion_kwargs)) == []
    assert list(_texts_with("prompt_cache_breakpoint", wire)) == []
