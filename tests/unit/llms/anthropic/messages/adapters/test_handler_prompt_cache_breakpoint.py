import json
from collections.abc import Iterator
from typing import Final, List, cast
from unittest.mock import Mock

import pytest

import litellm
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.llms.anthropic.messages.adapters.handler import LiteLLMMessagesToCompletionTransformationHandler
from litellm.types.llms.openai import AllMessageValues

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
