import json
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.anthropic.pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)

_CONVERSE_CLAUDE: Final = "bedrock/converse/us.anthropic.claude-haiku-5-5"
_BREAKPOINT: Final = {"mode": "explicit"}
_EPHEMERAL: Final = {"type": "ephemeral"}


def _system() -> list[dict[str, object]]:
    return [{"type": "text", "text": "system prompt", "prompt_cache_breakpoint": _BREAKPOINT}]


def _messages(*, breakpoint: bool) -> list[dict[str, object]]:
    marked: Final = {"prompt_cache_breakpoint": _BREAKPOINT} if breakpoint else {}
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "cached prefix", **marked},
                {"type": "text", "text": "native mark", "cache_control": _EPHEMERAL},
            ],
        }
    ]


def _tools() -> list[dict[str, object]]:
    return [{"name": "lookup", "input_schema": {"type": "object"}, "prompt_cache_breakpoint": _BREAKPOINT}]


def _completion_messages(
    model: str, extra_kwargs: Mapping[str, object], *, breakpoint: bool = True
) -> list[dict[str, object]]:
    completion_kwargs, _ = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=64,
        messages=_messages(breakpoint=breakpoint),
        model=model,
        system=_system() if breakpoint else None,
        tools=_tools() if breakpoint else None,
        extra_kwargs=extra_kwargs,
    )
    return json.loads(json.dumps(completion_kwargs["messages"]))


def test_claude_target_refuses_prompt_cache_breakpoint_without_drop_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", None)

    with pytest.raises(UnsupportedParamsError) as raised:
        _completion_messages(_CONVERSE_CLAUDE, {"custom_llm_provider": "bedrock"})

    assert raised.value.status_code == 400
    assert "prompt_cache_breakpoint" in raised.value.message and "drop_params" in raised.value.message


@pytest.mark.parametrize(
    ("global_drop_params", "extra_kwargs"),
    [
        (True, {"custom_llm_provider": "bedrock"}),
        (None, {"custom_llm_provider": "bedrock", "drop_params": True}),
    ],
    ids=["litellm.drop_params", "request_or_deployment_drop_params"],
)
def test_claude_target_drops_prompt_cache_breakpoint_under_drop_params(
    monkeypatch: pytest.MonkeyPatch, global_drop_params: bool | None, extra_kwargs: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)

    messages: Final = _completion_messages(_CONVERSE_CLAUDE, extra_kwargs)

    assert "prompt_cache_breakpoint" not in json.dumps(messages), messages
    assert messages[1]["content"] == [
        {"type": "text", "text": "cached prefix"},
        {"type": "text", "text": "native mark", "cache_control": _EPHEMERAL},
    ]


def test_claude_target_without_a_breakpoint_keeps_its_cache_control(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", None)

    messages: Final = _completion_messages(_CONVERSE_CLAUDE, {"custom_llm_provider": "bedrock"}, breakpoint=False)

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "cached prefix"},
                {"type": "text", "text": "native mark", "cache_control": _EPHEMERAL},
            ],
        }
    ]


def test_non_claude_target_still_carries_prompt_cache_breakpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", None)

    messages: Final = _completion_messages("openai/gpt-5.6", {"custom_llm_provider": "openai"})

    assert messages[1]["content"][0] == {
        "type": "text",
        "text": "cached prefix",
        "prompt_cache_breakpoint": _BREAKPOINT,
    }
