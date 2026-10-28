from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    removed_tool_names,
)
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail


@pytest.mark.parametrize(
    ("per_guardrail", "global_default", "expected"),
    [
        (True, False, True),
        (False, True, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_a_per_guardrail_skip_flag_wins_over_the_global_setting(
    monkeypatch: pytest.MonkeyPatch, per_guardrail: bool | None, global_default: bool, expected: bool
):
    monkeypatch.setattr(litellm, "skip_system_message_in_guardrail", global_default)
    monkeypatch.setattr(litellm, "skip_tool_message_in_guardrail", global_default)
    guardrail: Final = LakeraAIGuardrail(
        api_key="lakera-test-key",
        skip_system_message_in_guardrail=per_guardrail,
        skip_tool_message_in_guardrail=per_guardrail,
    )

    assert effective_skip_system_message_for_guardrail(guardrail) is expected
    assert effective_skip_tool_message_for_guardrail(guardrail) is expected


@pytest.mark.parametrize(
    "dropped_tool",
    [
        {"type": "function", "function": {"name": "get_time"}},
        {"type": "custom", "custom": {"name": "get_time"}},
        {"type": "function", "name": "get_time"},
        {"type": "custom", "name": "get_time"},
    ],
    ids=["chat_function", "chat_custom", "responses_function", "responses_custom"],
)
def test_a_dropped_function_or_custom_tool_is_named_as_removed(dropped_tool):
    kept_tool: Final = {"type": "function", "function": {"name": "get_weather"}}
    request_tools: Final = (kept_tool, dropped_tool, {"type": "mcp", "server_label": "deepwiki"})

    assert removed_tool_names(request_tools, (kept_tool,)) == {"get_time"}
