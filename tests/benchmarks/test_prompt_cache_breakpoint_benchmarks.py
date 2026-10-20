from collections.abc import Sequence
from typing import Final

import pytest

from litellm.llms.anthropic.chat.prompt_cache_breakpoints import ClaudePromptCache, ClaudeRoute, claude_prompt_cache

_EXPLICIT: Final = {"mode": "explicit"}

_MARKED_SYSTEM_TURNS: Final = (
    {"role": "user", "content": "hi"},
    *({"role": "system", "content": f"s{index}", "prompt_cache_breakpoint": _EXPLICIT} for index in range(2000)),
)

_ONE_HOUR_PARTS_BESIDE_A_BREAKPOINT: Final = (
    {
        "role": "user",
        "content": [
            *(
                {"type": "text", "text": f"p{index}", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
                for index in range(5000)
            ),
            {"type": "text", "text": "end", "prompt_cache_breakpoint": _EXPLICIT},
        ],
    },
)

_SYSTEM_TOOLS_BESIDE_A_MARKED_TOOL: Final = (
    *({"systemTool": {"name": f"s{index}"}} for index in range(20000)),
    {"type": "function", "function": {"name": "t"}, "prompt_cache_breakpoint": _EXPLICIT},
)


def _settled_under_drop_params(
    messages: Sequence[object], tools: Sequence[object] | None = None, route: ClaudeRoute = "anthropic"
) -> None:
    outcome: Final = claude_prompt_cache(
        messages,
        tools,
        {},
        drop_params=True,
        tool_history_as_text=False,
        tool_config_cache_control=None,
        route=route,
        adds_a_tool=False,
        pairs_unanswered_tool_calls=False,
    )
    assert isinstance(outcome, ClaudePromptCache)


@pytest.mark.benchmark
def test_settling_thousands_of_marked_system_turns_under_drop_params():
    _settled_under_drop_params(_MARKED_SYSTEM_TURNS)


@pytest.mark.benchmark
def test_settling_a_breakpoint_beside_thousands_of_one_hour_parts_under_drop_params():
    _settled_under_drop_params(_ONE_HOUR_PARTS_BESIDE_A_BREAKPOINT)


@pytest.mark.benchmark
def test_settling_a_tool_breakpoint_beside_thousands_of_bedrock_system_tools_under_drop_params():
    _settled_under_drop_params(({"role": "user", "content": "hi"},), _SYSTEM_TOOLS_BESIDE_A_MARKED_TOOL, "converse")
