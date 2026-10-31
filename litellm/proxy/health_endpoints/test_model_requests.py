"""The three-turn conversations ``POST /health/test_model`` sends, one per endpoint, in that endpoint's native shape.

Every conversation asks the same weather question in front of the same long system prompt (``test_model_requests/
system_prompt.txt``), sends the same tool result back, then asks the same follow-up. Chat Completions and Responses
carry no prompt-cache markers, so cached tokens on turns 2 and 3 can only come from the provider caching the repeated
prefix on its own. Messages mirrors Claude Code: ``CACHE_CONTROL`` on the latest user message and a mid-conversation
``SYSTEM_REMINDER``, so a proxy that hoists the reminder into the system prompt changes the cached prefix and shows up
as a cache miss. The ``TURN_*`` placeholders are filled from the model's earlier replies.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

SYSTEM_PROMPT: Final = (Path(__file__).resolve().parent / "test_model_requests" / "system_prompt.txt").read_text(
    encoding="utf-8"
)

TURN_ONE_ASSISTANT_OUTPUT: Final = "<turn 1 response: assistant output>"
TURN_ONE_TOOL_CALL_ID: Final = "<turn 1 response: tool call id>"
TURN_TWO_ASSISTANT_OUTPUT: Final = "<turn 2 response: assistant output>"

WEATHER_QUESTION: Final = (
    "What is the weather in Boston right now? First tell me in one sentence what you are about to do, then use the "
    "get_weather tool to find out."
)
WEATHER_TOOL_RESULT: Final = "72F and sunny"
FOLLOW_UP_QUESTION: Final = (
    "Thanks. Is that warm enough for a short walk? Repeat the temperature you found in your answer."
)
SYSTEM_REMINDER: Final = (
    "<system-reminder>The user prefers short answers. Keep the temperature in the reply.</system-reminder>"
)
CACHE_CONTROL: Final = MappingProxyType({"type": "ephemeral"})

WEATHER_TOOL_PARAMETERS: Final = MappingProxyType(
    {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, for example Boston"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    }
)
WEATHER_TOOL_DESCRIPTION: Final = "Get the current weather for a city"


@dataclass(frozen=True, slots=True)
class RequestFile:
    """One three-turn conversation: ``turn_1`` is the full first request body, ``turn_2_appended_messages`` the
    messages the second request adds after it, ``turn_3_appended_messages`` the ones the third adds after those."""

    turn_1: Mapping[str, object]
    turn_2_appended_messages: Sequence[object]
    turn_3_appended_messages: Sequence[object]


CHAT_COMPLETIONS: Final = RequestFile(
    turn_1={
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": WEATHER_QUESTION},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": WEATHER_TOOL_DESCRIPTION,
                    "parameters": dict(WEATHER_TOOL_PARAMETERS),
                },
            }
        ],
        "reasoning_effort": "low",
        "max_tokens": 2048,
    },
    turn_2_appended_messages=(
        TURN_ONE_ASSISTANT_OUTPUT,
        {"role": "tool", "tool_call_id": TURN_ONE_TOOL_CALL_ID, "content": WEATHER_TOOL_RESULT},
    ),
    turn_3_appended_messages=(
        TURN_TWO_ASSISTANT_OUTPUT,
        {"role": "system", "content": SYSTEM_REMINDER},
        {"role": "user", "content": FOLLOW_UP_QUESTION},
    ),
)

RESPONSES: Final = RequestFile(
    turn_1={
        "instructions": SYSTEM_PROMPT,
        "input": [{"role": "user", "content": WEATHER_QUESTION}],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": WEATHER_TOOL_DESCRIPTION,
                "parameters": dict(WEATHER_TOOL_PARAMETERS),
            }
        ],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 2048,
    },
    turn_2_appended_messages=(
        TURN_ONE_ASSISTANT_OUTPUT,
        {"type": "function_call_output", "call_id": TURN_ONE_TOOL_CALL_ID, "output": WEATHER_TOOL_RESULT},
    ),
    turn_3_appended_messages=(
        TURN_TWO_ASSISTANT_OUTPUT,
        {"role": "system", "content": SYSTEM_REMINDER},
        {"role": "user", "content": FOLLOW_UP_QUESTION},
    ),
)

MESSAGES: Final = RequestFile(
    turn_1={
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": WEATHER_QUESTION, "cache_control": dict(CACHE_CONTROL)}],
            }
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": WEATHER_TOOL_DESCRIPTION,
                "input_schema": dict(WEATHER_TOOL_PARAMETERS),
            }
        ],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "max_tokens": 2048,
    },
    turn_2_appended_messages=(
        {"role": "assistant", "content": TURN_ONE_ASSISTANT_OUTPUT},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": TURN_ONE_TOOL_CALL_ID,
                    "content": WEATHER_TOOL_RESULT,
                    "cache_control": dict(CACHE_CONTROL),
                }
            ],
        },
    ),
    turn_3_appended_messages=(
        {"role": "assistant", "content": TURN_TWO_ASSISTANT_OUTPUT},
        {
            "role": "user",
            "content": [{"type": "text", "text": FOLLOW_UP_QUESTION, "cache_control": dict(CACHE_CONTROL)}],
        },
        {"role": "system", "content": SYSTEM_REMINDER},
    ),
)

REQUEST_FILES: Final = MappingProxyType(
    {"chat_completions": CHAT_COMPLETIONS, "responses": RESPONSES, "messages": MESSAGES}
)
