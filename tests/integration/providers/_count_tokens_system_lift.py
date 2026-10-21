from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
from integration._support.client import Gateway
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
TOKEN_COUNTING_BETA: Final = "token-counting-2024-11-01"
ANTHROPIC_VERSION: Final = "2023-06-01"
REJECTION: Final = 'messages.0: Unexpected role "system". The Messages API accepts a top-level `system` parameter'

USER_TEXT: Final = "Count this message"
USER: Final[dict[str, JsonValue]] = {"role": "user", "content": USER_TEXT}
ASSISTANT: Final[dict[str, JsonValue]] = {"role": "assistant", "content": "One."}
FOLLOW_UP: Final[dict[str, JsonValue]] = {"role": "user", "content": "Again"}
INSTRUCTION: Final = "You are a terse assistant"
REMINDER: Final = "Answer in one sentence"
LEADING: Final[dict[str, JsonValue]] = {"role": "system", "content": INSTRUCTION}
LIFTED: Final[dict[str, JsonValue]] = {"type": "text", "text": INSTRUCTION}
MID_SYSTEM: Final[dict[str, JsonValue]] = {"role": "system", "content": REMINDER}
EPHEMERAL: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
ONE_HOUR: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
IMAGE_PART: Final[dict[str, JsonValue]] = {
    "type": "image_url",
    "image_url": {
        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    },
}
FIVE_KB: Final = "Answer in one sentence. " * 214
CALLER_SYSTEM: Final = "Prefer metric units"
CALLER_BLOCKS: Final[list[JsonValue]] = [
    {"type": "text", "text": CALLER_SYSTEM},
    {"type": "text", "text": "Never guess", "cache_control": EPHEMERAL},
]
TOOLS: Final[list[JsonValue]] = [
    {
        "name": "get_weather",
        "description": "Look up the current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City to look up"}},
            "required": ["city"],
        },
    }
]


@dataclass(frozen=True, slots=True)
class LiftCase:
    messages: tuple[dict[str, JsonValue], ...]
    system: JsonValue | None
    lifted: list[JsonValue] | None


LIFT_CASES: Final[Mapping[str, LiftCase]] = MappingProxyType(
    {
        "string": LiftCase((LEADING, USER), None, [LIFTED]),
        "run_with_cache_control": LiftCase(
            (
                {"role": "system", "content": INSTRUCTION, "cache_control": ONE_HOUR},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": REMINDER, "cache_control": EPHEMERAL},
                        {"type": "text", "text": ""},
                        IMAGE_PART,
                    ],
                },
                USER,
            ),
            None,
            [
                {"type": "text", "text": INSTRUCTION, "cache_control": ONE_HOUR},
                {"type": "text", "text": REMINDER, "cache_control": EPHEMERAL},
            ],
        ),
        "caller_system_string_first": LiftCase(
            (LEADING, USER), CALLER_SYSTEM, [{"type": "text", "text": CALLER_SYSTEM}, LIFTED]
        ),
        "caller_system_blocks_first": LiftCase((LEADING, USER), CALLER_BLOCKS, [*CALLER_BLOCKS, LIFTED]),
        "caller_empty_system_dropped": LiftCase((LEADING, USER), "", [LIFTED]),
        "empty_content_dropped": LiftCase(({"role": "system", "content": ""}, USER), None, None),
        "image_only_content_dropped": LiftCase(({"role": "system", "content": [IMAGE_PART]}, USER), None, None),
        "integer_content_dropped": LiftCase(({"role": "system", "content": 5}, USER), None, None),
        "integer_text_part_dropped": LiftCase(
            ({"role": "system", "content": [{"type": "text", "text": 7}]}, USER), None, None
        ),
        "five_kb": LiftCase(({"role": "system", "content": FIVE_KB}, USER), None, [{"type": "text", "text": FIVE_KB}]),
    }
)
STRING_CASE: Final = LIFT_CASES["string"]


def count_request(model: str, case: LiftCase) -> dict[str, JsonValue]:
    return {"model": model, "messages": list(case.messages), **({} if case.system is None else {"system": case.system})}


def expected_count_body(model: str, case: LiftCase) -> dict[str, JsonValue]:
    conversation: Final[list[JsonValue]] = [message for message in case.messages if message["role"] != "system"]
    return {"model": model, "messages": conversation, **({} if case.lifted is None else {"system": case.lifted})}


def _opening_message(message: JsonValue) -> bool:
    return isinstance(message, dict) and message.get("role") in ("user", "assistant")


def _conversation_message(message: JsonValue) -> bool:
    return isinstance(message, dict) and message.get("role") in ("user", "assistant", "system")


def _anthropic_tool(tool: JsonValue) -> bool:
    return isinstance(tool, dict) and isinstance(tool.get("name"), str) and isinstance(tool.get("input_schema"), dict)


def accepts_count_body(body: Mapping[str, JsonValue]) -> bool:
    # Anthropic, Azure AI Foundry and Bedrock Mantle count_tokens verdicts observed live on 2026-10-09: an empty
    # messages list and a system role at messages[0] answer 400 invalid_request_error, a later system role counts
    messages: Final = body.get("messages")
    tools: Final = body.get("tools", [])
    return (
        isinstance(messages, list)
        and len(messages) > 0
        and _opening_message(messages[0])
        and all(map(_conversation_message, messages))
        and isinstance(body.get("system", ""), (str, list))
        and isinstance(tools, list)
        and all(map(_anthropic_tool, tools))
    )


def anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def async_anthropic_client(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


def async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False),
    )
