"""A generated provider reply for one request: `plan_reply` reads the request (model, prompt size, output
cap, tools offered) and `render_json` / `render_chunks` write it in the wire format of the provider and
route. The words are synthetic; the schema, ids, event order, usage fields and delta and chunk sizes follow
what each provider sends, so the gateway's parsers, cost tracking and stream handling do the same work."""

from __future__ import annotations

from typing import Final, Literal, cast

import reply_anthropic
import reply_bedrock
import reply_openai
from reply_common import Provider, Reply, plan_reply

Route = Literal["chat", "responses", "messages", "converse", "invoke"]
_MAX_TOKEN_KEYS: Final = ("max_output_tokens", "max_completion_tokens", "max_tokens")
DEFAULT_MAX_TOKENS: Final = 1024


def _mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _items(value: object) -> tuple[object, ...]:
    return tuple(cast(list[object], value)) if isinstance(value, list) else ()


def max_tokens_of(body: object) -> int:
    mapping: Final = _mapping(body)
    found: Final = tuple(mapping[key] for key in _MAX_TOKEN_KEYS if isinstance(mapping.get(key), int))
    inference: Final = _mapping(mapping.get("inferenceConfig")).get("maxTokens")
    if found:
        return cast(int, found[0])
    return inference if isinstance(inference, int) else DEFAULT_MAX_TOKENS


def _tool_name(tool: object) -> str | None:
    mapping: Final = _mapping(tool)
    nested: Final = _mapping(mapping.get("function")) or _mapping(mapping.get("toolSpec"))
    name: Final = nested.get("name", mapping.get("name"))
    return name if isinstance(name, str) else None


def tool_names_of(body: object) -> tuple[str, ...]:
    mapping: Final = _mapping(body)
    tools: Final = _items(mapping.get("tools")) or _items(_mapping(mapping.get("toolConfig")).get("tools"))
    return tuple(name for name in map(_tool_name, tools) if name is not None)


def reply_for(*, provider: Provider, body: object, seed: str, model: str, prompt_tokens: int) -> Reply:
    return plan_reply(
        provider=provider,
        seed=seed,
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens_of(body),
        tool_names=tool_names_of(body),
    )


def render_json(reply: Reply, route: Route) -> bytes:
    if route == "chat":
        return reply_openai.chat_json(reply)
    if route == "responses":
        return reply_openai.responses_json(reply)
    if route in ("messages", "invoke"):
        return reply_anthropic.messages_json(reply)
    return reply_bedrock.converse_json(reply)


def render_chunks(reply: Reply, route: Route) -> tuple[bytes, ...]:
    if route == "chat":
        return reply_openai.chat_chunks(reply)
    if route == "responses":
        return reply_openai.responses_chunks(reply)
    if route == "messages":
        return reply_anthropic.messages_chunks(reply)
    if route == "invoke":
        return reply_bedrock.invoke_chunks(reply)
    return reply_bedrock.converse_chunks(reply)
