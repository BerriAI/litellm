"""Anthropic prompt-cache breakpoints on the v1 contract: a `before_send` body patch.

Legacy twin: `AnthropicCacheControlHook.apply_to_anthropic_messages_request` in
`litellm.integrations.anthropic_cache_control_hook`, the Messages-route half of that hook.

Not ported yet: the OpenAI `prompt_cache_breakpoint` dialect, `tool_config` points, the
default-injection seeding, and the chat-completions path (that route is not native).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from litellm.callbacks_v1 import JSONValue, RequestFactsV1, WirePatchV1
from litellm.callbacks_v1.builtin.port import InterceptorPort

MAX_CACHE_CONTROL_BLOCKS: Final = 4

Role: TypeAlias = Literal["user", "system", "assistant"]
Block: TypeAlias = Mapping[str, JSONValue]


@dataclass(frozen=True, slots=True)
class InjectionPoint:
    """Where to put a breakpoint: the system prompt, every message of `role`, or the message at `index`."""

    role: Role | None = None
    index: int | None = None
    ttl: Literal["5m", "1h"] | None = None

    @property
    def control(self) -> Block:
        if self.ttl is None:
            return MappingProxyType({"type": "ephemeral"})
        return MappingProxyType({"type": "ephemeral", "ttl": self.ttl})


@dataclass(frozen=True, slots=True)
class Config:
    points: tuple[InjectionPoint, ...]
    providers: frozenset[str] = frozenset({"anthropic"})


def _marked(block: JSONValue) -> bool:
    return isinstance(block, Mapping) and block.get("cache_control") is not None


def _blocks(content: JSONValue) -> Sequence[JSONValue]:
    return content if isinstance(content, Sequence) and not isinstance(content, str) else ()


def _breakpoints(message: JSONValue) -> int:
    if not isinstance(message, Mapping):
        return 0
    return int(_marked(message)) + sum(1 for block in _blocks(message.get("content")) if _marked(block))


def _with_last_block_marked(blocks: Sequence[JSONValue], control: Block) -> Sequence[JSONValue] | None:
    if not blocks or not isinstance(blocks[-1], Mapping):
        return None
    return (*blocks[:-1], MappingProxyType({**blocks[-1], "cache_control": control}))


def _as_blocks(message: Block) -> Block:
    content: Final = message.get("content")
    if not isinstance(content, str):
        return message
    return MappingProxyType({**message, "content": (MappingProxyType({"type": "text", "text": content}),)})


def _targets(point: InjectionPoint, messages: Sequence[Block]) -> tuple[int, ...]:
    if point.index is not None:
        index: Final = point.index + len(messages) if point.index < 0 else point.index
        return (index,) if 0 <= index < len(messages) else ()
    return tuple(i for i, message in enumerate(messages) if message.get("role") == point.role)


def _mark_system(system: JSONValue, control: Block) -> JSONValue:
    if isinstance(system, str):
        return (MappingProxyType({"type": "text", "text": system, "cache_control": control}),)
    blocks: Final = _blocks(system)
    if any(_marked(block) for block in blocks):
        return system
    return _with_last_block_marked(blocks, control) or system


def _mark_messages(messages: Sequence[Block], targets: Sequence[tuple[int, Block]], budget: int) -> Sequence[Block]:
    if budget <= 0 or not targets:
        return messages
    (index, control), rest = targets[0], targets[1:]
    marked: Final = (
        None
        if _breakpoints(messages[index])
        else _with_last_block_marked(_blocks(messages[index].get("content")), control)
    )
    if marked is None:
        return _mark_messages(messages, rest, budget)
    message: Final = MappingProxyType({**messages[index], "content": marked})
    updated: Final = (*messages[:index], message, *messages[index + 1 :])
    return _mark_messages(updated, rest, budget - 1)


@dataclass(frozen=True, slots=True)
class AnthropicCacheControl(InterceptorPort):
    """A body patch that adds the configured breakpoints to a Messages request."""

    config: Config

    def payload(self, request: RequestFactsV1) -> WirePatchV1 | None:
        """The patch this request needs, or `None` for a provider or body this port leaves alone."""
        if request["custom_llm_provider"] not in self.config.providers:
            return None
        body: Final = self.marked(request["body"])
        if body is None:
            return None
        patch: Final[WirePatchV1] = {"body": body}
        return patch

    def marked(self, body: JSONValue) -> JSONValue | None:
        """The Messages body with the configured breakpoints added, or `None` to leave it alone."""
        if not isinstance(body, Mapping) or not self.config.points:
            return None
        raw_messages: Final = body.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, str):
            return None
        messages: Final = tuple(_as_blocks(message) for message in raw_messages if isinstance(message, Mapping))
        if len(messages) != len(raw_messages):
            return None

        system: Final = body.get("system")
        system_point: Final = next((point for point in self.config.points if point.role == "system"), None)
        used: Final = sum(_breakpoints(message) for message in messages) + sum(1 for b in _blocks(system) if _marked(b))
        new_system: Final = (
            _mark_system(system, system_point.control)
            if system_point is not None and system is not None and used < MAX_CACHE_CONTROL_BLOCKS
            else system
        )
        used_after_system: Final = used + int(new_system != system)

        targets: Final = tuple(
            (index, point.control)
            for point in self.config.points
            if point.role != "system"
            for index in _targets(point, messages)
        )
        new_messages: Final = _mark_messages(messages, targets, MAX_CACHE_CONTROL_BLOCKS - used_after_system)
        replaced: Final = MappingProxyType({"messages": new_messages, "system": new_system})
        return MappingProxyType({key: replaced.get(key, value) for key, value in body.items()})
