"""
Wrapper around router cache. Meant to store model id when prompt caching supported prompt is called.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Final, cast

from pydantic import JsonValue, TypeAdapter
from pydantic_core import to_jsonable_python
from typing_extensions import TypedDict

from litellm.caching.caching import DualCache
from litellm.constants import PROMPT_CACHE_LOOKBACK_POSITIONS
from litellm.litellm_core_utils.logging_utils import truncate_base64_in_messages
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.router import Router

    litellm_router = Router
    Span = _Span
else:
    Span = Any
    litellm_router = Any


class PromptCachingCacheValue(TypedDict):
    model_id: str


PROMPT_CACHE_PIN_TTL_SECONDS: Final = 300
_TOOL_RUN_BLOCK_TYPES: Final = frozenset({"tool_use", "tool_result"})
_PREFIX_ADAPTER: Final = TypeAdapter(tuple[Mapping[str, JsonValue], ...])
_TOOLS_ADAPTER: Final = TypeAdapter(tuple[JsonValue, ...])
_PINS_ADAPTER: Final[TypeAdapter[tuple[JsonValue, ...] | None]] = TypeAdapter(tuple[JsonValue, ...] | None)


@dataclass(frozen=True, slots=True)
class PrefixPosition:
    cache_key: str
    position: int


def _sorted_pairs(pairs: Iterable[tuple[str, JsonValue]]) -> tuple[tuple[str, JsonValue], ...]:
    return tuple(sorted(pairs, key=lambda pair: pair[0]))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _block_unit(
    envelope: tuple[tuple[str, JsonValue], ...], message_run_type: str | None, block: JsonValue
) -> tuple[bytes, str | None]:
    if not isinstance(block, dict):
        return _canonical_bytes((envelope, block)), message_run_type
    block_type: Final = block.get("type")
    block_run_type: Final = block_type if isinstance(block_type, str) and block_type in _TOOL_RUN_BLOCK_TYPES else None
    stripped: Final = _sorted_pairs(item for item in block.items() if item[0] != "cache_control")
    return _canonical_bytes((envelope, stripped)), message_run_type or block_run_type


def _message_units(message: Mapping[str, JsonValue]) -> tuple[tuple[bytes, str | None], ...]:
    envelope: Final = _sorted_pairs(item for item in message.items() if item[0] not in ("content", "cache_control"))
    message_run_type: Final = "tool_result" if message.get("role") == "tool" else None
    content: Final = message.get("content")
    if isinstance(content, list) and content:
        return tuple(_block_unit(envelope, message_run_type, block) for block in content)
    if isinstance(content, str) and content:
        return ((_canonical_bytes((envelope, (("text", content), ("type", "text")))), message_run_type),)
    return ((_canonical_bytes((envelope, None)), message_run_type),)


def _chain_digest(digest: bytes, unit: bytes) -> bytes:
    return hashlib.sha256(digest + unit).digest()


def _seed(tools: Sequence[ChatCompletionToolParam] | None) -> bytes:
    if tools is None:
        return hashlib.sha256(b"").digest()
    return hashlib.sha256(
        _canonical_bytes(
            _TOOLS_ADAPTER.validate_python(to_jsonable_python(tools, serialize_unknown=True, bytes_mode="base64"))
        )
    ).digest()


def _positions_of(
    prefix: tuple[Mapping[str, JsonValue], ...], tools: Sequence[ChatCompletionToolParam] | None
) -> tuple[PrefixPosition, ...]:
    units: Final = tuple(unit for message in prefix for unit in _message_units(message))
    digests: Final = tuple(accumulate((unit_bytes for unit_bytes, _ in units), _chain_digest, initial=_seed(tools)))[1:]
    run_types: Final = tuple(run_type for _, run_type in units)
    positions: Final = accumulate(
        0 if run_type is not None and run_type == previous else 1
        for run_type, previous in zip(run_types, (None, *run_types[:-1]))
    )
    return tuple(
        PrefixPosition(cache_key=f"deployment:{digest.hex()}:prompt_caching", position=position)
        for digest, position in zip(digests, positions)
    )


def _lookback_keys(positions: tuple[PrefixPosition, ...]) -> tuple[str, ...]:
    if not positions:
        return ()
    oldest_probed_position: Final = positions[-1].position - PROMPT_CACHE_LOOKBACK_POSITIONS
    return tuple(entry.cache_key for entry in reversed(positions) if entry.position > oldest_probed_position)


def _pinned_value(value: JsonValue) -> PromptCachingCacheValue | None:
    if not isinstance(value, dict):
        return None
    model_id: Final = value.get("model_id")
    return PromptCachingCacheValue(model_id=model_id) if isinstance(model_id, str) else None


def _first_pin(values: tuple[JsonValue, ...] | None) -> PromptCachingCacheValue | None:
    if values is None:
        return None
    return next((pin for pin in map(_pinned_value, values) if pin is not None), None)


class PromptCachingCache:
    def __init__(self, cache: DualCache):
        self.cache = cache

    @staticmethod
    def extract_cacheable_prefix(
        messages: list[AllMessageValues],
    ) -> list[AllMessageValues]:
        """
        Extract the cacheable prefix from messages.

        The cacheable prefix is everything UP TO AND INCLUDING the LAST content block
        (across all messages) that has cache_control. This includes ALL blocks before
        the last cacheable block (even if they don't have cache_control).

        Args:
            messages: List of messages to extract cacheable prefix from

        Returns:
            List of messages containing only the cacheable prefix
        """
        if not messages:
            return messages

        # Find the last content block (across all messages) that has cache_control
        last_cacheable_message_idx = None
        last_cacheable_content_idx = None

        for msg_idx, message in enumerate(messages):
            content = message.get("content")

            # Check for cache_control at message level (when content is a string)
            # This handles the case where cache_control is a sibling of string content:
            # {"role": "user", "content": "...", "cache_control": {"type": "ephemeral"}}
            message_level_cache_control = message.get("cache_control")
            if (
                message_level_cache_control is not None
                and isinstance(message_level_cache_control, dict)
                and message_level_cache_control.get("type") == "ephemeral"
            ):
                last_cacheable_message_idx = msg_idx
                # Set to None to indicate the entire message content is cacheable
                # (not a specific content block index within a list)
                last_cacheable_content_idx = None

            # Also check for cache_control within content blocks (when content is a list)
            if not isinstance(content, list):
                continue

            for content_idx, content_block in enumerate(content):
                if isinstance(content_block, dict):
                    cache_control = content_block.get("cache_control")
                    if (
                        cache_control is not None
                        and isinstance(cache_control, dict)
                        and cache_control.get("type") == "ephemeral"
                    ):
                        last_cacheable_message_idx = msg_idx
                        last_cacheable_content_idx = content_idx

        # If no cacheable block found, return empty list (no cacheable prefix)
        if last_cacheable_message_idx is None:
            return []

        # Build the cacheable prefix: all messages up to and including the last cacheable message
        cacheable_prefix: Final = []

        for msg_idx, message in enumerate(messages):
            if msg_idx < last_cacheable_message_idx:
                # Include entire message (comes before last cacheable block)
                cacheable_prefix.append(message)
            elif msg_idx == last_cacheable_message_idx:
                # Include message but only up to and including the last cacheable content block
                content = message.get("content")
                if isinstance(content, list) and last_cacheable_content_idx is not None:
                    # Create a copy of the message with only cacheable content blocks
                    message_copy = cast(
                        AllMessageValues,
                        {
                            **message,
                            "content": content[: last_cacheable_content_idx + 1],
                        },
                    )
                    cacheable_prefix.append(message_copy)
                else:
                    # Content is not a list or cacheable content idx is None, include full message
                    cacheable_prefix.append(message)
            else:
                # Message comes after last cacheable block, don't include
                break

        return cacheable_prefix

    @staticmethod
    def prefix_positions(
        messages: list[AllMessageValues] | None,
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> tuple[PrefixPosition, ...]:
        """
        One cache key per content block of the cacheable prefix, oldest block first.

        Each key hashes the prefix content up to and including that block, with cache_control markers
        left out, so the key of a block is the same whichever turn's breakpoint the prefix ends at.
        String content hashes like a single text block, which is how the provider treats it and how
        Claude Code re-sends a previously marked message. `position` counts a run of consecutive
        tool_use (or tool_result) blocks as one, matching the provider's lookback window.

        The prefix is hashed in the shape the success event sees it, with long base64 data URIs
        already replaced by their size placeholder, so a request carrying the raw image bytes
        derives the same keys the write side stored.
        """
        if not messages:
            return ()
        return _positions_of(
            _PREFIX_ADAPTER.validate_python(
                to_jsonable_python(
                    truncate_base64_in_messages(PromptCachingCache.extract_cacheable_prefix(messages)),
                    serialize_unknown=True,
                    bytes_mode="base64",
                )
            ),
            tools,
        )

    @staticmethod
    async def async_prefix_positions(
        messages: list[AllMessageValues] | None,
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> tuple[PrefixPosition, ...]:
        if not messages:
            return ()
        return await offload_token_count(PromptCachingCache.prefix_positions)(messages, tools)

    @staticmethod
    def get_prompt_caching_cache_key(
        messages: list[AllMessageValues] | None,
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> str | None:
        positions: Final = PromptCachingCache.prefix_positions(messages, tools)
        return positions[-1].cache_key if positions else None

    def add_model_id(
        self,
        model_id: str,
        messages: list[AllMessageValues] | None,
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> None:
        cache_key: Final = PromptCachingCache.get_prompt_caching_cache_key(messages, tools)
        if cache_key is None:
            return

        self.cache.set_cache(cache_key, PromptCachingCacheValue(model_id=model_id), ttl=PROMPT_CACHE_PIN_TTL_SECONDS)

    async def async_add_model_id(
        self,
        model_id: str,
        messages: list[AllMessageValues] | None,
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> None:
        positions: Final = await PromptCachingCache.async_prefix_positions(messages, tools)
        if not positions:
            return

        await self.cache.async_set_cache(
            positions[-1].cache_key,
            PromptCachingCacheValue(model_id=model_id),
            ttl=PROMPT_CACHE_PIN_TTL_SECONDS,
        )

    async def async_get_model_id(
        self,
        messages: list[AllMessageValues] | None,
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> PromptCachingCacheValue | None:
        """
        Find the deployment that last served this prefix, walking back from the breakpoint the
        same way the provider cache does, so a breakpoint that moved forward since the last
        turn still lands on the deployment whose cache holds the earlier prefix.
        """
        cache_keys: Final = _lookback_keys(await PromptCachingCache.async_prefix_positions(messages, tools))
        if not cache_keys:
            return None

        return _first_pin(
            _PINS_ADAPTER.validate_python(
                await self.cache.async_batch_get_cache(
                    keys=list(cache_keys),  # mutable-ok: DualCache.async_batch_get_cache only takes a list
                )
            )
        )

    def get_model_id(
        self,
        messages: list[AllMessageValues] | None,
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> PromptCachingCacheValue | None:
        cache_keys: Final = _lookback_keys(PromptCachingCache.prefix_positions(messages, tools))
        if not cache_keys:
            return None

        return _first_pin(
            _PINS_ADAPTER.validate_python(
                self.cache.batch_get_cache(
                    keys=list(cache_keys),  # mutable-ok: DualCache.batch_get_cache only takes a list
                )
            )
        )
