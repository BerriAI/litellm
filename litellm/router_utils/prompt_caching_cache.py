"""
Wrapper around router cache. Meant to store model id when prompt caching supported prompt is called.
"""

import hashlib
import json
from typing import TYPE_CHECKING, Any, Final, cast

from typing_extensions import TypedDict

from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.types.llms.openai import AllMessageValues, ChatCompletionCachedContent, ChatCompletionToolParam

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


class PromptCachingCache:
    def __init__(self, cache: DualCache):
        self.cache = cache
        self.in_memory_cache = InMemoryCache()

    @staticmethod
    def serialize_object(obj: Any) -> object:
        """Helper function to serialize Pydantic objects, dictionaries, or fallback to string."""
        if hasattr(obj, "dict"):
            # If the object is a Pydantic model, use its `dict()` method
            return obj.dict()
        elif isinstance(obj, dict):
            # If the object is a dictionary, serialize it with sorted keys
            return json.dumps(obj, sort_keys=True, separators=(",", ":"))  # Standardize serialization

        elif isinstance(obj, list):
            # Serialize lists by ensuring each element is handled properly
            return [PromptCachingCache.serialize_object(item) for item in obj]
        elif isinstance(obj, (int, float, bool)):
            return obj  # Keep primitive types as-is
        return str(obj)

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
    def extract_cacheable_tools(
        tools: list[ChatCompletionToolParam],
    ) -> list[ChatCompletionToolParam]:
        cacheable_tool_index: Final = next(
            (
                index
                for index in range(len(tools) - 1, -1, -1)
                if isinstance(cache_control := tools[index].get("cache_control"), dict)
                and cache_control.get("type") == "ephemeral"
            ),
            None,
        )
        # A breakpoint truncates trailing, uncached tools out of the key. With no breakpoint,
        # every tool still precedes whatever the real provider prefix caches on, so the full
        # list has to stay in the key or distinct tool sets would collide on the same key.
        return tools[: cacheable_tool_index + 1] if cacheable_tool_index is not None else tools[:]

    @staticmethod
    def prepend_system_prompt(
        messages: list[AllMessageValues],
        system: object | None,
    ) -> list[AllMessageValues]:
        if system is None:
            return messages
        # Standard logging already prepends a string system prompt onto `messages` before this
        # runs, so re-prepending here would double it up and produce a different affinity key
        # than the one computed from the raw request messages.
        if messages and messages[0].get("role") == "system" and messages[0].get("content") == system:
            return messages
        return cast(  # cast-ok: system content is validated by the provider payload
            list[AllMessageValues],
            [{"role": "system", "content": system}, *messages],  # mutable-ok: cast target requires a concrete list
        )

    @staticmethod
    def get_prompt_caching_ttl(
        messages: list[AllMessageValues] | None,
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> int:
        cacheable_prefix: Final = (
            PromptCachingCache.extract_cacheable_prefix(messages)
            if messages is not None
            else []  # mutable-ok: TTL helper requires a concrete list
        )
        return PromptCachingCache.get_prompt_caching_ttl_from_prefix(cacheable_prefix, tools)

    @staticmethod
    def _ephemeral_cache_controls(
        message: AllMessageValues,
    ) -> tuple[ChatCompletionCachedContent | dict[str, object], ...]:
        content: Final = message.get("content")
        content_cache_controls: Final = (
            tuple(content_block.get("cache_control") for content_block in content if isinstance(content_block, dict))
            if isinstance(content, list)
            else ()
        )
        return tuple(
            cache_control
            for cache_control in (message.get("cache_control"), *content_cache_controls)
            if isinstance(cache_control, dict) and cache_control.get("type") == "ephemeral"
        )

    @staticmethod
    def get_prompt_caching_ttl_from_prefix(
        cacheable_prefix: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None,
    ) -> int:
        cacheable_tools: Final = PromptCachingCache.extract_cacheable_tools(
            tools or []  # mutable-ok: tool API requires a concrete list
        )
        cache_control_values: Final = tuple(
            cache_control
            for message in cacheable_prefix
            for cache_control in PromptCachingCache._ephemeral_cache_controls(message)
        ) + tuple(
            cache_control
            for tool in cacheable_tools
            if isinstance(cache_control := tool.get("cache_control"), dict) and cache_control.get("type") == "ephemeral"
        )
        # Prefer the shortest provider lifetime so affinity never outlives a cached segment
        return 3600 if cache_control_values and all(value.get("ttl") == "1h" for value in cache_control_values) else 300

    @staticmethod
    def get_prompt_caching_cache_key(
        messages: list[AllMessageValues] | None,
        tools: list[ChatCompletionToolParam] | None,
    ) -> str | None:
        cacheable_messages: Final = (
            PromptCachingCache.extract_cacheable_prefix(messages) if messages is not None else None
        )
        return PromptCachingCache.get_prompt_caching_cache_key_from_prefix(cacheable_messages, tools)

    @staticmethod
    def get_prompt_caching_cache_key_from_prefix(
        cacheable_messages: list[AllMessageValues] | None,
        tools: list[ChatCompletionToolParam] | None,
    ) -> str | None:
        if cacheable_messages is None and tools is None:
            return None
        if cacheable_messages is not None and not cacheable_messages:
            return None

        # Only exact cached-prefix matches can establish deployment affinity.
        # Partial matches must fall back to normal routing.
        # Use serialize_object for consistent and stable serialization
        data_to_hash: Final = {}
        if cacheable_messages is not None:
            serialized_messages: Final = PromptCachingCache.serialize_object(cacheable_messages)
            data_to_hash["messages"] = serialized_messages
        if tools is not None:
            cacheable_tools: Final = PromptCachingCache.extract_cacheable_tools(tools)
            serialized_tools: Final = PromptCachingCache.serialize_object(cacheable_tools)
            data_to_hash["tools"] = serialized_tools

        # Combine serialized data into a single string
        data_to_hash_str: Final = json.dumps(
            data_to_hash,
            sort_keys=True,
            separators=(",", ":"),
        )

        # Create a hash of the serialized data for a stable cache key
        hashed_data: Final = hashlib.sha256(data_to_hash_str.encode()).hexdigest()
        return f"deployment:{hashed_data}:prompt_caching"

    def add_model_id(
        self,
        model_id: str,
        messages: list[AllMessageValues] | None,
        tools: list[ChatCompletionToolParam] | None,
    ) -> None:
        if messages is None and tools is None:
            return

        cacheable_prefix: Final = (
            PromptCachingCache.extract_cacheable_prefix(messages) if messages is not None else None
        )
        cache_key: Final = PromptCachingCache.get_prompt_caching_cache_key_from_prefix(cacheable_prefix, tools)
        if cache_key is None:
            return

        self.cache.set_cache(
            cache_key,
            PromptCachingCacheValue(model_id=model_id),
            ttl=PromptCachingCache.get_prompt_caching_ttl_from_prefix(
                cacheable_prefix or [],  # mutable-ok: TTL helper requires a concrete list
                tools,
            ),
        )
        return

    async def async_add_model_id(
        self,
        model_id: str,
        messages: list[AllMessageValues] | None,
        tools: list[ChatCompletionToolParam] | None,
    ) -> None:
        if messages is None and tools is None:
            return

        cacheable_prefix: Final = (
            PromptCachingCache.extract_cacheable_prefix(messages) if messages is not None else None
        )
        cache_key: Final = PromptCachingCache.get_prompt_caching_cache_key_from_prefix(cacheable_prefix, tools)
        if cache_key is None:
            return

        await self.cache.async_set_cache(
            cache_key,
            PromptCachingCacheValue(model_id=model_id),
            ttl=PromptCachingCache.get_prompt_caching_ttl_from_prefix(
                cacheable_prefix or [],  # mutable-ok: TTL helper requires a concrete list
                tools,
            ),
        )
        return

    async def async_get_model_id(
        self,
        messages: list[AllMessageValues] | None,
        tools: list[ChatCompletionToolParam] | None,
    ) -> PromptCachingCacheValue | None:
        """
        Get model ID from cache using the cacheable prefix.

        The cache key is based on the cacheable prefix (everything up to and including
        the last cache_control block), so requests with the same cacheable prefix but
        different user messages will have the same cache key.
        """
        if messages is None and tools is None:
            return None

        # Generate cache key using cacheable prefix
        cache_key: Final = PromptCachingCache.get_prompt_caching_cache_key(messages, tools)
        if cache_key is None:
            return None

        # Perform cache lookup
        cache_result: Final = await self.cache.async_get_cache(key=cache_key)
        return cache_result

    def get_model_id(
        self,
        messages: list[AllMessageValues] | None,
        tools: list[ChatCompletionToolParam] | None,
    ) -> PromptCachingCacheValue | None:
        if messages is None and tools is None:
            return None

        cache_key: Final = PromptCachingCache.get_prompt_caching_cache_key(messages, tools)
        # If no cacheable prefix found, return None (can't cache)
        if cache_key is None:
            return None

        return self.cache.get_cache(cache_key)
