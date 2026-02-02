"""
Wrapper around router cache. Meant to store model id when prompt caching supported prompt is called.
"""

import hashlib
import json
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

from typing_extensions import TypedDict

from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.router import Router

    litellm_router = Router
    Span = Union[_Span, Any]
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
    def serialize_object(obj: Any) -> Any:
        """Helper function to serialize Pydantic objects, dictionaries, or fallback to string."""
        if hasattr(obj, "dict"):
            # If the object is a Pydantic model, use its `dict()` method
            return obj.dict()
        elif isinstance(obj, dict):
            # If the object is a dictionary, serialize it with sorted keys
            return json.dumps(
                obj, sort_keys=True, separators=(",", ":")
            )  # Standardize serialization

        elif isinstance(obj, list):
            # Serialize lists by ensuring each element is handled properly
            return [PromptCachingCache.serialize_object(item) for item in obj]
        elif isinstance(obj, (int, float, bool)):
            return obj  # Keep primitive types as-is
        return str(obj)

    @staticmethod
    def extract_cacheable_prefix(messages: List[AllMessageValues]) -> List[AllMessageValues]:
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
        cacheable_prefix = []
        
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
                        {**message, "content": content[: last_cacheable_content_idx + 1]},
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
    def get_prompt_caching_cache_key(
        messages: Optional[List[AllMessageValues]],
        tools: Optional[List[ChatCompletionToolParam]],
    ) -> Optional[str]:
        if messages is None and tools is None:
            return None
        
        # Extract cacheable prefix from messages (only include up to last cache_control block)
        cacheable_messages = None
        if messages is not None:
            cacheable_messages = PromptCachingCache.extract_cacheable_prefix(messages)
            # If no cacheable prefix found, return None (can't cache)
            if not cacheable_messages:
                return None
        
        # Use serialize_object for consistent and stable serialization
        data_to_hash = {}
        if cacheable_messages is not None:
            serialized_messages = PromptCachingCache.serialize_object(cacheable_messages)
            data_to_hash["messages"] = serialized_messages
        if tools is not None:
            serialized_tools = PromptCachingCache.serialize_object(tools)
            data_to_hash["tools"] = serialized_tools

        # Combine serialized data into a single string
        data_to_hash_str = json.dumps(
            data_to_hash,
            sort_keys=True,
            separators=(",", ":"),
        )

        # Create a hash of the serialized data for a stable cache key
        hashed_data = hashlib.sha256(data_to_hash_str.encode()).hexdigest()
        return f"deployment:{hashed_data}:prompt_caching"

    def add_model_id(
        self,
        model_id: str,
        messages: Optional[List[AllMessageValues]],
        tools: Optional[List[ChatCompletionToolParam]],
    ) -> None:
        if messages is None and tools is None:
            return None

        cache_key = PromptCachingCache.get_prompt_caching_cache_key(messages, tools)
        # If no cacheable prefix found, don't cache (can't generate cache key)
        if cache_key is None:
            return None

        self.cache.set_cache(
            cache_key, PromptCachingCacheValue(model_id=model_id), ttl=300
        )
        return None

    async def async_add_model_id(
        self,
        model_id: str,
        messages: Optional[List[AllMessageValues]],
        tools: Optional[List[ChatCompletionToolParam]],
    ) -> None:
        if messages is None and tools is None:
            return None

        cache_key = PromptCachingCache.get_prompt_caching_cache_key(messages, tools)
        # If no cacheable prefix found, don't cache (can't generate cache key)
        if cache_key is None:
            return None

        await self.cache.async_set_cache(
            cache_key,
            PromptCachingCacheValue(model_id=model_id),
            ttl=300,  # store for 5 minutes
        )
        return None

    async def async_get_model_id(
        self,
        messages: Optional[List[AllMessageValues]],
        tools: Optional[List[ChatCompletionToolParam]],
    ) -> Optional[PromptCachingCacheValue]:
        """
        Get model ID from cache using the cacheable prefix.
        
        The cache key is based on the cacheable prefix (everything up to and including
        the last cache_control block), so requests with the same cacheable prefix but
        different user messages will have the same cache key.
        """
        if messages is None and tools is None:
            return None

        # Generate cache key using cacheable prefix
        cache_key = PromptCachingCache.get_prompt_caching_cache_key(messages, tools)
        if cache_key is None:
            return None

        # Perform cache lookup
        cache_result = await self.cache.async_get_cache(key=cache_key)
        return cache_result

    def get_model_id(
        self,
        messages: Optional[List[AllMessageValues]],
        tools: Optional[List[ChatCompletionToolParam]],
    ) -> Optional[PromptCachingCacheValue]:
        if messages is None and tools is None:
            return None

        cache_key = PromptCachingCache.get_prompt_caching_cache_key(messages, tools)
        # If no cacheable prefix found, return None (can't cache)
        if cache_key is None:
            return None

        return self.cache.get_cache(cache_key)
