"""
Wrapper around router cache. Meant to store model id when prompt caching supported prompt is called.
"""

import hashlib
import json
from typing import TYPE_CHECKING, Any, List, Optional, TypedDict

from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
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
    def get_prompt_caching_cache_key(
        messages: Optional[List[AllMessageValues]],
        tools: Optional[List[ChatCompletionToolParam]],
    ) -> Optional[str]:
        if messages is None and tools is None:
            return None
        # Use serialize_object for consistent and stable serialization
        data_to_hash = {}
        if messages is not None:
            serialized_messages = PromptCachingCache.serialize_object(messages)
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
        if messages is not none
        - check full messages
        - check messages[:-1]
        - check messages[:-2]
        - check messages[:-3]

        use self.cache.async_batch_get_cache(keys=potential_cache_keys])
        """
        if messages is None and tools is None:
            return None

        # Generate potential cache keys by slicing messages

        potential_cache_keys = []

        if messages is not None:
            full_cache_key = PromptCachingCache.get_prompt_caching_cache_key(
                messages, tools
            )
            potential_cache_keys.append(full_cache_key)

            # Check progressively shorter message slices
            for i in range(1, min(4, len(messages))):
                partial_messages = messages[:-i]
                partial_cache_key = PromptCachingCache.get_prompt_caching_cache_key(
                    partial_messages, tools
                )
                potential_cache_keys.append(partial_cache_key)

        # Perform batch cache lookup
        cache_results = await self.cache.async_batch_get_cache(
            keys=potential_cache_keys
        )

        if cache_results is None:
            return None

        # Return the first non-None cache result
        for result in cache_results:
            if result is not None:
                return result

        return None

    def get_model_id(
        self,
        messages: Optional[List[AllMessageValues]],
        tools: Optional[List[ChatCompletionToolParam]],
    ) -> Optional[PromptCachingCacheValue]:
        if messages is None and tools is None:
            return None

        cache_key = PromptCachingCache.get_prompt_caching_cache_key(messages, tools)
        return self.cache.get_cache(cache_key)
