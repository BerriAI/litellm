"""
Response Polling Handler for Background Responses with Cache
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Final

from pydantic import JsonValue, TypeAdapter

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid4
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreakerOpenError
from litellm.redis.stores import response_polling
from litellm.redis.stores.response_polling import ResponsePollingStore, TransportResponsePollingStore
from litellm.redis.transport.contract import RedisFailure
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesAPIStatus

_RESPONSE_POLLING_TARGET: Final = "response_polling"
_JSON_LIST: Final = TypeAdapter(list[object])


class ResponsePollingHandler:
    """Handles polling-based responses with Redis cache"""

    CACHE_KEY_PREFIX = response_polling.KEY_PREFIX
    POLLING_ID_PREFIX = "litellm_poll_"  # Clear prefix to identify polling IDs

    def __init__(self, redis_cache: RedisCache | None = None, ttl: int = 3600):
        self.ttl = ttl
        self._store: Final[ResponsePollingStore | None] = (
            TransportResponsePollingStore(redis_cache.transport()) if redis_cache is not None else None
        )

    @classmethod
    def generate_polling_id(cls) -> str:
        """Generate a unique UUID for polling with clear prefix"""
        return f"{cls.POLLING_ID_PREFIX}{uuid4()}"

    @classmethod
    def is_polling_id(cls, response_id: str) -> bool:
        """Check if a response_id is a polling ID"""
        return response_id.startswith(cls.POLLING_ID_PREFIX)

    @classmethod
    def get_cache_key(cls, polling_id: str) -> str:
        """Get Redis cache key for a polling ID"""
        return response_polling.state_key(polling_id)

    def _ttl(self) -> timedelta:
        return timedelta(seconds=self.ttl)

    @with_service_target(_RESPONSE_POLLING_TARGET)
    async def create_initial_state(
        self,
        polling_id: str,
        request_data: Mapping[str, object],
    ) -> ResponsesAPIResponse:
        """
        Create initial state in Redis for a polling request

        Uses OpenAI ResponsesAPIResponse object:
        https://platform.openai.com/docs/api-reference/responses/object

        Args:
            polling_id: Unique identifier for this polling request
            request_data: Original request data

        Returns:
            ResponsesAPIResponse object following OpenAI spec
        """
        created_timestamp: Final = int(datetime.now(timezone.utc).timestamp())

        # Create OpenAI-compliant response object
        response: Final = ResponsesAPIResponse.model_validate(
            {
                "id": polling_id,
                "object": "response",
                "status": "queued",  # OpenAI native status
                "created_at": created_timestamp,
                "output": [],
                "metadata": request_data.get("metadata", {}),
                "usage": None,
            }
        )

        if self._store is not None:
            _raise_if_breaker_open(await self._store.create(polling_id, response, ttl=self._ttl()))
            verbose_proxy_logger.debug("Created initial polling state for %s with TTL=%ss", polling_id, self.ttl)

        return response

    @with_service_target(_RESPONSE_POLLING_TARGET)
    async def update_state(
        self,
        polling_id: str,
        status: ResponsesAPIStatus | None = None,
        usage: Mapping[str, object] | None = None,
        error: Mapping[str, object] | None = None,
        incomplete_details: Mapping[str, object] | None = None,
        reasoning: Mapping[str, object] | None = None,
        tool_choice: object | None = None,
        tools: Sequence[object] | None = None,
        output: Sequence[object] | None = None,
        # Additional ResponsesAPIResponse fields
        model: str | None = None,
        instructions: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        previous_response_id: str | None = None,
        text: Mapping[str, object] | None = None,
        truncation: str | None = None,
        parallel_tool_calls: bool | None = None,
        user: str | None = None,
        store: bool | None = None,
    ) -> None:
        """
        Update the polling state in Redis

        Uses OpenAI Response object format with native status types:
        https://platform.openai.com/docs/api-reference/responses/object

        Args:
            polling_id: Unique identifier for this polling request
            status: OpenAI ResponsesAPIStatus value
            usage: Usage information
            error: Error dict (automatically sets status to "failed")
            incomplete_details: Details for incomplete responses
            reasoning: Reasoning configuration from response.completed
            tool_choice: Tool choice configuration from response.completed
            tools: Tools list from response.completed
            output: Full output list to replace current output
            model: Model identifier
            instructions: System instructions
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_output_tokens: Maximum output tokens
            previous_response_id: ID of previous response in conversation
            text: Text configuration
            truncation: Truncation setting
            parallel_tool_calls: Whether parallel tool calls are enabled
            user: User identifier
            store: Whether to store the response
        """
        if self._store is None:
            return

        cached_state: Final = _unless_raising(await self._store.get(polling_id))
        if cached_state is None:
            verbose_proxy_logger.warning("No cached state found for polling_id: %s", polling_id)
            return

        updates: Final[Mapping[str, object]] = {
            "status": "failed" if error else status or None,
            "output": output,
            "usage": usage or None,
            "error": error or None,
            "incomplete_details": incomplete_details or None,
            "reasoning": reasoning,
            "tool_choice": tool_choice,
            "tools": tools,
            "model": model,
            "instructions": instructions,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "previous_response_id": previous_response_id,
            "text": text,
            "truncation": truncation,
            "parallel_tool_calls": parallel_tool_calls,
            "user": user,
            "store": store,
        }
        state: Final[Mapping[str, object]] = {
            **cached_state,
            **{key: value for key, value in updates.items() if value is not None},
        }

        _raise_if_breaker_open(await self._store.update(polling_id, state, ttl=self._ttl()))

        output_count: Final = len(_JSON_LIST.validate_python(state.get("output", [])))
        verbose_proxy_logger.debug(
            "Updated polling state for %s: status=%s, output_items=%s", polling_id, state["status"], output_count
        )

    @with_service_target(_RESPONSE_POLLING_TARGET)
    async def get_state(self, polling_id: str) -> Mapping[str, object] | None:
        """Get current polling state from Redis"""
        if self._store is None:
            return None

        return _unless_raising(await self._store.get(polling_id))

    async def cancel_polling(self, polling_id: str) -> bool:
        """
        Cancel a polling request

        Following OpenAI Response object format for cancelled status
        """
        await self.update_state(
            polling_id=polling_id,
            status="cancelled",
        )
        return True

    @with_service_target(_RESPONSE_POLLING_TARGET)
    async def delete_polling(self, polling_id: str) -> bool:
        """Delete a polling request from cache"""
        if self._store is None:
            return False

        deleted: Final = await self._store.delete(polling_id)
        if isinstance(deleted, RedisFailure):
            raise _cache_error(deleted)
        return True


def _cache_error(failure: RedisFailure) -> Exception:
    match failure.kind:
        case "breaker_open":
            return RedisCircuitBreakerOpenError(failure.message)
        case "decode":
            return ValueError(failure.message)
        case "transport" | "server":
            return RuntimeError(failure.message)


def _raise_if_breaker_open(written: RedisFailure | None) -> None:
    if isinstance(written, RedisFailure) and written.kind == "breaker_open":
        raise _cache_error(written)


def _unless_raising(read: Mapping[str, JsonValue] | None | RedisFailure) -> Mapping[str, JsonValue] | None:
    if not isinstance(read, RedisFailure):
        return read
    if read.kind in ("breaker_open", "decode"):
        raise _cache_error(read)
    return None


def should_use_polling_for_request(
    background_mode: bool,
    polling_via_cache_enabled,  # Can be False, "all", or List[str]
    redis_cache,  # RedisCache or None
    model: str,
    llm_router,  # Router instance or None
    native_background_mode: list[str] | None = None,  # List of models that should use native background mode
) -> bool:
    """
    Determine if polling via cache should be used for a request.

    Args:
        background_mode: Whether background=true was set in the request
        polling_via_cache_enabled: Config value - False, "all", or list of providers
        redis_cache: Redis cache instance (required for polling)
        model: Model name from the request (e.g., "gpt-5" or "openai/gpt-4o")
        llm_router: LiteLLM router instance for looking up model deployments
        native_background_mode: List of model names that should use native provider
            background mode instead of polling via cache

    Returns:
        True if polling should be used, False otherwise
    """
    # All conditions must be met
    if not (background_mode and polling_via_cache_enabled and redis_cache):
        return False

    # Check if model is in native_background_mode list - these use native provider background mode
    if native_background_mode and model in native_background_mode:
        verbose_proxy_logger.debug("Model %s is in native_background_mode list, skipping polling via cache", model)
        return False

    # "all" enables polling for all providers
    if polling_via_cache_enabled == "all":
        return True

    # Check if provider is in the enabled list
    if isinstance(polling_via_cache_enabled, list):
        # First, try to get provider from model string format "provider/model"
        if "/" in model:
            provider: Final = model.split("/")[0]
            if provider in polling_via_cache_enabled:
                return True
        # Otherwise, check ALL deployments for this model_name in router
        elif llm_router is not None:
            try:
                for deployment_dict in llm_router.registry().deployments(model, source="named") or ():
                    litellm_params = deployment_dict.get("litellm_params", {})

                    # Check custom_llm_provider first
                    dep_provider = litellm_params.get("custom_llm_provider")

                    # Then try to extract from model (e.g., "openai/gpt-5")
                    if not dep_provider:
                        dep_model = litellm_params.get("model", "")
                        if "/" in dep_model:
                            dep_provider = dep_model.split("/")[0]

                    # If ANY deployment's provider matches, enable polling
                    if dep_provider and dep_provider in polling_via_cache_enabled:
                        verbose_proxy_logger.debug("Polling enabled for model=%s, provider=%s", model, dep_provider)
                        return True
            except Exception as e:
                verbose_proxy_logger.debug("Could not resolve provider for model %s: %s", model, e)

    return False
