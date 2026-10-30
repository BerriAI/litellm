"""
Response Polling Handler for Background Responses with Cache
"""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

from pydantic import TypeAdapter

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid4
from litellm.caching.redis_cache import RedisCache
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
            written: Final = await self._store.put(polling_id, response.model_dump(mode="json"), ttl=self._ttl())
            if written is None:
                verbose_proxy_logger.debug("Created initial polling state for %s with TTL=%ss", polling_id, self.ttl)

        return response

    @with_service_target(_RESPONSE_POLLING_TARGET)
    async def update_state(
        self,
        polling_id: str,
        status: ResponsesAPIStatus | None = None,
        usage: dict | None = None,
        error: dict | None = None,
        incomplete_details: dict | None = None,
        reasoning: dict | None = None,
        tool_choice: object | None = None,
        tools: list | None = None,
        output: list | None = None,
        # Additional ResponsesAPIResponse fields
        model: str | None = None,
        instructions: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        previous_response_id: str | None = None,
        text: dict | None = None,
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

        cached_state: Final = await self._store.get(polling_id)
        if isinstance(cached_state, RedisFailure):
            return
        if cached_state is None:
            verbose_proxy_logger.warning("No cached state found for polling_id: %s", polling_id)
            return

        state: Final[dict[str, object]] = dict(cached_state)

        # Update status (using OpenAI native status values)
        if status:
            state["status"] = status

        # Replace full output list if provided
        if output is not None:
            state["output"] = output

        # Update usage
        if usage:
            state["usage"] = usage

        # Handle error (sets status to OpenAI's "failed")
        if error:
            state["status"] = "failed"
            state["error"] = error  # Use OpenAI's 'error' field

        # Handle incomplete details
        if incomplete_details:
            state["incomplete_details"] = incomplete_details

        # Update reasoning, tool_choice, tools from response.completed
        if reasoning is not None:
            state["reasoning"] = reasoning
        if tool_choice is not None:
            state["tool_choice"] = tool_choice
        if tools is not None:
            state["tools"] = tools

        # Update additional ResponsesAPIResponse fields
        if model is not None:
            state["model"] = model
        if instructions is not None:
            state["instructions"] = instructions
        if temperature is not None:
            state["temperature"] = temperature
        if top_p is not None:
            state["top_p"] = top_p
        if max_output_tokens is not None:
            state["max_output_tokens"] = max_output_tokens
        if previous_response_id is not None:
            state["previous_response_id"] = previous_response_id
        if text is not None:
            state["text"] = text
        if truncation is not None:
            state["truncation"] = truncation
        if parallel_tool_calls is not None:
            state["parallel_tool_calls"] = parallel_tool_calls
        if user is not None:
            state["user"] = user
        if store is not None:
            state["store"] = store

        written: Final = await self._store.put(polling_id, state, ttl=self._ttl())
        if written is not None:
            return

        output_count: Final = len(_JSON_LIST.validate_python(state.get("output", [])))
        verbose_proxy_logger.debug(
            "Updated polling state for %s: status=%s, output_items=%s", polling_id, state["status"], output_count
        )

    @with_service_target(_RESPONSE_POLLING_TARGET)
    async def get_state(self, polling_id: str) -> Mapping[str, object] | None:
        """Get current polling state from Redis"""
        if self._store is None:
            return None

        cached_state: Final = await self._store.get(polling_id)
        return None if isinstance(cached_state, RedisFailure) else cached_state

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

        return await self._store.delete(polling_id) is None


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
