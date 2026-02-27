"""
Response Polling Handler for Background Responses with Cache
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid4
from litellm.caching.redis_cache import RedisCache
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesAPIStatus


class ResponsePollingHandler:
    """Handles polling-based responses with Redis cache"""
    
    CACHE_KEY_PREFIX = "litellm:polling:response:"
    POLLING_ID_PREFIX = "litellm_poll_"  # Clear prefix to identify polling IDs
    
    def __init__(self, redis_cache: Optional[RedisCache] = None, ttl: int = 3600):
        self.redis_cache = redis_cache
        self.ttl = ttl  # Time-to-live for cache entries (default: 1 hour)
    
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
        return f"{cls.CACHE_KEY_PREFIX}{polling_id}"
    
    async def create_initial_state(
        self,
        polling_id: str,
        request_data: Dict[str, Any],
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
        created_timestamp = int(datetime.now(timezone.utc).timestamp())
        
        # Create OpenAI-compliant response object
        response = ResponsesAPIResponse(
            id=polling_id,
            object="response",
            status="queued",  # OpenAI native status
            created_at=created_timestamp,
            output=[],
            metadata=request_data.get("metadata", {}),
            usage=None,
        )
        
        cache_key = self.get_cache_key(polling_id)
        
        if self.redis_cache:
            # Store ResponsesAPIResponse directly in Redis
            await self.redis_cache.async_set_cache(
                key=cache_key,
                value=response.model_dump_json(),  # Pydantic v2 method
                ttl=self.ttl,
            )
            verbose_proxy_logger.debug(
                f"Created initial polling state for {polling_id} with TTL={self.ttl}s"
            )
        
        return response
    
    async def update_state(
        self,
        polling_id: str,
        status: Optional[ResponsesAPIStatus] = None,
        usage: Optional[Dict] = None,
        error: Optional[Dict] = None,
        incomplete_details: Optional[Dict] = None,
        reasoning: Optional[Dict] = None,
        tool_choice: Optional[Any] = None,
        tools: Optional[list] = None,
        output: Optional[list] = None,
        # Additional ResponsesAPIResponse fields
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        previous_response_id: Optional[str] = None,
        text: Optional[Dict] = None,
        truncation: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        user: Optional[str] = None,
        store: Optional[bool] = None,
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
        if not self.redis_cache:
            return
        
        cache_key = self.get_cache_key(polling_id)
        
        # Get current state
        cached_state = await self.redis_cache.async_get_cache(cache_key)
        if not cached_state:
            verbose_proxy_logger.warning(
                f"No cached state found for polling_id: {polling_id}"
            )
            return
        
        # Parse existing ResponsesAPIResponse from cache
        state = json.loads(cached_state)
        
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
        
        # Update cache with configured TTL
        await self.redis_cache.async_set_cache(
            key=cache_key,
            value=json.dumps(state),
            ttl=self.ttl,
        )
        
        output_count = len(state.get("output", []))
        verbose_proxy_logger.debug(
            f"Updated polling state for {polling_id}: status={state['status']}, output_items={output_count}"
        )
    
    async def get_state(self, polling_id: str) -> Optional[Dict[str, Any]]:
        """Get current polling state from Redis"""
        if not self.redis_cache:
            return None
        
        cache_key = self.get_cache_key(polling_id)
        cached_state = await self.redis_cache.async_get_cache(cache_key)
        
        if cached_state:
            return json.loads(cached_state)
        
        return None
    
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
    
    async def delete_polling(self, polling_id: str) -> bool:
        """Delete a polling request from cache"""
        if not self.redis_cache:
            return False
        
        cache_key = self.get_cache_key(polling_id)
        # Use RedisCache's async_delete_cache method which handles Redis/RedisCluster
        await self.redis_cache.async_delete_cache(cache_key)
        return True


def should_use_polling_for_request(
    background_mode: bool,
    polling_via_cache_enabled,  # Can be False, "all", or List[str]
    redis_cache,  # RedisCache or None
    model: str,
    llm_router,  # Router instance or None
    native_background_mode: Optional[List[str]] = None,  # List of models that should use native background mode
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
        verbose_proxy_logger.debug(
            f"Model {model} is in native_background_mode list, skipping polling via cache"
        )
        return False
    
    # "all" enables polling for all providers
    if polling_via_cache_enabled == "all":
        return True
    
    # Check if provider is in the enabled list
    if isinstance(polling_via_cache_enabled, list):
        # First, try to get provider from model string format "provider/model"
        if "/" in model:
            provider = model.split("/")[0]
            if provider in polling_via_cache_enabled:
                return True
        # Otherwise, check ALL deployments for this model_name in router
        elif llm_router is not None:
            try:
                # Get all deployment indices for this model name
                indices = llm_router.model_name_to_deployment_indices.get(model, [])
                for idx in indices:
                    deployment_dict = llm_router.model_list[idx]
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
                        verbose_proxy_logger.debug(
                            f"Polling enabled for model={model}, provider={dep_provider}"
                        )
                        return True
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Could not resolve provider for model {model}: {e}"
                )
    
    return False

