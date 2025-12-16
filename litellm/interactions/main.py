"""
LiteLLM Interactions API - Main Module

Per OpenAPI spec (https://ai.google.dev/static/api/interactions.openapi.json):
- Create interaction: POST /{api_version}/interactions
- Get interaction: GET /{api_version}/interactions/{interaction_id}
- Delete interaction: DELETE /{api_version}/interactions/{interaction_id}

Usage:
    import litellm
    
    # Create an interaction with a model
    response = litellm.interactions.create(
        model="gemini-2.5-flash",
        input="Hello, how are you?"
    )
    
    # Create an interaction with an agent
    response = litellm.interactions.create(
        agent="deep-research-pro-preview-12-2025",
        input="Research the current state of cancer research"
    )
    
    # Async version
    response = await litellm.interactions.acreate(...)
    
    # Get an interaction
    response = litellm.interactions.get(interaction_id="...")
    
    # Delete an interaction
    result = litellm.interactions.delete(interaction_id="...")
"""

import asyncio
import contextvars
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

import httpx

import litellm
from litellm.interactions.http_handler import interactions_http_handler
from litellm.interactions.utils import (
    InteractionsAPIRequestUtils,
    get_provider_interactions_api_config,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.interactions.main import (
    CancelInteractionResult,
    DeleteInteractionResult,
    InteractionInput,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
    InteractionTool,
    InteractionToolChoiceConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import client

# ============================================================
# SDK Methods - CREATE INTERACTION
# ============================================================


@client
async def acreate(
    # Model or Agent (one required per OpenAPI spec)
    model: Optional[str] = None,
    agent: Optional[str] = None,
    # Input (required)
    input: Optional[InteractionInput] = None,
    # Tools (for model interactions)
    tools: Optional[List[InteractionTool]] = None,
    tool_choice: Optional[InteractionToolChoiceConfig] = None,
    # Instructions
    instructions: Optional[str] = None,
    # Agent config (for agent interactions)
    agent_config: Optional[Dict[str, Any]] = None,
    # Streaming
    stream: Optional[bool] = None,
    # Extra params
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM params
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[InteractionsAPIResponse, AsyncIterator[InteractionsAPIStreamingResponse]]:
    """
    Async: Create a new interaction using Google's Interactions API.
    
    Per OpenAPI spec, provide either `model` or `agent`.
    
    Args:
        model: The model to use (e.g., "gemini-2.5-flash")
        agent: The agent to use (e.g., "deep-research-pro-preview-12-2025")
        input: The input content (string, content object, or list)
        tools: Tools available for the model
        tool_choice: Tool choice configuration
        instructions: Instructions for the model/agent
        agent_config: Agent configuration (for agent interactions)
        stream: Whether to stream the response
        extra_headers: Additional headers
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Override the LLM provider
        
    Returns:
        InteractionsAPIResponse or async iterator for streaming
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate_interaction"] = True
        
        if custom_llm_provider is None and model:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model, api_base=kwargs.get("api_base", None)
            )
        elif custom_llm_provider is None:
            custom_llm_provider = "gemini"
        
        func = partial(
            create,
            model=model,
            agent=agent,
            input=input,
            tools=tools,
            tool_choice=tool_choice,
            instructions=instructions,
            agent_config=agent_config,
            stream=stream,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        
        return response  # type: ignore
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def create(
    # Model or Agent (one required per OpenAPI spec)
    model: Optional[str] = None,
    agent: Optional[str] = None,
    # Input (required)
    input: Optional[InteractionInput] = None,
    # Tools (for model interactions)
    tools: Optional[List[InteractionTool]] = None,
    tool_choice: Optional[InteractionToolChoiceConfig] = None,
    # Instructions
    instructions: Optional[str] = None,
    # Agent config (for agent interactions)
    agent_config: Optional[Dict[str, Any]] = None,
    # Streaming
    stream: Optional[bool] = None,
    # Extra params
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM params
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[
    InteractionsAPIResponse,
    Iterator[InteractionsAPIStreamingResponse],
    Coroutine[Any, Any, Union[InteractionsAPIResponse, AsyncIterator[InteractionsAPIStreamingResponse]]],
]:
    """
    Sync: Create a new interaction using Google's Interactions API.
    
    Per OpenAPI spec, provide either `model` or `agent`.
    
    Args:
        model: The model to use (e.g., "gemini-2.5-flash")
        agent: The agent to use (e.g., "deep-research-pro-preview-12-2025")
        input: The input content (string, content object, or list)
        tools: Tools available for the model
        tool_choice: Tool choice configuration
        instructions: Instructions for the model/agent
        agent_config: Agent configuration (for agent interactions)
        stream: Whether to stream the response
        extra_headers: Additional headers
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Override the LLM provider
        
    Returns:
        InteractionsAPIResponse or iterator for streaming
    """
    local_vars = locals()
    
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("acreate_interaction", False) is True
        
        litellm_params = GenericLiteLLMParams(**kwargs)
        
        if model:
            model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=litellm_params.api_base,
            api_key=litellm_params.api_key,
        )
        else:
            custom_llm_provider = custom_llm_provider or "gemini"
        
        interactions_api_config = get_provider_interactions_api_config(
            provider=custom_llm_provider,
            model=model,
        )
        
        if interactions_api_config is None:
            raise ValueError(
                f"Interactions API is not supported for provider: {custom_llm_provider}. "
                "Currently only 'gemini' is supported."
            )
        
        # Get optional params using utility (similar to responses API pattern)
        local_vars.update(kwargs)
        optional_params = InteractionsAPIRequestUtils.get_requested_interactions_api_optional_params(
            local_vars
        )
        
        litellm_logging_obj.update_environment_variables(
            model=model,
            optional_params=dict(optional_params),
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )
        
        response = interactions_http_handler.create_interaction(
            model=model,
            agent=agent,
            input=input,
            interactions_api_config=interactions_api_config,
            optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            _is_async=_is_async,
            stream=stream,
        )
        
        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ============================================================
# SDK Methods - GET INTERACTION
# ============================================================


@client
async def aget(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> InteractionsAPIResponse:
    """Async: Get an interaction by its ID."""
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aget_interaction"] = True
        
        func = partial(
            get,
            interaction_id=interaction_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider or "gemini",
            **kwargs,
        )
        
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        
        return response  # type: ignore
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def get(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[InteractionsAPIResponse, Coroutine[Any, Any, InteractionsAPIResponse]]:
    """Sync: Get an interaction by its ID."""
    local_vars = locals()
    custom_llm_provider = custom_llm_provider or "gemini"
    
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("aget_interaction", False) is True
        
        litellm_params = GenericLiteLLMParams(**kwargs)
        
        interactions_api_config = get_provider_interactions_api_config(
            provider=custom_llm_provider,
        )
        
        if interactions_api_config is None:
            raise ValueError(f"Interactions API not supported for: {custom_llm_provider}")
        
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={"interaction_id": interaction_id},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )
        
        return interactions_http_handler.get_interaction(
            interaction_id=interaction_id,
            interactions_api_config=interactions_api_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ============================================================
# SDK Methods - DELETE INTERACTION
# ============================================================


@client
async def adelete(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> DeleteInteractionResult:
    """Async: Delete an interaction by its ID."""
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["adelete_interaction"] = True
        
        func = partial(
            delete,
            interaction_id=interaction_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider or "gemini",
            **kwargs,
        )
        
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        
        return response  # type: ignore
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def delete(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[DeleteInteractionResult, Coroutine[Any, Any, DeleteInteractionResult]]:
    """Sync: Delete an interaction by its ID."""
    local_vars = locals()
    custom_llm_provider = custom_llm_provider or "gemini"
    
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("adelete_interaction", False) is True
        
        litellm_params = GenericLiteLLMParams(**kwargs)
        
        interactions_api_config = get_provider_interactions_api_config(
            provider=custom_llm_provider,
        )
        
        if interactions_api_config is None:
            raise ValueError(f"Interactions API not supported for: {custom_llm_provider}")
        
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={"interaction_id": interaction_id},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )
        
        return interactions_http_handler.delete_interaction(
            interaction_id=interaction_id,
            interactions_api_config=interactions_api_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ============================================================
# SDK Methods - CANCEL INTERACTION
# ============================================================


@client
async def acancel(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> CancelInteractionResult:
    """Async: Cancel an interaction by its ID."""
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["acancel_interaction"] = True
        
        func = partial(
            cancel,
            interaction_id=interaction_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider or "gemini",
            **kwargs,
        )
        
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        
        return response  # type: ignore
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def cancel(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[CancelInteractionResult, Coroutine[Any, Any, CancelInteractionResult]]:
    """Sync: Cancel an interaction by its ID."""
    local_vars = locals()
    custom_llm_provider = custom_llm_provider or "gemini"
    
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("acancel_interaction", False) is True
        
        litellm_params = GenericLiteLLMParams(**kwargs)
        
        interactions_api_config = get_provider_interactions_api_config(
            provider=custom_llm_provider,
        )
        
        if interactions_api_config is None:
            raise ValueError(f"Interactions API not supported for: {custom_llm_provider}")
        
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={"interaction_id": interaction_id},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )
        
        return interactions_http_handler.cancel_interaction(
            interaction_id=interaction_id,
            interactions_api_config=interactions_api_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
