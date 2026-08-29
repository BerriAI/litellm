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
from collections.abc import AsyncIterator, Coroutine, Iterator
from functools import partial
from typing import Any, Final

import httpx

import litellm
from litellm.interactions.background_cost_polling import (
    maybe_schedule_background_interaction_cost_polling,
    maybe_settle_background_interaction_before_delete,
)
from litellm.interactions.http_handler import interactions_http_handler
from litellm.interactions.utils import (
    InteractionsAPIRequestUtils,
    get_provider_interactions_api_config,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.interactions import (
    CancelInteractionResult,
    DeleteInteractionResult,
    InteractionEnvironment,
    InteractionInput,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
    InteractionTool,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import client

# ============================================================
# SDK Methods - CREATE INTERACTION
# ============================================================


@client
async def acreate(
    # Model or Agent (one required per OpenAPI spec)
    model: str | None = None,
    agent: str | None = None,
    # Input (required)
    input: InteractionInput | None = None,
    # Tools (for model interactions)
    tools: list[InteractionTool] | None = None,
    # System instruction
    system_instruction: str | None = None,
    # Generation config
    generation_config: dict[str, Any] | None = None,
    # Streaming
    stream: bool | None = None,
    # Storage
    store: bool | None = None,
    # Background execution
    background: bool | None = None,
    # Agent execution environment ("remote", env id, or remote config object)
    environment: InteractionEnvironment | None = None,
    # Response format
    response_modalities: list[str] | None = None,
    response_format: dict[str, Any] | None = None,
    response_mime_type: str | None = None,
    # Continuation
    previous_interaction_id: str | None = None,
    # Extra params
    extra_headers: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM params
    custom_llm_provider: str | None = None,
    **kwargs,
) -> InteractionsAPIResponse | AsyncIterator[InteractionsAPIStreamingResponse]:
    """
    Async: Create a new interaction using Google's Interactions API.

    Per OpenAPI spec, provide either `model` or `agent`.

    Args:
        model: The model to use (e.g., "gemini-2.5-flash")
        agent: The agent to use (e.g., "deep-research-pro-preview-12-2025")
        input: The input content (string, content object, or list)
        tools: Tools available for the model
        system_instruction: System instruction for the interaction
        generation_config: Generation configuration
        stream: Whether to stream the response
        store: Whether to store the response for later retrieval
        background: Whether to run in background
        environment: Agent execution environment — ``"remote"``, an existing env id
            string, or a config object such as
            ``{"type": "remote", "sources": [...]}`` /
            ``{"type": "remote", "network": {...}}``
        response_modalities: Requested response modalities (TEXT, IMAGE, AUDIO)
        response_format: JSON schema for response format
        response_mime_type: MIME type of the response
        previous_interaction_id: ID of previous interaction for continuation
        extra_headers: Additional headers
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Override the LLM provider

    Returns:
        InteractionsAPIResponse or async iterator for streaming
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acreate_interaction"] = True

        if custom_llm_provider is None and model:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model, api_base=kwargs.get("api_base", None))
        elif custom_llm_provider is None:
            custom_llm_provider = "gemini"

        func: Final = partial(
            create,
            model=model,
            agent=agent,
            input=input,
            tools=tools,
            system_instruction=system_instruction,
            generation_config=generation_config,
            stream=stream,
            store=store,
            background=background,
            environment=environment,
            response_modalities=response_modalities,
            response_format=response_format,
            response_mime_type=response_mime_type,
            previous_interaction_id=previous_interaction_id,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        maybe_schedule_background_interaction_cost_polling(
            response=response,
            create_kwargs=kwargs,
            custom_llm_provider=custom_llm_provider,
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


@client
def create(
    # Model or Agent (one required per OpenAPI spec)
    model: str | None = None,
    agent: str | None = None,
    # Input (required)
    input: InteractionInput | None = None,
    # Tools (for model interactions)
    tools: list[InteractionTool] | None = None,
    # System instruction
    system_instruction: str | None = None,
    # Generation config
    generation_config: dict[str, Any] | None = None,
    # Streaming
    stream: bool | None = None,
    # Storage
    store: bool | None = None,
    # Background execution
    background: bool | None = None,
    # Agent execution environment ("remote", env id, or remote config object)
    environment: InteractionEnvironment | None = None,
    # Response format
    response_modalities: list[str] | None = None,
    response_format: dict[str, Any] | None = None,
    response_mime_type: str | None = None,
    # Continuation
    previous_interaction_id: str | None = None,
    # Extra params
    extra_headers: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM params
    custom_llm_provider: str | None = None,
    **kwargs,
) -> (
    InteractionsAPIResponse
    | Iterator[InteractionsAPIStreamingResponse]
    | Coroutine[Any, Any, InteractionsAPIResponse | AsyncIterator[InteractionsAPIStreamingResponse]]
):
    """
    Sync: Create a new interaction using Google's Interactions API.

    Per OpenAPI spec, provide either `model` or `agent`.

    Args:
        model: The model to use (e.g., "gemini-2.5-flash")
        agent: The agent to use (e.g., "deep-research-pro-preview-12-2025")
        input: The input content (string, content object, or list)
        tools: Tools available for the model
        system_instruction: System instruction for the interaction
        generation_config: Generation configuration
        stream: Whether to stream the response
        store: Whether to store the response for later retrieval
        background: Whether to run in background
        environment: Agent execution environment — ``"remote"``, an existing env id
            string, or a config object such as
            ``{"type": "remote", "sources": [...]}`` /
            ``{"type": "remote", "network": {...}}``
        response_modalities: Requested response modalities (TEXT, IMAGE, AUDIO)
        response_format: JSON schema for response format
        response_mime_type: MIME type of the response
        previous_interaction_id: ID of previous interaction for continuation
        extra_headers: Additional headers
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Override the LLM provider

    Returns:
        InteractionsAPIResponse or iterator for streaming
    """
    local_vars: Final = locals()

    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("acreate_interaction", False) is True

        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Routing logic:
        # - agent provided (no model, or model accidentally set to agent name) → gemini
        # - model provided → resolve provider via get_llm_provider (normal routing)
        if agent and model == agent:
            model = None
        if agent and not model:
            custom_llm_provider = custom_llm_provider or "gemini"
        elif model:
            model, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model,
                custom_llm_provider=custom_llm_provider,
                api_base=litellm_params.api_base,
                api_key=litellm_params.api_key,
            )
        else:
            custom_llm_provider = custom_llm_provider or "gemini"

        interactions_api_config: Final = get_provider_interactions_api_config(
            provider=custom_llm_provider,
            model=model,
        )

        # Get optional params using utility (similar to responses API pattern)
        local_vars.update(kwargs)
        optional_params: Final = InteractionsAPIRequestUtils.get_requested_interactions_api_optional_params(local_vars)

        # Check if this is a bridge provider (litellm_responses) - similar to responses API
        # Either provider is explicitly "litellm_responses" or no config found (bridge to responses)
        if custom_llm_provider == "litellm_responses" or interactions_api_config is None:
            # Bridge to litellm.responses() for non-native providers
            from litellm.interactions.litellm_responses_transformation.handler import (
                LiteLLMResponsesInteractionsHandler,
            )

            handler: Final = LiteLLMResponsesInteractionsHandler()
            return handler.interactions_api_handler(
                model=model or "",
                input=input,
                optional_params=optional_params,
                custom_llm_provider=custom_llm_provider,
                _is_async=_is_async,
                stream=stream,
                **kwargs,
            )

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=model,
            optional_params=dict(optional_params),
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = interactions_http_handler.create_interaction(
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
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> InteractionsAPIResponse:
    """Async: Get an interaction by its ID."""
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aget_interaction"] = True

        func: Final = partial(
            get,
            interaction_id=interaction_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider or "gemini",
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
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
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> InteractionsAPIResponse | Coroutine[Any, Any, InteractionsAPIResponse]:
    """Sync: Get an interaction by its ID."""
    local_vars: Final = locals()
    custom_llm_provider = custom_llm_provider or "gemini"

    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("aget_interaction", False) is True

        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        interactions_api_config: Final = get_provider_interactions_api_config(
            provider=custom_llm_provider,
        )

        if interactions_api_config is None:
            raise ValueError(f"Interactions API not supported for: {custom_llm_provider}")

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
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
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> DeleteInteractionResult:
    """Async: Delete an interaction by its ID."""
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["adelete_interaction"] = True

        await maybe_settle_background_interaction_before_delete(interaction_id=interaction_id)

        func: Final = partial(
            delete,
            interaction_id=interaction_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider or "gemini",
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
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
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> DeleteInteractionResult | Coroutine[Any, Any, DeleteInteractionResult]:
    """Sync: Delete an interaction by its ID."""
    local_vars: Final = locals()
    custom_llm_provider = custom_llm_provider or "gemini"

    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("adelete_interaction", False) is True

        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        interactions_api_config: Final = get_provider_interactions_api_config(
            provider=custom_llm_provider,
        )

        if interactions_api_config is None:
            raise ValueError(f"Interactions API not supported for: {custom_llm_provider}")

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
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
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> CancelInteractionResult:
    """Async: Cancel an interaction by its ID."""
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acancel_interaction"] = True

        func: Final = partial(
            cancel,
            interaction_id=interaction_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider or "gemini",
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
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
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> CancelInteractionResult | Coroutine[Any, Any, CancelInteractionResult]:
    """Sync: Cancel an interaction by its ID."""
    local_vars: Final = locals()
    custom_llm_provider = custom_llm_provider or "gemini"

    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("acancel_interaction", False) is True

        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        interactions_api_config: Final = get_provider_interactions_api_config(
            provider=custom_llm_provider,
        )

        if interactions_api_config is None:
            raise ValueError(f"Interactions API not supported for: {custom_llm_provider}")

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
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
