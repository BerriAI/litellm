"""
LiteLLM Agents API - Main Module

Usage:
    import litellm

    # Create a managed agent on the provider side
    response = litellm.interactions.agents.create(
        name="waverunner",
        custom_llm_provider="gemini",
        api_key="...",
        base_agent="gemini-2.5-flash",
        instructions="You are a helpful assistant.",
    )

    # Async version
    response = await litellm.interactions.agents.acreate(...)
"""

import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

import litellm
from litellm.interactions.agents.http_handler import agents_http_handler
from litellm.interactions.agents.utils import get_provider_agents_api_config
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.agents import AgentCreateResponse
from litellm.types.interactions import InteractionEnvironment
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import client


# ============================================================
# SDK Methods - CREATE AGENT
# ============================================================


@client
async def acreate(
    name: str,
    base_agent: Optional[str] = None,
    instructions: Optional[str] = None,
    base_environment: Optional[InteractionEnvironment] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> AgentCreateResponse:
    """
    Async: Create a managed agent on the provider side.

    Args:
        name: Name for the agent (required).
        base_agent: Base agent to derive from (e.g. "waverunner").
        instructions: System instructions for the agent.
        base_environment: Environment to use — either an env_id string to fork
            from an existing environment, or a dict such as
            ``{"type": "remote", "sources": [...]}`` to build from scratch.
        custom_llm_provider: Provider to use, e.g. "gemini".
        extra_headers: Additional HTTP headers.
        extra_body: Additional request body fields.
        timeout: Request timeout.
        **kwargs: Additional params forwarded to GenericLiteLLMParams
                  (api_key, api_base, etc.).

    Returns:
        AgentCreateResponse
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate_agent"] = True

        if custom_llm_provider is None:
            custom_llm_provider = "gemini"

        func = partial(
            create,
            name=name,
            base_agent=base_agent,
            instructions=instructions,
            base_environment=base_environment,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def create(
    name: str,
    base_agent: Optional[str] = None,
    instructions: Optional[str] = None,
    base_environment: Optional[InteractionEnvironment] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> Union[AgentCreateResponse, Coroutine[Any, Any, AgentCreateResponse]]:
    """
    Sync: Create a managed agent on the provider side.

    Args:
        name: Name for the agent (required).
        base_agent: Base agent to derive from (e.g. "waverunner").
        instructions: System instructions for the agent.
        base_environment: Environment to use — either an env_id string to fork
            from an existing environment, or a dict such as
            ``{"type": "remote", "sources": [...]}`` to build from scratch.
        custom_llm_provider: Provider to use, e.g. "gemini".
        extra_headers: Additional HTTP headers.
        extra_body: Additional request body fields.
        timeout: Request timeout.
        **kwargs: Additional params forwarded to GenericLiteLLMParams
                  (api_key, api_base, etc.).

    Returns:
        AgentCreateResponse
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("acreate_agent", False) is True

        if custom_llm_provider is None:
            custom_llm_provider = "gemini"

        # Inject explicit agent fields into kwargs so GenericLiteLLMParams
        # (extra="allow") stores them and GeminiAgentsConfig can read them.
        if base_agent is not None:
            kwargs["base_agent"] = base_agent
        if instructions is not None:
            kwargs["instructions"] = instructions
        if base_environment is not None:
            kwargs["base_environment"] = base_environment

        litellm_params = GenericLiteLLMParams(**kwargs)

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=name,
            optional_params={},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )

        agents_api_config = get_provider_agents_api_config(custom_llm_provider)
        if agents_api_config is None:
            raise litellm.BadRequestError(
                message=(
                    f"Provider '{custom_llm_provider}' does not have a native "
                    "agent-creation API. Use the proxy POST /v1/agents endpoint "
                    "to store agents locally."
                ),
                model=name,
                llm_provider=custom_llm_provider,
            )

        response = agents_http_handler.create_agent(
            agents_api_config=agents_api_config,
            name=name,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            _is_async=_is_async,
        )

        return response  # type: ignore
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
