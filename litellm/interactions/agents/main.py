"""
LiteLLM Agents API - Main Module

Usage:
    import litellm

    # Create
    response = litellm.interactions.agents.create(
        name="waverunner",
        custom_llm_provider="gemini",
        api_key="...",
        base_agent="gemini-2.5-flash",
        instructions="You are a helpful assistant.",
    )

    # List
    response = litellm.interactions.agents.list(api_key="...", custom_llm_provider="gemini")

    # Get
    response = litellm.interactions.agents.get(name="waverunner", api_key="...")

    # Delete
    result = litellm.interactions.agents.delete(name="waverunner", api_key="...")

    # List versions
    result = litellm.interactions.agents.list_versions(name="waverunner", api_key="...")

    # Async versions: acreate, alist, aget, adelete, alist_versions
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
from litellm.types.agents import (
    AgentCreateResponse,
    AgentDeleteResult,
    AgentListResponse,
    AgentVersionsResponse,
)
from litellm.types.interactions import InteractionEnvironment
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import client

# ------------------------------------------------------------------ #
# Shared helpers                                                       #
# ------------------------------------------------------------------ #


def _get_agents_api_config(custom_llm_provider: str):
    config = get_provider_agents_api_config(custom_llm_provider)
    if config is None:
        raise litellm.BadRequestError(
            message=(
                f"Provider '{custom_llm_provider}' does not have a native "
                "agents API. Use the proxy POST /v1/agents endpoint to store "
                "agents locally."
            ),
            model="",
            llm_provider=custom_llm_provider,
        )
    return config


def _make_logging_obj(
    kwargs: Dict[str, Any],
    model: str,
    custom_llm_provider: str,
    call_type: str,
    optional_params: Dict[str, Any],
) -> LiteLLMLoggingObj:
    litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
    litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
    litellm_logging_obj.update_from_kwargs(
        kwargs=kwargs,
        model=model,
        optional_params=optional_params,
        litellm_params={"litellm_call_id": litellm_call_id},
        custom_llm_provider=custom_llm_provider,
    )
    return litellm_logging_obj


# ================================================================== #
# CREATE                                                               #
# ================================================================== #


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
    """Async: Create a managed agent on the provider side."""
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate_agent"] = True
        func = partial(
            create,
            name=name,
            base_agent=base_agent,
            instructions=instructions,
            base_environment=base_environment,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            **kwargs,
        )
        ctx = contextvars.copy_context()
        init_response = await loop.run_in_executor(None, partial(ctx.run, func))
        if asyncio.iscoroutine(init_response):
            return await init_response
        return init_response
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider or "gemini",
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
        base_environment: Environment to fork from — an env_id string or a
            dict like ``{"type": "remote", "sources": [...]}``.
        custom_llm_provider: Provider to use, e.g. "gemini".
        extra_headers: Additional HTTP headers.
        extra_body: Additional request body fields.
        timeout: Request timeout.
        **kwargs: Forwarded to GenericLiteLLMParams (api_key, api_base, etc.).
    """
    local_vars = locals()
    custom_llm_provider = (
        custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    )
    try:
        _is_async = kwargs.pop("acreate_agent", False) is True
        if base_agent is not None:
            kwargs["base_agent"] = base_agent
        if instructions is not None:
            kwargs["instructions"] = instructions
        if base_environment is not None:
            kwargs["base_environment"] = base_environment
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params = GenericLiteLLMParams(**kwargs)
        logging_obj = _make_logging_obj(
            kwargs, name, custom_llm_provider, "create_agent", {}
        )
        config = _get_agents_api_config(custom_llm_provider)
        return agents_http_handler.create_agent(
            agents_api_config=config,
            name=name,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ================================================================== #
# LIST                                                                 #
# ================================================================== #


@client
async def alist(
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> AgentListResponse:
    """Async: List all agents on the provider side."""
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["alist_agents"] = True
        func = partial(
            list,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            timeout=timeout,
            **kwargs,
        )
        ctx = contextvars.copy_context()
        init_response = await loop.run_in_executor(None, partial(ctx.run, func))
        if asyncio.iscoroutine(init_response):
            return await init_response
        return init_response
    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def list(
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> Union[AgentListResponse, Coroutine[Any, Any, AgentListResponse]]:
    """Sync: List all agents on the provider side."""
    local_vars = locals()
    custom_llm_provider = (
        custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    )
    try:
        _is_async = kwargs.pop("alist_agents", False) is True
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params = GenericLiteLLMParams(**kwargs)
        logging_obj = _make_logging_obj(
            kwargs, "", custom_llm_provider, "list_agents", {}
        )
        config = _get_agents_api_config(custom_llm_provider)
        return agents_http_handler.list_agents(
            agents_api_config=config,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ================================================================== #
# GET                                                                  #
# ================================================================== #


@client
async def aget(
    name: str,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> AgentCreateResponse:
    """Async: Get a specific agent by name."""
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aget_agent"] = True
        func = partial(
            get,
            name=name,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            timeout=timeout,
            **kwargs,
        )
        ctx = contextvars.copy_context()
        init_response = await loop.run_in_executor(None, partial(ctx.run, func))
        if asyncio.iscoroutine(init_response):
            return await init_response
        return init_response
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def get(
    name: str,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> Union[AgentCreateResponse, Coroutine[Any, Any, AgentCreateResponse]]:
    """Sync: Get a specific agent by name."""
    local_vars = locals()
    custom_llm_provider = (
        custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    )
    try:
        _is_async = kwargs.pop("aget_agent", False) is True
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params = GenericLiteLLMParams(**kwargs)
        logging_obj = _make_logging_obj(
            kwargs, name, custom_llm_provider, "get_agent", {"name": name}
        )
        config = _get_agents_api_config(custom_llm_provider)
        return agents_http_handler.get_agent(
            agents_api_config=config,
            name=name,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ================================================================== #
# DELETE                                                               #
# ================================================================== #


@client
async def adelete(
    name: str,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> AgentDeleteResult:
    """Async: Delete a specific agent by name."""
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["adelete_agent"] = True
        func = partial(
            delete,
            name=name,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            timeout=timeout,
            **kwargs,
        )
        ctx = contextvars.copy_context()
        init_response = await loop.run_in_executor(None, partial(ctx.run, func))
        if asyncio.iscoroutine(init_response):
            return await init_response
        return init_response
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def delete(
    name: str,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> Union[AgentDeleteResult, Coroutine[Any, Any, AgentDeleteResult]]:
    """Sync: Delete a specific agent by name."""
    local_vars = locals()
    custom_llm_provider = (
        custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    )
    try:
        _is_async = kwargs.pop("adelete_agent", False) is True
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params = GenericLiteLLMParams(**kwargs)
        logging_obj = _make_logging_obj(
            kwargs, name, custom_llm_provider, "delete_agent", {"name": name}
        )
        config = _get_agents_api_config(custom_llm_provider)
        return agents_http_handler.delete_agent(
            agents_api_config=config,
            name=name,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ================================================================== #
# LIST VERSIONS                                                        #
# ================================================================== #


@client
async def alist_versions(
    name: str,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> AgentVersionsResponse:
    """Async: List versions of a specific agent."""
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["alist_agent_versions"] = True
        func = partial(
            list_versions,
            name=name,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            timeout=timeout,
            **kwargs,
        )
        ctx = contextvars.copy_context()
        init_response = await loop.run_in_executor(None, partial(ctx.run, func))
        if asyncio.iscoroutine(init_response):
            return await init_response
        return init_response
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def list_versions(
    name: str,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> Union[AgentVersionsResponse, Coroutine[Any, Any, AgentVersionsResponse]]:
    """Sync: List versions of a specific agent."""
    local_vars = locals()
    custom_llm_provider = (
        custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    )
    try:
        _is_async = kwargs.pop("alist_agent_versions", False) is True
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params = GenericLiteLLMParams(**kwargs)
        logging_obj = _make_logging_obj(
            kwargs, name, custom_llm_provider, "list_agent_versions", {"name": name}
        )
        config = _get_agents_api_config(custom_llm_provider)
        return agents_http_handler.list_agent_versions(
            agents_api_config=config,
            name=name,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=name,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
