"""
LiteLLM Agents API - Main Module

Usage:
    import litellm

    # Create
    response = litellm.interactions.agents.create(
        id="waverunner",
        custom_llm_provider="gemini",
        api_key="...",
        base_agent="antigravity-preview-09-2026",
        system_instruction="You are a helpful assistant.",
        agent_config={"type": "antigravity", "model": "gemini-3.8-flash"},
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
from collections.abc import Coroutine, Mapping, Sequence
from functools import partial
from typing import Any, Final

import httpx
from pydantic import ValidationError

import litellm
from litellm.interactions.agents.http_handler import agents_http_handler
from litellm.interactions.agents.utils import get_provider_agents_api_config
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.agents import (
    AgentCreateRequest,
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
    config: Final = get_provider_agents_api_config(custom_llm_provider)
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
    kwargs: dict[str, Any],
    model: str,
    custom_llm_provider: str,
    call_type: str,
    optional_params: dict[str, object],
) -> LiteLLMLoggingObj:
    litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
    litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
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


def _alias_conflict(field: str, value: str | None, alias: str, alias_value: str | None) -> str | None:
    if value is None or alias_value is None or value == alias_value:
        return None
    return f"'{alias}' is an alias of '{field}', but the request gives them different values. Pass only '{field}'."


def _create_request_or_error(
    id: str | None,
    name: str | None,
    base_agent: str | None,
    agent_config: Mapping[str, object] | None,
    system_instruction: str | None,
    instructions: str | None,
    base_environment: InteractionEnvironment | None,
    description: str | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> AgentCreateRequest | str:
    conflict: Final = _alias_conflict("id", id, "name", name) or _alias_conflict(
        "system_instruction", system_instruction, "instructions", instructions
    )
    if conflict is not None:
        return conflict
    agent_id: Final = id if id is not None else name
    if agent_id is None:
        return "Missing required parameter: 'id' (or its alias 'name')."
    try:
        return AgentCreateRequest(
            id=agent_id,
            base_agent=base_agent,
            agent_config=agent_config,
            system_instruction=system_instruction if system_instruction is not None else instructions,
            base_environment=base_environment,
            description=description,
            tools=tools,
        )
    except ValidationError as e:
        return str(e)


@client
async def acreate(
    id: str | None = None,
    base_agent: str | None = None,
    instructions: str | None = None,
    base_environment: InteractionEnvironment | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: Mapping[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    *,
    name: str | None = None,
    agent_config: Mapping[str, object] | None = None,
    system_instruction: str | None = None,
    description: str | None = None,
    tools: Sequence[Mapping[str, object]] | None = None,
    **kwargs,
) -> AgentCreateResponse:
    """Async: Create a managed agent on the provider side."""
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acreate_agent"] = True
        func: Final = partial(
            create,
            id=id,
            base_agent=base_agent,
            instructions=instructions,
            base_environment=base_environment,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            name=name,
            agent_config=agent_config,
            system_instruction=system_instruction,
            description=description,
            tools=tools,
            **kwargs,
        )
        ctx: Final = contextvars.copy_context()
        init_response: Final = await loop.run_in_executor(None, partial(ctx.run, func))
        if asyncio.iscoroutine(init_response):
            return await init_response
        return init_response
    except Exception as e:
        raise litellm.exception_type(
            model=id or name or "",
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def create(
    id: str | None = None,
    base_agent: str | None = None,
    instructions: str | None = None,
    base_environment: InteractionEnvironment | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: Mapping[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    *,
    name: str | None = None,
    agent_config: Mapping[str, object] | None = None,
    system_instruction: str | None = None,
    description: str | None = None,
    tools: Sequence[Mapping[str, object]] | None = None,
    **kwargs,
) -> AgentCreateResponse | Coroutine[object, object, AgentCreateResponse]:
    """
    Sync: Create a managed agent on the provider side.

    Args:
        id: Identifier for the agent. Exactly one of ``id`` or its alias ``name`` is required.
        base_agent: Base agent to derive from (e.g. "antigravity-preview-09-2026").
        instructions: Alias of ``system_instruction``.
        base_environment: Environment to fork from — an env_id string or a
            dict like ``{"type": "remote", "sources": [...]}``.
        custom_llm_provider: Provider to use, e.g. "gemini".
        extra_headers: Additional HTTP headers.
        extra_body: Additional request body fields.
        timeout: Request timeout.
        name: Alias of ``id``.
        agent_config: Agent runtime config, e.g. ``{"type": "antigravity", "model": "gemini-3.8-flash"}``.
        system_instruction: System instruction for the agent.
        description: Description of the agent for developers.
        tools: Tools available to the agent, e.g. ``[{"type": "code_execution"}]``.
        **kwargs: Forwarded to GenericLiteLLMParams (api_key, api_base, etc.).
    """
    local_vars: Final = locals()
    custom_llm_provider = custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    try:
        _is_async: Final = kwargs.pop("acreate_agent", False) is True
        create_request: Final = _create_request_or_error(
            id=id,
            name=name,
            base_agent=base_agent,
            agent_config=agent_config,
            system_instruction=system_instruction,
            instructions=instructions,
            base_environment=base_environment,
            description=description,
            tools=tools,
        )
        if isinstance(create_request, str):
            raise litellm.BadRequestError(
                message=create_request, model=id or name or "", llm_provider=custom_llm_provider
            )
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params: Final = GenericLiteLLMParams(**kwargs)
        logging_obj: Final = _make_logging_obj(kwargs, create_request.id, custom_llm_provider, "create_agent", {})
        config: Final = _get_agents_api_config(custom_llm_provider)
        return agents_http_handler.create_agent(
            agents_api_config=config,
            create_request=create_request,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=id or name or "",
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
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    **kwargs,
) -> AgentListResponse:
    """Async: List all agents on the provider side."""
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["alist_agents"] = True
        func: Final = partial(
            list,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            timeout=timeout,
            **kwargs,
        )
        ctx: Final = contextvars.copy_context()
        init_response: Final = await loop.run_in_executor(None, partial(ctx.run, func))
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
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    **kwargs,
) -> AgentListResponse | Coroutine[object, object, AgentListResponse]:
    """Sync: List all agents on the provider side."""
    local_vars: Final = locals()
    custom_llm_provider = custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    try:
        _is_async: Final = kwargs.pop("alist_agents", False) is True
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params: Final = GenericLiteLLMParams(**kwargs)
        logging_obj: Final = _make_logging_obj(kwargs, "", custom_llm_provider, "list_agents", {})
        config: Final = _get_agents_api_config(custom_llm_provider)
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
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    **kwargs,
) -> AgentCreateResponse:
    """Async: Get a specific agent by name."""
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aget_agent"] = True
        func: Final = partial(
            get,
            name=name,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            timeout=timeout,
            **kwargs,
        )
        ctx: Final = contextvars.copy_context()
        init_response: Final = await loop.run_in_executor(None, partial(ctx.run, func))
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
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    **kwargs,
) -> AgentCreateResponse | Coroutine[object, object, AgentCreateResponse]:
    """Sync: Get a specific agent by name."""
    local_vars: Final = locals()
    custom_llm_provider = custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    try:
        _is_async: Final = kwargs.pop("aget_agent", False) is True
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params: Final = GenericLiteLLMParams(**kwargs)
        logging_obj: Final = _make_logging_obj(kwargs, name, custom_llm_provider, "get_agent", {"name": name})
        config: Final = _get_agents_api_config(custom_llm_provider)
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
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    **kwargs,
) -> AgentDeleteResult:
    """Async: Delete a specific agent by name."""
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["adelete_agent"] = True
        func: Final = partial(
            delete,
            name=name,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            timeout=timeout,
            **kwargs,
        )
        ctx: Final = contextvars.copy_context()
        init_response: Final = await loop.run_in_executor(None, partial(ctx.run, func))
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
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    **kwargs,
) -> AgentDeleteResult | Coroutine[object, object, AgentDeleteResult]:
    """Sync: Delete a specific agent by name."""
    local_vars: Final = locals()
    custom_llm_provider = custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    try:
        _is_async: Final = kwargs.pop("adelete_agent", False) is True
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params: Final = GenericLiteLLMParams(**kwargs)
        logging_obj: Final = _make_logging_obj(kwargs, name, custom_llm_provider, "delete_agent", {"name": name})
        config: Final = _get_agents_api_config(custom_llm_provider)
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
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    **kwargs,
) -> AgentVersionsResponse:
    """Async: List versions of a specific agent."""
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["alist_agent_versions"] = True
        func: Final = partial(
            list_versions,
            name=name,
            custom_llm_provider=custom_llm_provider or "gemini",
            extra_headers=extra_headers,
            timeout=timeout,
            **kwargs,
        )
        ctx: Final = contextvars.copy_context()
        init_response: Final = await loop.run_in_executor(None, partial(ctx.run, func))
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
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    **kwargs,
) -> AgentVersionsResponse | Coroutine[object, object, AgentVersionsResponse]:
    """Sync: List versions of a specific agent."""
    local_vars: Final = locals()
    custom_llm_provider = custom_llm_provider or kwargs.get("custom_llm_provider") or "gemini"
    try:
        _is_async: Final = kwargs.pop("alist_agent_versions", False) is True
        kwargs.setdefault("custom_llm_provider", custom_llm_provider)
        litellm_params: Final = GenericLiteLLMParams(**kwargs)
        logging_obj: Final = _make_logging_obj(kwargs, name, custom_llm_provider, "list_agent_versions", {"name": name})
        config: Final = _get_agents_api_config(custom_llm_provider)
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
