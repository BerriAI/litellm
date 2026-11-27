"""
Public entrypoints for running model-generated code in a sandbox.

Low-level lifecycle:
    acreate_container -> arun_code -> adelete_container

High-level convenience:
    acode_interpreter_tool  (ephemeral: create -> run -> delete)

Each entrypoint is `@client`-decorated, so every operation is logged the same
way `litellm.asearch` is.
"""

import asyncio
import time
from collections.abc import Mapping
from typing import Final

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.sandbox.transformation import (
    BaseSandboxConfig,
    CodeExecutionResult,
    ContainerHandle,
    SandboxCost,
    estimate_sandbox_cost,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import SandboxProviders
from litellm.utils import ProviderConfigManager, client

__all__ = [
    "acode_interpreter_tool",
    "acreate_sandbox",
    "adelete_sandbox",
    "arun_code",
    "asandbox_cost",
]

COST_BEFORE_DELETE_TIMEOUT_SECONDS: Final = 10.0
_LITELLM_INTERNAL_KWARGS: Final = {
    "litellm_logging_obj",
    "litellm_call_id",
    "litellm_trace_id",
    "litellm_metadata",
}


def _get_config(provider: str) -> BaseSandboxConfig:
    config: Final = ProviderConfigManager.get_provider_sandbox_config(SandboxProviders(provider))
    if config is None:
        raise ValueError(f"Code execution is not supported for provider: {provider}")
    return config


def _forward_kwargs(kwargs: Mapping[str, object]) -> dict:
    return {k: v for k, v in kwargs.items() if k not in _LITELLM_INTERNAL_KWARGS}


def _update_logging(kwargs: dict, provider: str, operation: str) -> None:
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if logging_obj is None:
        return
    logging_obj.update_from_kwargs(
        kwargs=kwargs,
        model=f"{provider}/{operation}",
        optional_params={},
        litellm_params={"litellm_call_id": kwargs.get("litellm_call_id")},
        custom_llm_provider=provider,
    )


@client
async def acreate_sandbox(
    provider: str,
    template: str | None = None,
    timeout: int | None = None,
    allow_internet_access: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs: object,
) -> ContainerHandle:
    _update_logging(kwargs, provider, "create_sandbox")
    return await _get_config(provider).acreate_sandbox(
        template=template,
        timeout=timeout,
        allow_internet_access=allow_internet_access,
        api_key=api_key,
        api_base=api_base,
        **_forward_kwargs(kwargs),
    )


@client
async def arun_code(
    provider: str,
    container: ContainerHandle | str,
    code: str,
    api_key: str | None = None,
    **kwargs: object,
) -> CodeExecutionResult:
    _update_logging(kwargs, provider, "run_code")
    return await _get_config(provider).arun_code(
        container=container,
        code=code,
        api_key=api_key,
        **_forward_kwargs(kwargs),
    )


async def asandbox_cost(
    provider: str,
    container: ContainerHandle | str,
    api_key: str | None = None,
    api_base: str | None = None,
    client: AsyncHTTPHandler | None = None,
) -> SandboxCost:
    """
    What the sandbox has cost so far. The provider's own dollars when it reports them,
    otherwise an estimate from `sandbox_prices.json`. After delete it returns the final cost.
    """
    if isinstance(container, ContainerHandle) and container.final_cost is not None:
        return container.final_cost
    return await _get_config(provider).aget_sandbox_cost(
        container=container, api_key=api_key, api_base=api_base, client=client
    )


async def _reported_cost(
    config: BaseSandboxConfig,
    container: ContainerHandle,
    api_key: str | None,
    api_base: str | None,
    client: AsyncHTTPHandler | None,
) -> SandboxCost:
    try:
        return await asyncio.wait_for(
            config.aget_sandbox_cost(container=container, api_key=api_key, api_base=api_base, client=client),
            timeout=COST_BEFORE_DELETE_TIMEOUT_SECONDS,
        )
    except (httpx.HTTPError, ValueError, asyncio.TimeoutError) as e:
        verbose_logger.debug("sandbox: failed to read cost before delete: %s", e)
        return SandboxCost()


async def _delete_with_final_cost(
    config: BaseSandboxConfig,
    container: ContainerHandle | str,
    api_key: str | None,
    api_base: str | None,
    client: AsyncHTTPHandler | None,
    kwargs: Mapping[str, object],
) -> bool:
    if not isinstance(container, ContainerHandle) or container.final_cost is not None:
        return await config.adelete_sandbox(
            container=container, api_key=api_key, api_base=api_base, client=client, **_forward_kwargs(kwargs)
        )
    reported: Final = await _reported_cost(config, container, api_key, api_base, client)
    deleted: Final = await config.adelete_sandbox(
        container=container, api_key=api_key, api_base=api_base, client=client, **_forward_kwargs(kwargs)
    )
    container.ended_at = time.time()  # rebind-ok: the handle carries its own final cost after delete
    final: Final = reported if reported.source == "provider" else estimate_sandbox_cost(container)
    container.final_cost = final  # rebind-ok: the handle carries its own final cost after delete
    return deleted


@client
async def adelete_sandbox(
    provider: str,
    container: ContainerHandle | str,
    api_key: str | None = None,
    api_base: str | None = None,
    client: AsyncHTTPHandler | None = None,
    **kwargs: object,
) -> bool:
    _update_logging(kwargs, provider, "delete_sandbox")
    return await _delete_with_final_cost(_get_config(provider), container, api_key, api_base, client, kwargs)


@client
async def acode_interpreter_tool(
    provider: str,
    code: str,
    template: str | None = None,
    timeout: int | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client: AsyncHTTPHandler | None = None,
    **kwargs: object,
) -> CodeExecutionResult:
    _update_logging(kwargs, provider, "code_interpreter_tool")
    config: Final = _get_config(provider)
    forwarded: Final = _forward_kwargs(kwargs)

    container: Final = await config.acreate_sandbox(
        template=template,
        timeout=timeout,
        api_key=api_key,
        api_base=api_base,
        client=client,
        **forwarded,
    )
    try:
        result: Final = await config.arun_code(
            container=container, code=code, api_key=api_key, client=client, **forwarded
        )
    except BaseException:
        await _delete_ephemeral(config, container, api_key, api_base, client, kwargs)
        raise
    await _delete_ephemeral(config, container, api_key, api_base, client, kwargs)
    if container.final_cost is not None and container.final_cost.cost_usd is not None:
        result.hidden_params["response_cost"] = container.final_cost.cost_usd
    return result


async def _delete_ephemeral(
    config: BaseSandboxConfig,
    container: ContainerHandle,
    api_key: str | None,
    api_base: str | None,
    client: AsyncHTTPHandler | None,
    kwargs: Mapping[str, object],
) -> None:
    try:
        await _delete_with_final_cost(config, container, api_key, api_base, client, kwargs)
    except Exception as e:
        verbose_logger.debug("sandbox: failed to delete ephemeral container: %s", e)
