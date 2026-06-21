"""
Public entrypoints for running model-generated code in a sandbox.

Low-level lifecycle:
    acreate_container -> arun_code -> adelete_container

High-level convenience:
    acode_interpreter_tool  (ephemeral: create -> run -> delete)

Each entrypoint is `@client`-decorated, so every operation is logged the same
way `litellm.asearch` is.
"""

from typing import Union

import litellm
from litellm.llms.base_llm.sandbox.transformation import (
    BaseSandboxConfig,
    CodeExecutionResult,
    ContainerHandle,
)
from litellm.types.utils import SandboxProviders
from litellm.utils import ProviderConfigManager, client

__all__ = [
    "acreate_sandbox",
    "arun_code",
    "adelete_sandbox",
    "acode_interpreter_tool",
]

_LITELLM_INTERNAL_KWARGS = {
    "litellm_logging_obj",
    "litellm_call_id",
    "litellm_trace_id",
    "litellm_metadata",
}


def _get_config(provider: str) -> BaseSandboxConfig:
    config = ProviderConfigManager.get_provider_sandbox_config(
        SandboxProviders(provider)
    )
    if config is None:
        raise ValueError(f"Code execution is not supported for provider: {provider}")
    return config


def _forward_kwargs(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in _LITELLM_INTERNAL_KWARGS}


def _update_logging(kwargs: dict, provider: str, operation: str) -> None:
    logging_obj = kwargs.get("litellm_logging_obj")
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
    allow_internet_access: bool = True,
    api_key: str | None = None,
    **kwargs,
) -> ContainerHandle:
    _update_logging(kwargs, provider, "create_sandbox")
    return await _get_config(provider).acreate_sandbox(
        template=template,
        timeout=timeout,
        allow_internet_access=allow_internet_access,
        api_key=api_key,
        **_forward_kwargs(kwargs),
    )


@client
async def arun_code(
    provider: str,
    container: Union[ContainerHandle, str],
    code: str,
    api_key: str | None = None,
    **kwargs,
) -> CodeExecutionResult:
    _update_logging(kwargs, provider, "run_code")
    return await _get_config(provider).arun_code(
        container=container,
        code=code,
        api_key=api_key,
        **_forward_kwargs(kwargs),
    )


@client
async def adelete_sandbox(
    provider: str,
    container: Union[ContainerHandle, str],
    api_key: str | None = None,
    **kwargs,
) -> bool:
    _update_logging(kwargs, provider, "delete_sandbox")
    return await _get_config(provider).adelete_sandbox(
        container=container,
        api_key=api_key,
        **_forward_kwargs(kwargs),
    )


@client
async def acode_interpreter_tool(
    provider: str,
    code: str,
    template: str | None = None,
    timeout: int | None = None,
    api_key: str | None = None,
    **kwargs,
) -> CodeExecutionResult:
    _update_logging(kwargs, provider, "code_interpreter_tool")
    config = _get_config(provider)
    forwarded = _forward_kwargs(kwargs)

    container = await config.acreate_sandbox(
        template=template, timeout=timeout, api_key=api_key, **forwarded
    )
    try:
        return await config.arun_code(
            container=container, code=code, api_key=api_key, **forwarded
        )
    finally:
        try:
            await config.adelete_sandbox(
                container=container, api_key=api_key, **forwarded
            )
        except Exception as e:
            litellm._logging.verbose_logger.debug(
                f"sandbox: failed to delete ephemeral container: {e}"
            )
