from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, TypeAlias

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode, SupportedGuardrailIntegrations

from .client import (
    DEFAULT_CLIENT_PLATFORM,
    DEFAULT_ENDPOINT,
    DEFAULT_TIMEOUT_SECONDS,
    InvalidSettings,
    ObsidianClient,
    ObsidianSettings,
    validate_client_platform,
)
from .obsidian_policy import ObsidianPolicyGuardrail

_EventHook: TypeAlias = GuardrailEventHooks | list[GuardrailEventHooks] | Mode

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def resolve_settings(litellm_params: "LitellmParams") -> ObsidianSettings | InvalidSettings:
    fail_open: Final = litellm_params.unreachable_fallback == "fail_open"
    gateway_token: Final = litellm_params.api_key or get_secret_str("OBSIDIAN_GATEWAY_TOKEN")
    if not gateway_token:
        return InvalidSettings("set api_key or the OBSIDIAN_GATEWAY_TOKEN environment variable", fail_open)

    client_platform: Final = (
        litellm_params.client_platform or get_secret_str("OBSIDIAN_CLIENT_PLATFORM") or DEFAULT_CLIENT_PLATFORM
    ).strip()
    platform_error: Final = validate_client_platform(client_platform)
    if platform_error:
        return InvalidSettings(platform_error, fail_open)

    return ObsidianSettings(
        endpoint=litellm_params.api_base or get_secret_str("OBSIDIAN_ENDPOINT") or DEFAULT_ENDPOINT,
        gateway_token=gateway_token,
        client_platform=client_platform,
        timeout_seconds=litellm_params.timeout or DEFAULT_TIMEOUT_SECONDS,
        fail_open=fail_open,
    )


def _event_hook(mode: str | Sequence[str] | Mode) -> _EventHook | None:
    supported: Final = ObsidianPolicyGuardrail.get_supported_event_hooks()
    match mode:
        case Mode():
            return mode
        case str():
            return GuardrailEventHooks(mode) if mode in supported else None
        case _:
            return [GuardrailEventHooks(name) for name in mode] if mode and all(n in supported for n in mode) else None


def build_guardrail(
    litellm_params: "LitellmParams", guardrail_name: str, http: httpx.AsyncClient
) -> ObsidianPolicyGuardrail:
    """An invalid configuration still yields a guardrail, one that applies unreachable_fallback to every call.

    LiteLLM skips a guardrail whose initializer raises and starts without it, which would leave MCP calls unchecked.
    """
    event_hook: Final = _event_hook(litellm_params.mode)
    settings: Final = (
        resolve_settings(litellm_params)
        if event_hook is not None
        else InvalidSettings(
            f"mode must be pre_mcp_call, post_mcp_call or both, got {litellm_params.mode!r}",
            litellm_params.unreachable_fallback == "fail_open",
        )
    )
    if isinstance(settings, InvalidSettings):
        verbose_proxy_logger.error(
            "Obsidian policy guardrail '%s' is misconfigured, every checked MCP call will be %s until it is fixed: %s",
            guardrail_name,
            "allowed" if settings.fail_open else "denied",
            settings.reason,
        )
    return ObsidianPolicyGuardrail(
        client=ObsidianClient(settings, http),
        guardrail_name=guardrail_name,
        event_hook=event_hook,
        default_on=litellm_params.default_on or False,
    )


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> ObsidianPolicyGuardrail:
    import litellm
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # helper is untyped in http_handler
        httpxSpecialProvider,
    )

    callback: Final = build_guardrail(
        litellm_params,
        guardrail["guardrail_name"],
        get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback).client,
    )
    litellm.logging_callback_manager.add_litellm_callback(callback)
    litellm.logging_callback_manager.add_litellm_success_callback(callback)
    return callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.OBSIDIAN_POLICY.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.OBSIDIAN_POLICY.value: ObsidianPolicyGuardrail,
}


__all__ = [
    "ObsidianPolicyGuardrail",
    "build_guardrail",
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
]
