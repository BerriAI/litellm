from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, TypeAlias

from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode, SupportedGuardrailIntegrations

from .client import (
    DEFAULT_CLIENT_PLATFORM,
    DEFAULT_ENDPOINT,
    DEFAULT_TIMEOUT_SECONDS,
    ObsidianClient,
    ObsidianSettings,
    validate_client_platform,
)
from .obsidian_policy import ObsidianPolicyGuardrail

_EventHook: TypeAlias = GuardrailEventHooks | list[GuardrailEventHooks] | Mode

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def resolve_settings(litellm_params: "LitellmParams") -> ObsidianSettings:
    gateway_token: Final = litellm_params.api_key or get_secret_str("OBSIDIAN_GATEWAY_TOKEN")
    if not gateway_token:
        raise ValueError("Obsidian policy guardrail: set api_key or the OBSIDIAN_GATEWAY_TOKEN environment variable")

    client_platform: Final = (
        litellm_params.client_platform or get_secret_str("OBSIDIAN_CLIENT_PLATFORM") or DEFAULT_CLIENT_PLATFORM
    ).strip()
    platform_error: Final = validate_client_platform(client_platform)
    if platform_error:
        raise ValueError(f"Obsidian policy guardrail: {platform_error}")

    return ObsidianSettings(
        endpoint=litellm_params.api_base or get_secret_str("OBSIDIAN_ENDPOINT") or DEFAULT_ENDPOINT,
        gateway_token=gateway_token,
        client_platform=client_platform,
        timeout_seconds=litellm_params.timeout or DEFAULT_TIMEOUT_SECONDS,
        fail_open=litellm_params.unreachable_fallback == "fail_open",
    )


def _event_hook(mode: str | Sequence[str] | Mode) -> _EventHook:
    match mode:
        case Mode():
            return mode
        case str():
            return GuardrailEventHooks(mode)
        case _:
            return [GuardrailEventHooks(name) for name in mode]


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> ObsidianPolicyGuardrail:
    import litellm
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # helper is untyped in http_handler
        httpxSpecialProvider,
    )

    http: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback).client
    callback: Final = ObsidianPolicyGuardrail(
        client=ObsidianClient(resolve_settings(litellm_params), http),
        guardrail_name=guardrail["guardrail_name"],
        event_hook=_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on or False,
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
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
]
