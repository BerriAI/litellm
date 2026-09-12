from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .conduct import ConductGuardrail

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import Guardrail, LitellmParams

DEFAULT_TIMEOUT_SECONDS: Final = 8.0
_NO_EXTRAS: Final[Mapping[str, object]] = MappingProxyType({})


def initialize_guardrail(
    litellm_params: LitellmParams,
    guardrail: Guardrail,
    guardrail_cls: type[CustomGuardrail] = ConductGuardrail,
) -> CustomGuardrail:
    import litellm

    extras: Final = litellm_params.model_extra or _NO_EXTRAS
    _callback: Final = guardrail_cls(
        api_url=litellm_params.api_base,
        agent_token=litellm_params.api_key,
        workspace_id=extras.get("workspace_id"),
        tool_name=extras.get("tool_name", "llm_call"),
        unreachable_fallback=litellm_params.unreachable_fallback,
        timeout=DEFAULT_TIMEOUT_SECONDS if litellm_params.timeout is None else litellm_params.timeout,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        supported_event_hooks=guardrail_cls.get_supported_event_hooks(),
    )
    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry: Final = {  # mutable-ok: module-level registry, built once and never mutated
    SupportedGuardrailIntegrations.CONDUCT.value: initialize_guardrail,
}

guardrail_class_registry: Final = {  # mutable-ok: module-level registry, built once and never mutated
    SupportedGuardrailIntegrations.CONDUCT.value: ConductGuardrail,
}
