from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.integrations.custom_guardrail import GuardrailSetupError
from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.collieai import DEFAULT_API_BASE

from .collieai import CollieAiGuardrail

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import Guardrail, LitellmParams

_NO_EXTRAS: Final[Mapping[str, object]] = MappingProxyType({})


def initialize_guardrail(
    litellm_params: LitellmParams,
    guardrail: Guardrail,
    guardrail_cls: Callable[..., CustomGuardrail] = CollieAiGuardrail,
) -> CustomGuardrail:
    import litellm
    from litellm.secret_managers.main import get_secret_str

    extras: Final = MappingProxyType(
        {key: value for key, value in (litellm_params.model_extra or _NO_EXTRAS).items() if value is not None}
    )
    try:
        _callback: Final = guardrail_cls(
            **{
                **extras,
                "api_base": litellm_params.api_base or DEFAULT_API_BASE,
                "api_key": litellm_params.api_key or get_secret_str("COLLIEAI_API_KEY"),
                "guardrail_name": guardrail["guardrail_name"],
                "event_hook": litellm_params.mode,
                "default_on": litellm_params.default_on,
                "stream_scope": litellm_params.stream_scope,
            }
        )
    except Exception as setup_error:
        raise GuardrailSetupError(str(setup_error)) from setup_error
    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.COLLIEAI.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.COLLIEAI.value: CollieAiGuardrail,
}
