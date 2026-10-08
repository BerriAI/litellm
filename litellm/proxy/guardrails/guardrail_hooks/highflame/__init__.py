from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .highflame import HighflameGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

_NO_EXTRAS: Final[Mapping[str, object]] = MappingProxyType({})


def _optional_str(extras: Mapping[str, object], key: str) -> str | None:
    value: Final = extras.get(key)
    return value if isinstance(value, str) and value else None


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> HighflameGuardrail:
    import litellm

    extras: Final = litellm_params.model_extra or _NO_EXTRAS
    _highflame_callback: Final = HighflameGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        token_url=_optional_str(extras, "token_url"),
        shield_mode=_optional_str(extras, "shield_mode"),
        unreachable_fallback=litellm_params.unreachable_fallback,
        timeout=litellm_params.timeout,
        streaming_buffer_until_moderated=extras.get("streaming_buffer_until_moderated") is not False,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(_highflame_callback)
    return _highflame_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.HIGHFLAME.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.HIGHFLAME.value: HighflameGuardrail,
}
