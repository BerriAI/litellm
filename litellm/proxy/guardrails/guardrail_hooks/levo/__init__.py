from typing import TYPE_CHECKING, Final

import litellm
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .levo import LevoGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

#: Forwarded to LevoGuardrail only when set, so unset values fall through to the
#: constructor defaults rather than being pinned to None.
_OPTIONAL_INIT_FIELDS: Final = (
    "timeout",
    "unreachable_fallback",
    "extra_headers",
    "buffer_streaming_until_moderated",
    "additional_provider_specific_params",
)


def _get_config_value(litellm_params: "LitellmParams", optional_params: object, attribute_name: str) -> object:
    """Read a field from optional_params if present, else from litellm_params.

    The UI submits provider settings under optional_params while YAML puts them
    at the top level, so both shapes have to resolve.
    """
    from_optional: Final = (
        optional_params.get(attribute_name)
        if isinstance(optional_params, dict)
        else getattr(optional_params, attribute_name, None)
        if optional_params is not None
        else None
    )
    if from_optional is not None:
        return from_optional
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    optional_params: Final = getattr(litellm_params, "optional_params", None)

    api_base: Final = litellm_params.api_base
    if not api_base:
        raise ValueError(
            "api_base is required for levo — point it at your Levo AI Gateway, e.g. http://levo-gateway:8080"
        )

    kwargs: Final[dict[str, object]] = {
        field: value
        for field in _OPTIONAL_INIT_FIELDS
        for value in [_get_config_value(litellm_params, optional_params, field)]
        if value is not None
    }

    _callback: Final = LevoGuardrail(
        api_base=api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", "levo"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        **kwargs,
    )

    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.LEVO.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.LEVO.value: LevoGuardrail,
}
