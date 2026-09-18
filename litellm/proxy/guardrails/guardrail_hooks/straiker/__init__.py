from typing import TYPE_CHECKING, Final

import litellm
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .straiker import StraikerGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

_OPTIONAL_INIT_FIELDS: Final = (
    "timeout",
    "max_retries",
    "initial_backoff",
    "max_backoff",
    "unreachable_fallback",
    "fail_on_error",
    "max_payload_bytes",
    "custom_headers",
    "metadata",
    "verbose",
)


def _get_config_value(litellm_params: "LitellmParams", optional_params: object, attribute_name: str) -> object:
    if optional_params is not None:
        if isinstance(optional_params, dict):
            value = optional_params.get(attribute_name)
        else:
            value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    optional_params: Final = getattr(litellm_params, "optional_params", None)
    api_key: Final = litellm_params.api_key
    if not api_key:
        raise ValueError("api_key is required for straiker")

    api_base: Final = litellm_params.api_base or "https://api.prod.straiker.ai"
    default_app: Final = getattr(litellm_params, "default_app", None) or getattr(litellm_params, "source", None)
    source: Final = default_app if isinstance(default_app, str) and default_app else "LiteLLM Gateway"
    kwargs: Final[dict[str, object]] = {
        field: value
        for field in _OPTIONAL_INIT_FIELDS
        for value in [_get_config_value(litellm_params, optional_params, field)]
        if value is not None
    }
    _callback: Final = StraikerGuardrail(
        api_key=api_key,
        api_base=api_base if isinstance(api_base, str) else "https://api.prod.straiker.ai",
        source=source,
        guardrail_name=guardrail.get("guardrail_name", "straiker"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        **kwargs,
    )

    litellm.logging_callback_manager.add_litellm_callback(_callback)
    return _callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.STRAIKER.value: initialize_guardrail,
}

guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.STRAIKER.value: StraikerGuardrail,
}
