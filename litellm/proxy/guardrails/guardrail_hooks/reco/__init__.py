from typing import TYPE_CHECKING, Final

from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .reco import RecoGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_config_value(optional_params: object, attribute_name: str) -> str | None:
    """Read one param from a dict or a pydantic object, resolving an
    ``os.environ/<VAR>`` reference the way guardrail api_key/api_base are resolved."""
    if isinstance(optional_params, dict):
        value = optional_params.get(attribute_name)
    else:
        value = getattr(optional_params, attribute_name, None)
    if isinstance(value, str) and value.startswith("os.environ/"):
        return get_secret_str(value)
    return value


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    optional_params: Final = getattr(litellm_params, "optional_params", None)

    _reco_callback: Final = RecoGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        reco_tenant_id=_get_config_value(optional_params, "reco_tenant_id"),
        api_base=_get_config_value(optional_params, "api_base"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_reco_callback)

    return _reco_callback


guardrail_initializer_registry: Final = {  # mutable-ok: guardrail auto-discovery requires an actual dict instance
    SupportedGuardrailIntegrations.RECO.value: initialize_guardrail,
}


guardrail_class_registry: Final = {  # mutable-ok: guardrail auto-discovery requires an actual dict instance
    SupportedGuardrailIntegrations.RECO.value: RecoGuardrail,
}
