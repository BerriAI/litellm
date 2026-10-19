from typing import TYPE_CHECKING, Final

from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.reco import RecoOptionalParams

from .reco import RecoGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _get_config_value(litellm_params: "LitellmParams", optional_params: object, attribute_name: str) -> object:
    """Read one param from optional_params (a dict or a pydantic object), falling back to a top-level
    litellm_params field only when it was set explicitly, since those carry fail-closed defaults.
    Resolves an ``os.environ/<VAR>`` reference the way guardrail api_key/api_base are resolved."""
    nested: Final = (
        optional_params.get(attribute_name)
        if isinstance(optional_params, dict)
        else getattr(optional_params, attribute_name, None)
    )
    explicitly_set_top_level: Final = attribute_name in getattr(litellm_params, "model_fields_set", frozenset())
    value: Final = getattr(litellm_params, attribute_name) if nested is None and explicitly_set_top_level else nested
    if isinstance(value, str) and value.startswith("os.environ/"):
        return get_secret_str(value)
    return value


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> RecoGuardrail:
    import litellm

    optional_params: Final = getattr(litellm_params, "optional_params", None)
    configured: Final = (
        (field_name, _get_config_value(litellm_params, optional_params, field_name))
        for field_name in RecoOptionalParams.model_fields
    )
    reco_params: Final = RecoOptionalParams.model_validate(
        {field_name: value for field_name, value in configured if value is not None}
    )

    _reco_callback: Final = RecoGuardrail(
        guardrail_name=guardrail.get("guardrail_name", ""),
        reco_tenant_id=reco_params.reco_tenant_id,
        api_base=reco_params.api_base,
        unreachable_fallback=reco_params.unreachable_fallback,
        fail_on_error=reco_params.fail_on_error,
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
