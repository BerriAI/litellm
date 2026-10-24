from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, TypeAdapter

from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.reco import RecoOptionalParams

from .reco import RecoGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

_CONFIG_MAPPING: Final = TypeAdapter(Mapping[str, object])


def _optional_params_mapping(optional_params: object) -> Mapping[str, object]:
    """optional_params arrives as a pydantic object from YAML, or as a plain dict when reconciled from the DB."""
    if optional_params is None:
        return MappingProxyType({})
    raw: Final = (
        optional_params.model_dump(exclude_unset=True) if isinstance(optional_params, BaseModel) else optional_params
    )
    return _CONFIG_MAPPING.validate_python(raw)


def _first_configured(field_name: str, sources: tuple[Mapping[str, object], ...]) -> object:
    value: Final = next((source[field_name] for source in sources if source.get(field_name) is not None), None)
    if isinstance(value, str) and value.startswith("os.environ/"):
        return get_secret_str(value)
    return value


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> RecoGuardrail:
    import litellm

    # Top-level fields only count when set explicitly: their defaults are fail-closed and would flip Reco's.
    explicit_top_level: Final = _CONFIG_MAPPING.validate_python(
        litellm_params.model_dump(include=litellm_params.model_fields_set & RecoOptionalParams.model_fields.keys())
    )
    sources: Final = (_optional_params_mapping(litellm_params.optional_params), explicit_top_level)
    configured: Final = (
        (field_name, _first_configured(field_name, sources)) for field_name in RecoOptionalParams.model_fields
    )
    reco_params: Final = RecoOptionalParams.model_validate(
        {field_name: value for field_name, value in configured if value is not None}
    )

    _reco_callback: Final = RecoGuardrail(
        guardrail_name=guardrail["guardrail_name"],
        reco_tenant_id=reco_params.reco_tenant_id,
        api_base=reco_params.api_base,
        unreachable_fallback=reco_params.unreachable_fallback,
        fail_on_error=reco_params.fail_on_error,
        event_hook=litellm_params.mode,
        default_on=bool(litellm_params.default_on),
    )
    litellm.logging_callback_manager.add_litellm_callback(_reco_callback)

    return _reco_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.RECO.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.RECO.value: RecoGuardrail,
}
