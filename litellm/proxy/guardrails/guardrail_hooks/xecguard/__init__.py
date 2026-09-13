from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .xecguard import XecGuardGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
):
    import litellm

    _cb: Final = XecGuardGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        xecguard_model=litellm_params.xecguard_model,
        policy_names=litellm_params.policy_names,
        apply_to_aliases=litellm_params.apply_to_aliases,
        except_aliases=litellm_params.except_aliases,
        send_meta=litellm_params.send_meta,
        meta_data_fields=litellm_params.meta_data_fields,
        meta_identity_format=litellm_params.meta_identity_format,
        block_on_error=litellm_params.block_on_error,
        grounding_strictness=litellm_params.grounding_strictness,
        guardrail_name=guardrail.get(
            "guardrail_name",
            "",
        ),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(
        _cb,
    )

    return _cb


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.XECGUARD.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.XECGUARD.value: XecGuardGuardrail,
}
