from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .ibm_detector import IBMGuardrailDetector

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm

    if not litellm_params.auth_token:
        raise ValueError("IBM Guardrails: auth_token is required")
    if not litellm_params.base_url:
        raise ValueError("IBM Guardrails: base_url is required")
    if not litellm_params.detector_id:
        raise ValueError("IBM Guardrails: detector_id is required")

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("IBM Guardrails: guardrail_name is required")

    # Get optional params
    detector_params = getattr(litellm_params, "detector_params", {})
    score_threshold = getattr(litellm_params, "score_threshold", None)
    block_on_detection = getattr(litellm_params, "block_on_detection", True)
    verify_ssl = getattr(litellm_params, "verify_ssl", True)
    is_detector_server = litellm_params.is_detector_server
    if is_detector_server is None:
        is_detector_server = True

    ibm_guardrail = IBMGuardrailDetector(
        guardrail_name=guardrail_name,
        auth_token=litellm_params.auth_token,
        base_url=litellm_params.base_url,
        detector_id=litellm_params.detector_id,
        is_detector_server=is_detector_server,
        detector_params=detector_params,
        score_threshold=score_threshold,
        block_on_detection=block_on_detection,
        verify_ssl=verify_ssl,
        default_on=litellm_params.default_on,
        event_hook=litellm_params.mode,
    )

    litellm.logging_callback_manager.add_litellm_callback(ibm_guardrail)
    return ibm_guardrail


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.IBM_GUARDRAILS.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.IBM_GUARDRAILS.value: IBMGuardrailDetector,
}


__all__ = ["IBMGuardrailDetector", "initialize_guardrail"]
