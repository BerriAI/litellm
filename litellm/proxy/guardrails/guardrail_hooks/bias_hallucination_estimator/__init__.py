from __future__ import annotations

from typing import TYPE_CHECKING

from litellm import logging_callback_manager
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .bias_hallucination_estimator import (
    BiasHallucinationEstimatorGuardrail,
    GuardrailBehaviorConfig,
    GuardrailConfig,
    GuardrailSessionConfig,
)
from .risk_scorer import RiskThresholds, RiskWeights

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: LitellmParams,
    guardrail: Guardrail,
) -> BiasHallucinationEstimatorGuardrail:
    guardrail_id = guardrail["guardrail_id"] if "guardrail_id" in guardrail else None
    callback = BiasHallucinationEstimatorGuardrail(
        guardrail_name=guardrail["guardrail_name"],
        guardrail_id=guardrail_id,
        event_hook=litellm_params.mode,
        config=GuardrailConfig(
            thresholds=RiskThresholds(
                bias_threshold=getattr(litellm_params, "bias_threshold", 0.5),
                hallucination_threshold=getattr(litellm_params, "hallucination_threshold", 0.5),
                flag_threshold=getattr(litellm_params, "risk_flag_threshold", 0.25),
                block_threshold=getattr(litellm_params, "risk_block_threshold", 0.5),
            ),
            weights=RiskWeights(
                bias_weight=getattr(litellm_params, "bias_weight", 0.4),
                hallucination_weight=getattr(litellm_params, "hallucination_weight", 0.6),
            ),
            behavior=GuardrailBehaviorConfig(
                block_on_high_risk=getattr(litellm_params, "block_on_high_risk", True),
                log_only=getattr(litellm_params, "log_only", False),
                check_request=getattr(litellm_params, "check_request", False),
                check_response=getattr(litellm_params, "check_response", True),
                violation_message=getattr(litellm_params, "violation_message", None),
            ),
            session=GuardrailSessionConfig(
                violation_message_template=getattr(litellm_params, "violation_message_template", None),
            ),
        ),
    )
    logging_callback_manager.add_litellm_callback(callback)  # pyright: ignore[reportUnknownMemberType]
    return callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.BIAS_HALLUCINATION_ESTIMATOR.value: initialize_guardrail,
}


guardrail_class_registry = {
    SupportedGuardrailIntegrations.BIAS_HALLUCINATION_ESTIMATOR.value: BiasHallucinationEstimatorGuardrail,
}
