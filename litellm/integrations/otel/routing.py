from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, TypeVar

from litellm.litellm_core_utils.core_helpers import as_str_mapping

RoutingAttributeValue: TypeAlias = str | int | float | bool
_Model = TypeVar("_Model")  # rebind-ok: a TypeVar cannot be Final

_ROUTING_SCALAR_FIELDS: Final = frozenset(
    {
        "router_model_name",
        "router_type",
        "router_config_id",
        "router_config_updated_at",
        "router_config_fingerprint",
        "routed_model",
        "cause",
        "tier",
        "tier_label",
        "request_type",
        "score",
        "classifier_model",
        "classifier_cost",
        "classifier_failure_reason",
        "classifier_error_type",
        "classifier_input_budget_tokens",
        "classifier_input_tokens",
        "classifier_input_count_method",
        "classifier_input_truncated",
        "classifier_input_budget_status",
        "classifier_confidence",
        "classifier_primary_rule",
        "classifier_capability_boundary",
        "classifier_p_solve",
        "classifier_calibrated_p_solve",
        "classifier_calibration_version",
        "classifier_efficient_p_solve",
        "classifier_capable_p_solve",
        "classifier_calibrated_efficient_p_solve",
        "classifier_calibrated_capable_p_solve",
        "classifier_max_quality_gap",
        "classifier_prompt_version",
        "classifier_threshold",
        "escalated",
        "context_escalated",
        "context_escalation_original_tier",
        "reasoning_override_min_score",
        "conversation_continuing",
        "savings_baseline_model",
        "savings_baseline_deployment_id",
    }
)


def routing_decision_attributes(decision: Mapping[str, object] | None) -> Mapping[str, RoutingAttributeValue]:
    if not isinstance(decision, Mapping):
        return MappingProxyType({})
    return MappingProxyType(
        {
            f"litellm.routing.{key}": value
            for key in _ROUTING_SCALAR_FIELDS
            if isinstance(value := decision.get(key), (str, int, float, bool))
        }
    )


def routed_model_name(model: _Model, requested_model: object, decision: object) -> _Model | str:
    """The model an auto router picked when ``model`` is only the name the caller requested, else ``model``."""
    fields: Final = as_str_mapping(decision)
    routed: Final = fields.get("routed_model") if fields is not None else None
    if model and model == requested_model and isinstance(routed, str) and routed:
        return routed
    return model
