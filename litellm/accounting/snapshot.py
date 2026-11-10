from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_json_structure
from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens
from litellm.proxy.spend_tracking.savings import SavingsSpend, compute_savings_spend, marks_gateway_injection

if TYPE_CHECKING:
    from litellm.router import Router

_FIELDS: Final = TypeAdapter(dict[str, object])
_JSON_SAFE_INTEGER: Final = 2**53 - 1


def _mapping(value: object) -> Mapping[str, object] | None:
    return _FIELDS.validate_python(value) if isinstance(value, Mapping) else None


_METADATA_FIELDS: Final = frozenset(
    {
        "usage_object",
        "status",
        "guardrail_information",
        "standard_logging_guardrail_information",
        "compression_savings",
        "internal_call_origin",
        "billed_retry_attempt",
    }
)


_ACCOUNTING_FIELDS: Final = frozenset(
    {
        "request_id",
        "call_type",
        "api_key",
        "spend",
        "prompt_tokens",
        "completion_tokens",
        "startTime",
        "model",
        "model_group",
        "custom_llm_provider",
        "user",
        "team_id",
        "organization_id",
        "end_user",
        "agent_id",
        "request_tags",
        "status",
        "request_duration_ms",
        "mcp_namespaced_tool_name",
    }
)


def build_accounting_payload(
    payload: SpendLogsPayload,
    *,
    entities: Mapping[str, str | None],
    model_access_groups: Sequence[str],
    windows: Sequence[Mapping[str, object]],
    router: "Callable[[], Router | None]",
) -> Mapping[str, object]:
    metadata: Final = _FIELDS.validate_json(payload["metadata"])
    savings: Final = _savings_or_zero(payload, metadata, router)
    accounting_metadata: Final = {
        **{key: _accounting_metadata_field(key, value) for key, value in metadata.items() if key in _METADATA_FIELDS},
        "accounting_entities": entities,
        "accounting_savings": savings._asdict(),
        "accounting_model_access_groups": model_access_groups,
        "accounting_windows": windows,
    }
    return _FIELDS.validate_python(
        safe_json_structure(
            {
                **{
                    key: value.isoformat() if isinstance(value, datetime) else value
                    for key, value in payload.items()
                    if key in _ACCOUNTING_FIELDS
                },
                "metadata": accounting_metadata,
            },
            non_finite_as_null=True,
            integer_limit=_JSON_SAFE_INTEGER,
        )
    )


def _savings_or_zero(
    payload: SpendLogsPayload, metadata: Mapping[str, object], router: "Callable[[], Router | None]"
) -> SavingsSpend:
    """Savings are analytics computed from caller-supplied counts, so one that cannot be computed is stored as zero
    rather than losing the charge it rides with."""
    try:
        return _savings(payload, metadata, router)
    except (ArithmeticError, ValueError, TypeError) as error:
        verbose_proxy_logger.warning(
            "Recording zero savings for request_id=%s, savings could not be computed: %s", payload["request_id"], error
        )
        return SavingsSpend(compression=0.0, prompt_caching=0.0)


def _savings(
    payload: SpendLogsPayload, metadata: Mapping[str, object], router: "Callable[[], Router | None]"
) -> SavingsSpend:
    return compute_savings_spend(
        model=payload["model"],
        custom_llm_provider=payload["custom_llm_provider"],
        compression_saved_tokens=extract_compression_saved_tokens(metadata),
        gateway_injected_cache=marks_gateway_injection(metadata, payload["model_id"]),
        routing_decision=_mapping(metadata.get("routing_decision")),
        model_id=payload["model_id"],
        llm_router=router,
        usage_object=_mapping(metadata.get("usage_object")),
        cost_breakdown=_mapping(metadata.get("cost_breakdown")),
        recorded_autorouter_savings=metadata.get("autorouter_savings"),
        recorded_autorouter_savings_estimate=_mapping(metadata.get("autorouter_savings_estimate")),
        billed_at=payload["endTime"],
    )


_GUARDRAIL_ENTRIES: Final = TypeAdapter(tuple[dict[str, object], ...])
_GUARDRAIL_METADATA_FIELDS: Final = frozenset({"guardrail_information", "standard_logging_guardrail_information"})
_GUARDRAIL_ACCOUNTING_FIELDS: Final = frozenset(
    {
        "guardrail_id",
        "guardrail_name",
        "guardrail_provider",
        "guardrail_usage",
        "guardrail_cost_by_unit",
        "guardrail_cost_in_spend",
    }
)
_GUARDRAIL_RESPONSE: Final = TypeAdapter(dict[str, object])


def _accounting_guardrail_entries(value: object) -> object:
    try:
        entries: Final = _GUARDRAIL_ENTRIES.validate_python((value,) if isinstance(value, dict) else value)
    except ValidationError:
        return None
    return [_accounting_guardrail_entry(entry) for entry in entries]


def _accounting_guardrail_entry(entry: Mapping[str, object]) -> Mapping[str, object]:
    kept: Final = {key: item for key, item in entry.items() if key in _GUARDRAIL_ACCOUNTING_FIELDS}
    response: Final = entry.get("guardrail_response")
    if not isinstance(response, dict) or "tokens_saved" not in response:
        return kept
    return {
        **kept,
        "guardrail_response": {"tokens_saved": _GUARDRAIL_RESPONSE.validate_python(response)["tokens_saved"]},
    }


def _accounting_metadata_field(key: str, value: object) -> object:
    return _accounting_guardrail_entries(value) if key in _GUARDRAIL_METADATA_FIELDS else value
