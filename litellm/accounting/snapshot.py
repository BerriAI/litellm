import copy
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens
from litellm.proxy.spend_tracking.savings import compute_savings_spend, marks_gateway_injection

if TYPE_CHECKING:
    from litellm.router import Router

_FIELDS: Final = TypeAdapter(dict[str, object])


def _mapping(value: object) -> Mapping[str, object] | None:
    return _FIELDS.validate_python(value) if isinstance(value, Mapping) else None


def accounting_metadata(
    payload: SpendLogsPayload,
    *,
    entities: Mapping[str, str | None],
    model_access_groups: Sequence[str],
    windows: Sequence[Mapping[str, object]],
    router: "Callable[[], Router | None]",
) -> str:
    metadata: Final = _FIELDS.validate_json(payload["metadata"])
    savings: Final = compute_savings_spend(
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
    return safe_dumps(
        {
            **metadata,
            "accounting_entities": entities,
            "accounting_savings": savings._asdict(),
            "accounting_model_access_groups": model_access_groups,
            "accounting_windows": windows,
        },
        non_finite_as_null=True,
    )


ACCOUNTING_METADATA_FIELDS: Final = frozenset(
    {
        "accounting_windows",
        "accounting_lean",
        "accounting_model_access_groups",
        "accounting_savings",
        "accounting_entities",
        "accounting_proxy_budget_name",
        "user_api_key_project_id",
        "usage_object",
        "status",
        "guardrail_information",
        "standard_logging_guardrail_information",
        "compression_savings",
        "internal_call_origin",
        "billed_retry_attempt",
    }
)


def lean_accounting_payload(payload: SpendLogsPayload) -> SpendLogsPayload:
    lean: Final = copy.deepcopy(payload)
    metadata: Final = _FIELDS.validate_json(payload["metadata"])
    lean["messages"] = None
    lean["response"] = None
    lean["proxy_server_request"] = None
    lean["requester_ip_address"] = None
    lean["session_id"] = None
    lean["api_base"] = ""
    lean["cache_key"] = ""
    lean["metadata"] = safe_dumps(
        {
            **{key: value for key, value in metadata.items() if key in ACCOUNTING_METADATA_FIELDS},
            "accounting_lean": True,
        },
        non_finite_as_null=True,
    )
    return lean


def is_lean_accounting_payload(payload: Mapping[str, object]) -> bool:
    metadata: Final = payload.get("metadata")
    fields: Final = _FIELDS.validate_json(metadata) if isinstance(metadata, str) else _mapping(metadata)
    return fields is not None and fields.get("accounting_lean") is True
