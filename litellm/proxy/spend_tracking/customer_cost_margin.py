import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.llm_cost_calc.finalization import FinalizedCost, finalize_cost
from litellm.proxy.common_utils.customer_usage import (
    CUSTOMER_USAGE_METADATA_KEY,
    CustomerUsageAttribution,
    customer_usage_key_fields,
    customer_usage_requested,
)
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth
from litellm.types.utils import StandardLoggingPayload


class _RecordedSurcharge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)

    customer_margin_amount: StrictFloat | StrictInt


_RECORDED_SURCHARGE: Final = TypeAdapter(_RecordedSurcharge)
_REQUEST_FIELDS: Final = TypeAdapter(Mapping[str, object])
CUSTOMER_MARGIN_KEY: Final = "customer"
MAX_CUSTOMER_MARGIN: Final = 10


def is_valid_customer_margin(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isfinite(value) and 0 <= value <= MAX_CUSTOMER_MARGIN


def customer_cost_margin(has_customer: bool, app_margin: float | None = None) -> float | None:
    """The fraction added to a request's cost because it runs for a customer, e.g. 0.02 adds 2%.

    The app's own surcharge replaces the gateway-wide one, so an app set to 0 is not surcharged."""
    if not has_customer:
        return None
    if app_margin is not None and is_valid_customer_margin(app_margin):
        return float(app_margin)
    value: Final = litellm.cost_margin_config.get(CUSTOMER_MARGIN_KEY)
    return float(value) if isinstance(value, (int, float)) and is_valid_customer_margin(value) else None


class _AppSurcharge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)

    app_customer_margin: StrictFloat | StrictInt | None = None


_APP_SURCHARGE: Final = TypeAdapter(_AppSurcharge)


def request_app_margin(metadata: Mapping[str, object]) -> float | None:
    """The surcharge of the app the request was authenticated as, read from the hierarchy the proxy stamped."""
    try:
        return _APP_SURCHARGE.validate_python(metadata.get("user_api_key_customer_hierarchy")).app_customer_margin
    except ValidationError:
        return None


def request_customer_cost_margin(metadata: Mapping[str, object]) -> float | None:
    return customer_cost_margin(request_has_customer(metadata), request_app_margin(metadata))


def _has_customer_usage_attribution(metadata: Mapping[str, object]) -> bool:
    serialized: Final = metadata.get(CUSTOMER_USAGE_METADATA_KEY)
    if not isinstance(serialized, str):
        return False
    try:
        CustomerUsageAttribution.model_validate_json(serialized)
    except ValidationError:
        return False
    return True


def request_has_customer(metadata: Mapping[str, object]) -> bool:
    """Whether the request is charged to a customer organization, by request identity or by the key's binding."""
    return bool(metadata.get("user_api_key_customer_org_id")) or _has_customer_usage_attribution(metadata)


def _request_fields(value: object) -> Mapping[str, object]:
    try:
        return _REQUEST_FIELDS.validate_python(value)
    except ValidationError:
        return {}


def request_customer_attribution(litellm_params: object, identity: Mapping[str, object]) -> Mapping[str, object]:
    """``identity`` plus the key's customer binding, which the proxy stamps into ``litellm_metadata`` on routes
    whose ``metadata`` belongs to the caller."""
    buckets: Final = _request_fields(litellm_params)
    bindings: Final = tuple(
        _request_fields(buckets.get(bucket)).get(CUSTOMER_USAGE_METADATA_KEY)
        for bucket in ("litellm_metadata", "metadata")
    )
    return {
        CUSTOMER_USAGE_METADATA_KEY: next((binding for binding in bindings if binding is not None), None),
        **identity,
    }


def failed_request_customer_attribution(row_kwargs: object) -> Mapping[str, object]:
    """A failed request keeps the proxy's ``litellm_metadata`` beside ``litellm_params`` on routes whose
    ``metadata`` belongs to the caller, so both places are read."""
    row: Final = _request_fields(row_kwargs)
    litellm_params: Final = _request_fields(row.get("litellm_params"))
    buckets: Final = {
        "metadata": _request_fields(litellm_params.get("metadata")),
        "litellm_metadata": {
            **_request_fields(row.get("litellm_metadata")),
            **_request_fields(litellm_params.get("litellm_metadata")),
        },
    }
    return request_customer_attribution(buckets, {**buckets["metadata"], **buckets["litellm_metadata"]})


def key_customer_cost_margin(valid_token: object, hierarchy: CustomerHierarchyAuth | None) -> float | None:
    """The surcharge to reserve for, before the request settles: the resolved hierarchy's, else the key's binding."""
    has_customer: Final = (hierarchy is not None and hierarchy.customer_org_id is not None) or customer_usage_requested(
        {}, customer_usage_key_fields(valid_token)
    )
    return customer_cost_margin(has_customer, hierarchy.app_customer_margin if hierarchy is not None else None)


def _cost_is_finalized(payload: Mapping[str, object]) -> bool:
    return bool(payload.get("cost_finalized", False))


def apply_customer_cost_margin(cost: FinalizedCost, margin: float | None) -> FinalizedCost:
    if margin is None or margin == 0:
        return cost
    if cost.breakdown is not None and "customer_margin_amount" in cost.breakdown:
        return cost
    surcharge: Final = cost.amount * margin
    amount: Final = cost.amount + surcharge
    return FinalizedCost(
        amount=amount,
        breakdown=MappingProxyType(
            {
                **(cost.breakdown or {}),
                "customer_margin_percent": margin,
                "customer_margin_amount": surcharge,
                "total_cost": amount,
            }
        ),
        allowance_discount=cost.allowance_discount,
    )


def apply_customer_cost_margin_to_estimate(amount: float | None, margin: float | None) -> float | None:
    if amount is None or margin is None:
        return amount
    return amount * (1 + margin)


def _callback_cost(payload: StandardLoggingPayload, amount: float) -> FinalizedCost:
    try:
        return finalize_cost(amount, payload["cost_breakdown"])
    except KeyError:
        return finalize_cost(amount, None)


def apply_callback_customer_cost_margin(
    payload: StandardLoggingPayload | None, amount: float | None, margin: float | None
) -> float | None:
    if payload is None or amount is None:
        return amount
    if _cost_is_finalized(payload):
        return amount
    original: Final = _callback_cost(payload, amount)
    cost: Final = apply_customer_cost_margin(original, margin)
    if cost is original:
        return amount
    payload["response_cost"] = cost.amount  # rebind-ok: the legacy spend writer consumes this callback payload
    payload["cost_breakdown"] = cost.logging_breakdown()  # rebind-ok: persist reconciliation with the billed cost
    return cost.amount


def snapshotless_cost_with_customer_margin(amount: float, margin: float | None) -> FinalizedCost:
    """A billed cost that has no logging payload to carry its breakdown, surcharged with the surcharge recorded."""
    return apply_customer_cost_margin(finalize_cost(amount, None), margin)


def customer_margin_share_of_spend(spend_log_metadata: Mapping[str, object], spend: float) -> float:
    """The part of a spend row's ``spend`` that is the customer surcharge, capped at the row's spend."""
    try:
        recorded: Final = _RECORDED_SURCHARGE.validate_python(spend_log_metadata.get("cost_breakdown"))
    except ValidationError:
        return 0.0
    amount: Final = recorded.customer_margin_amount
    if not math.isfinite(amount):
        return 0.0
    return min(max(amount, 0.0), max(spend, 0.0))
