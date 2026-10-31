import json
import math
from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm._internal_context import pinned_billing_time
from litellm._logging import verbose_proxy_logger
from litellm.accounting import CostEvent, DeliveryTarget, DimensionName, OutboundUsageEvent, PricingSnapshot, Unpriced
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_cost_calc.finalization import finalize_cost
from litellm.litellm_core_utils.llm_cost_calc.pricing_context import RetainedPricing, retained_pricing
from litellm.proxy.db.accounting_ledger import AccountingLedger
from litellm.proxy.spend_tracking.usage_meter import usage_delivery_targets
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import (
    CallTypes,
    CostPerToken,
    LiteLLMRealtimeStreamLoggingObject,
    ModelResponse,
    StandardBuiltInToolsParams,
    Usage,
    is_server_derived_pricing_key,
)

_FIELDS: Final = TypeAdapter(dict[str, JsonValue])


def _fields(value: object) -> Mapping[str, JsonValue]:
    return _FIELDS.validate_python(value) if isinstance(value, dict) else {}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _UsageEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    event: OutboundUsageEvent
    usage: Mapping[str, JsonValue] | None = None
    pricing: Mapping[str, JsonValue]
    charge_eligible: bool
    finalized_cost: Mapping[str, JsonValue] | None = None
    allocation: Mapping[str, JsonValue] | None = None


def price_usage(envelope: str) -> PricingSnapshot | Unpriced:
    captured: Final = _UsageEnvelope.model_validate_json(envelope)
    event: Final = captured.event
    if not captured.charge_eligible:
        return Unpriced(usage_event_id=event.event_id, reason="not_charge_owner")
    fixed_call: Final = event.call_type in {CallTypes.call_mcp_tool, CallTypes.send_message, CallTypes.asend_message}
    realtime_events: Final = TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(
        captured.pricing.get("realtime_events") or ()
    )
    realtime_usage: Final = event.call_type is CallTypes.arealtime and any(
        bool(item.get("usage")) for item in realtime_events
    )
    if captured.usage is None and not fixed_call and not realtime_usage:
        return Unpriced(usage_event_id=event.event_id, reason="usage_unknown")
    pricing: Final = captured.pricing
    mapped: Final = _fields(pricing.get("model_map_information"))
    rates: Final = _fields(mapped.get("model_map_value"))
    model: Final = next(
        (d.value for d in event.dimensions if d.name is DimensionName.MODEL),
        event.call_type.value if fixed_call else None,
    )
    provider: Final = next((d.value for d in event.dimensions if d.name is DimensionName.PROVIDER), None)
    logged: Final = _fields(pricing.get("logging"))
    reported: Final = logged.get("provider_reported_cost")
    upstream: Final = (
        isinstance(reported, (int, float))
        and not isinstance(reported, bool)
        and math.isfinite(reported)
        and reported >= 0
        and logged.get("custom_pricing") is not True
    )
    key: Final = mapped.get("model_map_key") or (model if fixed_call or upstream else None)
    has_rates: Final = any(is_server_derived_pricing_key(name) and value is not None for name, value in rates.items())
    if model is None or not isinstance(key, str) or (not has_rates and not fixed_call and not upstream):
        return Unpriced(usage_event_id=event.event_id, reason="pricing_inputs_missing")
    info: Final = {
        "key": key,
        "litellm_provider": provider,
        "mode": "chat",
        "max_tokens": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "input_cost_per_token": None,
        "output_cost_per_token": None,
        "supported_openai_params": None,
        **rates,
    }
    adjustments: Final = _fields(pricing.get("adjustments"))
    discount: Final = adjustments.get("discount")
    margin: Final = adjustments.get("margin")
    secondary: Final = {
        name: {
            **info,
            **_fields(_fields(snapshot).get("model_map_value")),
            "key": _fields(snapshot).get("model_map_key") or name,
        }
        for name, snapshot in _fields(pricing.get("realtime_models")).items()
        if _fields(snapshot).get("model_map_value") is not None
    }
    context: Final = RetainedPricing(
        _json(
            {
                **secondary,
                **{
                    alias: info
                    for alias in dict.fromkeys((model, key, f"{provider}/{model}", model.removeprefix(f"{provider}/")))
                },
            }
        ),
        _json({provider: discount} if provider and discount is not None else {}),
        _json({"global": margin} if margin is not None else {}),
    )
    response_inputs: Final = _fields(pricing.get("response_inputs"))
    tool_response: Final = _fields(pricing.get("tool_response"))
    token_response: Final = ModelResponse(
        model=model,
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "annotations": tuple(
                        {"type": annotation}
                        for flag, annotation in (("web_search", "url_citation"),)
                        if tool_response.get(flag)
                    ),
                }
            }
        ],
        usage=Usage.model_validate({name: value for name, value in (captured.usage or {}).items() if name != "cost"}),
        service_tier=response_inputs.get("service_tier"),
        meta={"billed_units": response_inputs.get("billed_units")},
        hidden_params={
            "region_name": response_inputs.get("region_name"),
            "audio_transcription_duration": response_inputs.get("audio_transcription_duration"),
        },
        response_ms=logged.get("response_ms"),
    )
    response: Final = (
        LiteLLMRealtimeStreamLoggingObject.model_validate(
            {
                "results": pricing.get("realtime_events", []),
                "usage": captured.usage or {},
                "service_tier": response_inputs.get("service_tier"),
            }
        )
        if event.call_type in {CallTypes.arealtime, CallTypes.aresponses_websocket}
        else ResponsesAPIResponse.model_validate(
            {
                "id": event.event_id,
                "created_at": int(event.started_at.timestamp()),
                "model": model,
                "output": tool_response.get("output", []),
                "usage": tool_response.get("usage"),
            }
        )
        if tool_response.get("responses_api")
        else token_response
    )
    logger: Final = Logging(
        model=model,
        messages=[],
        stream=False,
        call_type=event.call_type.value,
        start_time=event.started_at,
        litellm_call_id=event.event_id,
        function_id=event.attempt_id,
    )
    request: Final = _fields(pricing.get("request"))
    deployment: Final = _fields(pricing.get("deployment"))
    logger.update_environment_variables(  # pyright: ignore[reportUnknownMemberType]  # legacy Logging environment method accepts validated JSON fields
        model=model,
        custom_llm_provider=provider,
        litellm_params={**deployment, **request, "metadata": {"model_info": _fields(pricing.get("model_info"))}},
        additional_response_cost=request.get("additional_response_cost"),
        usage=Usage.model_validate({name: value for name, value in (captured.usage or {}).items() if name != "cost"}),
        end_time=event.ended_at,
        mcp_tool_call_metadata={
            "name": _fields(pricing.get("tool")).get("name"),
            "mcp_server_cost_info": {"default_cost_per_query": _fields(pricing.get("tool")).get("cost_per_query")},
        },
        optional_params=dict(_fields(pricing.get("optional_params"))),
    )
    if logged.get("tool_response_cost") is not None:
        logger.model_call_details["response_cost"] = logged["tool_response_cost"]
    logger.request_pricing = context if logged.get("custom_pricing") is True else None
    logger.custom_cost_per_token = TypeAdapter(CostPerToken | None).validate_python(logged.get("custom_cost_per_token"))
    logger.custom_cost_per_second = TypeAdapter(float | None).validate_python(logged.get("custom_cost_per_second"))
    logger.standard_built_in_tools_params = TypeAdapter(StandardBuiltInToolsParams).validate_python(
        {
            "web_search_options": adjustments.get("web_search_options"),
            "file_search": adjustments.get("file_search") or None,
        }
    )
    logger.record_provider_reported_cost(
        TypeAdapter[float | None](float | None).validate_python(reported if upstream else None)
    )
    billed_at: Final = TypeAdapter(datetime).validate_python(logged.get("current_billing_time", event.ended_at))
    if event.call_type is CallTypes.arealtime and captured.usage and captured.usage.get("metered_usage") is not None:
        from litellm.constants import REALTIME_METERED_USAGE_KEY
        from litellm.types.llms.xai import RealtimeMeteredUsage

        logger.model_call_details[REALTIME_METERED_USAGE_KEY] = TypeAdapter(RealtimeMeteredUsage).validate_python(
            captured.usage["metered_usage"]
        )
    logger.cost_billing_time = billed_at
    with retained_pricing(context), pinned_billing_time(billed_at):
        amount: Final = logger.response_cost_calculator(response, ignore_stamped_cost=True)
    if amount is None:
        return Unpriced(usage_event_id=event.event_id, reason="calculation_failed")
    retained_final: Final = captured.finalized_cost or {}
    breakdown: Final = _fields(retained_final.get("cost_breakdown"))
    guardrail: Final = TypeAdapter(float).validate_python(breakdown.get("guardrail_cost", 0.0))
    allowance: Final = TypeAdapter(float).validate_python(retained_final.get("allowance_discount", 0.0))
    if allowance and captured.allocation is None:
        return Unpriced(usage_event_id=event.event_id, reason="allowance_allocation_missing")
    finalized: Final = finalize_cost(amount + guardrail, logger.cost_breakdown, allowance)
    if retained_final.get("response_cost") is not None and Decimal(str(finalized.amount)) != Decimal(
        str(retained_final["response_cost"])
    ):
        return Unpriced(usage_event_id=event.event_id, reason="finalized_cost_mismatch")
    final_breakdown: Final = dict(finalized.breakdown or {})
    components: Final = {
        name: Decimal(str(value))
        for name, value in final_breakdown.items()
        if name in _COST_COMPONENTS and isinstance(value, (int, float))
    }
    return PricingSnapshot.model_validate(
        {
            **components,
            "calculator_version": "logging-v1",
            "model": model,
            "provider": provider,
            "pricing_source": "upstream_header"
            if upstream
            else "per_request"
            if request
            else "deployment"
            if deployment
            else "cost_map",
            "priced_model_key": key,
            "total_cost": Decimal(str(finalized.amount)),
            "guardrail_cost": Decimal(str(guardrail)),
            "upstream_reported_cost": Decimal(str(logged["provider_reported_cost"])) if upstream else None,
            "additional_costs": tuple(
                {"name": name, "amount": Decimal(str(value))}
                for name, value in {
                    **_fields(final_breakdown.get("additional_costs")),
                    **(
                        {"request": request["additional_response_cost"]}
                        if request.get("additional_response_cost")
                        else {}
                    ),
                }.items()
            ),
        }
    )


_COST_COMPONENTS: Final = frozenset(
    {
        "input_cost",
        "output_cost",
        "cache_read_cost",
        "cache_creation_cost",
        "reasoning_cost",
        "tool_usage_cost",
        "original_cost",
        "discount_percent",
        "discount_amount",
        "margin_percent",
        "margin_fixed_amount",
        "margin_total_amount",
    }
)


class LoggingCostCompiler:
    def __init__(
        self,
        ledger: AccountingLedger,
        price_usage: Callable[[str], PricingSnapshot | Unpriced] = price_usage,
        delivery_targets: tuple[DeliveryTarget, ...] | None = None,
    ) -> None:
        self._ledger: Final = ledger
        self._price_usage: Final = price_usage
        self._delivery_targets: Final = delivery_targets

    async def compile(self, event: OutboundUsageEvent) -> CostEvent | Unpriced:
        targets: Final = self._delivery_targets if self._delivery_targets is not None else cost_delivery_targets(event)
        return await self._ledger.compile_cost(event, self._price_usage, targets)

    async def compile_pending(self, limit: int = 100) -> None:
        for event in await self._ledger.pending_compilation(limit):
            try:
                await self.compile(event)
            except Exception:  # noqa: BLE001  # retain pending work for retry without starving other events
                verbose_proxy_logger.exception("Cost compilation failed for %s", event.event_id)


_SPEND_TARGETS: Final = (
    (DimensionName.KEY, DeliveryTarget.KEY_SPEND),
    (DimensionName.USER, DeliveryTarget.USER_SPEND),
    (DimensionName.TEAM, DeliveryTarget.TEAM_SPEND),
    (DimensionName.ORGANIZATION, DeliveryTarget.ORGANIZATION_SPEND),
    (DimensionName.END_USER, DeliveryTarget.END_USER_SPEND),
    (DimensionName.AGENT, DeliveryTarget.AGENT_SPEND),
    (DimensionName.TAG, DeliveryTarget.TAG_SPEND),
    (DimensionName.PROJECT, DeliveryTarget.PROJECT_SPEND),
    (DimensionName.MODEL_ACCESS_GROUP, DeliveryTarget.MODEL_ACCESS_GROUP_SPEND),
)


def cost_delivery_targets(event: OutboundUsageEvent) -> tuple[DeliveryTarget, ...]:
    names: Final = frozenset(d.name for d in event.dimensions)
    return (
        *usage_delivery_targets(event.dimensions),
        *(target for dimension, target in _SPEND_TARGETS if dimension in names),
        *((DeliveryTarget.TEAM_MEMBER_SPEND,) if {DimensionName.TEAM, DimensionName.USER} <= names else ()),
        *(
            (DeliveryTarget.ORGANIZATION_MEMBER_SPEND,)
            if {DimensionName.ORGANIZATION, DimensionName.USER} <= names
            else ()
        ),
    )
