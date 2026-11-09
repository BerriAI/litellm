from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING, Final

from pydantic import ConfigDict, InstanceOf, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import (
    provider_reported_usage_cost,
    set_response_cost_in_hidden_params,
    stamp_litellm_bill_on_usage,
)
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

_CHAT_USAGE: Final[TypeAdapter[Usage | None]] = TypeAdapter(Usage | None)
_USAGE_MAPPING: Final = TypeAdapter(Mapping[str, object])
_WIRE_TARGET: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(InstanceOf[dict[str, object]])
_COST: Final[TypeAdapter[float | None]] = TypeAdapter(float | None, config=ConfigDict(strict=True))


def validated_usage_cost(value: object) -> float | None:
    try:
        cost: Final = _COST.validate_python(value)
    except ValidationError:
        return None
    return cost if cost is not None and isfinite(cost) else None


class UsageCostStamper(ABC):
    @abstractmethod
    def stamp(self, cost: float) -> None: ...


class ChatUsageCostStamper(UsageCostStamper):
    def __init__(self, response: ModelResponse | ModelResponseStream) -> None:
        self.response = response

    def stamp(self, cost: float) -> None:
        try:
            usage: Final = _CHAT_USAGE.validate_python(getattr(self.response, "usage", None))
        except ValidationError:
            return
        if usage is not None:
            stamp_litellm_bill_on_usage(self.response, usage, cost)


class ResponsesUsageCostStamper(UsageCostStamper):
    def __init__(self, response: ResponsesAPIResponse) -> None:
        self.response = response

    def stamp(self, cost: float) -> None:
        if self.response.usage is not None:
            stamp_litellm_bill_on_usage(self.response, self.response.usage, cost)

    def stamp_calculated(self, logging_obj: Logging | None) -> None:
        from litellm.responses.utils import ResponseAPILoggingUtils

        usage: Final = self.response.usage
        if usage is None:
            return
        try:
            normalized: Final = ResponseAPIUsage.model_validate(usage)
        except ValidationError:
            return
        self.response.usage = normalized
        if logging_obj is None:
            return
        priced: Final = self.response.model_copy(
            update={"usage": ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(normalized)}
        )
        priced.hidden_params = {}
        set_response_cost_in_hidden_params(
            priced, provider_reported_usage_cost(normalized, logging_obj.model_call_details.get("custom_llm_provider"))
        )
        try:
            bill: Final = logging_obj.response_cost_calculator(result=priced)
        except Exception:  # noqa: BLE001  # optional response metadata must not fail a completed response
            return
        if isinstance(bill, bool) or not isinstance(bill, (int, float)):
            return
        if bill > 0 or isinstance(normalized.cost, (int, float)):
            self.stamp(float(bill))


class MessagesUsageCostStamper(UsageCostStamper):
    def __init__(self, response: dict[str, object]) -> None:  # mutable-ok: wire body is stamped in place [LIT001]
        self.response = response

    @classmethod
    def for_response(cls, response: object) -> MessagesUsageCostStamper | None:
        if not isinstance(response, dict):
            return None
        return cls(_WIRE_TARGET.validate_python(response))

    def stamp(self, cost: float) -> None:
        usage: Final = self.response.get("usage")
        if isinstance(usage, Mapping):
            self.response["usage"] = {**_USAGE_MAPPING.validate_python(usage), "cost": cost}

    def stamp_calculated(
        self,
        logging_obj: Logging,
        cache_hit: bool | None = None,
        *,
        billing_body: Mapping[str, object] | None = None,
    ) -> bool:
        try:
            body: Final = self.response if billing_body is None else billing_body
            if body.get("type") != "message" or not isinstance(body.get("usage"), Mapping):
                return False
            priced: Final = logging_obj.normalize_endpoint_logging_result(body, for_billing=True)
            cost: Final = validated_usage_cost(logging_obj.response_cost_calculator(result=priced, cache_hit=cache_hit))
            if cost is None:
                return False
            self.stamp(cost)
            if isinstance(priced, ModelResponse):
                stamp_usage_cost(priced, cost)
                logging_obj.record_bridged_model_response(priced)
        except Exception:  # noqa: BLE001  # optional billing metadata must not fail a completed response or stream
            return False
        return True


def stamp_usage_cost(response: object, cost: float | None) -> None:
    if cost is None or not isfinite(cost):
        return
    if isinstance(response, (ModelResponse, ModelResponseStream)):
        ChatUsageCostStamper(response).stamp(float(cost))
        return
    if isinstance(response, ResponsesAPIResponse):
        ResponsesUsageCostStamper(response).stamp(float(cost))
        return
    stamper: Final = MessagesUsageCostStamper.for_response(response)
    if stamper is not None and stamper.response.get("type") in ("message", "message_delta"):
        stamper.stamp(float(cost))


def stamp_responses_usage_cost(response_obj: ResponsesAPIResponse | None, logging_obj: Logging | None) -> None:
    if response_obj is not None:
        ResponsesUsageCostStamper(response_obj).stamp_calculated(logging_obj)
