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
_WIRE_BODY: Final = TypeAdapter(Mapping[str, object])
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
        if getattr(logging_obj, "call_type", None) == "anthropic_messages":
            logging_obj.record_bridged_responses_api_response(self.response)


class MessagesUsageCostStamper(UsageCostStamper):
    def __init__(self, response: object) -> None:
        self.response = _WIRE_TARGET.validate_python(response)

    def stamp(self, cost: float) -> None:
        body: Final = _WIRE_BODY.validate_python(self.response)
        usage: Final = body.get("usage")
        if isinstance(usage, Mapping):
            self.response["usage"] = {**usage, "cost": cost}


def stamp_usage_cost(response: object, cost: float | None) -> None:
    if cost is None or not isfinite(cost):
        return
    if isinstance(response, (ModelResponse, ModelResponseStream)):
        ChatUsageCostStamper(response).stamp(float(cost))
        return
    if isinstance(response, ResponsesAPIResponse):
        ResponsesUsageCostStamper(response).stamp(float(cost))
        return
    if not isinstance(response, dict):
        return
    wire_body: Final = _WIRE_BODY.validate_python(response)
    if wire_body.get("type") in ("message", "message_delta"):
        MessagesUsageCostStamper(_WIRE_TARGET.validate_python(response)).stamp(float(cost))


def stamp_responses_usage_cost(response_obj: ResponsesAPIResponse | None, logging_obj: Logging | None) -> None:
    if response_obj is not None:
        ResponsesUsageCostStamper(response_obj).stamp_calculated(logging_obj)
