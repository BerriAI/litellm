from collections.abc import Mapping
from datetime import datetime
from typing import Final, cast

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,  # pyright: ignore[reportUnknownVariableType]  # legacy helper has an untyped signature
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import ModelResponse, StandardPassThroughResponseObject, Usage


class _TypeSafeUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class _TypeSafeResponse(BaseModel):
    model: str | None = None
    usage: _TypeSafeUsage | None = None


_TYPESAFE_RESPONSE_ADAPTER: Final = TypeAdapter(_TypeSafeResponse)
_MODEL_COST_ENTRY_ADAPTER: Final = TypeAdapter(dict[str, object])


def _parse_typesafe_response(response_body: Mapping[str, object]) -> _TypeSafeResponse:
    try:
        return _TYPESAFE_RESPONSE_ADAPTER.validate_python(response_body)
    except ValidationError:
        return _TypeSafeResponse()


def _get_model_cost_entry(model_key: str) -> Mapping[str, object] | None:
    model_cost: Final[Mapping[str, object]] = cast(Mapping[str, object], litellm.model_cost)
    entry: Final[object] = model_cost.get(model_key)
    try:
        return _MODEL_COST_ENTRY_ADAPTER.validate_python(entry)
    except ValidationError:
        return None


class TypeSafePassthroughLoggingHandler:
    @staticmethod
    def typesafe_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Mapping[str, object],
        **kwargs: object,
    ) -> PassThroughEndpointLoggingTypedDict:
        response: Final = _parse_typesafe_response(response_body)
        response_model: Final = response.model
        request_model_value: Final = request_body.get("model")
        request_model: Final = request_model_value if isinstance(request_model_value, str) else None
        logged_model: Final = response_model or request_model or "jev-latest"
        model_name: Final = f"typesafe/{logged_model}"
        usage: Final = response.usage or _TypeSafeUsage()
        input_tokens: Final = usage.input_tokens
        output_tokens: Final = usage.output_tokens
        candidate_model_keys: Final = tuple(
            f"typesafe/{model}" for model in (response_model, request_model) if model is not None
        )
        cost_entry: Final = next(
            (entry for model_key in candidate_model_keys if (entry := _get_model_cost_entry(model_key)) is not None),
            None,
        )
        input_cost_per_token: Final = (
            cost_entry.get("input_cost_per_token", 0.0) if isinstance(cost_entry, Mapping) else 0.0
        )
        output_cost_per_token: Final = (
            cost_entry.get("output_cost_per_token", 0.0) if isinstance(cost_entry, Mapping) else 0.0
        )
        response_cost: Final = (
            input_tokens * float(input_cost_per_token) + output_tokens * float(output_cost_per_token)
            if isinstance(input_cost_per_token, (int, float)) and isinstance(output_cost_per_token, (int, float))
            else 0.0
        )
        usage_object: Final = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        updated_kwargs: Final = {
            **kwargs,
            "model": model_name,
            "custom_llm_provider": "typesafe",
            "response_cost": response_cost,
            "combined_usage_object": usage_object,
        }
        logging_obj.model_call_details.update(
            model=model_name,
            custom_llm_provider="typesafe",
            response_cost=response_cost,
        )
        standard_logging_object: Final = get_standard_logging_object_payload(
            kwargs=updated_kwargs,
            init_response_obj=ModelResponse(model=model_name, usage=usage_object),
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
            status="success",
        )
        return {
            "result": StandardPassThroughResponseObject(response=result),
            "kwargs": {**updated_kwargs, "standard_logging_object": standard_logging_object},
        }
