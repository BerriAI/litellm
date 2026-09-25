import math
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import set_response_cost_in_hidden_params
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

_SUPPORTED_OPENAI_PARAMS: Final = (
    "extra_headers",
    "frequency_penalty",
    "max_retries",
    "max_tokens",
    "parallel_tool_calls",
    "presence_penalty",
    "response_format",
    "service_tier",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "user",
)


_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])


def _pricing_failed(reported: Mapping[str, object]) -> bool:
    try:
        return bool(_JSON_OBJECT.validate_python(reported.get("cost_breakdown")).get("pricing_failed"))
    except ValidationError:
        return False


def _reported_cost_usd(raw_response: httpx.Response) -> float | None:
    try:
        reported: Final = _JSON_OBJECT.validate_python(raw_response.json()["nadir_metadata"]["cost"])
    except (ValueError, KeyError, TypeError):
        return None
    # An unpriced call still carries a total (0.0, or the sum of the attempts Nadir could price),
    # flagged in its breakdown. Logging that total would record a paid call as free or cheaper than it was.
    if _pricing_failed(reported):
        return None
    cost: Final = reported.get("total_cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    if not math.isfinite(cost) or cost < 0:
        return None
    return float(cost)


class NadirConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: return type fixed by the base interface
        return list(_SUPPORTED_OPENAI_PARAMS)  # mutable-ok: the base interface returns a list

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: object,
        request_data: dict,  # mutable-ok: signature fixed by the base interface
        messages: list[AllMessageValues],  # mutable-ok: signature fixed by the base interface
        optional_params: dict,  # mutable-ok: signature fixed by the base interface
        litellm_params: dict,  # mutable-ok: signature fixed by the base interface
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        transformed: Final = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        set_response_cost_in_hidden_params(transformed, _reported_cost_usd(raw_response))
        return transformed
