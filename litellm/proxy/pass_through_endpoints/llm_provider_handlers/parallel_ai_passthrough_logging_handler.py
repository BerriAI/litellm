import json
from collections.abc import Mapping
from typing import Final
from urllib.parse import urlparse

from typing_extensions import TypedDict

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.parallel_ai.extract.cost_calculator import (
    PARALLEL_AI_EXTRACT_MODEL,
    parallel_ai_extract_cost,
)
from litellm.types.utils import StandardPassThroughResponseObject


class ParallelAIPassthroughLoggingResult(TypedDict):
    result: StandardPassThroughResponseObject
    kwargs: dict[str, object]


class ParallelAIPassthroughLoggingHandler:
    @staticmethod
    def is_extract_route(url_route: str, custom_llm_provider: str | None) -> bool:
        path: Final = urlparse(url_route).path.rstrip("/")
        return custom_llm_provider == "parallel_ai" and path.endswith("/v1/extract")

    @staticmethod
    def parallel_ai_extract_handler(
        response_body: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        request_body: Mapping[str, object],
        **kwargs: object,
    ) -> ParallelAIPassthroughLoggingResult:
        response_cost: Final = parallel_ai_extract_cost(
            request_body=request_body,
            response_body=response_body,
        )
        response: Final = StandardPassThroughResponseObject(response=json.dumps(response_body))

        kwargs["model"] = PARALLEL_AI_EXTRACT_MODEL
        kwargs["custom_llm_provider"] = "parallel_ai"
        kwargs["response_cost"] = response_cost

        logging_obj.model_call_details["model"] = PARALLEL_AI_EXTRACT_MODEL
        logging_obj.model_call_details["custom_llm_provider"] = "parallel_ai"
        logging_obj.model_call_details["response_cost"] = response_cost

        return {"result": response, "kwargs": kwargs}
