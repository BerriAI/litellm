import json
from collections.abc import Mapping
from typing import Final
from urllib.parse import urlparse

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.parallel_ai.extract.cost_calculator import (
    PARALLEL_AI_EXTRACT_MODEL,
    parallel_ai_extract_cost,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import StandardPassThroughResponseObject


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
        **kwargs: object,  # kwargs-ok: the passthrough logging dispatch forwards shared logging kwargs to every handler
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Prices a Parallel AI Extract call from the URLs the provider reports as
        billed (falling back to the requested URL count) and records model,
        provider, and cost on the logging payload.
        """
        response_cost: Final = parallel_ai_extract_cost(
            request_body=request_body,
            response_body=response_body,
        )
        logging_obj.model_call_details.update(
            model=PARALLEL_AI_EXTRACT_MODEL,
            custom_llm_provider="parallel_ai",
            response_cost=response_cost,
        )

        return {
            "result": StandardPassThroughResponseObject(response=json.dumps(response_body)),
            "kwargs": {  # mutable-ok: the logging pipeline requires a plain kwargs dict
                **kwargs,
                "model": PARALLEL_AI_EXTRACT_MODEL,
                "custom_llm_provider": "parallel_ai",
                "response_cost": response_cost,
            },
        }
