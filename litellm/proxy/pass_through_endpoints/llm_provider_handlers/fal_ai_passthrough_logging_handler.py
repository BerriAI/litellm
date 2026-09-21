from collections.abc import Mapping, Sequence
from typing import Final
from urllib.parse import urlparse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.fal_ai.cost_calculator import fal_ai_passthrough_cost
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import ImageObject, ImageResponse

FAL_AI_PROVIDER: Final[str] = litellm.LlmProviders.FAL_AI.value


def _url_parts(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        return (value,) if isinstance(value.get("url"), str) else ()
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(item for item in value if isinstance(item, Mapping) and isinstance(item.get("url"), str))
    return ()


class FalAIPassthroughLoggingHandler:
    @staticmethod
    def is_fal_ai_route(url_route: str, custom_llm_provider: str | None) -> bool:
        return custom_llm_provider == FAL_AI_PROVIDER

    def fal_ai_passthrough_handler(
        self,
        response_body: Mapping[str, object],
        request_body: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        kwargs: Mapping[str, object],
    ) -> PassThroughEndpointLoggingTypedDict:
        model: Final = urlparse(url_route).path.lstrip("/")
        response: Final = ImageResponse(
            data=tuple(
                ImageObject(url=url)
                for value in response_body.values()
                for part in _url_parts(value)
                if isinstance((url := part.get("url")), str)
            )
        )
        response_cost: Final = fal_ai_passthrough_cost(model, request_body)
        response._hidden_params["response_cost"] = response_cost  # pyright: ignore[reportPrivateUsage]  # the logger reads a precomputed cost off the response's hidden params
        logging_obj.model = model  # rebind-ok: the spend logger reads model and cost off the shared logging object
        logging_obj.model_call_details["model"] = model  # rebind-ok: same shared logging object
        logging_obj.model_call_details["custom_llm_provider"] = FAL_AI_PROVIDER  # rebind-ok: same shared logging object
        logging_obj.model_call_details["response_cost"] = response_cost  # rebind-ok: same shared logging object
        verbose_proxy_logger.debug(
            "Fal AI passthrough cost tracking: model %s, cost %s",
            model,
            response_cost,
        )
        logging_result: Final[PassThroughEndpointLoggingTypedDict] = {
            "result": response,
            "kwargs": {
                **kwargs,
                "model": model,
                "custom_llm_provider": FAL_AI_PROVIDER,
                "response_cost": response_cost,
            },
        }
        return logging_result
