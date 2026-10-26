import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.deepgram.common_utils import (
    deepgram_speak_character_rate,
    deepgram_speak_model,
    deepgram_speak_registry_key,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.passthrough_endpoints.pass_through_endpoints import WEBSOCKET_CLIENT_TEXT_COUNT_KEY
from litellm.types.utils import StandardPassThroughResponseObject

DEEPGRAM_SPEAK_ROUTE_SUFFIX: Final = "/speak"


def _billed_characters(request_body: Mapping[str, object]) -> int:
    characters: Final = request_body.get(WEBSOCKET_CLIENT_TEXT_COUNT_KEY)
    return characters if isinstance(characters, int) else 0


def _speech_cost(upstream_url: str, characters: int) -> float | None:
    rate: Final = deepgram_speak_character_rate(upstream_url)
    if rate is None:
        verbose_proxy_logger.warning(
            "Deepgram speak passthrough: no per-character registry entry '%s'",
            deepgram_speak_registry_key(upstream_url),
        )
        return None
    return characters * rate


class DeepgramSpeakPassthroughLoggingHandler:
    """Bills a Deepgram speak session by the characters of Speak text the client sent. The relay hands over only the
    count, never the text, so the spend row and every callback see a model, a count and a cost and nothing the caller
    asked to have spoken"""

    @staticmethod
    def is_deepgram_speak_route(url_route: str, request_method: str) -> bool:
        path: Final = urlparse(url_route).path
        return request_method == "WEBSOCKET" and "/deepgram/" in path and path.endswith(DEEPGRAM_SPEAK_ROUTE_SUFFIX)

    @staticmethod
    def deepgram_speak_passthrough_handler(
        request_body: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        upstream_url: str,
        kwargs: Mapping[str, object] = MappingProxyType({}),
    ) -> PassThroughEndpointLoggingTypedDict:
        model: Final = deepgram_speak_model(upstream_url)
        characters: Final = _billed_characters(request_body)
        response_cost: Final = _speech_cost(upstream_url, characters)

        provider: Final = litellm.LlmProviders.DEEPGRAM.value
        logging_obj.model = model  # rebind-ok: the spend logger reads model and cost off the shared logging object
        logging_obj.model_call_details["model"] = model  # rebind-ok: same shared logging object
        logging_obj.model_call_details["custom_llm_provider"] = provider  # rebind-ok: same shared logging object
        logging_obj.model_call_details["response_cost"] = response_cost  # rebind-ok: same shared logging object
        verbose_proxy_logger.debug(
            "Deepgram speak passthrough cost tracking: model %s, characters %s, cost %s",
            model,
            characters,
            response_cost,
        )
        logging_result: Final[PassThroughEndpointLoggingTypedDict] = {
            "result": StandardPassThroughResponseObject(response=json.dumps({"characters": characters})),
            "kwargs": {
                **kwargs,
                "model": model,
                "custom_llm_provider": provider,
                "response_cost": response_cost,
            },
        }
        return logging_result
