import json
from typing import Final
from urllib.parse import urlparse

import litellm
from litellm.constants import WEBSOCKET_CLIENT_TEXT_COUNT_KEY
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.deepgram.common_utils import deepgram_speak_character_rate, deepgram_speak_model
from litellm.types.utils import StandardPassThroughResponseObject

DEEPGRAM_SPEAK_ROUTE_SUFFIX: Final = "/speak"


class DeepgramSpeakPassthroughLoggingHandler:
    @staticmethod
    def is_deepgram_speak_route(url_route: str) -> bool:
        path: Final = urlparse(url_route).path
        return "/deepgram/" in path and path.endswith(DEEPGRAM_SPEAK_ROUTE_SUFFIX)

    @staticmethod
    def deepgram_speak_passthrough_handler(
        logging_obj: LiteLLMLoggingObj, upstream_url: str
    ) -> StandardPassThroughResponseObject:
        counted: Final = logging_obj.model_call_details.get(WEBSOCKET_CLIENT_TEXT_COUNT_KEY)
        characters: Final = counted if isinstance(counted, int) else 0
        rate: Final = deepgram_speak_character_rate(upstream_url)
        model: Final = deepgram_speak_model(upstream_url)
        logging_obj.model = model  # rebind-ok: the spend logger reads model and cost off the shared logging object
        logging_obj.model_call_details["model"] = model  # rebind-ok: same shared logging object
        logging_obj.model_call_details["custom_llm_provider"] = litellm.LlmProviders.DEEPGRAM.value  # rebind-ok: same
        logging_obj.model_call_details["response_cost"] = None if rate is None else characters * rate  # rebind-ok: same
        return StandardPassThroughResponseObject(response=json.dumps({"characters": characters}))
