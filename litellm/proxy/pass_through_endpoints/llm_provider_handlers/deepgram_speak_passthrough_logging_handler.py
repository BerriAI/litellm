import json
from typing import Final
from urllib.parse import urlparse

import litellm
from litellm.constants import DEEPGRAM_SPEAK_DEFAULT_MODEL
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.deepgram.common_utils import deepgram_speak_character_rate, deepgram_upstream_model
from litellm.types.passthrough_endpoints.pass_through_endpoints import WEBSOCKET_CLIENT_TEXT_COUNT_KEY
from litellm.types.utils import StandardPassThroughResponseObject


class DeepgramSpeakPassthroughLoggingHandler:
    """Bills a Deepgram speak session by the characters of Speak text the client sent. The relay hands over only the
    count, never the text, so the spend row and every callback see a model, a count and a cost and nothing the caller
    asked to have spoken"""

    @staticmethod
    def is_deepgram_speak_route(url_route: str, request_method: str) -> bool:
        path: Final = urlparse(url_route).path
        return request_method == "WEBSOCKET" and "/deepgram/" in path and path.endswith("/speak")

    @staticmethod
    def deepgram_speak_passthrough_handler(
        logging_obj: LiteLLMLoggingObj, upstream_url: str
    ) -> StandardPassThroughResponseObject:
        counted: Final = logging_obj.model_call_details.get(WEBSOCKET_CLIENT_TEXT_COUNT_KEY)
        characters: Final = counted if isinstance(counted, int) else 0
        rate: Final = deepgram_speak_character_rate(upstream_url)
        model: Final = deepgram_upstream_model(upstream_url, DEEPGRAM_SPEAK_DEFAULT_MODEL)
        logging_obj.model = model  # rebind-ok: the spend logger reads model and cost off the shared logging object
        logging_obj.model_call_details["model"] = model  # rebind-ok: same shared logging object
        logging_obj.model_call_details["custom_llm_provider"] = litellm.LlmProviders.DEEPGRAM.value  # rebind-ok: same
        logging_obj.model_call_details["response_cost"] = None if rate is None else characters * rate  # rebind-ok: same
        return StandardPassThroughResponseObject(response=json.dumps({"characters": characters}))
