from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import WEBSOCKET_PASSTHROUGH_CLIENT_FRAMES_KEY
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.elevenlabs.common_utils import (
    ELEVENLABS_TTS_WEBSOCKET_ROUTE_SUFFIXES,
    elevenlabs_tts_characters,
    elevenlabs_tts_is_priced,
    elevenlabs_tts_model,
    elevenlabs_tts_registry_key,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import StandardPassThroughResponseObject

_RECORDED_FRAMES: Final = TypeAdapter(tuple[dict[str, object], ...])


def _speech_cost(upstream_url: str, characters: int) -> float | None:
    if not elevenlabs_tts_is_priced(upstream_url):
        verbose_proxy_logger.warning(
            "ElevenLabs TTS passthrough: no per-character registry entry '%s'",
            elevenlabs_tts_registry_key(upstream_url),
        )
        return None
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=elevenlabs_tts_model(upstream_url),
        custom_llm_provider=litellm.LlmProviders.ELEVENLABS.value,
        prompt_characters=characters,
        call_type="speech",
    )
    return prompt_cost + completion_cost


class ElevenLabsTextToSpeechPassthroughLoggingHandler:
    @staticmethod
    def is_elevenlabs_tts_websocket_route(url_route: str) -> bool:
        path: Final = urlparse(url_route).path
        return "/elevenlabs/" in path and path.endswith(ELEVENLABS_TTS_WEBSOCKET_ROUTE_SUFFIXES)

    def elevenlabs_tts_passthrough_handler(
        self,
        request_body: Mapping[str, object],
        upstream_frame_count: int,
        logging_obj: LiteLLMLoggingObj,
        upstream_url: str,
        kwargs: Mapping[str, object] = MappingProxyType({}),
    ) -> PassThroughEndpointLoggingTypedDict:
        client_frames: Final = _RECORDED_FRAMES.validate_python(
            request_body.get(WEBSOCKET_PASSTHROUGH_CLIENT_FRAMES_KEY, ())
        )
        model: Final = elevenlabs_tts_model(upstream_url)
        characters: Final = elevenlabs_tts_characters(client_frames)
        response_cost: Final = _speech_cost(upstream_url, characters)

        provider: Final = litellm.LlmProviders.ELEVENLABS.value
        logging_obj.model = model  # rebind-ok: the spend logger reads model and cost off the shared logging object
        logging_obj.model_call_details["model"] = model  # rebind-ok: same shared logging object
        logging_obj.model_call_details["custom_llm_provider"] = provider  # rebind-ok: same shared logging object
        logging_obj.model_call_details["response_cost"] = response_cost  # rebind-ok: same shared logging object
        verbose_proxy_logger.debug(
            "ElevenLabs TTS passthrough cost tracking: model %s, characters %s, cost %s",
            model,
            characters,
            response_cost,
        )
        logging_result: Final[PassThroughEndpointLoggingTypedDict] = {
            "result": StandardPassThroughResponseObject(
                response={
                    "model": model,
                    "characters": characters,
                    "client_text_frames": len(client_frames),
                    "upstream_frames": upstream_frame_count,
                }
            ),
            "kwargs": {
                **kwargs,
                "model": model,
                "custom_llm_provider": provider,
                "response_cost": response_cost,
            },
        }
        return logging_result
