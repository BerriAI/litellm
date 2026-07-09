from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.openrouter.common_utils import (
    OpenRouterException,
    get_openrouter_endpoint,
    get_openrouter_headers,
    raise_openrouter_error,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenRouterTextToSpeechConfig(BaseTextToSpeechConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return ["voice", "response_format", "speed", "instructions"]

    def _resolve_voice(self, voice: Optional[Union[str, Dict[str, Any]]]) -> Optional[str]:
        if voice is None:
            return None
        if isinstance(voice, str):
            return voice
        for key in ("voice_id", "id", "name"):
            candidate = voice.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        raise ValueError("OpenRouter TTS voice must be a string or a dict with voice_id, id, or name.")

    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        voice: Optional[Union[str, Dict[str, Any]]] = None,
        drop_params: bool = False,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict]:
        mapped_params: Dict[str, Any] = {}
        supported_params = set(self.get_supported_openai_params(model))
        for key, value in optional_params.items():
            if value is None:
                continue
            if key in supported_params or not drop_params:
                mapped_params[key] = value
        extra_body = (kwargs or {}).get("extra_body")
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is not None:
                    mapped_params[key] = value
        return self._resolve_voice(voice), mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return get_openrouter_headers(api_key=api_key, headers=headers)

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        return get_openrouter_endpoint(api_base, "audio/speech")

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: Optional[str],
        optional_params: Dict,
        litellm_params: Dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        request_body: Dict[str, Any] = {
            "model": model,
            "input": input,
        }
        if voice is not None:
            request_body["voice"] = voice
        request_body.update({k: v for k, v in optional_params.items() if v is not None})
        return TextToSpeechRequestData(dict_body=request_body)

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> HttpxBinaryResponseContent:
        raise_openrouter_error(raw_response)
        return HttpxBinaryResponseContent(raw_response)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
