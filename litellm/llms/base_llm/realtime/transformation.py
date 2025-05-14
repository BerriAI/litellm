import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import httpx

from litellm.types.llms.gemini import BidiGenerateContentServerContent
from litellm.types.llms.openai import OpenAIRealtimeStreamResponseBaseObject

from ..chat.transformation import BaseLLMException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseRealtimeConfig(ABC):
    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        pass

    @abstractmethod
    def get_complete_url(
        self, api_base: Optional[str], model: str, api_key: Optional[str] = None
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    @abstractmethod
    def transform_realtime_request(self, message: str) -> str:
        pass

    def requires_session_configuration(
        self,
    ) -> bool:  # initial configuration message sent to setup the realtime session
        return False

    def session_configuration_request(
        self, model: str
    ) -> Optional[str]:  # message sent to setup the realtime session
        return None

    def transform_realtime_response(
        self, message: str
    ) -> str:  # message sent to setup the realtime session
        try:
            message = json.loads(message)
        except json.JSONDecodeError:
            return message

        if "modelTurn" in message:
            message = BidiGenerateContentServerContent(**message)  # type: ignore

        return message
