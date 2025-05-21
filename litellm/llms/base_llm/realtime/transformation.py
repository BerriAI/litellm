from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.types.realtime import (
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)

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
    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: Optional[str] = None,
    ) -> List[str]:
        pass

    def requires_session_configuration(
        self,
    ) -> bool:  # initial configuration message sent to setup the realtime session
        return False

    def session_configuration_request(
        self, model: str
    ) -> Optional[str]:  # message sent to setup the realtime session
        return None

    @abstractmethod
    def transform_realtime_response(
        self,
        message: Union[str, bytes],
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:  # message sent to setup the realtime session
        """
        Keep this state less - leave the state management (e.g. tracking current_output_item_id, current_response_id, current_conversation_id, current_delta_chunks) to the caller.
        """
        pass
