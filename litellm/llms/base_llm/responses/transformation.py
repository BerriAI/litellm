import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from litellm.types.llms.openai import ResponsesAPIRequestParams
from litellm.types.utils import ModelInfo

from ..chat.transformation import BaseLLMException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseResponsesAPIConfig(ABC):
    def __init__(self):
        pass

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("_abc")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    @abstractmethod
    def get_supported_openai_params(self, model: str) -> list:
        pass

    @abstractmethod
    def map_openai_params(
        self,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> ResponsesAPIRequestParams:

        pass

    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        return {}

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    # @abstractmethod
    # def transform_request(
    #     self,
    #     model: str,
    #     optional_params: dict,
    #     litellm_params: dict,
    #     headers: dict,
    # ) -> dict:
    #     pass

    # @abstractmethod
    # def transform_response(
    #     self,
    # ):
    #     pass
