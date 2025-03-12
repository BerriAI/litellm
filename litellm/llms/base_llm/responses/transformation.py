import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import httpx

from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.router import GenericLiteLLMParams
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
    ) -> ResponsesAPIOptionalRequestParams:

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

    @abstractmethod
    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: ResponsesAPIOptionalRequestParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> ResponsesAPIRequestParams:
        pass

    @abstractmethod
    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        pass

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
