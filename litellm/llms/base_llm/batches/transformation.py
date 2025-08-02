from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.types.llms.openai import (
    AllMessageValues,
    CreateBatchRequest,
)
from litellm.types.utils import LlmProviders

from ..chat.transformation import BaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.router import Router as _Router
    from openai.types import Batch

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    Span = Any
    Router = _Router
else:
    LiteLLMLoggingObj = Any
    Span = Any
    Router = Any


class BaseBatchesConfig(BaseConfig):
    @property
    @abstractmethod
    def custom_llm_provider(self) -> LlmProviders:
        pass

    @abstractmethod
    def get_supported_openai_params(
        self, model: str
    ) -> List[str]:
        pass

    def get_complete_batch_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        data: CreateBatchRequest,
    ) -> Optional[str]:
        """
        Returns the complete URL for batch creation.
        """
        return api_base

    @abstractmethod
    def transform_create_batch_request(
        self,
        model: str,
        create_batch_data: CreateBatchRequest,
        litellm_params: dict,
        optional_params: dict,
    ) -> Union[bytes, str, dict]:
        """
        Transform the create batch request to the provider's format.
        """
        pass

    @abstractmethod
    def transform_create_batch_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> "Batch":
        """
        Transform the provider's response to OpenAI batch format.
        """
        pass

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validates the environment for batch creation.
        """
        return headers