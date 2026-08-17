"""
Bridge for transforming API requests to another API requests
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from pydantic import BaseModel

    from litellm import LiteLLMLoggingObj, ModelResponse
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
    from litellm.types.llms.openai import AllMessageValues


class CompletionTransformationBridge(ABC):
    @abstractmethod
    def transform_request(
        self,
        model: str,
        messages: list["AllMessageValues"],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
        litellm_logging_obj: "LiteLLMLoggingObj",
    ) -> dict:
        """Transform /chat/completions api request to another request"""

    @abstractmethod
    def transform_response(
        self,
        model: str,
        raw_response: "BaseModel",  # the response from the other API
        model_response: "ModelResponse",
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: list["AllMessageValues"],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> "ModelResponse":
        """Transform another response to /chat/completions api response"""

    @abstractmethod
    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], "ModelResponse"],
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> "BaseModelResponseIterator":
        pass
