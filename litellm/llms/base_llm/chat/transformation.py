"""
Common base config for all LLM providers
"""

import types
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Type,
    Union,
    cast,
)

import httpx
from pydantic import BaseModel

from litellm.constants import DEFAULT_MAX_TOKENS, RESPONSE_FORMAT_TOOL_NAME
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

from ..base_utils import (
    map_developer_role_to_system_role,
    type_to_response_format_param,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseLLMException(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, httpx.Headers]] = None,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        body: Optional[dict] = None,
    ):
        self.status_code = status_code
        self.message: str = message
        self.headers = headers
        if request:
            self.request = request
        else:
            self.request = httpx.Request(
                method="POST", url="https://docs.litellm.ai/docs"
            )
        if response:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        self.body = body
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class BaseConfig(ABC):
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

    def get_json_schema_from_pydantic_object(
        self, response_format: Optional[Union[Type[BaseModel], dict]]
    ) -> Optional[dict]:
        return type_to_response_format_param(response_format=response_format)

    def is_thinking_enabled(self, non_default_params: dict) -> bool:
        return (
            non_default_params.get("thinking", {}).get("type") == "enabled"
            or non_default_params.get("reasoning_effort") is not None
        )

    def update_optional_params_with_thinking_tokens(
        self, non_default_params: dict, optional_params: dict
    ):
        """
        Handles scenario where max tokens is not specified. For anthropic models (anthropic api/bedrock/vertex ai), this requires having the max tokens being set and being greater than the thinking token budget.

        Checks 'non_default_params' for 'thinking' and 'max_tokens'

        if 'thinking' is enabled and 'max_tokens' is not specified, set 'max_tokens' to the thinking token budget + DEFAULT_MAX_TOKENS
        """
        is_thinking_enabled = self.is_thinking_enabled(optional_params)
        if is_thinking_enabled and "max_tokens" not in non_default_params:
            thinking_token_budget = cast(dict, optional_params["thinking"]).get(
                "budget_tokens", None
            )
            if thinking_token_budget is not None:
                optional_params["max_tokens"] = (
                    thinking_token_budget + DEFAULT_MAX_TOKENS
                )

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        Returns True if the model/provider should fake stream
        """
        return False

    def _add_tools_to_optional_params(self, optional_params: dict, tools: List) -> dict:
        """
        Helper util to add tools to optional_params.
        """
        if "tools" not in optional_params:
            optional_params["tools"] = tools
        else:
            optional_params["tools"] = [
                *optional_params["tools"],
                *tools,
            ]
        return optional_params

    def translate_developer_role_to_system_role(
        self,
        messages: List[AllMessageValues],
    ) -> List[AllMessageValues]:
        """
        Translate `developer` role to `system` role for non-OpenAI providers.

        Overriden by OpenAI/Azure
        """
        return map_developer_role_to_system_role(messages=messages)

    def should_retry_llm_api_inside_llm_translation_on_http_error(
        self, e: httpx.HTTPStatusError, litellm_params: dict
    ) -> bool:
        """
        Returns True if the model/provider should retry the LLM API on UnprocessableEntityError

        Overriden by azure ai - where different models support different parameters
        """
        return False

    def transform_request_on_unprocessable_entity_error(
        self, e: httpx.HTTPStatusError, request_data: dict
    ) -> dict:
        """
        Transform the request data on UnprocessableEntityError
        """
        return request_data

    @property
    def max_retry_on_unprocessable_entity_error(self) -> int:
        """
        Returns the max retry count for UnprocessableEntityError

        Used if `should_retry_llm_api_inside_llm_translation_on_http_error` is True
        """
        return 0

    @abstractmethod
    def get_supported_openai_params(self, model: str) -> list:
        pass

    def _add_response_format_to_tools(
        self,
        optional_params: dict,
        value: dict,
        is_response_format_supported: bool,
        enforce_tool_choice: bool = True,
    ) -> dict:
        """
        Follow similar approach to anthropic - translate to a single tool call.

        When using tools in this way: - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
        - You usually want to provide a single tool
        - You should set tool_choice (see Forcing tool use) to instruct the model to explicitly use that tool
        - Remember that the model will pass the input to the tool, so the name of the tool and description should be from the model’s perspective.

        Add response format to tools

        This is used to translate response_format to a tool call, for models/APIs that don't support response_format directly.
        """
        json_schema: Optional[dict] = None
        if "response_schema" in value:
            json_schema = value["response_schema"]
        elif "json_schema" in value:
            json_schema = value["json_schema"]["schema"]

        if json_schema and not is_response_format_supported:
            _tool_choice = ChatCompletionToolChoiceObjectParam(
                type="function",
                function=ChatCompletionToolChoiceFunctionParam(
                    name=RESPONSE_FORMAT_TOOL_NAME
                ),
            )

            _tool = ChatCompletionToolParam(
                type="function",
                function=ChatCompletionToolParamFunctionChunk(
                    name=RESPONSE_FORMAT_TOOL_NAME, parameters=json_schema
                ),
            )

            optional_params.setdefault("tools", [])
            optional_params["tools"].append(_tool)
            if enforce_tool_choice:
                optional_params["tool_choice"] = _tool_choice

            optional_params["json_mode"] = True
        elif is_response_format_supported:
            optional_params["response_format"] = value
        return optional_params

    @abstractmethod
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        pass

    @abstractmethod
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
        pass

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> dict:
        """
        Some providers like Bedrock require signing the request. The sign request funtion needs access to `request_data` and `complete_url`
        Args:
            headers: dict
            optional_params: dict
            request_data: dict - the request body being sent in http request
            api_base: str - the complete url being sent in http request
        Returns:
            dict - the signed headers

        Update the headers with the signed headers in this function. The return values will be sent as headers in the http request.
        """
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
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
    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        pass

    @abstractmethod
    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        pass

    @abstractmethod
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        pass

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        pass

    def get_async_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[AsyncHTTPHandler] = None,
        json_mode: Optional[bool] = None,
    ) -> CustomStreamWrapper:
        raise NotImplementedError

    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        json_mode: Optional[bool] = None,
    ) -> CustomStreamWrapper:
        raise NotImplementedError

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return None

    @property
    def has_custom_stream_wrapper(self) -> bool:
        return False

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        """
        Some providers like Bedrock invoke do not support the stream parameter in the request body.

        By default, this is true for almost all providers.
        """
        return True
