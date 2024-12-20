from typing import TYPE_CHECKING, Any, List, Optional, Union

from httpx._models import Headers, Response

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_azure_openai_messages,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import ModelResponse

from ....exceptions import UnsupportedParamsError
from ....types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from ...base_llm.chat.transformation import BaseConfig
from ..common_utils import AzureOpenAIError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class AzureOpenAIConfig(BaseConfig):
    """
    Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions

    The class `AzureOpenAIConfig` provides configuration for the OpenAI's Chat API interface, for use with Azure. Below are the parameters::

    - `frequency_penalty` (number or null): Defaults to 0. Allows a value between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, thereby minimizing repetition.

    - `function_call` (string or object): This optional parameter controls how the model calls functions.

    - `functions` (array): An optional parameter. It is a list of functions for which the model may generate JSON inputs.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion.

    - `n` (integer or null): This optional parameter helps to set how many chat completion choices to generate for each input message.

    - `presence_penalty` (number or null): Defaults to 0. It penalizes new tokens based on if they appear in the text so far, hence increasing the model's likelihood to talk about new topics.

    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling.
    """

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "temperature",
            "n",
            "stream",
            "stream_options",
            "stop",
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "function_call",
            "functions",
            "tools",
            "tool_choice",
            "top_p",
            "logprobs",
            "top_logprobs",
            "response_format",
            "seed",
            "extra_headers",
            "parallel_tool_calls",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        api_version: str = "",
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model)

        api_version_times = api_version.split("-")
        api_version_year = api_version_times[0]
        api_version_month = api_version_times[1]
        api_version_day = api_version_times[2]
        for param, value in non_default_params.items():
            if param == "tool_choice":
                """
                This parameter requires API version 2023-12-01-preview or later

                tool_choice='required' is not supported as of 2024-05-01-preview
                """
                ## check if api version supports this param ##
                if (
                    api_version_year < "2023"
                    or (api_version_year == "2023" and api_version_month < "12")
                    or (
                        api_version_year == "2023"
                        and api_version_month == "12"
                        and api_version_day < "01"
                    )
                ):
                    if litellm.drop_params is True or (
                        drop_params is not None and drop_params is True
                    ):
                        pass
                    else:
                        raise UnsupportedParamsError(
                            status_code=400,
                            message=f"""Azure does not support 'tool_choice', for api_version={api_version}. Bump your API version to '2023-12-01-preview' or later. This parameter requires 'api_version="2023-12-01-preview"' or later. Azure API Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions""",
                        )
                elif value == "required" and (
                    api_version_year == "2024" and api_version_month <= "05"
                ):  ## check if tool_choice value is supported ##
                    if litellm.drop_params is True or (
                        drop_params is not None and drop_params is True
                    ):
                        pass
                    else:
                        raise UnsupportedParamsError(
                            status_code=400,
                            message=f"Azure does not support '{value}' as a {param} param, for api_version={api_version}. To drop 'tool_choice=required' for calls with this Azure API version, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\nAzure API Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions",
                        )
                else:
                    optional_params["tool_choice"] = value
            elif param == "response_format" and isinstance(value, dict):
                json_schema: Optional[dict] = None
                schema_name: str = ""
                if "response_schema" in value:
                    json_schema = value["response_schema"]
                    schema_name = "json_tool_call"
                elif "json_schema" in value:
                    json_schema = value["json_schema"]["schema"]
                    schema_name = value["json_schema"]["name"]
                """
                Follow similar approach to anthropic - translate to a single tool call. 

                When using tools in this way: - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
                - You usually want to provide a single tool
                - You should set tool_choice (see Forcing tool use) to instruct the model to explicitly use that tool
                - Remember that the model will pass the input to the tool, so the name of the tool and description should be from the model’s perspective.
                """
                if json_schema is not None and (
                    (api_version_year <= "2024" and api_version_month < "08")
                    or "gpt-4o" not in model
                ):  # azure api version "2024-08-01-preview" onwards supports 'json_schema' only for gpt-4o
                    _tool_choice = ChatCompletionToolChoiceObjectParam(
                        type="function",
                        function=ChatCompletionToolChoiceFunctionParam(
                            name=schema_name
                        ),
                    )

                    _tool = ChatCompletionToolParam(
                        type="function",
                        function=ChatCompletionToolParamFunctionChunk(
                            name=schema_name, parameters=json_schema
                        ),
                    )

                    optional_params["tools"] = [_tool]
                    optional_params["tool_choice"] = _tool_choice
                    optional_params["json_mode"] = True
                else:
                    optional_params["response_format"] = value
            elif param in supported_openai_params:
                optional_params[param] = value

        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        messages = convert_to_azure_openai_messages(messages)
        return {
            "model": model,
            "messages": messages,
            **optional_params,
        }

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        raise NotImplementedError(
            "Azure OpenAI handler.py has custom logic for transforming response, as it uses the OpenAI SDK."
        )

    def get_mapped_special_auth_params(self) -> dict:
        return {"token": "azure_ad_token"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "token":
                optional_params["azure_ad_token"] = value
        return optional_params

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4-and-gpt-4-turbo-model-availability
        """
        return ["europe", "sweden", "switzerland", "france", "uk"]

    def get_us_regions(self) -> List[str]:
        """
        Source: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4-and-gpt-4-turbo-model-availability
        """
        return [
            "us",
            "eastus",
            "eastus2",
            "eastus2euap",
            "eastus3",
            "southcentralus",
            "westus",
            "westus2",
            "westus3",
            "westus4",
        ]

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return AzureOpenAIError(
            message=error_message, status_code=status_code, headers=headers
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
    ) -> dict:
        raise NotImplementedError(
            "Azure OpenAI has custom logic for validating environment, as it uses the OpenAI SDK."
        )
