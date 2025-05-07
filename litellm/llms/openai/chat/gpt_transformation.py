"""
Support for gpt model family 
"""

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

import httpx

import litellm
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _extract_reasoning_content,
    _handle_invalid_parallel_tool_calls,
    _should_convert_tool_call_to_json_mode,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import get_tool_call_names
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionFileObject,
    ChatCompletionFileObjectFile,
    ChatCompletionImageObject,
    ChatCompletionImageUrlObject,
    OpenAIChatCompletionChoices,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
)
from litellm.utils import convert_to_model_response_object

from ..common_utils import OpenAIError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenAIGPTConfig(BaseLLMModelInfo, BaseConfig):
    """
    Reference: https://platform.openai.com/docs/api-reference/chat/create

    The class `OpenAIConfig` provides configuration for the OpenAI's Chat API interface. Below are the parameters:

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

    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None

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
        response_format: Optional[dict] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        base_params = [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "max_completion_tokens",
            "modalities",
            "prediction",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "function_call",
            "functions",
            "max_retries",
            "extra_headers",
            "parallel_tool_calls",
            "audio",
        ]  # works across all models

        model_specific_params = []
        if (
            model != "gpt-3.5-turbo-16k" and model != "gpt-4"
        ):  # gpt-4 does not support 'response_format'
            model_specific_params.append("response_format")

        if (
            model in litellm.open_ai_chat_completion_models
        ) or model in litellm.open_ai_text_completion_models:
            model_specific_params.append(
                "user"
            )  # user is not a param supported by all openai-compatible endpoints - e.g. azure ai
        return base_params + model_specific_params

    def _map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        If any supported_openai_params are in non_default_params, add them to optional_params, so they are use in API call

        Args:
            non_default_params (dict): Non-default parameters to filter.
            optional_params (dict): Optional parameters to update.
            model (str): Model name for parameter support check.

        Returns:
            dict: Updated optional_params with supported non-default parameters.
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return self._map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str
    ) -> List[AllMessageValues]:
        """OpenAI no longer supports image_url as a string, so we need to convert it to a dict"""
        for message in messages:
            message_content = message.get("content")
            if message_content and isinstance(message_content, list):
                for content_item in message_content:
                    litellm_specific_params = {"format"}
                    if content_item.get("type") == "image_url":
                        content_item = cast(ChatCompletionImageObject, content_item)
                        if isinstance(content_item["image_url"], str):
                            content_item["image_url"] = {
                                "url": content_item["image_url"],
                            }
                        elif isinstance(content_item["image_url"], dict):
                            new_image_url_obj = ChatCompletionImageUrlObject(
                                **{  # type: ignore
                                    k: v
                                    for k, v in content_item["image_url"].items()
                                    if k not in litellm_specific_params
                                }
                            )
                            content_item["image_url"] = new_image_url_obj
                    elif content_item.get("type") == "file":
                        content_item = cast(ChatCompletionFileObject, content_item)
                        file_obj = content_item["file"]
                        new_file_obj = ChatCompletionFileObjectFile(
                            **{  # type: ignore
                                k: v
                                for k, v in file_obj.items()
                                if k not in litellm_specific_params
                            }
                        )
                        content_item["file"] = new_file_obj
        return messages

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the overall request to be sent to the API.

        Returns:
            dict: The transformed request. Sent as the body of the API call.
        """
        messages = self._transform_messages(messages=messages, model=model)
        return {
            "model": model,
            "messages": messages,
            **optional_params,
        }

    def _passed_in_tools(self, optional_params: dict) -> bool:
        return optional_params.get("tools", None) is not None

    def _check_and_fix_if_content_is_tool_call(
        self, content: str, optional_params: dict
    ) -> Optional[ChatCompletionMessageToolCall]:
        """
        Check if the content is a tool call
        """
        import json

        if not self._passed_in_tools(optional_params):
            return None
        tool_call_names = get_tool_call_names(optional_params.get("tools", []))
        try:
            json_content = json.loads(content)
            if (
                json_content.get("type") == "function"
                and json_content.get("name") in tool_call_names
            ):
                return ChatCompletionMessageToolCall(
                    function=Function(
                        name=json_content.get("name"),
                        arguments=json_content.get("arguments"),
                    )
                )
        except Exception:
            return None

        return None

    def _get_finish_reason(self, message: Message, received_finish_reason: str) -> str:
        if message.tool_calls is not None:
            return "tool_calls"
        else:
            return received_finish_reason

    def _transform_choices(
        self,
        choices: List[OpenAIChatCompletionChoices],
        json_mode: Optional[bool] = None,
        optional_params: Optional[dict] = None,
    ) -> List[Choices]:
        transformed_choices = []

        for choice in choices:
            ## HANDLE JSON MODE - anthropic returns single function call]
            tool_calls = choice["message"].get("tool_calls", None)
            new_tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
            message_content = choice["message"].get("content", None)
            if tool_calls is not None:
                _openai_tool_calls = []
                for _tc in tool_calls:
                    _openai_tc = ChatCompletionMessageToolCall(**_tc)  # type: ignore
                    _openai_tool_calls.append(_openai_tc)
                fixed_tool_calls = _handle_invalid_parallel_tool_calls(
                    _openai_tool_calls
                )

                if fixed_tool_calls is not None:
                    new_tool_calls = fixed_tool_calls
            elif (
                optional_params is not None
                and message_content
                and isinstance(message_content, str)
            ):
                new_tool_call = self._check_and_fix_if_content_is_tool_call(
                    message_content, optional_params
                )
                if new_tool_call is not None:
                    choice["message"]["content"] = None  # remove the content
                    new_tool_calls = [new_tool_call]

            translated_message: Optional[Message] = None
            finish_reason: Optional[str] = None
            if new_tool_calls and _should_convert_tool_call_to_json_mode(
                tool_calls=new_tool_calls,
                convert_tool_call_to_json_mode=json_mode,
            ):
                # to support response_format on claude models
                json_mode_content_str: Optional[str] = (
                    str(new_tool_calls[0]["function"].get("arguments", "")) or None
                )
                if json_mode_content_str is not None:
                    translated_message = Message(content=json_mode_content_str)
                    finish_reason = "stop"

            if translated_message is None:
                ## get the reasoning content
                (
                    reasoning_content,
                    content_str,
                ) = _extract_reasoning_content(cast(dict, choice["message"]))

                translated_message = Message(
                    role="assistant",
                    content=content_str,
                    reasoning_content=reasoning_content,
                    thinking_blocks=None,
                    tool_calls=new_tool_calls,
                )

            if finish_reason is None:
                finish_reason = choice["finish_reason"]

            translated_choice = Choices(
                finish_reason=finish_reason,
                index=choice["index"],
                message=translated_message,
                logprobs=None,
                enhancements=None,
            )

            translated_choice.finish_reason = self._get_finish_reason(
                translated_message, choice["finish_reason"]
            )
            transformed_choices.append(translated_choice)

        return transformed_choices

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
        """
        Transform the response from the API.

        Returns:
            dict: The transformed response.
        """

        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except Exception as e:
            response_headers = getattr(raw_response, "headers", None)
            raise OpenAIError(
                message="Unable to get json response - {}, Original Response: {}".format(
                    str(e), raw_response.text
                ),
                status_code=raw_response.status_code,
                headers=response_headers,
            )
        raw_response_headers = dict(raw_response.headers)
        final_response_obj = convert_to_model_response_object(
            response_object=completion_response,
            model_response_object=model_response,
            hidden_params={"headers": raw_response_headers},
            _response_headers=raw_response_headers,
        )

        return cast(ModelResponse, final_response_obj)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=cast(httpx.Headers, headers),
        )

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
        Get the complete URL for the API call.

        Returns:
            str: The complete URL for the API call.
        """
        if api_base is None:
            api_base = "https://api.openai.com"
        endpoint = "chat/completions"

        # Remove trailing slash from api_base if present
        api_base = api_base.rstrip("/")

        # Check if endpoint is already in the api_base
        if endpoint in api_base:
            return api_base

        return f"{api_base}/{endpoint}"

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
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        # Ensure Content-Type is set to application/json
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        return headers

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        """
        Calls OpenAI's `/v1/models` endpoint and returns the list of models.
        """

        if api_base is None:
            api_base = "https://api.openai.com"
        if api_key is None:
            api_key = get_secret_str("OPENAI_API_KEY")

        response = litellm.module_level_client.get(
            url=f"{api_base}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get models: {response.text}")

        models = response.json()["data"]
        return [model["id"] for model in models]

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

    @staticmethod
    def get_base_model(model: Optional[str] = None) -> Optional[str]:
        return model

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return OpenAIChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class OpenAIChatCompletionStreamingHandler(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            return ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                model=chunk["model"],
                choices=chunk["choices"],
            )
        except Exception as e:
            raise e
