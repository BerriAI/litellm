import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.base_model_iterator import FakeStreamResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)
from litellm.utils import token_counter

from ..common_utils import ClarifaiError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class ClarifaiConfig(BaseConfig):
    """
    Reference: https://clarifai.com/meta/Llama-2/models/llama2-70b-chat
    """

    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_k: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "temperature",
            "max_tokens",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "temperature":
                optional_params["temperature"] = value
            elif param == "max_tokens":
                optional_params["max_tokens"] = value

        return optional_params

    def _completions_to_model(self, prompt: str, optional_params: dict) -> dict:
        params = {}
        if temperature := optional_params.get("temperature"):
            params["temperature"] = temperature
        if max_tokens := optional_params.get("max_tokens"):
            params["max_tokens"] = max_tokens
        return {
            "inputs": [{"data": {"text": {"raw": prompt}}}],
            "model": {"output_info": {"params": params}},
        }

    def _convert_model_to_url(self, model: str, api_base: str):
        user_id, app_id, model_id = model.split(".")
        return f"{api_base}/users/{user_id}/apps/{app_id}/models/{model_id}/outputs"

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        prompt = " ".join(convert_content_list_to_str(message) for message in messages)

        ## Load Config
        config = self.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        data = self._completions_to_model(
            prompt=prompt, optional_params=optional_params
        )

        return data

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return ClarifaiError(message=error_message, status_code=status_code)

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: str,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )
        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except httpx.HTTPStatusError as e:
            raise ClarifaiError(
                message=str(e),
                status_code=raw_response.status_code,
            )
        except Exception as e:
            raise ClarifaiError(
                message=str(e),
                status_code=422,
            )
        # print(completion_response)
        try:
            choices_list = []
            for idx, item in enumerate(completion_response["outputs"]):
                if len(item["data"]["text"]["raw"]) > 0:
                    message_obj = Message(content=item["data"]["text"]["raw"])
                else:
                    message_obj = Message(content=None)
                choice_obj = Choices(
                    finish_reason="stop",
                    index=idx + 1,  # check
                    message=message_obj,
                )
                choices_list.append(choice_obj)
            model_response.choices = choices_list  # type: ignore

        except Exception as e:
            raise ClarifaiError(
                message=str(e),
                status_code=422,
            )

        # Calculate Usage
        prompt_tokens = token_counter(model=model, messages=messages)
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content"))
        )
        model_response.model = model
        setattr(
            model_response,
            "usage",
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return ClarifaiModelResponseIterator(
            model_response=streaming_response,
            json_mode=json_mode,
        )


class ClarifaiModelResponseIterator(FakeStreamResponseIterator):
    def __init__(
        self,
        model_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        json_mode: Optional[bool] = False,
    ):
        super().__init__(
            model_response=model_response,
            json_mode=json_mode,
        )

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None

            text = (
                chunk.get("outputs", "")[0]
                .get("data", "")
                .get("text", "")
                .get("raw", "")
            )

            index: int = 0

            return GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
                provider_specific_fields=provider_specific_fields,
            )
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")
