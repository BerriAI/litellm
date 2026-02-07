import json
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
    version,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import LlmProviders
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

from ..common_utils import API_BASE, BytezError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


# 5 minute timeout (models may need to load)
STREAMING_TIMEOUT = 60 * 5


class BytezChatConfig(BaseConfig):
    """
    Configuration class for Bytez's API interface.
    """

    def __init__(
        self,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
        # mark the class as using a custom stream wrapper because the default only iterates on lines
        setattr(self.__class__, "has_custom_stream_wrapper", True)

        self.openai_to_bytez_param_map = {
            "stream": "stream",
            "max_tokens": "max_new_tokens",
            "max_completion_tokens": "max_new_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "n": "num_return_sequences",
            "max_retries": "max_retries",
            "seed": False,  # TODO requires backend changes
            "stop": False,  # TODO requires backend changes
            "logit_bias": False,  # TODO requires backend changes
            "logprobs": False,  # TODO requires backend changes
            "frequency_penalty": False,
            "presence_penalty": False,
            "top_logprobs": False,
            "modalities": False,
            "prediction": False,
            "stream_options": False,
            "tools": False,
            "tool_choice": False,
            "function_call": False,
            "functions": False,
            "extra_headers": False,
            "parallel_tool_calls": False,
            "audio": False,
            "web_search_options": False,
        }

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = []
        for key, value in self.openai_to_bytez_param_map.items():
            if value:
                supported_params.append(key)

        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:

        adapted_params = {}

        all_params = {**non_default_params, **optional_params}

        for key, value in all_params.items():

            alias = self.openai_to_bytez_param_map.get(key)

            if alias is False:
                if drop_params:
                    continue

                raise Exception(f"param `{key}` is not supported on Bytez")

            if alias is None:
                adapted_params[key] = value
                continue

            adapted_params[alias] = value

        return adapted_params

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

        headers.update(
            {
                "content-type": "application/json",
                "Authorization": f"Key {api_key}",
                "user-agent": f"litellm/{version}",
            }
        )

        if not messages:
            raise Exception(
                "kwarg `messages` must be an array of messages that follow the openai chat standard"
            )

        if not api_key:
            raise Exception("Missing api_key, make sure you pass in your api key")


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
        return f"{API_BASE}/{model}"

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        stream = optional_params.get("stream", False)

        # we add stream not as an additional param, but as a primary prop on the request body, this is always defined if stream == True
        if optional_params.get("stream"):
            del optional_params["stream"]

        messages = adapt_messages_to_bytez_standard(messages=messages)  # type: ignore

        data = {
            "messages": messages,
            "stream": stream,
            "params": optional_params,
        }

        return data

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

        json = raw_response.json()  # noqa: F811

        error = json.get("error")

        if error is not None:
            raise BytezError(
                message=str(json["error"]),
                status_code=raw_response.status_code,
            )

        # set meta data here
        model_response.created = int(time.time())
        model_response.model = model

        # Add the output
        output = json.get("output")

        message = model_response.choices[0].message  # type: ignore

        message.content = output["content"][0]["text"]

        messages = adapt_messages_to_bytez_standard(messages=messages)  # type: ignore

        # NOTE We are approximating tokens, to get the true values we will need to update our BE
        prompt_tokens = get_tokens_from_messages(messages)  # type: ignore

        output_messages = adapt_messages_to_bytez_standard(messages=[output])

        completion_tokens = get_tokens_from_messages(output_messages)

        total_tokens = prompt_tokens + completion_tokens

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        model_response.usage = usage  # type: ignore

        model_response._hidden_params["additional_headers"] = raw_response.headers
        message.provider_specific_fields = {
            "ratelimit-limit": raw_response.headers.get("ratelimit-limit"),
            "ratelimit-remaining": raw_response.headers.get("ratelimit-remaining"),
            "ratelimit-reset": raw_response.headers.get("ratelimit-reset"),
            "inference-meter": raw_response.headers.get("inference-meter"),
            "inference-time": raw_response.headers.get("inference-time"),
        }

        # TODO additional data when supported
        # message.tool_calls
        # message.function_call

        return model_response

    @track_llm_api_timing()
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
        signed_json_body: Optional[bytes] = None,
    ) -> "BytezCustomStreamWrapper":
        if client is None or isinstance(client, AsyncHTTPHandler):
            client = _get_httpx_client(params={})

        try:
            response = client.post(
                api_base,
                headers=headers,
                data=json.dumps(data),
                stream=True,
                logging_obj=logging_obj,
                timeout=STREAMING_TIMEOUT,
            )
        except httpx.HTTPStatusError as e:
            raise BytezError(
                status_code=e.response.status_code, message=e.response.text
            )

        if response.status_code != 200:
            raise BytezError(status_code=response.status_code, message=response.text)

        completion_stream = response.iter_text()

        streaming_response = BytezCustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return streaming_response

    @track_llm_api_timing()
    async def get_async_custom_stream_wrapper(
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
        signed_json_body: Optional[bytes] = None,
    ) -> "BytezCustomStreamWrapper":
        if client is None or isinstance(client, HTTPHandler):
            client = get_async_httpx_client(llm_provider=LlmProviders.BYTEZ, params={})

        try:
            response = await client.post(
                api_base,
                headers=headers,
                data=json.dumps(data),
                stream=True,
                logging_obj=logging_obj,
                timeout=STREAMING_TIMEOUT,
            )
        except httpx.HTTPStatusError as e:
            raise BytezError(
                status_code=e.response.status_code, message=e.response.text
            )

        if response.status_code != 200:
            raise BytezError(status_code=response.status_code, message=response.text)

        completion_stream = response.aiter_text()

        streaming_response = BytezCustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return streaming_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BytezError(status_code=status_code, message=error_message)


class BytezCustomStreamWrapper(CustomStreamWrapper):
    def chunk_creator(self, chunk: Any):
        try:
            model_response = self.model_response_creator()
            response_obj: Dict[str, Any] = {}

            response_obj = {
                "text": chunk,
                "is_finished": False,
                "finish_reason": "",
            }

            completion_obj: Dict[str, Any] = {"content": chunk}

            return self.return_processed_chunk_logic(
                completion_obj=completion_obj,
                model_response=model_response,  # type: ignore
                response_obj=response_obj,
            )

        except StopIteration:
            raise StopIteration
        except Exception as e:
            traceback.format_exc()
            setattr(e, "message", str(e))
            raise exception_type(
                model=self.model,
                custom_llm_provider=self.custom_llm_provider,
                original_exception=e,
            )


# litellm/types/llms/openai.py is a good reference for what is supported
open_ai_to_bytez_content_item_map = {
    "text": {"type": "text", "value_name": "text"},
    "image_url": {"type": "image", "value_name": "url"},
    "input_audio": {"type": "audio", "value_name": "url"},
    "video_url": {"type": "video", "value_name": "url"},
    "document": None,
    "file": None,
}


def adapt_messages_to_bytez_standard(messages: List[Dict]):

    messages = _adapt_string_only_content_to_lists(messages)

    new_messages = []

    for message in messages:

        role = message["role"]
        content: list = message["content"]

        new_content = []

        for content_item in content:
            type: Union[str, None] = content_item.get("type")

            if not type:
                raise Exception("Prop `type` is not a string")

            content_item_map = open_ai_to_bytez_content_item_map[type]

            if not content_item_map:
                raise Exception(f"Prop `{type}` is not supported")

            new_type = content_item_map["type"]

            value_name = content_item_map["value_name"]

            value: Union[str, None] = content_item.get(value_name)

            if not value:
                raise Exception(f"Prop `{value_name}` is not a string")

            new_content.append({"type": new_type, value_name: value})

        new_messages.append({"role": role, "content": new_content})

    return new_messages


# "content": "The cat ran so fast"
# becomes
# "content": [{"type": "text", "text": "The cat ran so fast"}]
def _adapt_string_only_content_to_lists(messages: List[Dict]):
    new_messages = []

    for message in messages:

        role = message.get("role")
        content = message.get("content")

        new_content = []

        if isinstance(content, str):
            new_content.append({"type": "text", "text": content})

        elif isinstance(content, dict):
            new_content.append(content)

        elif isinstance(content, list):

            new_content_items = []
            for content_item in content:
                if isinstance(content_item, str):
                    new_content_items.append({"type": "text", "text": content_item})
                elif isinstance(content_item, dict):
                    new_content_items.append(content_item)
                else:
                    raise Exception(
                        "`content` can only contain strings or openai content dicts"
                    )

            new_content += new_content_items
        else:
            raise Exception("Content must be a string")

        new_messages.append({"role": role, "content": new_content})

    return new_messages


# TODO get this from the api instead of doing it here, will require backend work
def get_tokens_from_messages(messages: List[dict]):
    total = 0

    for message in messages:
        content: List[dict] = message["content"]

        for content_item in content:
            type = content_item["type"]
            if type == "text":
                value: str = content_item["text"]
                words = value.split(" ")
                total += len(words)
                continue
            # we'll count media as single tokens for now
            total += 1

    return total
