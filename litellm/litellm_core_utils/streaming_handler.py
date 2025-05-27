import asyncio
import collections.abc
import datetime
import json
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Union, cast

import httpx
from pydantic import BaseModel

import litellm
from litellm import verbose_logger
from litellm.litellm_core_utils.redact_messages import LiteLLMLoggingObject
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.types.llms.openai import ChatCompletionChunk
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import Delta
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from ..exceptions import OpenAIError
from .core_helpers import map_finish_reason, process_response_headers
from .exception_mapping_utils import exception_type
from .llm_response_utils.get_api_base import get_api_base
from .rules import Rules


def is_async_iterable(obj: Any) -> bool:
    """
    Check if an object is an async iterable (can be used with 'async for').

    Args:
        obj: Any Python object to check

    Returns:
        bool: True if the object is async iterable, False otherwise
    """
    return isinstance(obj, collections.abc.AsyncIterable)


def print_verbose(print_statement):
    try:
        if litellm.set_verbose:
            print(print_statement)  # noqa
    except Exception:
        pass


class CustomStreamWrapper:
    def __init__(
        self,
        completion_stream,
        model,
        logging_obj: Any,
        custom_llm_provider: Optional[str] = None,
        stream_options=None,
        make_call: Optional[Callable] = None,
        _response_headers: Optional[dict] = None,
    ):
        self.model = model
        self.make_call = make_call
        self.custom_llm_provider = custom_llm_provider
        self.logging_obj: LiteLLMLoggingObject = logging_obj
        self.completion_stream = completion_stream
        self.sent_first_chunk = False
        self.sent_last_chunk = False

        litellm_params: GenericLiteLLMParams = GenericLiteLLMParams(
            **self.logging_obj.model_call_details.get("litellm_params", {})
        )
        self.merge_reasoning_content_in_choices: bool = (
            litellm_params.merge_reasoning_content_in_choices or False
        )
        self.sent_first_thinking_block = False
        self.sent_last_thinking_block = False
        self.thinking_content = ""

        self.system_fingerprint: Optional[str] = None
        self.received_finish_reason: Optional[str] = None
        self.intermittent_finish_reason: Optional[
            str
        ] = None  # finish reasons that show up mid-stream
        self.special_tokens = [
            "<|assistant|>",
            "<|system|>",
            "<|user|>",
            "<s>",
            "</s>",
            "<|im_end|>",
            "<|im_start|>",
        ]
        self.holding_chunk = ""
        self.complete_response = ""
        self.response_uptil_now = ""
        _model_info: Dict = litellm_params.model_info or {}

        _api_base = get_api_base(
            model=model or "",
            optional_params=self.logging_obj.model_call_details.get(
                "litellm_params", {}
            ),
        )

        self._hidden_params = {
            "model_id": (_model_info.get("id", None)),
            "api_base": _api_base,
        }  # returned as x-litellm-model-id response header in proxy

        self._hidden_params["additional_headers"] = process_response_headers(
            _response_headers or {}
        )  # GUARANTEE OPENAI HEADERS IN RESPONSE

        self._response_headers = _response_headers
        self.response_id: Optional[str] = None
        self.logging_loop = None
        self.rules = Rules()
        self.stream_options = stream_options or getattr(
            logging_obj, "stream_options", None
        )
        self.messages = getattr(logging_obj, "messages", None)
        self.sent_stream_usage = False
        self.send_stream_usage = (
            True if self.check_send_stream_usage(self.stream_options) else False
        )
        self.tool_call = False
        self.chunks: List = (
            []
        )  # keep track of the returned chunks - used for calculating the input/output tokens for stream options
        self.is_function_call = self.check_is_function_call(logging_obj=logging_obj)

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def check_send_stream_usage(self, stream_options: Optional[dict]):
        return (
            stream_options is not None
            and stream_options.get("include_usage", False) is True
        )

    def check_is_function_call(self, logging_obj) -> bool:
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            is_function_call,
        )

        if hasattr(logging_obj, "optional_params") and isinstance(
            logging_obj.optional_params, dict
        ):
            if is_function_call(logging_obj.optional_params):
                return True

        return False

    def process_chunk(self, chunk: str):
        """
        NLP Cloud streaming returns the entire response, for each chunk. Process this, to only return the delta.
        """
        try:
            chunk = chunk.strip()
            self.complete_response = self.complete_response.strip()

            if chunk.startswith(self.complete_response):
                # Remove last_sent_chunk only if it appears at the start of the new chunk
                chunk = chunk[len(self.complete_response) :]

            self.complete_response += chunk
            return chunk
        except Exception as e:
            raise e

    def safety_checker(self) -> None:
        """
        Fixes - https://github.com/BerriAI/litellm/issues/5158

        if the model enters a loop and starts repeating the same chunk again, break out of loop and raise an internalservererror - allows for retries.

        Raises - InternalServerError, if LLM enters infinite loop while streaming
        """
        if len(self.chunks) >= litellm.REPEATED_STREAMING_CHUNK_LIMIT:
            # Get the last n chunks
            last_chunks = self.chunks[-litellm.REPEATED_STREAMING_CHUNK_LIMIT :]

            # Extract the relevant content from the chunks
            last_contents = [chunk.choices[0].delta.content for chunk in last_chunks]

            # Check if all extracted contents are identical
            if all(content == last_contents[0] for content in last_contents):
                if (
                    last_contents[0] is not None
                    and isinstance(last_contents[0], str)
                    and len(last_contents[0]) > 2
                ):  # ignore empty content - https://github.com/BerriAI/litellm/issues/5158#issuecomment-2287156946
                    # All last n chunks are identical
                    raise litellm.InternalServerError(
                        message="The model is repeating the same chunk = {}.".format(
                            last_contents[0]
                        ),
                        model="",
                        llm_provider="",
                    )

    def check_special_tokens(self, chunk: str, finish_reason: Optional[str]):
        """
        Output parse <s> / </s> special tokens for sagemaker + hf streaming.
        """
        hold = False
        if self.custom_llm_provider != "sagemaker":
            return hold, chunk

        if finish_reason:
            for token in self.special_tokens:
                if token in chunk:
                    chunk = chunk.replace(token, "")
            return hold, chunk

        if self.sent_first_chunk is True:
            return hold, chunk

        curr_chunk = self.holding_chunk + chunk
        curr_chunk = curr_chunk.strip()

        for token in self.special_tokens:
            if len(curr_chunk) < len(token) and curr_chunk in token:
                hold = True
                self.holding_chunk = curr_chunk
            elif len(curr_chunk) >= len(token):
                if token in curr_chunk:
                    self.holding_chunk = curr_chunk.replace(token, "")
                    hold = True
            else:
                pass

        if hold is False:  # reset
            self.holding_chunk = ""
        return hold, curr_chunk

    def handle_predibase_chunk(self, chunk):
        try:
            if not isinstance(chunk, str):
                chunk = chunk.decode(
                    "utf-8"
                )  # DO NOT REMOVE this: This is required for HF inference API + Streaming
            text = ""
            is_finished = False
            finish_reason = ""
            print_verbose(f"chunk: {chunk}")
            if chunk.startswith("data:"):
                data_json = json.loads(chunk[5:])
                print_verbose(f"data json: {data_json}")
                if "token" in data_json and "text" in data_json["token"]:
                    text = data_json["token"]["text"]
                if data_json.get("details", False) and data_json["details"].get(
                    "finish_reason", False
                ):
                    is_finished = True
                    finish_reason = data_json["details"]["finish_reason"]
                elif data_json.get(
                    "generated_text", False
                ):  # if full generated text exists, then stream is complete
                    text = ""  # don't return the final bos token
                    is_finished = True
                    finish_reason = "stop"
                elif data_json.get("error", False):
                    raise Exception(data_json.get("error"))
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            elif "error" in chunk:
                raise ValueError(chunk)
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except Exception as e:
            raise e

    def handle_ai21_chunk(self, chunk):  # fake streaming
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            text = data_json["completions"][0]["data"]["text"]
            is_finished = True
            finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except Exception:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_maritalk_chunk(self, chunk):  # fake streaming
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            text = data_json["answer"]
            is_finished = True
            finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except Exception:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_nlp_cloud_chunk(self, chunk):
        text = ""
        is_finished = False
        finish_reason = ""
        try:
            if self.model and "dolphin" in self.model:
                chunk = self.process_chunk(chunk=chunk)
            else:
                data_json = json.loads(chunk)
                chunk = data_json["generated_text"]
            text = chunk
            if "[DONE]" in text:
                text = text.replace("[DONE]", "")
                is_finished = True
                finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except Exception:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_aleph_alpha_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            text = data_json["completions"][0]["completion"]
            is_finished = True
            finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except Exception:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_azure_chunk(self, chunk):
        is_finished = False
        finish_reason = ""
        text = ""
        print_verbose(f"chunk: {chunk}")
        if "data: [DONE]" in chunk:
            text = ""
            is_finished = True
            finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        elif chunk.startswith("data:"):
            data_json = json.loads(chunk[5:])  # chunk.startswith("data:"):
            try:
                if len(data_json["choices"]) > 0:
                    delta = data_json["choices"][0]["delta"]
                    text = "" if delta is None else delta.get("content", "")
                    if data_json["choices"][0].get("finish_reason", None):
                        is_finished = True
                        finish_reason = data_json["choices"][0]["finish_reason"]
                print_verbose(
                    f"text: {text}; is_finished: {is_finished}; finish_reason: {finish_reason}"
                )
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            except Exception:
                raise ValueError(
                    f"Unable to parse response. Original response: {chunk}"
                )
        elif "error" in chunk:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")
        else:
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }

    def handle_replicate_chunk(self, chunk):
        try:
            text = ""
            is_finished = False
            finish_reason = ""
            if "output" in chunk:
                text = chunk["output"]
            if "status" in chunk:
                if chunk["status"] == "succeeded":
                    is_finished = True
                    finish_reason = "stop"
            elif chunk.get("error", None):
                raise Exception(chunk["error"])
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except Exception:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_openai_chat_completion_chunk(self, chunk):
        try:
            print_verbose(f"\nRaw OpenAI Chunk\n{chunk}\n")
            str_line = chunk
            text = ""
            is_finished = False
            finish_reason = None
            logprobs = None
            usage = None

            if str_line and str_line.choices and len(str_line.choices) > 0:
                if (
                    str_line.choices[0].delta is not None
                    and str_line.choices[0].delta.content is not None
                ):
                    text = str_line.choices[0].delta.content
                else:  # function/tool calling chunk - when content is None. in this case we just return the original chunk from openai
                    pass
                if str_line.choices[0].finish_reason:
                    is_finished = True
                    finish_reason = str_line.choices[0].finish_reason

                # checking for logprobs
                if (
                    hasattr(str_line.choices[0], "logprobs")
                    and str_line.choices[0].logprobs is not None
                ):
                    logprobs = str_line.choices[0].logprobs
                else:
                    logprobs = None

            usage = getattr(str_line, "usage", None)

            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
                "logprobs": logprobs,
                "original_chunk": str_line,
                "usage": usage,
            }
        except Exception as e:
            raise e

    def handle_azure_text_completion_chunk(self, chunk):
        try:
            print_verbose(f"\nRaw OpenAI Chunk\n{chunk}\n")
            text = ""
            is_finished = False
            finish_reason = None
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                text = choices[0].text
                if choices[0].finish_reason is not None:
                    is_finished = True
                    finish_reason = choices[0].finish_reason
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }

        except Exception as e:
            raise e

    def handle_openai_text_completion_chunk(self, chunk):
        try:
            print_verbose(f"\nRaw OpenAI Chunk\n{chunk}\n")
            text = ""
            is_finished = False
            finish_reason = None
            usage = None
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                text = choices[0].text
                if choices[0].finish_reason is not None:
                    is_finished = True
                    finish_reason = choices[0].finish_reason
            usage = getattr(chunk, "usage", None)
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
                "usage": usage,
            }

        except Exception as e:
            raise e

    def handle_baseten_chunk(self, chunk):
        try:
            chunk = chunk.decode("utf-8")
            if len(chunk) > 0:
                if chunk.startswith("data:"):
                    data_json = json.loads(chunk[5:])
                    if "token" in data_json and "text" in data_json["token"]:
                        return data_json["token"]["text"]
                    else:
                        return ""
                data_json = json.loads(chunk)
                if "model_output" in data_json:
                    if (
                        isinstance(data_json["model_output"], dict)
                        and "data" in data_json["model_output"]
                        and isinstance(data_json["model_output"]["data"], list)
                    ):
                        return data_json["model_output"]["data"][0]
                    elif isinstance(data_json["model_output"], str):
                        return data_json["model_output"]
                    elif "completion" in data_json and isinstance(
                        data_json["completion"], str
                    ):
                        return data_json["completion"]
                    else:
                        raise ValueError(
                            f"Unable to parse response. Original response: {chunk}"
                        )
                else:
                    return ""
            else:
                return ""
        except Exception as e:
            verbose_logger.exception(
                "litellm.CustomStreamWrapper.handle_baseten_chunk(): Exception occured - {}".format(
                    str(e)
                )
            )
            return ""

    def handle_triton_stream(self, chunk):
        try:
            if isinstance(chunk, dict):
                parsed_response = chunk
            elif isinstance(chunk, (str, bytes)):
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                if "text_output" in chunk:
                    response = (
                        CustomStreamWrapper._strip_sse_data_from_chunk(chunk) or ""
                    )
                    response = response.strip()
                    parsed_response = json.loads(response)
                else:
                    return {
                        "text": "",
                        "is_finished": False,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }
            else:
                print_verbose(f"chunk: {chunk} (Type: {type(chunk)})")
                raise ValueError(
                    f"Unable to parse response. Original response: {chunk}"
                )
            text = parsed_response.get("text_output", "")
            finish_reason = parsed_response.get("stop_reason")
            is_finished = parsed_response.get("is_finished", False)
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
                "prompt_tokens": parsed_response.get("input_token_count", 0),
                "completion_tokens": parsed_response.get("generated_token_count", 0),
            }
            return {"text": "", "is_finished": False}
        except Exception as e:
            raise e

    def model_response_creator(
        self, chunk: Optional[dict] = None, hidden_params: Optional[dict] = None
    ):
        _model = self.model
        _received_llm_provider = self.custom_llm_provider
        _logging_obj_llm_provider = self.logging_obj.model_call_details.get("custom_llm_provider", None)  # type: ignore
        if (
            _received_llm_provider == "openai"
            and _received_llm_provider != _logging_obj_llm_provider
        ):
            _model = "{}/{}".format(_logging_obj_llm_provider, _model)
        if chunk is None:
            chunk = {}
        else:
            # pop model keyword
            chunk.pop("model", None)

        chunk_dict = {}
        for key, value in chunk.items():
            if key != "stream":
                chunk_dict[key] = value

        args = {
            "model": _model,
            "stream_options": self.stream_options,
            **chunk_dict,
        }

        model_response = ModelResponseStream(**args)
        if self.response_id is not None:
            model_response.id = self.response_id
        else:
            self.response_id = model_response.id  # type: ignore
        if self.system_fingerprint is not None:
            model_response.system_fingerprint = self.system_fingerprint
        if hidden_params is not None:
            model_response._hidden_params = hidden_params
        model_response._hidden_params["custom_llm_provider"] = _logging_obj_llm_provider
        model_response._hidden_params["created_at"] = time.time()
        model_response._hidden_params = {
            **model_response._hidden_params,
            **self._hidden_params,
        }

        if (
            len(model_response.choices) > 0
            and getattr(model_response.choices[0], "delta") is not None
        ):
            # do nothing, if object instantiated
            pass
        else:
            model_response.choices = [StreamingChoices(finish_reason=None)]
        return model_response

    def is_delta_empty(self, delta: Delta) -> bool:
        is_empty = True
        if delta.content:
            is_empty = False
        elif delta.tool_calls is not None:
            is_empty = False
        elif delta.function_call is not None:
            is_empty = False
        return is_empty

    def set_model_id(
        self, id: str, model_response: ModelResponseStream
    ) -> ModelResponseStream:
        """
        Set the model id and response id to the given id.

        Ensure model id is always the same across all chunks.

        If a valid ID is received in any chunk, use it for the response.
        """
        if self.response_id is None and id and isinstance(id, str) and id.strip():
            self.response_id = id

        if id and isinstance(id, str) and id.strip():
            model_response._hidden_params["received_model_id"] = id

        if self.response_id is not None and isinstance(self.response_id, str):
            model_response.id = self.response_id
        return model_response

    def copy_model_response_level_provider_specific_fields(
        self,
        original_chunk: Union[ModelResponseStream, ChatCompletionChunk],
        model_response: ModelResponseStream,
    ) -> ModelResponseStream:
        """
        Copy provider_specific_fields from original_chunk to model_response.
        """
        provider_specific_fields = getattr(
            original_chunk, "provider_specific_fields", None
        )
        if provider_specific_fields is not None:
            model_response.provider_specific_fields = provider_specific_fields
            for k, v in provider_specific_fields.items():
                setattr(model_response, k, v)
        return model_response

    def is_chunk_non_empty(
        self,
        completion_obj: Dict[str, Any],
        model_response: ModelResponseStream,
        response_obj: Dict[str, Any],
    ) -> bool:
        if (
            "content" in completion_obj
            and (
                isinstance(completion_obj["content"], str)
                and len(completion_obj["content"]) > 0
            )
            or (
                "tool_calls" in completion_obj
                and completion_obj["tool_calls"] is not None
                and len(completion_obj["tool_calls"]) > 0
            )
            or (
                "function_call" in completion_obj
                and completion_obj["function_call"] is not None
            )
            or (
                "reasoning_content" in model_response.choices[0].delta
                and model_response.choices[0].delta.reasoning_content is not None
            )
            or (model_response.choices[0].delta.provider_specific_fields is not None)
            or (
                "provider_specific_fields" in model_response
                and model_response.choices[0].delta.provider_specific_fields is not None
            )
            or (
                "provider_specific_fields" in response_obj
                and response_obj["provider_specific_fields"] is not None
            )
            or (
                "annotations" in model_response.choices[0].delta
                and model_response.choices[0].delta.annotations is not None
            )
        ):
            return True
        else:
            return False

    def return_processed_chunk_logic(  # noqa
        self,
        completion_obj: Dict[str, Any],
        model_response: ModelResponseStream,
        response_obj: Dict[str, Any],
    ):
        print_verbose(
            f"completion_obj: {completion_obj}, model_response.choices[0]: {model_response.choices[0]}, response_obj: {response_obj}"
        )
        is_chunk_non_empty = self.is_chunk_non_empty(
            completion_obj, model_response, response_obj
        )
        if (
            is_chunk_non_empty
        ):  # cannot set content of an OpenAI Object to be an empty string
            self.safety_checker()
            hold, model_response_str = self.check_special_tokens(
                chunk=completion_obj["content"],
                finish_reason=model_response.choices[0].finish_reason,
            )  # filter out bos/eos tokens from openai-compatible hf endpoints
            print_verbose(f"hold - {hold}, model_response_str - {model_response_str}")
            if hold is False:
                ## check if openai/azure chunk
                original_chunk = response_obj.get("original_chunk", None)
                if original_chunk:
                    if len(original_chunk.choices) > 0:
                        choices = []
                        for choice in original_chunk.choices:
                            try:
                                if isinstance(choice, BaseModel):
                                    choice_json = choice.model_dump()  # type: ignore
                                    choice_json.pop(
                                        "finish_reason", None
                                    )  # for mistral etc. which return a value in their last chunk (not-openai compatible).
                                    print_verbose(f"choice_json: {choice_json}")
                                    choices.append(StreamingChoices(**choice_json))
                            except Exception:
                                choices.append(StreamingChoices())
                        print_verbose(f"choices in streaming: {choices}")
                        setattr(model_response, "choices", choices)
                    else:
                        return
                    model_response.system_fingerprint = (
                        original_chunk.system_fingerprint
                    )
                    setattr(
                        model_response,
                        "citations",
                        getattr(original_chunk, "citations", None),
                    )
                    print_verbose(f"self.sent_first_chunk: {self.sent_first_chunk}")
                    if self.sent_first_chunk is False:
                        model_response.choices[0].delta["role"] = "assistant"
                        self.sent_first_chunk = True
                    elif self.sent_first_chunk is True and hasattr(
                        model_response.choices[0].delta, "role"
                    ):
                        _initial_delta = model_response.choices[0].delta.model_dump()

                        _initial_delta.pop("role", None)
                        model_response.choices[0].delta = Delta(**_initial_delta)
                    verbose_logger.debug(
                        f"model_response.choices[0].delta: {model_response.choices[0].delta}"
                    )
                else:
                    ## else
                    completion_obj["content"] = model_response_str
                    if self.sent_first_chunk is False:
                        completion_obj["role"] = "assistant"
                        self.sent_first_chunk = True
                    if response_obj.get("provider_specific_fields") is not None:
                        completion_obj["provider_specific_fields"] = response_obj[
                            "provider_specific_fields"
                        ]
                    model_response.choices[0].delta = Delta(**completion_obj)
                    _index: Optional[int] = completion_obj.get("index")
                    if _index is not None:
                        model_response.choices[0].index = _index

                self._optional_combine_thinking_block_in_choices(
                    model_response=model_response
                )
                print_verbose(f"returning model_response: {model_response}")
                return model_response
            else:
                return
        elif self.received_finish_reason is not None:
            if self.sent_last_chunk is True:
                # Bedrock returns the guardrail trace in the last chunk - we want to return this here
                if self.custom_llm_provider == "bedrock" and "trace" in model_response:
                    return model_response

                # Default - return StopIteration
                if hasattr(model_response, "usage"):
                    self.chunks.append(model_response)
                raise StopIteration
            # flush any remaining holding chunk
            if len(self.holding_chunk) > 0:
                if model_response.choices[0].delta.content is None:
                    model_response.choices[0].delta.content = self.holding_chunk
                else:
                    model_response.choices[0].delta.content = (
                        self.holding_chunk + model_response.choices[0].delta.content
                    )
                self.holding_chunk = ""
            # if delta is None
            _is_delta_empty = self.is_delta_empty(delta=model_response.choices[0].delta)

            if _is_delta_empty:
                model_response.choices[0].delta = Delta(
                    content=None
                )  # ensure empty delta chunk returned
                # get any function call arguments
                model_response.choices[0].finish_reason = map_finish_reason(
                    finish_reason=self.received_finish_reason
                )  # ensure consistent output to openai

                self.sent_last_chunk = True

            return model_response
        elif (
            model_response.choices[0].delta.tool_calls is not None
            or model_response.choices[0].delta.function_call is not None
        ):
            if self.sent_first_chunk is False:
                model_response.choices[0].delta["role"] = "assistant"
                self.sent_first_chunk = True
            return model_response
        elif (
            len(model_response.choices) > 0
            and hasattr(model_response.choices[0].delta, "audio")
            and model_response.choices[0].delta.audio is not None
        ):
            return model_response
        else:
            if hasattr(model_response, "usage"):
                self.chunks.append(model_response)
            return

    def _optional_combine_thinking_block_in_choices(
        self, model_response: ModelResponseStream
    ) -> None:
        """
        UI's Like OpenWebUI expect to get 1 chunk with <think>...</think> tags in the chunk content

        In place updates the model_response object with reasoning_content in content with <think>...</think> tags

        Enabled when `merge_reasoning_content_in_choices=True` passed in request params


        """
        if self.merge_reasoning_content_in_choices is True:
            reasoning_content = getattr(
                model_response.choices[0].delta, "reasoning_content", None
            )
            if reasoning_content:
                if self.sent_first_thinking_block is False:
                    model_response.choices[0].delta.content += (
                        "<think>" + reasoning_content
                    )
                    self.sent_first_thinking_block = True
                elif (
                    self.sent_first_thinking_block is True
                    and hasattr(model_response.choices[0].delta, "reasoning_content")
                    and model_response.choices[0].delta.reasoning_content
                ):
                    model_response.choices[0].delta.content = reasoning_content
            elif (
                self.sent_first_thinking_block is True
                and not self.sent_last_thinking_block
                and model_response.choices[0].delta.content
            ):
                model_response.choices[0].delta.content = (
                    "</think>" + model_response.choices[0].delta.content
                )
                self.sent_last_thinking_block = True

            if hasattr(model_response.choices[0].delta, "reasoning_content"):
                del model_response.choices[0].delta.reasoning_content
        return

    def chunk_creator(self, chunk: Any):  # type: ignore  # noqa: PLR0915
        model_response = self.model_response_creator()
        response_obj: Dict[str, Any] = {}

        try:
            # return this for all models
            completion_obj: Dict[str, Any] = {"content": ""}
            from litellm.types.utils import GenericStreamingChunk as GChunk

            if (
                isinstance(chunk, dict)
                and generic_chunk_has_all_required_fields(
                    chunk=chunk
                )  # check if chunk is a generic streaming chunk
            ) or (
                self.custom_llm_provider
                and self.custom_llm_provider in litellm._custom_providers
            ):
                if self.received_finish_reason is not None:
                    if "provider_specific_fields" not in chunk:
                        raise StopIteration
                anthropic_response_obj: GChunk = cast(GChunk, chunk)
                completion_obj["content"] = anthropic_response_obj["text"]
                if anthropic_response_obj["is_finished"]:
                    self.received_finish_reason = anthropic_response_obj[
                        "finish_reason"
                    ]

                if anthropic_response_obj["finish_reason"]:
                    self.intermittent_finish_reason = anthropic_response_obj[
                        "finish_reason"
                    ]

                if anthropic_response_obj["usage"] is not None:
                    setattr(
                        model_response,
                        "usage",
                        litellm.Usage(**anthropic_response_obj["usage"]),
                    )

                if (
                    "tool_use" in anthropic_response_obj
                    and anthropic_response_obj["tool_use"] is not None
                ):
                    completion_obj["tool_calls"] = [anthropic_response_obj["tool_use"]]

                if (
                    "provider_specific_fields" in anthropic_response_obj
                    and anthropic_response_obj["provider_specific_fields"] is not None
                ):
                    for key, value in anthropic_response_obj[
                        "provider_specific_fields"
                    ].items():
                        setattr(model_response, key, value)

                response_obj = cast(Dict[str, Any], anthropic_response_obj)
            elif self.model == "replicate" or self.custom_llm_provider == "replicate":
                response_obj = self.handle_replicate_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider and self.custom_llm_provider == "predibase":
                response_obj = self.handle_predibase_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif (
                self.custom_llm_provider and self.custom_llm_provider == "baseten"
            ):  # baseten doesn't provide streaming
                completion_obj["content"] = self.handle_baseten_chunk(chunk)
            elif (
                self.custom_llm_provider and self.custom_llm_provider == "ai21"
            ):  # ai21 doesn't provide streaming
                response_obj = self.handle_ai21_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider and self.custom_llm_provider == "maritalk":
                response_obj = self.handle_maritalk_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider and self.custom_llm_provider == "vllm":
                completion_obj["content"] = chunk[0].outputs[0].text
            elif (
                self.custom_llm_provider and self.custom_llm_provider == "aleph_alpha"
            ):  # aleph alpha doesn't provide streaming
                response_obj = self.handle_aleph_alpha_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "nlp_cloud":
                try:
                    response_obj = self.handle_nlp_cloud_chunk(chunk)
                    completion_obj["content"] = response_obj["text"]
                    if response_obj["is_finished"]:
                        self.received_finish_reason = response_obj["finish_reason"]
                except Exception as e:
                    if self.received_finish_reason:
                        raise e
                    else:
                        if self.sent_first_chunk is False:
                            raise Exception("An unknown error occurred with the stream")
                        self.received_finish_reason = "stop"
            elif self.custom_llm_provider == "vertex_ai" and not isinstance(
                chunk, ModelResponseStream
            ):
                import proto  # type: ignore

                if hasattr(chunk, "candidates") is True:
                    try:
                        try:
                            completion_obj["content"] = chunk.text  # type: ignore
                        except Exception as e:
                            original_exception = e
                            if "Part has no text." in str(e):
                                ## check for function calling
                                function_call = (
                                    chunk.candidates[0].content.parts[0].function_call  # type: ignore
                                )

                                args_dict = {}

                                # Check if it's a RepeatedComposite instance
                                for key, val in function_call.args.items():
                                    if isinstance(
                                        val,
                                        proto.marshal.collections.repeated.RepeatedComposite,  # type: ignore
                                    ):
                                        # If so, convert to list
                                        args_dict[key] = [v for v in val]
                                    else:
                                        args_dict[key] = val

                                try:
                                    args_str = json.dumps(args_dict)
                                except Exception as e:
                                    raise e
                                _delta_obj = litellm.utils.Delta(
                                    content=None,
                                    tool_calls=[
                                        {
                                            "id": f"call_{str(uuid.uuid4())}",
                                            "function": {
                                                "arguments": args_str,
                                                "name": function_call.name,
                                            },
                                            "type": "function",
                                        }
                                    ],
                                )
                                _streaming_response = StreamingChoices(delta=_delta_obj)
                                _model_response = ModelResponse(stream=True)
                                _model_response.choices = [_streaming_response]
                                response_obj = {"original_chunk": _model_response}
                            else:
                                raise original_exception
                        if (
                            hasattr(chunk.candidates[0], "finish_reason")  # type: ignore
                            and chunk.candidates[0].finish_reason.name  # type: ignore
                            != "FINISH_REASON_UNSPECIFIED"
                        ):  # every non-final chunk in vertex ai has this
                            self.received_finish_reason = chunk.candidates[  # type: ignore
                                0
                            ].finish_reason.name
                    except Exception:
                        if chunk.candidates[0].finish_reason.name == "SAFETY":  # type: ignore
                            raise Exception(
                                f"The response was blocked by VertexAI. {str(chunk)}"
                            )
                else:
                    completion_obj["content"] = str(chunk)
            elif self.custom_llm_provider == "petals":
                if len(self.completion_stream) == 0:
                    if self.received_finish_reason is not None:
                        raise StopIteration
                    else:
                        self.received_finish_reason = "stop"
                chunk_size = 30
                new_chunk = self.completion_stream[:chunk_size]
                completion_obj["content"] = new_chunk
                self.completion_stream = self.completion_stream[chunk_size:]
            elif self.custom_llm_provider == "palm":
                # fake streaming
                response_obj = {}
                if len(self.completion_stream) == 0:
                    if self.received_finish_reason is not None:
                        raise StopIteration
                    else:
                        self.received_finish_reason = "stop"
                chunk_size = 30
                new_chunk = self.completion_stream[:chunk_size]
                completion_obj["content"] = new_chunk
                self.completion_stream = self.completion_stream[chunk_size:]
            elif self.custom_llm_provider == "triton":
                response_obj = self.handle_triton_stream(chunk)
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "text-completion-openai":
                response_obj = self.handle_openai_text_completion_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
                if response_obj["usage"] is not None:
                    setattr(
                        model_response,
                        "usage",
                        litellm.Usage(
                            prompt_tokens=response_obj["usage"].prompt_tokens,
                            completion_tokens=response_obj["usage"].completion_tokens,
                            total_tokens=response_obj["usage"].total_tokens,
                        ),
                    )
            elif self.custom_llm_provider == "text-completion-codestral":
                if not isinstance(chunk, str):
                    raise ValueError(f"chunk is not a string: {chunk}")
                response_obj = cast(
                    Dict[str, Any],
                    litellm.CodestralTextCompletionConfig()._chunk_parser(chunk),
                )
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
                if "usage" in response_obj is not None:
                    setattr(
                        model_response,
                        "usage",
                        litellm.Usage(
                            prompt_tokens=response_obj["usage"].prompt_tokens,
                            completion_tokens=response_obj["usage"].completion_tokens,
                            total_tokens=response_obj["usage"].total_tokens,
                        ),
                    )
            elif self.custom_llm_provider == "azure_text":
                response_obj = self.handle_azure_text_completion_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "cached_response":
                chunk = cast(ModelResponseStream, chunk)
                response_obj = {
                    "text": chunk.choices[0].delta.content,
                    "is_finished": True,
                    "finish_reason": chunk.choices[0].finish_reason,
                    "original_chunk": chunk,
                    "tool_calls": (
                        chunk.choices[0].delta.tool_calls
                        if hasattr(chunk.choices[0].delta, "tool_calls")
                        else None
                    ),
                }

                completion_obj["content"] = response_obj["text"]
                if response_obj["tool_calls"] is not None:
                    completion_obj["tool_calls"] = response_obj["tool_calls"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if hasattr(chunk, "id"):
                    model_response.id = chunk.id
                    self.response_id = chunk.id
                if hasattr(chunk, "system_fingerprint"):
                    self.system_fingerprint = chunk.system_fingerprint
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            else:  # openai / azure chat model
                if self.custom_llm_provider == "azure":
                    if isinstance(chunk, BaseModel) and hasattr(chunk, "model"):
                        # for azure, we need to pass the model from the orignal chunk
                        self.model = getattr(chunk, "model", self.model)
                response_obj = self.handle_openai_chat_completion_chunk(chunk)
                if response_obj is None:
                    return
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    if response_obj["finish_reason"] == "error":
                        raise Exception(
                            "{} raised a streaming error - finish_reason: error, no content string given. Received Chunk={}".format(
                                self.custom_llm_provider, response_obj
                            )
                        )
                    self.received_finish_reason = response_obj["finish_reason"]
                if response_obj.get("original_chunk", None) is not None:
                    if hasattr(response_obj["original_chunk"], "id"):
                        model_response = self.set_model_id(
                            response_obj["original_chunk"].id, model_response
                        )
                    if hasattr(response_obj["original_chunk"], "system_fingerprint"):
                        model_response.system_fingerprint = response_obj[
                            "original_chunk"
                        ].system_fingerprint
                        self.system_fingerprint = response_obj[
                            "original_chunk"
                        ].system_fingerprint
                if response_obj["logprobs"] is not None:
                    model_response.choices[0].logprobs = response_obj["logprobs"]

                if response_obj["usage"] is not None:
                    if isinstance(response_obj["usage"], dict):
                        setattr(
                            model_response,
                            "usage",
                            litellm.Usage(
                                prompt_tokens=response_obj["usage"].get(
                                    "prompt_tokens", None
                                )
                                or None,
                                completion_tokens=response_obj["usage"].get(
                                    "completion_tokens", None
                                )
                                or None,
                                total_tokens=response_obj["usage"].get(
                                    "total_tokens", None
                                )
                                or None,
                            ),
                        )
                    elif isinstance(response_obj["usage"], BaseModel):
                        setattr(
                            model_response,
                            "usage",
                            litellm.Usage(**response_obj["usage"].model_dump()),
                        )

            model_response.model = self.model
            print_verbose(
                f"model_response finish reason 3: {self.received_finish_reason}; response_obj={response_obj}"
            )
            ## FUNCTION CALL PARSING
            if (
                response_obj is not None
                and response_obj.get("original_chunk", None) is not None
            ):  # function / tool calling branch - only set for openai/azure compatible endpoints
                # enter this branch when no content has been passed in response
                original_chunk = response_obj.get("original_chunk", None)
                if hasattr(original_chunk, "id"):
                    model_response = self.set_model_id(
                        original_chunk.id, model_response
                    )
                if hasattr(original_chunk, "provider_specific_fields"):
                    model_response = (
                        self.copy_model_response_level_provider_specific_fields(
                            original_chunk, model_response
                        )
                    )
                if original_chunk.choices and len(original_chunk.choices) > 0:
                    delta = original_chunk.choices[0].delta
                    if delta is not None and (
                        delta.function_call is not None or delta.tool_calls is not None
                    ):
                        try:
                            model_response.system_fingerprint = (
                                original_chunk.system_fingerprint
                            )
                            ## AZURE - check if arguments is not None
                            if (
                                original_chunk.choices[0].delta.function_call
                                is not None
                            ):
                                if (
                                    getattr(
                                        original_chunk.choices[0].delta.function_call,
                                        "arguments",
                                    )
                                    is None
                                ):
                                    original_chunk.choices[
                                        0
                                    ].delta.function_call.arguments = ""
                            elif original_chunk.choices[0].delta.tool_calls is not None:
                                if isinstance(
                                    original_chunk.choices[0].delta.tool_calls, list
                                ):
                                    for t in original_chunk.choices[0].delta.tool_calls:
                                        if hasattr(t, "functions") and hasattr(
                                            t.functions, "arguments"
                                        ):
                                            if (
                                                getattr(
                                                    t.function,
                                                    "arguments",
                                                )
                                                is None
                                            ):
                                                t.function.arguments = ""
                            _json_delta = delta.model_dump()
                            print_verbose(f"_json_delta: {_json_delta}")
                            if "role" not in _json_delta or _json_delta["role"] is None:
                                _json_delta[
                                    "role"
                                ] = "assistant"  # mistral's api returns role as None
                            if "tool_calls" in _json_delta and isinstance(
                                _json_delta["tool_calls"], list
                            ):
                                for tool in _json_delta["tool_calls"]:
                                    if (
                                        isinstance(tool, dict)
                                        and "function" in tool
                                        and isinstance(tool["function"], dict)
                                        and ("type" not in tool or tool["type"] is None)
                                    ):
                                        # if function returned but type set to None - mistral's api returns type: None
                                        tool["type"] = "function"
                            model_response.choices[0].delta = Delta(**_json_delta)
                        except Exception as e:
                            verbose_logger.exception(
                                "litellm.CustomStreamWrapper.chunk_creator(): Exception occured - {}".format(
                                    str(e)
                                )
                            )
                            model_response.choices[0].delta = Delta()
                    elif (
                        delta is not None and getattr(delta, "audio", None) is not None
                    ):
                        model_response.choices[0].delta.audio = delta.audio
                    else:
                        try:
                            delta = (
                                dict()
                                if original_chunk.choices[0].delta is None
                                else dict(original_chunk.choices[0].delta)
                            )
                            print_verbose(f"original delta: {delta}")
                            model_response.choices[0].delta = Delta(**delta)
                            print_verbose(
                                f"new delta: {model_response.choices[0].delta}"
                            )
                        except Exception:
                            model_response.choices[0].delta = Delta()
                else:
                    if (
                        self.stream_options is not None
                        and self.stream_options["include_usage"] is True
                    ):
                        return model_response
                    return
            print_verbose(
                f"model_response.choices[0].delta: {model_response.choices[0].delta}; completion_obj: {completion_obj}"
            )
            print_verbose(f"self.sent_first_chunk: {self.sent_first_chunk}")

            ## CHECK FOR TOOL USE
            if "tool_calls" in completion_obj and len(completion_obj["tool_calls"]) > 0:
                if self.is_function_call is True:  # user passed in 'functions' param
                    completion_obj["function_call"] = completion_obj["tool_calls"][0][
                        "function"
                    ]
                    completion_obj["tool_calls"] = None

                self.tool_call = True

            ## RETURN ARG
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

    def set_logging_event_loop(self, loop):
        """
        import litellm, asyncio

        loop = asyncio.get_event_loop() # 👈 gets the current event loop

        response = litellm.completion(.., stream=True)

        response.set_logging_event_loop(loop=loop) # 👈 enables async_success callbacks for sync logging

        for chunk in response:
            ...
        """
        self.logging_loop = loop

    def cache_streaming_response(self, processed_chunk, cache_hit: bool):
        """
        Caches the streaming response
        """
        if not cache_hit and self.logging_obj._llm_caching_handler is not None:
            self.logging_obj._llm_caching_handler._sync_add_streaming_response_to_cache(
                processed_chunk
            )

    async def async_cache_streaming_response(self, processed_chunk, cache_hit: bool):
        """
        Caches the streaming response
        """
        if not cache_hit and self.logging_obj._llm_caching_handler is not None:
            await self.logging_obj._llm_caching_handler._add_streaming_response_to_cache(
                processed_chunk
            )

    def run_success_logging_and_cache_storage(self, processed_chunk, cache_hit: bool):
        """
        Runs success logging in a thread and adds the response to the cache
        """
        if litellm.disable_streaming_logging is True:
            """
            [NOT RECOMMENDED]
            Set this via `litellm.disable_streaming_logging = True`.

            Disables streaming logging.
            """
            return
        ## ASYNC LOGGING
        # Create an event loop for the new thread
        if self.logging_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self.logging_obj.async_success_handler(
                    processed_chunk, None, None, cache_hit
                ),
                loop=self.logging_loop,
            )
            future.result()
        else:
            asyncio.run(
                self.logging_obj.async_success_handler(
                    processed_chunk, None, None, cache_hit
                )
            )
        ## SYNC LOGGING
        self.logging_obj.success_handler(processed_chunk, None, None, cache_hit)

    def finish_reason_handler(self):
        model_response = self.model_response_creator()
        _finish_reason = self.received_finish_reason or self.intermittent_finish_reason
        if _finish_reason is not None:
            model_response.choices[0].finish_reason = _finish_reason
        else:
            model_response.choices[0].finish_reason = "stop"

        ## if tool use
        if (
            model_response.choices[0].finish_reason == "stop" and self.tool_call
        ):  # don't overwrite for other - potential error finish reasons
            model_response.choices[0].finish_reason = "tool_calls"
        return model_response

    def __next__(self):  # noqa: PLR0915
        cache_hit = False
        if (
            self.custom_llm_provider is not None
            and self.custom_llm_provider == "cached_response"
        ):
            cache_hit = True
        try:
            if self.completion_stream is None:
                self.fetch_sync_stream()
            while True:
                if (
                    isinstance(self.completion_stream, str)
                    or isinstance(self.completion_stream, bytes)
                    or isinstance(self.completion_stream, ModelResponse)
                ):
                    chunk = self.completion_stream
                else:
                    chunk = next(self.completion_stream)
                if chunk is not None and chunk != b"":
                    print_verbose(
                        f"PROCESSED CHUNK PRE CHUNK CREATOR: {chunk}; custom_llm_provider: {self.custom_llm_provider}"
                    )
                    response: Optional[ModelResponseStream] = self.chunk_creator(
                        chunk=chunk
                    )
                    print_verbose(f"PROCESSED CHUNK POST CHUNK CREATOR: {response}")

                    if response is None:
                        continue
                    if self.logging_obj.completion_start_time is None:
                        self.logging_obj._update_completion_start_time(
                            completion_start_time=datetime.datetime.now()
                        )
                    ## LOGGING
                    executor.submit(
                        self.run_success_logging_and_cache_storage,
                        response,
                        cache_hit,
                    )  # log response
                    choice = response.choices[0]
                    if isinstance(choice, StreamingChoices):
                        self.response_uptil_now += choice.delta.get("content", "") or ""
                    else:
                        self.response_uptil_now += ""
                    self.rules.post_call_rules(
                        input=self.response_uptil_now, model=self.model
                    )
                    # HANDLE STREAM OPTIONS
                    self.chunks.append(response)
                    if hasattr(
                        response, "usage"
                    ):  # remove usage from chunk, only send on final chunk
                        # Convert the object to a dictionary
                        obj_dict = response.dict()

                        # Remove an attribute (e.g., 'attr2')
                        if "usage" in obj_dict:
                            del obj_dict["usage"]

                        # Create a new object without the removed attribute
                        response = self.model_response_creator(
                            chunk=obj_dict, hidden_params=response._hidden_params
                        )
                    # add usage as hidden param
                    if self.sent_last_chunk is True and self.stream_options is None:
                        usage = calculate_total_usage(chunks=self.chunks)
                        response._hidden_params["usage"] = usage
                    # RETURN RESULT
                    return response

        except StopIteration:
            if self.sent_last_chunk is True:
                complete_streaming_response = litellm.stream_chunk_builder(
                    chunks=self.chunks, messages=self.messages
                )
                response = self.model_response_creator()
                if complete_streaming_response is not None:
                    setattr(
                        response,
                        "usage",
                        getattr(complete_streaming_response, "usage"),
                    )
                    self.cache_streaming_response(
                        processed_chunk=complete_streaming_response.model_copy(
                            deep=True
                        ),
                        cache_hit=cache_hit,
                    )
                    executor.submit(
                        self.logging_obj.success_handler,
                        complete_streaming_response.model_copy(deep=True),
                        None,
                        None,
                        cache_hit,
                    )
                else:
                    executor.submit(
                        self.logging_obj.success_handler,
                        response,
                        None,
                        None,
                        cache_hit,
                    )
                if self.sent_stream_usage is False and self.send_stream_usage is True:
                    self.sent_stream_usage = True
                    return response
                raise  # Re-raise StopIteration
            else:
                self.sent_last_chunk = True
                processed_chunk = self.finish_reason_handler()
                if self.stream_options is None:  # add usage as hidden param
                    usage = calculate_total_usage(chunks=self.chunks)
                    processed_chunk._hidden_params["usage"] = usage
                ## LOGGING
                executor.submit(
                    self.run_success_logging_and_cache_storage,
                    processed_chunk,
                    cache_hit,
                )  # log response
                return processed_chunk
        except Exception as e:
            traceback_exception = traceback.format_exc()
            # LOG FAILURE - handle streaming failure logging in the _next_ object, remove `handle_failure` once it's deprecated
            threading.Thread(
                target=self.logging_obj.failure_handler, args=(e, traceback_exception)
            ).start()
            if isinstance(e, OpenAIError):
                raise e
            else:
                raise exception_type(
                    model=self.model,
                    original_exception=e,
                    custom_llm_provider=self.custom_llm_provider,
                )

    def fetch_sync_stream(self):
        if self.completion_stream is None and self.make_call is not None:
            # Call make_call to get the completion stream
            self.completion_stream = self.make_call(client=litellm.module_level_client)
            self._stream_iter = self.completion_stream.__iter__()

        return self.completion_stream

    async def fetch_stream(self):
        if self.completion_stream is None and self.make_call is not None:
            # Call make_call to get the completion stream
            self.completion_stream = await self.make_call(
                client=litellm.module_level_aclient
            )
            self._stream_iter = self.completion_stream.__aiter__()

        return self.completion_stream

    async def __anext__(self):  # noqa: PLR0915
        cache_hit = False
        if (
            self.custom_llm_provider is not None
            and self.custom_llm_provider == "cached_response"
        ):
            cache_hit = True
        try:
            if self.completion_stream is None:
                await self.fetch_stream()

            if is_async_iterable(self.completion_stream):
                async for chunk in self.completion_stream:
                    if chunk == "None" or chunk is None:
                        raise Exception
                    elif (
                        self.custom_llm_provider == "gemini"
                        and hasattr(chunk, "parts")
                        and len(chunk.parts) == 0
                    ):
                        continue
                    # chunk_creator() does logging/stream chunk building. We need to let it know its being called in_async_func, so we don't double add chunks.
                    # __anext__ also calls async_success_handler, which does logging
                    print_verbose(f"PROCESSED ASYNC CHUNK PRE CHUNK CREATOR: {chunk}")

                    processed_chunk: Optional[ModelResponseStream] = self.chunk_creator(
                        chunk=chunk
                    )
                    verbose_logger.debug(
                        f"PROCESSED ASYNC CHUNK POST CHUNK CREATOR: {processed_chunk}"
                    )
                    if processed_chunk is None:
                        continue

                    if self.logging_obj.completion_start_time is None:
                        self.logging_obj._update_completion_start_time(
                            completion_start_time=datetime.datetime.now()
                        )

                    choice = processed_chunk.choices[0]
                    if isinstance(choice, StreamingChoices):
                        self.response_uptil_now += choice.delta.get("content", "") or ""
                    else:
                        self.response_uptil_now += ""
                    self.rules.post_call_rules(
                        input=self.response_uptil_now, model=self.model
                    )
                    self.chunks.append(processed_chunk)
                    if hasattr(
                        processed_chunk, "usage"
                    ):  # remove usage from chunk, only send on final chunk
                        # Convert the object to a dictionary
                        obj_dict = processed_chunk.dict()

                        # Remove an attribute (e.g., 'attr2')
                        if "usage" in obj_dict:
                            del obj_dict["usage"]

                        # Create a new object without the removed attribute
                        processed_chunk = self.model_response_creator(chunk=obj_dict)
                    print_verbose(f"final returned processed chunk: {processed_chunk}")
                    return processed_chunk
                raise StopAsyncIteration
            else:  # temporary patch for non-aiohttp async calls
                # example - boto3 bedrock llms
                while True:
                    if isinstance(self.completion_stream, str) or isinstance(
                        self.completion_stream, bytes
                    ):
                        chunk = self.completion_stream
                    else:
                        chunk = next(self.completion_stream)
                    if chunk is not None and chunk != b"":
                        print_verbose(f"PROCESSED CHUNK PRE CHUNK CREATOR: {chunk}")
                        processed_chunk: Optional[
                            ModelResponseStream
                        ] = self.chunk_creator(chunk=chunk)
                        print_verbose(
                            f"PROCESSED CHUNK POST CHUNK CREATOR: {processed_chunk}"
                        )
                        if processed_chunk is None:
                            continue

                        choice = processed_chunk.choices[0]
                        if isinstance(choice, StreamingChoices):
                            self.response_uptil_now += (
                                choice.delta.get("content", "") or ""
                            )
                        else:
                            self.response_uptil_now += ""
                        self.rules.post_call_rules(
                            input=self.response_uptil_now, model=self.model
                        )
                        # RETURN RESULT
                        self.chunks.append(processed_chunk)
                        return processed_chunk
        except (StopAsyncIteration, StopIteration):
            if self.sent_last_chunk is True:
                # log the final chunk with accurate streaming values
                complete_streaming_response = litellm.stream_chunk_builder(
                    chunks=self.chunks, messages=self.messages
                )
                response = self.model_response_creator()
                if complete_streaming_response is not None:
                    setattr(
                        response,
                        "usage",
                        getattr(complete_streaming_response, "usage"),
                    )
                    asyncio.create_task(
                        self.async_cache_streaming_response(
                            processed_chunk=complete_streaming_response.model_copy(
                                deep=True
                            ),
                            cache_hit=cache_hit,
                        )
                    )
                if self.sent_stream_usage is False and self.send_stream_usage is True:
                    self.sent_stream_usage = True
                    return response

                asyncio.create_task(
                    self.logging_obj.async_success_handler(
                        complete_streaming_response,
                        cache_hit=cache_hit,
                        start_time=None,
                        end_time=None,
                    )
                )

                executor.submit(
                    self.logging_obj.success_handler,
                    complete_streaming_response,
                    cache_hit=cache_hit,
                    start_time=None,
                    end_time=None,
                )

                raise StopAsyncIteration  # Re-raise StopIteration
            else:
                self.sent_last_chunk = True
                processed_chunk = self.finish_reason_handler()
                return processed_chunk
        except httpx.TimeoutException as e:  # if httpx read timeout error occues
            traceback_exception = traceback.format_exc()
            ## ADD DEBUG INFORMATION - E.G. LITELLM REQUEST TIMEOUT
            traceback_exception += "\nLiteLLM Default Request Timeout - {}".format(
                litellm.request_timeout
            )
            if self.logging_obj is not None:
                ## LOGGING
                threading.Thread(
                    target=self.logging_obj.failure_handler,
                    args=(e, traceback_exception),
                ).start()  # log response
                # Handle any exceptions that might occur during streaming
                asyncio.create_task(
                    self.logging_obj.async_failure_handler(e, traceback_exception)
                )
            raise e
        except Exception as e:
            traceback_exception = traceback.format_exc()
            if self.logging_obj is not None:
                ## LOGGING
                threading.Thread(
                    target=self.logging_obj.failure_handler,
                    args=(e, traceback_exception),
                ).start()  # log response
                # Handle any exceptions that might occur during streaming
                asyncio.create_task(
                    self.logging_obj.async_failure_handler(e, traceback_exception)  # type: ignore
                )
            ## Map to OpenAI Exception
            raise exception_type(
                model=self.model,
                custom_llm_provider=self.custom_llm_provider,
                original_exception=e,
                completion_kwargs={},
                extra_kwargs={},
            )

    @staticmethod
    def _strip_sse_data_from_chunk(chunk: Optional[str]) -> Optional[str]:
        """
        Strips the 'data: ' prefix from Server-Sent Events (SSE) chunks.

        Some providers like sagemaker send it as `data:`, need to handle both

        SSE messages are prefixed with 'data: ' which is part of the protocol,
        not the actual content from the LLM. This method removes that prefix
        and returns the actual content.

        Args:
            chunk: The SSE chunk that may contain the 'data: ' prefix (string or bytes)

        Returns:
            The chunk with the 'data: ' prefix removed, or the original chunk
            if no prefix was found. Returns None if input is None.

        See OpenAI Python Ref for this: https://github.com/openai/openai-python/blob/041bf5a8ec54da19aad0169671793c2078bd6173/openai/api_requestor.py#L100
        """
        if chunk is None:
            return None

        if isinstance(chunk, str):
            # OpenAI sends `data: `
            if chunk.startswith("data: "):
                # Strip the prefix and any leading whitespace that might follow it
                _length_of_sse_data_prefix = len("data: ")
                return chunk[_length_of_sse_data_prefix:]
            elif chunk.startswith("data:"):
                # Sagemaker sends `data:`, no trailing whitespace
                _length_of_sse_data_prefix = len("data:")
                return chunk[_length_of_sse_data_prefix:]

        return chunk


def calculate_total_usage(chunks: List[ModelResponse]) -> Usage:
    """Assume most recent usage chunk has total usage uptil then."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    for chunk in chunks:
        if "usage" in chunk:
            if "prompt_tokens" in chunk["usage"]:
                prompt_tokens = chunk["usage"].get("prompt_tokens", 0) or 0
            if "completion_tokens" in chunk["usage"]:
                completion_tokens = chunk["usage"].get("completion_tokens", 0) or 0

    returned_usage_chunk = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    return returned_usage_chunk


def generic_chunk_has_all_required_fields(chunk: dict) -> bool:
    """
    Checks if the provided chunk dictionary contains all required fields for GenericStreamingChunk.

    :param chunk: The dictionary to check.
    :return: True if all required fields are present, False otherwise.
    """
    _all_fields = GChunk.__annotations__

    decision = all(key in _all_fields for key in chunk)
    return decision
