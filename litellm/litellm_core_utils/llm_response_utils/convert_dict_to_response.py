import asyncio
import json
import time
import traceback
import uuid
from typing import Dict, Iterable, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    EmbeddingResponse,
    Function,
    HiddenParams,
    ImageResponse,
)
from litellm.types.utils import Logprobs as TextCompletionLogprobs
from litellm.types.utils import (
    Message,
    ModelResponse,
    RerankResponse,
    StreamingChoices,
    TextChoices,
    TextCompletionResponse,
    TranscriptionResponse,
    Usage,
)

from .get_headers import get_response_headers


async def convert_to_streaming_response_async(response_object: Optional[dict] = None):
    """
    Asynchronously converts a response object to a streaming response.

    Args:
        response_object (Optional[dict]): The response object to be converted. Defaults to None.

    Raises:
        Exception: If the response object is None.

    Yields:
        ModelResponse: The converted streaming response object.

    Returns:
        None
    """
    if response_object is None:
        raise Exception("Error in response object format")

    model_response_object = ModelResponse(stream=True)

    if model_response_object is None:
        raise Exception("Error in response creating model response object")

    choice_list = []

    for idx, choice in enumerate(response_object["choices"]):
        if (
            choice["message"].get("tool_calls", None) is not None
            and isinstance(choice["message"]["tool_calls"], list)
            and len(choice["message"]["tool_calls"]) > 0
            and isinstance(choice["message"]["tool_calls"][0], dict)
        ):
            pydantic_tool_calls = []
            for index, t in enumerate(choice["message"]["tool_calls"]):
                if "index" not in t:
                    t["index"] = index
                pydantic_tool_calls.append(ChatCompletionDeltaToolCall(**t))
            choice["message"]["tool_calls"] = pydantic_tool_calls
        delta = Delta(
            content=choice["message"].get("content", None),
            role=choice["message"]["role"],
            function_call=choice["message"].get("function_call", None),
            tool_calls=choice["message"].get("tool_calls", None),
        )
        finish_reason = choice.get("finish_reason", None)

        if finish_reason is None:
            finish_reason = choice.get("finish_details")

        logprobs = choice.get("logprobs", None)

        choice = StreamingChoices(
            finish_reason=finish_reason, index=idx, delta=delta, logprobs=logprobs
        )
        choice_list.append(choice)

    model_response_object.choices = choice_list

    if "usage" in response_object and response_object["usage"] is not None:
        setattr(
            model_response_object,
            "usage",
            Usage(
                completion_tokens=response_object["usage"].get("completion_tokens", 0),
                prompt_tokens=response_object["usage"].get("prompt_tokens", 0),
                total_tokens=response_object["usage"].get("total_tokens", 0),
            ),
        )

    if "id" in response_object:
        model_response_object.id = response_object["id"]

    if "created" in response_object:
        model_response_object.created = response_object["created"]

    if "system_fingerprint" in response_object:
        model_response_object.system_fingerprint = response_object["system_fingerprint"]

    if "model" in response_object:
        model_response_object.model = response_object["model"]

    yield model_response_object
    await asyncio.sleep(0)


def convert_to_streaming_response(response_object: Optional[dict] = None):
    # used for yielding Cache hits when stream == True
    if response_object is None:
        raise Exception("Error in response object format")

    model_response_object = ModelResponse(stream=True)
    choice_list = []
    for idx, choice in enumerate(response_object["choices"]):
        delta = Delta(
            content=choice["message"].get("content", None),
            role=choice["message"]["role"],
            function_call=choice["message"].get("function_call", None),
            tool_calls=choice["message"].get("tool_calls", None),
        )
        finish_reason = choice.get("finish_reason", None)
        if finish_reason is None:
            # gpt-4 vision can return 'finish_reason' or 'finish_details'
            finish_reason = choice.get("finish_details")
        logprobs = choice.get("logprobs", None)
        enhancements = choice.get("enhancements", None)
        choice = StreamingChoices(
            finish_reason=finish_reason,
            index=idx,
            delta=delta,
            logprobs=logprobs,
            enhancements=enhancements,
        )

        choice_list.append(choice)
    model_response_object.choices = choice_list

    if "usage" in response_object and response_object["usage"] is not None:
        setattr(model_response_object, "usage", Usage())
        model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
        model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
        model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

    if "id" in response_object:
        model_response_object.id = response_object["id"]

    if "created" in response_object:
        model_response_object.created = response_object["created"]

    if "system_fingerprint" in response_object:
        model_response_object.system_fingerprint = response_object["system_fingerprint"]

    if "model" in response_object:
        model_response_object.model = response_object["model"]
    yield model_response_object


from collections import defaultdict


def _handle_invalid_parallel_tool_calls(
    tool_calls: List[ChatCompletionMessageToolCall],
):
    """
    Handle hallucinated parallel tool call from openai - https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653

    Code modified from: https://github.com/phdowling/openai_multi_tool_use_parallel_patch/blob/main/openai_multi_tool_use_parallel_patch.py
    """

    if tool_calls is None:
        return
    try:
        replacements: Dict[int, List[ChatCompletionMessageToolCall]] = defaultdict(list)
        for i, tool_call in enumerate(tool_calls):
            current_function = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            if current_function == "multi_tool_use.parallel":
                verbose_logger.debug(
                    "OpenAI did a weird pseudo-multi-tool-use call, fixing call structure.."
                )
                for _fake_i, _fake_tool_use in enumerate(function_args["tool_uses"]):
                    _function_args = _fake_tool_use["parameters"]
                    _current_function = _fake_tool_use["recipient_name"]
                    if _current_function.startswith("functions."):
                        _current_function = _current_function[len("functions.") :]

                    fixed_tc = ChatCompletionMessageToolCall(
                        id=f"{tool_call.id}_{_fake_i}",
                        type="function",
                        function=Function(
                            name=_current_function, arguments=json.dumps(_function_args)
                        ),
                    )
                    replacements[i].append(fixed_tc)

        shift = 0
        for i, replacement in replacements.items():
            tool_calls[:] = (
                tool_calls[: i + shift] + replacement + tool_calls[i + shift + 1 :]
            )
            shift += len(replacement)

        return tool_calls
    except json.JSONDecodeError:
        # if there is a JSONDecodeError, return the original tool_calls
        return tool_calls


class LiteLLMResponseObjectHandler:

    @staticmethod
    def convert_to_image_response(
        response_object: dict,
        model_response_object: Optional[ImageResponse] = None,
        hidden_params: Optional[dict] = None,
    ) -> ImageResponse:

        response_object.update({"hidden_params": hidden_params})

        if model_response_object is None:
            model_response_object = ImageResponse(**response_object)
            return model_response_object
        else:
            model_response_dict = model_response_object.model_dump()

            model_response_dict.update(response_object)
            model_response_object = ImageResponse(**model_response_dict)
            return model_response_object

    @staticmethod
    def convert_chat_to_text_completion(
        response: ModelResponse,
        text_completion_response: TextCompletionResponse,
        custom_llm_provider: Optional[str] = None,
    ) -> TextCompletionResponse:
        """
        Converts a chat completion response to a text completion response format.

        Note: This is used for huggingface. For OpenAI / Azure Text the providers files directly return TextCompletionResponse which we then send to user

        Args:
            response (ModelResponse): The chat completion response to convert

        Returns:
            TextCompletionResponse: The converted text completion response

        Example:
            chat_response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi"}])
            text_response = convert_chat_to_text_completion(chat_response)
        """
        transformed_logprobs = LiteLLMResponseObjectHandler._convert_provider_response_logprobs_to_text_completion_logprobs(
            response=response,
            custom_llm_provider=custom_llm_provider,
        )

        text_completion_response["id"] = response.get("id", None)
        text_completion_response["object"] = "text_completion"
        text_completion_response["created"] = response.get("created", None)
        text_completion_response["model"] = response.get("model", None)
        choices_list: List[TextChoices] = []

        # Convert each choice to TextChoices
        for choice in response["choices"]:
            text_choices = TextChoices()
            text_choices["text"] = choice["message"]["content"]
            text_choices["index"] = choice["index"]
            text_choices["logprobs"] = transformed_logprobs
            text_choices["finish_reason"] = choice["finish_reason"]
            choices_list.append(text_choices)

        text_completion_response["choices"] = choices_list
        text_completion_response["usage"] = response.get("usage", None)
        text_completion_response._hidden_params = HiddenParams(
            **response._hidden_params
        )
        return text_completion_response

    @staticmethod
    def _convert_provider_response_logprobs_to_text_completion_logprobs(
        response: ModelResponse,
        custom_llm_provider: Optional[str] = None,
    ) -> Optional[TextCompletionLogprobs]:
        """
        Convert logprobs from provider to OpenAI.Completion() format

        Only supported for HF TGI models
        """
        transformed_logprobs: Optional[TextCompletionLogprobs] = None
        if custom_llm_provider == "huggingface":
            # only supported for TGI models
            try:
                raw_response = response._hidden_params.get("original_response", None)
                transformed_logprobs = litellm.huggingface._transform_logprobs(
                    hf_response=raw_response
                )
            except Exception as e:
                verbose_logger.exception(f"LiteLLM non blocking exception: {e}")

        return transformed_logprobs


def convert_to_model_response_object(  # noqa: PLR0915
    response_object: Optional[dict] = None,
    model_response_object: Optional[
        Union[
            ModelResponse,
            EmbeddingResponse,
            ImageResponse,
            TranscriptionResponse,
            RerankResponse,
        ]
    ] = None,
    response_type: Literal[
        "completion", "embedding", "image_generation", "audio_transcription", "rerank"
    ] = "completion",
    stream=False,
    start_time=None,
    end_time=None,
    hidden_params: Optional[dict] = None,
    _response_headers: Optional[dict] = None,
    convert_tool_call_to_json_mode: Optional[
        bool
    ] = None,  # used for supporting 'json_schema' on older models
):
    received_args = locals()

    additional_headers = get_response_headers(_response_headers)

    if hidden_params is None:
        hidden_params = {}
    hidden_params["additional_headers"] = additional_headers

    ### CHECK IF ERROR IN RESPONSE ### - openrouter returns these in the dictionary
    if (
        response_object is not None
        and "error" in response_object
        and response_object["error"] is not None
    ):
        error_args = {"status_code": 422, "message": "Error in response object"}
        if isinstance(response_object["error"], dict):
            if "code" in response_object["error"]:
                error_args["status_code"] = response_object["error"]["code"]
            if "message" in response_object["error"]:
                if isinstance(response_object["error"]["message"], dict):
                    message_str = json.dumps(response_object["error"]["message"])
                else:
                    message_str = str(response_object["error"]["message"])
                error_args["message"] = message_str
        raised_exception = Exception()
        setattr(raised_exception, "status_code", error_args["status_code"])
        setattr(raised_exception, "message", error_args["message"])
        raise raised_exception

    try:
        if response_type == "completion" and (
            model_response_object is None
            or isinstance(model_response_object, ModelResponse)
        ):
            if response_object is None or model_response_object is None:
                raise Exception("Error in response object format")
            if stream is True:
                # for returning cached responses, we need to yield a generator
                return convert_to_streaming_response(response_object=response_object)
            choice_list = []

            assert response_object["choices"] is not None and isinstance(
                response_object["choices"], Iterable
            )

            for idx, choice in enumerate(response_object["choices"]):
                ## HANDLE JSON MODE - anthropic returns single function call]
                tool_calls = choice["message"].get("tool_calls", None)
                if tool_calls is not None:
                    _openai_tool_calls = []
                    for _tc in tool_calls:
                        _openai_tc = ChatCompletionMessageToolCall(**_tc)
                        _openai_tool_calls.append(_openai_tc)
                    fixed_tool_calls = _handle_invalid_parallel_tool_calls(
                        _openai_tool_calls
                    )

                    if fixed_tool_calls is not None:
                        tool_calls = fixed_tool_calls

                message: Optional[Message] = None
                finish_reason: Optional[str] = None
                if (
                    convert_tool_call_to_json_mode
                    and tool_calls is not None
                    and len(tool_calls) == 1
                ):
                    # to support 'json_schema' logic on older models
                    json_mode_content_str: Optional[str] = tool_calls[0][
                        "function"
                    ].get("arguments")
                    if json_mode_content_str is not None:
                        message = litellm.Message(content=json_mode_content_str)
                        finish_reason = "stop"
                if message is None:
                    message = Message(
                        content=choice["message"].get("content", None),
                        role=choice["message"]["role"] or "assistant",
                        function_call=choice["message"].get("function_call", None),
                        tool_calls=tool_calls,
                        audio=choice["message"].get("audio", None),
                    )
                    finish_reason = choice.get("finish_reason", None)
                if finish_reason is None:
                    # gpt-4 vision can return 'finish_reason' or 'finish_details'
                    finish_reason = choice.get("finish_details") or "stop"
                logprobs = choice.get("logprobs", None)
                enhancements = choice.get("enhancements", None)
                choice = Choices(
                    finish_reason=finish_reason,
                    index=idx,
                    message=message,
                    logprobs=logprobs,
                    enhancements=enhancements,
                )
                choice_list.append(choice)
            model_response_object.choices = choice_list

            if "usage" in response_object and response_object["usage"] is not None:
                usage_object = litellm.Usage(**response_object["usage"])
                setattr(model_response_object, "usage", usage_object)
            if "created" in response_object:
                model_response_object.created = response_object["created"] or int(
                    time.time()
                )

            if "id" in response_object:
                model_response_object.id = response_object["id"] or str(uuid.uuid4())

            if "system_fingerprint" in response_object:
                model_response_object.system_fingerprint = response_object[
                    "system_fingerprint"
                ]

            if "model" in response_object:
                if model_response_object.model is None:
                    model_response_object.model = response_object["model"]
                elif (
                    "/" in model_response_object.model
                    and response_object["model"] is not None
                ):
                    openai_compatible_provider = model_response_object.model.split("/")[
                        0
                    ]
                    model_response_object.model = (
                        openai_compatible_provider + "/" + response_object["model"]
                    )

            if start_time is not None and end_time is not None:
                if isinstance(start_time, type(end_time)):
                    model_response_object._response_ms = (  # type: ignore
                        end_time - start_time
                    ).total_seconds() * 1000

            if hidden_params is not None:
                if model_response_object._hidden_params is None:
                    model_response_object._hidden_params = {}
                model_response_object._hidden_params.update(hidden_params)

            if _response_headers is not None:
                model_response_object._response_headers = _response_headers

            special_keys = list(litellm.ModelResponse.model_fields.keys())
            special_keys.append("usage")
            for k, v in response_object.items():
                if k not in special_keys:
                    setattr(model_response_object, k, v)

            return model_response_object
        elif response_type == "embedding" and (
            model_response_object is None
            or isinstance(model_response_object, EmbeddingResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = EmbeddingResponse()

            if "model" in response_object:
                model_response_object.model = response_object["model"]

            if "object" in response_object:
                model_response_object.object = response_object["object"]

            model_response_object.data = response_object["data"]

            if "usage" in response_object and response_object["usage"] is not None:
                model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
                model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
                model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

            if start_time is not None and end_time is not None:
                model_response_object._response_ms = (  # type: ignore
                    end_time - start_time
                ).total_seconds() * 1000  # return response latency in ms like openai

            if hidden_params is not None:
                model_response_object._hidden_params = hidden_params

            if _response_headers is not None:
                model_response_object._response_headers = _response_headers

            return model_response_object
        elif response_type == "image_generation" and (
            model_response_object is None
            or isinstance(model_response_object, ImageResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            return LiteLLMResponseObjectHandler.convert_to_image_response(
                response_object=response_object,
                model_response_object=model_response_object,
                hidden_params=hidden_params,
            )

        elif response_type == "audio_transcription" and (
            model_response_object is None
            or isinstance(model_response_object, TranscriptionResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = TranscriptionResponse()

            if "text" in response_object:
                model_response_object.text = response_object["text"]

            optional_keys = ["language", "task", "duration", "words", "segments"]
            for key in optional_keys:  # not guaranteed to be in response
                if key in response_object:
                    setattr(model_response_object, key, response_object[key])

            if hidden_params is not None:
                model_response_object._hidden_params = hidden_params

            if _response_headers is not None:
                model_response_object._response_headers = _response_headers

            return model_response_object
        elif response_type == "rerank" and (
            model_response_object is None
            or isinstance(model_response_object, RerankResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = RerankResponse(**response_object)
                return model_response_object

            if "id" in response_object:
                model_response_object.id = response_object["id"]

            if "meta" in response_object:
                model_response_object.meta = response_object["meta"]

            if "results" in response_object:
                model_response_object.results = response_object["results"]

            return model_response_object
    except Exception:
        raise Exception(
            f"Invalid response object {traceback.format_exc()}\n\nreceived_args={received_args}"
        )
