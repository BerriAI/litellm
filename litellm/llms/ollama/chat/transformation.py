import json
import time
from litellm._uuid import uuid
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

from httpx._models import Headers, Response
from pydantic import BaseModel

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _extract_reasoning_content,
    convert_content_list_to_str,
    extract_images_from_message,
)
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.ollama import (
    OllamaChatCompletionMessage,
    OllamaToolCall,
    OllamaToolCallFunction,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantToolCall,
    ChatCompletionUsageBlock,
)
from litellm.types.utils import ModelResponse, ModelResponseStream

from ..common_utils import OllamaError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OllamaChatConfig(BaseConfig):
    """
    Reference: https://github.com/ollama/ollama/blob/main/docs/api.md#parameters

    The class `OllamaConfig` provides the configuration for the Ollama's API interface. Below are the parameters:

    - `mirostat` (int): Enable Mirostat sampling for controlling perplexity. Default is 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0. Example usage: mirostat 0

    - `mirostat_eta` (float): Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. Default: 0.1. Example usage: mirostat_eta 0.1

    - `mirostat_tau` (float): Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. Default: 5.0. Example usage: mirostat_tau 5.0

    - `num_ctx` (int): Sets the size of the context window used to generate the next token. Default: 2048. Example usage: num_ctx 4096

    - `num_gqa` (int): The number of GQA groups in the transformer layer. Required for some models, for example it is 8 for llama2:70b. Example usage: num_gqa 1

    - `num_gpu` (int): The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable. Example usage: num_gpu 0

    - `num_thread` (int): Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores). Example usage: num_thread 8

    - `repeat_last_n` (int): Sets how far back for the model to look back to prevent repetition. Default: 64, 0 = disabled, -1 = num_ctx. Example usage: repeat_last_n 64

    - `repeat_penalty` (float): Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. Default: 1.1. Example usage: repeat_penalty 1.1

    - `temperature` (float): The temperature of the model. Increasing the temperature will make the model answer more creatively. Default: 0.8. Example usage: temperature 0.7

    - `seed` (int): Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. Example usage: seed 42

    - `stop` (string[]): Sets the stop sequences to use. Example usage: stop "AI assistant:"

    - `tfs_z` (float): Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. Default: 1. Example usage: tfs_z 1

    - `num_predict` (int): Maximum number of tokens to predict when generating text. Default: 128, -1 = infinite generation, -2 = fill context. Example usage: num_predict 42

    - `top_k` (int): Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. Default: 40. Example usage: top_k 40

    - `top_p` (float): Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. Default: 0.9. Example usage: top_p 0.9

    - `system` (string): system prompt for model (overrides what is defined in the Modelfile)

    - `template` (string): the full prompt or prompt template (overrides what is defined in the Modelfile)
    """

    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    num_ctx: Optional[int] = None
    num_gqa: Optional[int] = None
    num_thread: Optional[int] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None
    tfs_z: Optional[float] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    system: Optional[str] = None
    template: Optional[str] = None

    def __init__(
        self,
        mirostat: Optional[int] = None,
        mirostat_eta: Optional[float] = None,
        mirostat_tau: Optional[float] = None,
        num_ctx: Optional[int] = None,
        num_gqa: Optional[int] = None,
        num_thread: Optional[int] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[list] = None,
        tfs_z: Optional[float] = None,
        num_predict: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
        return [
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "top_p",
            "temperature",
            "seed",
            "frequency_penalty",
            "stop",
            "tools",
            "tool_choice",
            "functions",
            "response_format",
            "reasoning_effort",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["num_predict"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "seed":
                optional_params["seed"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "frequency_penalty":
                optional_params["repeat_penalty"] = value
            if param == "stop":
                optional_params["stop"] = value
            if (
                param == "response_format"
                and isinstance(value, dict)
                and value.get("type") == "json_object"
            ):
                optional_params["format"] = "json"
            if (
                param == "response_format"
                and isinstance(value, dict)
                and value.get("type") == "json_schema"
            ):
                if value.get("json_schema") and value["json_schema"].get("schema"):
                    optional_params["format"] = value["json_schema"]["schema"]
            if param == "reasoning_effort" and value is not None:
                if model.startswith("gpt-oss"):
                    optional_params["think"] = value
                else:
                    optional_params["think"] = value in {"low", "medium", "high"}
            ### FUNCTION CALLING LOGIC ###
            if param == "tools":
                ## CHECK IF MODEL SUPPORTS TOOL CALLING ##
                try:
                    model_info = litellm.get_model_info(
                        model=model, custom_llm_provider="ollama"
                    )
                    if model_info.get("supports_function_calling") is True:
                        optional_params["tools"] = value
                    else:
                        raise Exception
                except Exception:
                    optional_params["format"] = "json"
                    litellm.add_function_to_prompt = (
                        True  # so that main.py adds the function call to the prompt
                    )
                    optional_params["functions_unsupported_model"] = value

                    if len(optional_params["functions_unsupported_model"]) == 1:
                        optional_params["function_name"] = optional_params[
                            "functions_unsupported_model"
                        ][0]["function"]["name"]

            if param == "functions":
                ## CHECK IF MODEL SUPPORTS TOOL CALLING ##
                try:
                    model_info = litellm.get_model_info(
                        model=model, custom_llm_provider="ollama"
                    )
                    if model_info.get("supports_function_calling") is True:
                        optional_params["tools"] = value
                    else:
                        raise Exception
                except Exception:
                    optional_params["format"] = "json"
                    litellm.add_function_to_prompt = (
                        True  # so that main.py adds the function call to the prompt
                    )
                    optional_params["functions_unsupported_model"] = (
                        non_default_params.get("functions")
                    )
        non_default_params.pop("tool_choice", None)  # causes ollama requests to hang
        non_default_params.pop("functions", None)  # causes ollama requests to hang
        return optional_params

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
        if api_key is not None and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"
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
            api_base = "http://localhost:11434"
        if api_base.endswith("/api/chat"):
            url = api_base
        else:
            url = f"{api_base}/api/chat"

        return url

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        stream = optional_params.pop("stream", False)
        format = optional_params.pop("format", None)
        keep_alive = optional_params.pop("keep_alive", None)
        think = optional_params.pop("think", None)
        function_name = optional_params.pop("function_name", None)
        litellm_params["function_name"] = function_name
        tools = optional_params.pop("tools", None)

        new_messages = []
        for m in messages:
            if isinstance(
                m, BaseModel
            ):  # avoid message serialization issues - https://github.com/BerriAI/litellm/issues/5319
                m = m.model_dump(exclude_none=True)
            tool_calls = m.get("tool_calls")
            if tool_calls is not None and isinstance(tool_calls, list):
                new_tools: List[OllamaToolCall] = []
                for tool in tool_calls:
                    typed_tool = ChatCompletionAssistantToolCall(**tool)  # type: ignore
                    if typed_tool["type"] == "function":
                        arguments = {}
                        if "arguments" in typed_tool["function"]:
                            arguments = json.loads(typed_tool["function"]["arguments"])
                        ollama_tool_call = OllamaToolCall(
                            function=OllamaToolCallFunction(
                                name=typed_tool["function"].get("name") or "",
                                arguments=arguments,
                            )
                        )
                        new_tools.append(ollama_tool_call)
                cast(dict, m)["tool_calls"] = new_tools
            reasoning_content, parsed_content = _extract_reasoning_content(
                cast(dict, m)
            )
            content_str = convert_content_list_to_str(cast(AllMessageValues, m))
            images = extract_images_from_message(cast(AllMessageValues, m))

            ollama_message = OllamaChatCompletionMessage(
                role=cast(str, m.get("role")),
            )
            if reasoning_content is not None:
                ollama_message["thinking"] = reasoning_content
            if content_str is not None:
                ollama_message["content"] = content_str
            if images is not None:
                ollama_message["images"] = images

            new_messages.append(ollama_message)

        # Load Config
        config = self.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        data = {
            "model": model,
            "messages": new_messages,
            "options": optional_params,
            "stream": stream,
        }
        if format is not None:
            data["format"] = format
        if tools is not None:
            data["tools"] = tools
        if keep_alive is not None:
            data["keep_alive"] = keep_alive
        if think is not None:
            data["think"] = think

        return data

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: str,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=raw_response.text,
            additional_args={
                "headers": None,
                "api_base": litellm_params.get("api_base"),
            },
        )

        response_json = raw_response.json()

        ## RESPONSE OBJECT
        model_response.choices[0].finish_reason = "stop"
        response_json_message = response_json.get("message")
        if response_json_message is not None:
            if "thinking" in response_json_message:
                # remap 'thinking' to 'reasoning_content'
                response_json_message["reasoning_content"] = response_json_message[
                    "thinking"
                ]
                del response_json_message["thinking"]
            elif response_json_message.get("content") is not None:
                # parse reasoning content from content
                from litellm.litellm_core_utils.prompt_templates.common_utils import (
                    _parse_content_for_reasoning,
                )

                reasoning_content, content = _parse_content_for_reasoning(
                    response_json_message["content"]
                )
                response_json_message["reasoning_content"] = reasoning_content
                response_json_message["content"] = content

        if (
            request_data.get("format", "") == "json"
            and litellm_params.get("function_name") is not None
        ):
            function_call = json.loads(response_json_message["content"])
            message = litellm.Message(
                content=None,
                tool_calls=[
                    {
                        "id": f"call_{str(uuid.uuid4())}",
                        "function": {
                            "name": function_call.get(
                                "name", litellm_params.get("function_name")
                            ),
                            "arguments": json.dumps(
                                function_call.get("arguments", function_call)
                            ),
                        },
                        "type": "function",
                    }
                ],
                reasoning_content=response_json_message.get("reasoning_content"),
            )
            model_response.choices[0].message = message  # type: ignore
            model_response.choices[0].finish_reason = "tool_calls"
        else:

            _message = litellm.Message(**response_json_message)
            model_response.choices[0].message = _message  # type: ignore
        model_response.created = int(time.time())
        model_response.model = "ollama_chat/" + model
        prompt_tokens = response_json.get("prompt_eval_count", litellm.token_counter(messages=messages))  # type: ignore
        completion_tokens = response_json.get(
            "eval_count",
            litellm.token_counter(text=response_json["message"]["content"]),
        )
        setattr(
            model_response,
            "usage",
            litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return OllamaError(
            status_code=status_code, message=error_message, headers=headers
        )

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return OllamaChatCompletionResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class OllamaChatCompletionResponseIterator(BaseModelResponseIterator):
    started_reasoning_content: bool = False
    finished_reasoning_content: bool = False

    def _is_function_call_complete(self, function_args: Union[str, dict]) -> bool:
        if isinstance(function_args, dict):
            return True
        try:
            json.loads(function_args)
            return True
        except Exception:
            return False

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            """
            Expected chunk format:
            {
                "model": "llama3.1",
                "created_at": "2025-05-24T02:12:05.859654Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "get_latest_album_ratings",
                            "arguments": {
                                "artist_name": "Taylor Swift"
                            }
                        }
                    }]
                },
                "done_reason": "stop",
                "done": true,
                ...
            }

            Need to:
            - convert 'message' to 'delta'
            - return finish_reason when done is true
            - return usage when done is true

            """
            from litellm.types.utils import Delta, StreamingChoices

            # process tool calls - if complete function arg - add id to tool call
            tool_calls = chunk["message"].get("tool_calls")
            if tool_calls is not None:
                for tool_call in tool_calls:
                    function_args = tool_call.get("function").get("arguments")
                    if function_args is not None and len(function_args) > 0:
                        is_function_call_complete = self._is_function_call_complete(
                            function_args
                        )
                        if is_function_call_complete:
                            tool_call["id"] = str(uuid.uuid4())

            # PROCESS REASONING CONTENT
            reasoning_content: Optional[str] = None
            content: Optional[str] = None
            if chunk["message"].get("thinking") is not None:
                if self.started_reasoning_content is False:
                    reasoning_content = chunk["message"].get("thinking")
                    self.started_reasoning_content = True
                elif self.finished_reasoning_content is False:
                    reasoning_content = chunk["message"].get("thinking")
                    self.finished_reasoning_content = True
            elif chunk["message"].get("content") is not None:
                message_content = chunk["message"].get("content")
                if "<think>" in message_content:
                    message_content = message_content.replace("<think>", "")

                    self.started_reasoning_content = True

                if "</think>" in message_content and self.started_reasoning_content:
                    message_content = message_content.replace("</think>", "")
                    self.finished_reasoning_content = True

                if (
                    self.started_reasoning_content
                    and not self.finished_reasoning_content
                ):
                    reasoning_content = message_content
                else:
                    content = message_content

            delta = Delta(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
            )

            if chunk["done"] is True:
                finish_reason = chunk.get("done_reason", "stop")
                choices = [
                    StreamingChoices(
                        delta=delta,
                        finish_reason=finish_reason,
                    )
                ]
            else:
                choices = [
                    StreamingChoices(
                        delta=delta,
                    )
                ]

            usage = ChatCompletionUsageBlock(
                prompt_tokens=chunk.get("prompt_eval_count", 0),
                completion_tokens=chunk.get("eval_count", 0),
                total_tokens=chunk.get("prompt_eval_count", 0)
                + chunk.get("eval_count", 0),
            )

            return ModelResponseStream(
                id=str(uuid.uuid4()),
                object="chat.completion.chunk",
                created=int(time.time()),  # ollama created_at is in UTC
                usage=usage,
                model=chunk["model"],
                choices=choices,
            )
        except KeyError as e:
            raise OllamaError(
                message=f"KeyError: {e}, Got unexpected response from Ollama: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e
