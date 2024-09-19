import json
import time
import traceback
import types
import uuid
from itertools import chain
from typing import List, Optional

import aiohttp
import httpx
import requests
from pydantic import BaseModel

import litellm
from litellm import verbose_logger
from litellm.types.llms.ollama import OllamaToolCall, OllamaToolCallFunction
from litellm.types.llms.openai import ChatCompletionAssistantToolCall


class OllamaError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="http://localhost:11434")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class OllamaChatConfig:
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
    temperature: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[list] = (
        None  # stop is a list based on this - https://github.com/ollama/ollama/pull/442
    )
    tfs_z: Optional[float] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
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
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and k != "function_name"  # special param for function calling
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

    def get_supported_openai_params(
        self,
    ):
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
        ]

    def map_openai_params(
        self, model: str, non_default_params: dict, optional_params: dict
    ):
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
            if param == "response_format" and value["type"] == "json_object":
                optional_params["format"] = "json"
            ### FUNCTION CALLING LOGIC ###
            if param == "tools":
                # ollama actually supports json output
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
                # ollama actually supports json output
                optional_params["format"] = "json"
                litellm.add_function_to_prompt = (
                    True  # so that main.py adds the function call to the prompt
                )
                optional_params["functions_unsupported_model"] = non_default_params.get(
                    "functions"
                )
        non_default_params.pop("tool_choice", None)  # causes ollama requests to hang
        non_default_params.pop("functions", None)  # causes ollama requests to hang
        return optional_params


# ollama implementation
def get_ollama_response(
    model_response: litellm.ModelResponse,
    messages: list,
    optional_params: dict,
    api_base="http://localhost:11434",
    api_key: Optional[str] = None,
    model="llama2",
    logging_obj=None,
    acompletion: bool = False,
    encoding=None,
):
    if api_base.endswith("/api/chat"):
        url = api_base
    else:
        url = f"{api_base}/api/chat"

    ## Load Config
    config = litellm.OllamaChatConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    stream = optional_params.pop("stream", False)
    format = optional_params.pop("format", None)
    function_name = optional_params.pop("function_name", None)
    tools = optional_params.pop("tools", None)

    new_messages = []
    for m in messages:
        if isinstance(
            m, BaseModel
        ):  # avoid message serialization issues - https://github.com/BerriAI/litellm/issues/5319
            m = m.model_dump(exclude_none=True)
        if m.get("tool_calls") is not None and isinstance(m["tool_calls"], list):
            new_tools: List[OllamaToolCall] = []
            for tool in m["tool_calls"]:
                typed_tool = ChatCompletionAssistantToolCall(**tool)  # type: ignore
                if typed_tool["type"] == "function":
                    ollama_tool_call = OllamaToolCall(
                        function=OllamaToolCallFunction(
                            name=typed_tool["function"]["name"],
                            arguments=json.loads(typed_tool["function"]["arguments"]),
                        )
                    )
                    new_tools.append(ollama_tool_call)
            m["tool_calls"] = new_tools
        new_messages.append(m)

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
    ## LOGGING
    logging_obj.pre_call(
        input=None,
        api_key=None,
        additional_args={
            "api_base": url,
            "complete_input_dict": data,
            "headers": {},
            "acompletion": acompletion,
        },
    )
    if acompletion is True:
        if stream is True:
            response = ollama_async_streaming(
                url=url,
                api_key=api_key,
                data=data,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
            )
        else:
            response = ollama_acompletion(
                url=url,
                api_key=api_key,
                data=data,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
                function_name=function_name,
            )
        return response
    elif stream is True:
        return ollama_completion_stream(
            url=url, api_key=api_key, data=data, logging_obj=logging_obj
        )

    _request = {
        "url": f"{url}",
        "json": data,
    }
    if api_key is not None:
        _request["headers"] = {"Authorization": "Bearer {}".format(api_key)}
    response = requests.post(**_request)  # type: ignore
    if response.status_code != 200:
        raise OllamaError(status_code=response.status_code, message=response.text)

    ## LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=response.text,
        additional_args={
            "headers": None,
            "api_base": api_base,
        },
    )

    response_json = response.json()

    ## RESPONSE OBJECT
    model_response.choices[0].finish_reason = "stop"
    if data.get("format", "") == "json" and function_name is not None:
        function_call = json.loads(response_json["message"]["content"])
        message = litellm.Message(
            content=None,
            tool_calls=[
                {
                    "id": f"call_{str(uuid.uuid4())}",
                    "function": {
                        "name": function_call.get("name", function_name),
                        "arguments": json.dumps(
                            function_call.get("arguments", function_call)
                        ),
                    },
                    "type": "function",
                }
            ],
        )
        model_response.choices[0].message = message  # type: ignore
        model_response.choices[0].finish_reason = "tool_calls"
    else:
        _message = litellm.Message(**response_json["message"])
        model_response.choices[0].message = _message  # type: ignore
    model_response.created = int(time.time())
    model_response.model = "ollama_chat/" + model
    prompt_tokens = response_json.get("prompt_eval_count", litellm.token_counter(messages=messages))  # type: ignore
    completion_tokens = response_json.get(
        "eval_count", litellm.token_counter(text=response_json["message"]["content"])
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


def ollama_completion_stream(url, api_key, data, logging_obj):
    _request = {
        "url": f"{url}",
        "json": data,
        "method": "POST",
        "timeout": litellm.request_timeout,
        "follow_redirects": True,
    }
    if api_key is not None:
        _request["headers"] = {"Authorization": "Bearer {}".format(api_key)}
    with httpx.stream(**_request) as response:
        try:
            if response.status_code != 200:
                raise OllamaError(
                    status_code=response.status_code, message=response.iter_lines()
                )

            streamwrapper = litellm.CustomStreamWrapper(
                completion_stream=response.iter_lines(),
                model=data["model"],
                custom_llm_provider="ollama_chat",
                logging_obj=logging_obj,
            )

            # If format is JSON, this was a function call
            # Gather all chunks and return the function call as one delta to simplify parsing
            if data.get("format", "") == "json":
                first_chunk = next(streamwrapper)
                response_content = "".join(
                    chunk.choices[0].delta.content
                    for chunk in chain([first_chunk], streamwrapper)
                    if chunk.choices[0].delta.content
                )

                function_call = json.loads(response_content)
                delta = litellm.utils.Delta(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "name": function_call["name"],
                                "arguments": json.dumps(function_call["arguments"]),
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response = first_chunk
                model_response.choices[0].delta = delta  # type: ignore
                model_response.choices[0].finish_reason = "tool_calls"
                yield model_response
            else:
                for transformed_chunk in streamwrapper:
                    yield transformed_chunk
        except Exception as e:
            raise e


async def ollama_async_streaming(
    url, api_key, data, model_response, encoding, logging_obj
):
    try:
        client = httpx.AsyncClient()
        _request = {
            "url": f"{url}",
            "json": data,
            "method": "POST",
            "timeout": litellm.request_timeout,
        }
        if api_key is not None:
            _request["headers"] = {"Authorization": "Bearer {}".format(api_key)}
        async with client.stream(**_request) as response:
            if response.status_code != 200:
                raise OllamaError(
                    status_code=response.status_code, message=response.text
                )

            streamwrapper = litellm.CustomStreamWrapper(
                completion_stream=response.aiter_lines(),
                model=data["model"],
                custom_llm_provider="ollama_chat",
                logging_obj=logging_obj,
            )

            # If format is JSON, this was a function call
            # Gather all chunks and return the function call as one delta to simplify parsing
            if data.get("format", "") == "json":
                first_chunk = await anext(streamwrapper)
                first_chunk_content = first_chunk.choices[0].delta.content or ""
                response_content = first_chunk_content + "".join(
                    [
                        chunk.choices[0].delta.content
                        async for chunk in streamwrapper
                        if chunk.choices[0].delta.content
                    ]
                )
                function_call = json.loads(response_content)
                delta = litellm.utils.Delta(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "name": function_call.get(
                                    "name", function_call.get("function", None)
                                ),
                                "arguments": json.dumps(function_call["arguments"]),
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response = first_chunk
                model_response.choices[0].delta = delta  # type: ignore
                model_response.choices[0].finish_reason = "tool_calls"
                yield model_response
            else:
                async for transformed_chunk in streamwrapper:
                    yield transformed_chunk
    except Exception as e:
        verbose_logger.exception(
            "LiteLLM.ollama(): Exception occured - {}".format(str(e))
        )


async def ollama_acompletion(
    url,
    api_key: Optional[str],
    data,
    model_response: litellm.ModelResponse,
    encoding,
    logging_obj,
    function_name,
):
    data["stream"] = False
    try:
        timeout = aiohttp.ClientTimeout(total=litellm.request_timeout)  # 10 minutes
        async with aiohttp.ClientSession(timeout=timeout) as session:
            _request = {
                "url": f"{url}",
                "json": data,
            }
            if api_key is not None:
                _request["headers"] = {"Authorization": "Bearer {}".format(api_key)}
            resp = await session.post(**_request)

            if resp.status != 200:
                text = await resp.text()
                raise OllamaError(status_code=resp.status, message=text)

            response_json = await resp.json()

            ## LOGGING
            logging_obj.post_call(
                input=data,
                api_key="",
                original_response=response_json,
                additional_args={
                    "headers": None,
                    "api_base": url,
                },
            )

            ## RESPONSE OBJECT
            model_response.choices[0].finish_reason = "stop"

            if data.get("format", "") == "json" and function_name is not None:
                function_call = json.loads(response_json["message"]["content"])
                message = litellm.Message(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "name": function_call.get("name", function_name),
                                "arguments": json.dumps(
                                    function_call.get("arguments", function_call)
                                ),
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response.choices[0].message = message  # type: ignore
                model_response.choices[0].finish_reason = "tool_calls"
            else:
                _message = litellm.Message(**response_json["message"])
                model_response.choices[0].message = _message  # type: ignore

            model_response.created = int(time.time())
            model_response.model = "ollama_chat/" + data["model"]
            prompt_tokens = response_json.get("prompt_eval_count", litellm.token_counter(messages=data["messages"]))  # type: ignore
            completion_tokens = response_json.get(
                "eval_count",
                litellm.token_counter(
                    text=response_json["message"]["content"], count_response_tokens=True
                ),
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
    except Exception as e:
        raise e  # don't use verbose_logger.exception, if exception is raised
