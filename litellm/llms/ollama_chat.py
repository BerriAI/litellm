import requests, types, time
import json, uuid
import traceback
from typing import Optional
import litellm
import httpx, aiohttp, asyncio
from .prompt_templates.factory import prompt_factory, custom_prompt


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
    Reference: https://github.com/jmorganca/ollama/blob/main/docs/api.md#parameters

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
    stop: Optional[list] = (
        None  # stop is a list based on this - https://github.com/jmorganca/ollama/pull/442
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
            "stream",
            "top_p",
            "temperature",
            "frequency_penalty",
            "stop",
            "tools",
            "tool_choice",
            "functions",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["num_predict"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "frequency_penalty":
                optional_params["repeat_penalty"] = param
            if param == "stop":
                optional_params["stop"] = value
            ### FUNCTION CALLING LOGIC ###
            if param == "tools":
                # ollama actually supports json output
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
                optional_params["functions_unsupported_model"] = non_default_params.pop(
                    "functions"
                )
        non_default_params.pop("tool_choice", None)  # causes ollama requests to hang
        return optional_params


# ollama implementation
def get_ollama_response(
    api_base="http://localhost:11434",
    model="llama2",
    messages=None,
    optional_params=None,
    logging_obj=None,
    acompletion: bool = False,
    model_response=None,
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

    for m in messages:
        if "role" in m and m["role"] == "tool":
            m["role"] = "assistant"

    data = {
        "model": model,
        "messages": messages,
        "options": optional_params,
        "stream": stream,
    }
    if format is not None:
        data["format"] = format
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
        if stream == True:
            response = ollama_async_streaming(
                url=url,
                data=data,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
            )
        else:
            response = ollama_acompletion(
                url=url,
                data=data,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
                function_name=function_name,
            )
        return response
    elif stream == True:
        return ollama_completion_stream(url=url, data=data, logging_obj=logging_obj)

    response = requests.post(
        url=f"{url}",
        json=data,
    )
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
    model_response["choices"][0]["finish_reason"] = "stop"
    if data.get("format", "") == "json":
        message = litellm.Message(
            content=None,
            tool_calls=[
                {
                    "id": f"call_{str(uuid.uuid4())}",
                    "function": {
                        "arguments": response_json["message"]["content"],
                        "name": "",
                    },
                    "type": "function",
                }
            ],
        )
        model_response["choices"][0]["message"] = message
    else:
        model_response["choices"][0]["message"] = response_json["message"]
    model_response["created"] = int(time.time())
    model_response["model"] = "ollama/" + model
    prompt_tokens = response_json.get("prompt_eval_count", litellm.token_counter(messages=messages))  # type: ignore
    completion_tokens = response_json.get(
        "eval_count", litellm.token_counter(text=response_json["message"]["content"])
    )
    model_response["usage"] = litellm.Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return model_response


def ollama_completion_stream(url, data, logging_obj):
    with httpx.stream(
        url=url, json=data, method="POST", timeout=litellm.request_timeout
    ) as response:
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
            for transformed_chunk in streamwrapper:
                yield transformed_chunk
        except Exception as e:
            raise e


async def ollama_async_streaming(url, data, model_response, encoding, logging_obj):
    try:
        client = httpx.AsyncClient()
        async with client.stream(
            url=f"{url}", json=data, method="POST", timeout=litellm.request_timeout
        ) as response:
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
            async for transformed_chunk in streamwrapper:
                yield transformed_chunk
    except Exception as e:
        traceback.print_exc()


async def ollama_acompletion(
    url, data, model_response, encoding, logging_obj, function_name
):
    data["stream"] = False
    try:
        timeout = aiohttp.ClientTimeout(total=litellm.request_timeout)  # 10 minutes
        async with aiohttp.ClientSession(timeout=timeout) as session:
            resp = await session.post(url, json=data)

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
            model_response["choices"][0]["finish_reason"] = "stop"
            if data.get("format", "") == "json":
                message = litellm.Message(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "arguments": response_json["message"]["content"],
                                "name": function_name or "",
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response["choices"][0]["message"] = message
            else:
                model_response["choices"][0]["message"] = response_json["message"]

            model_response["created"] = int(time.time())
            model_response["model"] = "ollama_chat/" + data["model"]
            prompt_tokens = response_json.get("prompt_eval_count", litellm.token_counter(messages=data["messages"]))  # type: ignore
            completion_tokens = response_json.get(
                "eval_count",
                litellm.token_counter(
                    text=response_json["message"]["content"], count_response_tokens=True
                ),
            )
            model_response["usage"] = litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            return model_response
    except Exception as e:
        traceback.print_exc()
        raise e
