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


class OllamaConfig:
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
    stop: Optional[
        list
    ] = None  # stop is a list based on this - https://github.com/jmorganca/ollama/pull/442
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


# ollama implementation
def get_ollama_response(
    api_base="http://localhost:11434",
    model="llama2",
    prompt="Why is the sky blue?",
    optional_params=None,
    logging_obj=None,
    acompletion: bool = False,
    model_response=None,
    encoding=None,
):
    if api_base.endswith("/api/generate"):
        url = api_base
    else:
        url = f"{api_base}/api/generate"

    ## Load Config
    config = litellm.OllamaConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    stream = optional_params.pop("stream", False)
    format = optional_params.pop("format", None)
    images = optional_params.pop("images", None)
    data = {
        "model": model,
        "prompt": prompt,
        "options": optional_params,
        "stream": stream,
    }
    if format is not None:
        data["format"] = format
    if images is not None:
        data["images"] = images

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
            )
        return response
    elif stream == True:
        return ollama_completion_stream(url=url, data=data, logging_obj=logging_obj)

    response = requests.post(
        url=f"{url}", json={**data, "stream": stream}, timeout=litellm.request_timeout
    )
    if response.status_code != 200:
        raise OllamaError(status_code=response.status_code, message=response.text)

    ## LOGGING
    logging_obj.post_call(
        input=prompt,
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
    if optional_params.get("format", "") == "json":
        message = litellm.Message(
            content=None,
            tool_calls=[
                {
                    "id": f"call_{str(uuid.uuid4())}",
                    "function": {"arguments": response_json["response"], "name": ""},
                    "type": "function",
                }
            ],
        )
        model_response["choices"][0]["message"] = message
    else:
        model_response["choices"][0]["message"]["content"] = response_json["response"]
    model_response["created"] = int(time.time())
    model_response["model"] = "ollama/" + model
    prompt_tokens = response_json.get("prompt_eval_count", len(encoding.encode(prompt)))  # type: ignore
    completion_tokens = response_json["eval_count"]
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
                    status_code=response.status_code, message=response.text
                )

            streamwrapper = litellm.CustomStreamWrapper(
                completion_stream=response.iter_lines(),
                model=data["model"],
                custom_llm_provider="ollama",
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
                    status_code=response.status_code, message=await response.aread()
                )

            streamwrapper = litellm.CustomStreamWrapper(
                completion_stream=response.aiter_lines(),
                model=data["model"],
                custom_llm_provider="ollama",
                logging_obj=logging_obj,
            )
            async for transformed_chunk in streamwrapper:
                yield transformed_chunk
    except Exception as e:
        traceback.print_exc()
        raise e


async def ollama_acompletion(url, data, model_response, encoding, logging_obj):
    data["stream"] = False
    try:
        timeout = aiohttp.ClientTimeout(total=litellm.request_timeout)  # 10 minutes
        async with aiohttp.ClientSession(timeout=timeout) as session:
            resp = await session.post(url, json=data)

            if resp.status != 200:
                text = await resp.text()
                raise OllamaError(status_code=resp.status, message=text)

            ## LOGGING
            logging_obj.post_call(
                input=data["prompt"],
                api_key="",
                original_response=resp.text,
                additional_args={
                    "headers": None,
                    "api_base": url,
                },
            )

            response_json = await resp.json()
            ## RESPONSE OBJECT
            model_response["choices"][0]["finish_reason"] = "stop"
            if data.get("format", "") == "json":
                message = litellm.Message(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "arguments": response_json["response"],
                                "name": "",
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response["choices"][0]["message"] = message
            else:
                model_response["choices"][0]["message"]["content"] = response_json[
                    "response"
                ]
            model_response["created"] = int(time.time())
            model_response["model"] = "ollama/" + data["model"]
            prompt_tokens = response_json.get("prompt_eval_count", len(encoding.encode(data["prompt"])))  # type: ignore
            completion_tokens = response_json["eval_count"]
            model_response["usage"] = litellm.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            return model_response
    except Exception as e:
        traceback.print_exc()
        raise e


async def ollama_aembeddings(
    api_base="http://localhost:11434",
    model="llama2",
    prompt="Why is the sky blue?",
    optional_params=None,
    logging_obj=None,
    model_response=None,
    encoding=None,
):
    if api_base.endswith("/api/embeddings"):
        url = api_base
    else:
        url = f"{api_base}/api/embeddings"

    ## Load Config
    config = litellm.OllamaConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    data = {
        "model": model,
        "prompt": prompt,
    }
    ## LOGGING
    logging_obj.pre_call(
        input=None,
        api_key=None,
        additional_args={"api_base": url, "complete_input_dict": data, "headers": {}},
    )
    timeout = aiohttp.ClientTimeout(total=litellm.request_timeout)  # 10 minutes
    async with aiohttp.ClientSession(timeout=timeout) as session:
        response = await session.post(url, json=data)

        if response.status != 200:
            text = await response.text()
            raise OllamaError(status_code=response.status, message=text)

        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=response.text,
            additional_args={
                "headers": None,
                "api_base": api_base,
            },
        )

        response_json = await response.json()
        embeddings = response_json["embedding"]
        ## RESPONSE OBJECT
        output_data = []
        for idx, embedding in enumerate(embeddings):
            output_data.append(
                {"object": "embedding", "index": idx, "embedding": embedding}
            )
        model_response["object"] = "list"
        model_response["data"] = output_data
        model_response["model"] = model

        input_tokens = len(encoding.encode(prompt))

        model_response["usage"] = {
            "prompt_tokens": input_tokens,
            "total_tokens": input_tokens,
        }
        return model_response
