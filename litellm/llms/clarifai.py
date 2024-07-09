import os, types, traceback
import json
import requests
import time
from typing import Callable, Optional
from litellm.utils import ModelResponse, Usage, Choices, Message, CustomStreamWrapper
import litellm
import httpx
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from .prompt_templates.factory import prompt_factory, custom_prompt


class ClarifaiError(Exception):
    def __init__(self, status_code, message, url):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url=url)
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(self.message)


class ClarifaiConfig:
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


def validate_environment(api_key):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def completions_to_model(payload):
    # if payload["n"] != 1:
    #     raise HTTPException(
    #         status_code=422,
    #         detail="Only one generation is supported. Please set candidate_count to 1.",
    #     )

    params = {}
    if temperature := payload.get("temperature"):
        params["temperature"] = temperature
    if max_tokens := payload.get("max_tokens"):
        params["max_tokens"] = max_tokens
    return {
        "inputs": [{"data": {"text": {"raw": payload["prompt"]}}}],
        "model": {"output_info": {"params": params}},
    }


def process_response(
    model, prompt, response, model_response, api_key, data, encoding, logging_obj
):
    logging_obj.post_call(
        input=prompt,
        api_key=api_key,
        original_response=response.text,
        additional_args={"complete_input_dict": data},
    )
    ## RESPONSE OBJECT
    try:
        completion_response = response.json()
    except Exception:
        raise ClarifaiError(
            message=response.text, status_code=response.status_code, url=model
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
        model_response["choices"] = choices_list

    except Exception as e:
        raise ClarifaiError(
            message=traceback.format_exc(), status_code=response.status_code, url=model
        )

    # Calculate Usage
    prompt_tokens = len(encoding.encode(prompt))
    completion_tokens = len(
        encoding.encode(model_response["choices"][0]["message"].get("content"))
    )
    model_response["model"] = model
    model_response["usage"] = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return model_response


def convert_model_to_url(model: str, api_base: str):
    user_id, app_id, model_id = model.split(".")
    model_id = model_id.lower()
    return f"{api_base}/users/{user_id}/apps/{app_id}/models/{model_id}/outputs"


def get_prompt_model_name(url: str):
    clarifai_model_name = url.split("/")[-2]
    if "claude" in clarifai_model_name:
        return "anthropic", clarifai_model_name.replace("_", ".")
    if ("llama" in clarifai_model_name) or ("mistral" in clarifai_model_name):
        return "", "meta-llama/llama-2-chat"
    else:
        return "", clarifai_model_name


async def async_completion(
    model: str,
    prompt: str,
    api_base: str,
    custom_prompt_dict: dict,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    data=None,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
    headers={},
):

    async_handler = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))
    response = await async_handler.post(
        url=model, headers=headers, data=json.dumps(data)
    )

    logging_obj.post_call(
        input=prompt,
        api_key=api_key,
        original_response=response.text,
        additional_args={"complete_input_dict": data},
    )
    ## RESPONSE OBJECT
    try:
        completion_response = response.json()
    except Exception:
        raise ClarifaiError(
            message=response.text, status_code=response.status_code, url=model
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
        model_response["choices"] = choices_list

    except Exception as e:
        raise ClarifaiError(
            message=traceback.format_exc(), status_code=response.status_code, url=model
        )

    # Calculate Usage
    prompt_tokens = len(encoding.encode(prompt))
    completion_tokens = len(
        encoding.encode(model_response["choices"][0]["message"].get("content"))
    )
    model_response["model"] = model
    model_response["usage"] = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return model_response


def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    custom_prompt_dict={},
    acompletion=False,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    model = convert_model_to_url(model, api_base)
    prompt = " ".join(message["content"] for message in messages)  # TODO

    ## Load Config
    config = litellm.ClarifaiConfig.get_config()
    for k, v in config.items():
        if k not in optional_params:
            optional_params[k] = v

    custom_llm_provider, orig_model_name = get_prompt_model_name(model)
    if custom_llm_provider == "anthropic":
        prompt = prompt_factory(
            model=orig_model_name,
            messages=messages,
            api_key=api_key,
            custom_llm_provider="clarifai",
        )
    else:
        prompt = prompt_factory(
            model=orig_model_name,
            messages=messages,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider,
        )
    # print(prompt); exit(0)

    data = {
        "prompt": prompt,
        **optional_params,
    }
    data = completions_to_model(data)

    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": model,
        },
    )
    if acompletion == True:
        return async_completion(
            model=model,
            prompt=prompt,
            api_base=api_base,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            data=data,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
        )
    else:
        ## COMPLETION CALL
        response = requests.post(
            model,
            headers=headers,
            data=json.dumps(data),
        )
    # print(response.content); exit()

    if response.status_code != 200:
        raise ClarifaiError(
            status_code=response.status_code, message=response.text, url=model
        )

    if "stream" in optional_params and optional_params["stream"] == True:
        completion_stream = response.iter_lines()
        stream_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="clarifai",
            logging_obj=logging_obj,
        )
        return stream_response

    else:
        return process_response(
            model=model,
            prompt=prompt,
            response=response,
            model_response=model_response,
            api_key=api_key,
            data=data,
            encoding=encoding,
            logging_obj=logging_obj,
        )


class ModelResponseIterator:
    def __init__(self, model_response):
        self.model_response = model_response
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self.model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self.model_response
