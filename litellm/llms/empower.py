from typing import Optional, Union, Any, BinaryIO
import types, time, json, traceback
import httpx
from .base import BaseLLM
from litellm.utils import (
    ModelResponse,
    Choices,
    Message,
    CustomStreamWrapper,
    convert_to_model_response_object,
    Usage,
    TranscriptionResponse,
)
from typing import Callable, Optional
import aiohttp, requests
import litellm
from .prompt_templates.factory import prompt_factory, custom_prompt
from openai import OpenAI, AsyncOpenAI


class EmpowerError(Exception):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ):
        self.status_code = status_code
        self.message = message
        if request:
            self.request = request
        else:
            self.request = httpx.Request(method="POST", url="https://app.empower.dev/api/v1")
        if response:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class EmpowerConfig:
    """
    Reference: https://docs.empower.dev/api-reference/chat-completions/completions

    The class `EmpowerConfig` provides configuration for the Empower's Chat API interface. Below are the parameters:

    - `best_of` (integer or null): This optional parameter generates server-side completions and returns the one with the highest log probability per token.

    - `frequency_penalty` (number or null): Defaults to 0. Allows a value between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, thereby minimizing repetition.

    - `max_tokens` (integer or null): Defaults to 256. This optional parameter helps to set the maximum number of tokens to generate in the chat completion.

    - `n` (integer or null): This optional parameter helps to set how many chat completion choices to generate for each input message.

    - `presence_penalty` (number or null): Defaults to 0. It penalizes new tokens based on if they appear in the text so far, hence increasing the model's likelihood to talk about new topics.

    - `temperature` (number or null): Defaults to 1. Defines the sampling temperature to use, varying between 0 and 2.

    - `tools` (list or null): Defaults to null. This optional parameter helps to set the tools to use for the completion.

    - `tool_choice` (string or null): Controls which (if any) function is called by the model. `none` means the model will not call a function and instead generates a message. `auto` means the model can pick between generating a message or calling a function.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling.
    """

    best_of: Optional[int] = None
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    temperature: Optional[float] = None
    tools: Optional[list] = None
    tool_choice: Optional[str] = None
    top_p: Optional[float] = None

    def __init__(
        self,
        best_of: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        top_p: Optional[float] = None,
    ):
        locals_ = locals()
        for key in locals_.keys():
            if key != "self":
                setattr(self, key, locals_[key])

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
    if api_key is None:
        raise ValueError(
            "Missing Empower API Key - A call is being made to Empower but no key is set either in the environment variables or via params"
        )
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    return headers


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
        optional_params=None):

    print('calling completion')
    headers = validate_environment(api_key)
    config = litellm.EmpowerConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > togetherai_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    print_verbose(f"CUSTOM PROMPT DICT: {custom_prompt_dict}; model: {model}")
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_details = custom_prompt_dict[model]
        prompt = custom_prompt(
            role_dict=model_prompt_details.get("roles", {}),
            initial_prompt_value=model_prompt_details.get("initial_prompt_value", ""),
            final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
            bos_token=model_prompt_details.get("bos_token", ""),
            eos_token=model_prompt_details.get("eos_token", ""),
            messages=messages,
        )
    else:
        prompt = prompt_factory(
            model=model,
            messages=messages,
            api_key=api_key,
            custom_llm_provider="together_ai",
        )  # api key required to query together ai model list

    data = {
        "model": model,
        "prompt": prompt,
        "request_type": "language-model-inference",
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": api_base,
        },
    )
    ## COMPLETION CALL
    if "stream_tokens" in optional_params and optional_params["stream_tokens"] == True:
        response = requests.post(
            api_base,
            headers=headers,
            data=json.dumps(data),
            stream=optional_params["stream_tokens"],
        )
        return response.iter_lines()
    else:
        response = requests.post(api_base, headers=headers, data=json.dumps(data))
        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        if response.status_code != 200:
            raise EmpowerError(
                status_code=response.status_code, message=response.text
            )
        completion_response = response.json()

        if "error" in completion_response:
            raise EmpowerError(
                message=json.dumps(completion_response),
                status_code=response.status_code,
            )
        elif "error" in completion_response["output"]:
            raise EmpowerError(
                message=json.dumps(completion_response["output"]),
                status_code=response.status_code,
            )

        if len(completion_response["output"]["choices"][0]["text"]) >= 0:
            model_response["choices"][0]["message"]["content"] = completion_response[
                "output"
            ]["choices"][0]["text"]

        ## CALCULATING USAGE
        print_verbose(
            f"CALCULATING TOGETHERAI TOKEN USAGE. Model Response: {model_response}; model_response['choices'][0]['message'].get('content', ''): {model_response['choices'][0]['message'].get('content', None)}"
        )
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
        )
        if "finish_reason" in completion_response["output"]["choices"][0]:
            model_response.choices[0].finish_reason = completion_response["output"][
                "choices"
            ][0]["finish_reason"]
        model_response["created"] = int(time.time())
        model_response["model"] = "together_ai/" + model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        model_response.usage = usage
        return model_response

def embedding():
    pass