import os, types
import json
from enum import Enum
import requests
import time
from typing import Callable, Optional
import litellm
import httpx
from litellm.utils import ModelResponse, Usage
from .prompt_templates.factory import prompt_factory, custom_prompt


class CloudflareError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.cloudflare.com")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class CloudflareConfig:
    max_tokens: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
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
    if api_key is None:
        raise ValueError(
            "Missing CloudflareError API Key - A call is being made to cloudflare but no key is set either in the environment variables or via params"
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
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)

    ## Load Config
    config = litellm.CloudflareConfig.get_config()
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
            raise CloudflareError(
                status_code=response.status_code, message=response.text
            )
        completion_response = response.json()

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
    # logic for parsing in - calling - parsing out model embedding calls
    pass
