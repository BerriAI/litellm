import asyncio
import json
import os
import time
import types
from typing import Any, Callable, Optional, Tuple, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

from .prompt_templates.factory import custom_prompt, prompt_factory


class ReplicateError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.replicate.com/v1/deployments"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class ReplicateConfig:
    """
    Reference: https://replicate.com/meta/llama-2-70b-chat/api
    - `prompt` (string): The prompt to send to the model.

    - `system_prompt` (string): The system prompt to send to the model. This is prepended to the prompt and helps guide system behavior. Default value: `You are a helpful assistant`.

    - `max_new_tokens` (integer): Maximum number of tokens to generate. Typically, a word is made up of 2-3 tokens. Default value: `128`.

    - `min_new_tokens` (integer): Minimum number of tokens to generate. To disable, set to `-1`. A word is usually 2-3 tokens. Default value: `-1`.

    - `temperature` (number): Adjusts the randomness of outputs. Values greater than 1 increase randomness, 0 is deterministic, and 0.75 is a reasonable starting value. Default value: `0.75`.

    - `top_p` (number): During text decoding, it samples from the top `p` percentage of most likely tokens. Reduce this to ignore less probable tokens. Default value: `0.9`.

    - `top_k` (integer): During text decoding, samples from the top `k` most likely tokens. Reduce this to ignore less probable tokens. Default value: `50`.

    - `stop_sequences` (string): A comma-separated list of sequences to stop generation at. For example, inputting '<end>,<stop>' will cease generation at the first occurrence of either 'end' or '<stop>'.

    - `seed` (integer): This is the seed for the random generator. Leave it blank to randomize the seed.

    - `debug` (boolean): If set to `True`, it provides debugging output in logs.

    Please note that Replicate's mapping of these parameters can be inconsistent across different models, indicating that not all of these parameters may be available for use with all models.
    """

    system_prompt: Optional[str] = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[str] = None
    seed: Optional[int] = None
    debug: Optional[bool] = None

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[str] = None,
        seed: Optional[int] = None,
        debug: Optional[bool] = None,
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


# Function to start a prediction and get the prediction URL
def start_prediction(
    version_id, input_data, api_token, api_base, logging_obj, print_verbose
):
    base_url = api_base
    if "deployments" in version_id:
        print_verbose("\nLiteLLM: Request to custom replicate deployment")
        version_id = version_id.replace("deployments/", "")
        base_url = f"https://api.replicate.com/v1/deployments/{version_id}"
        print_verbose(f"Deployment base URL: {base_url}\n")
    else:  # assume it's a model
        base_url = f"https://api.replicate.com/v1/models/{version_id}"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }

    initial_prediction_data = {
        "input": input_data,
    }

    if ":" in version_id and len(version_id) > 64:
        model_parts = version_id.split(":")
        if (
            len(model_parts) > 1 and len(model_parts[1]) == 64
        ):  ## checks if model name has a 64 digit code - e.g. "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
            initial_prediction_data["version"] = model_parts[1]

    ## LOGGING
    logging_obj.pre_call(
        input=input_data["prompt"],
        api_key="",
        additional_args={
            "complete_input_dict": initial_prediction_data,
            "headers": headers,
            "api_base": base_url,
        },
    )

    response = requests.post(
        f"{base_url}/predictions", json=initial_prediction_data, headers=headers
    )
    if response.status_code == 201:
        response_data = response.json()
        return response_data.get("urls", {}).get("get")
    else:
        raise ReplicateError(
            response.status_code, f"Failed to start prediction {response.text}"
        )


async def async_start_prediction(
    version_id,
    input_data,
    api_token,
    api_base,
    logging_obj,
    print_verbose,
    http_handler: AsyncHTTPHandler,
) -> str:
    base_url = api_base
    if "deployments" in version_id:
        print_verbose("\nLiteLLM: Request to custom replicate deployment")
        version_id = version_id.replace("deployments/", "")
        base_url = f"https://api.replicate.com/v1/deployments/{version_id}"
        print_verbose(f"Deployment base URL: {base_url}\n")
    else:  # assume it's a model
        base_url = f"https://api.replicate.com/v1/models/{version_id}"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }

    initial_prediction_data = {
        "input": input_data,
    }

    if ":" in version_id and len(version_id) > 64:
        model_parts = version_id.split(":")
        if (
            len(model_parts) > 1 and len(model_parts[1]) == 64
        ):  ## checks if model name has a 64 digit code - e.g. "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
            initial_prediction_data["version"] = model_parts[1]

    ## LOGGING
    logging_obj.pre_call(
        input=input_data["prompt"],
        api_key="",
        additional_args={
            "complete_input_dict": initial_prediction_data,
            "headers": headers,
            "api_base": base_url,
        },
    )

    response = await http_handler.post(
        url="{}/predictions".format(base_url),
        data=json.dumps(initial_prediction_data),
        headers=headers,
    )

    if response.status_code == 201:
        response_data = response.json()
        return response_data.get("urls", {}).get("get")
    else:
        raise ReplicateError(
            response.status_code, f"Failed to start prediction {response.text}"
        )


# Function to handle prediction response (non-streaming)
def handle_prediction_response(prediction_url, api_token, print_verbose):
    output_string = ""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }

    status = ""
    logs = ""
    while True and (status not in ["succeeded", "failed", "canceled"]):
        print_verbose(f"replicate: polling endpoint: {prediction_url}")
        time.sleep(0.5)
        response = requests.get(prediction_url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            if "output" in response_data:
                output_string = "".join(response_data["output"])
                print_verbose(f"Non-streamed output:{output_string}")
            status = response_data.get("status", None)
            logs = response_data.get("logs", "")
            if status == "failed":
                replicate_error = response_data.get("error", "")
                raise ReplicateError(
                    status_code=400,
                    message=f"Error: {replicate_error}, \nReplicate logs:{logs}",
                )
        else:
            # this can fail temporarily but it does not mean the replicate request failed, replicate request fails when status=="failed"
            print_verbose("Replicate: Failed to fetch prediction status and output.")
    return output_string, logs


async def async_handle_prediction_response(
    prediction_url, api_token, print_verbose, http_handler: AsyncHTTPHandler
) -> Tuple[str, Any]:
    output_string = ""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }

    status = ""
    logs = ""
    while True and (status not in ["succeeded", "failed", "canceled"]):
        print_verbose(f"replicate: polling endpoint: {prediction_url}")
        await asyncio.sleep(0.5)  # prevent replicate rate limit errors
        response = await http_handler.get(prediction_url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            if "output" in response_data:
                output_string = "".join(response_data["output"])
                print_verbose(f"Non-streamed output:{output_string}")
            status = response_data.get("status", None)
            logs = response_data.get("logs", "")
            if status == "failed":
                replicate_error = response_data.get("error", "")
                raise ReplicateError(
                    status_code=400,
                    message=f"Error: {replicate_error}, \nReplicate logs:{logs}",
                )
        else:
            # this can fail temporarily but it does not mean the replicate request failed, replicate request fails when status=="failed"
            print_verbose("Replicate: Failed to fetch prediction status and output.")
    return output_string, logs


# Function to handle prediction response (streaming)
def handle_prediction_response_streaming(prediction_url, api_token, print_verbose):
    previous_output = ""
    output_string = ""

    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    status = ""
    while True and (status not in ["succeeded", "failed", "canceled"]):
        time.sleep(0.5)  # prevent being rate limited by replicate
        print_verbose(f"replicate: polling endpoint: {prediction_url}")
        response = requests.get(prediction_url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            status = response_data["status"]
            if "output" in response_data:
                try:
                    output_string = "".join(response_data["output"])
                except Exception as e:
                    raise ReplicateError(
                        status_code=422,
                        message="Unable to parse response. Got={}".format(
                            response_data["output"]
                        ),
                    )
                new_output = output_string[len(previous_output) :]
                print_verbose(f"New chunk: {new_output}")
                yield {"output": new_output, "status": status}
                previous_output = output_string
            status = response_data["status"]
            if status == "failed":
                replicate_error = response_data.get("error", "")
                raise ReplicateError(
                    status_code=400, message=f"Error: {replicate_error}"
                )
        else:
            # this can fail temporarily but it does not mean the replicate request failed, replicate request fails when status=="failed"
            print_verbose(
                f"Replicate: Failed to fetch prediction status and output.{response.status_code}{response.text}"
            )


# Function to handle prediction response (streaming)
async def async_handle_prediction_response_streaming(
    prediction_url, api_token, print_verbose
):
    http_handler = AsyncHTTPHandler(concurrent_limit=1)
    previous_output = ""
    output_string = ""

    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    status = ""
    while True and (status not in ["succeeded", "failed", "canceled"]):
        await asyncio.sleep(0.5)  # prevent being rate limited by replicate
        print_verbose(f"replicate: polling endpoint: {prediction_url}")
        response = await http_handler.get(prediction_url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            status = response_data["status"]
            if "output" in response_data:
                try:
                    output_string = "".join(response_data["output"])
                except Exception as e:
                    raise ReplicateError(
                        status_code=422,
                        message="Unable to parse response. Got={}".format(
                            response_data["output"]
                        ),
                    )
                new_output = output_string[len(previous_output) :]
                print_verbose(f"New chunk: {new_output}")
                yield {"output": new_output, "status": status}
                previous_output = output_string
            status = response_data["status"]
            if status == "failed":
                replicate_error = response_data.get("error", "")
                raise ReplicateError(
                    status_code=400, message=f"Error: {replicate_error}"
                )
        else:
            # this can fail temporarily but it does not mean the replicate request failed, replicate request fails when status=="failed"
            print_verbose(
                f"Replicate: Failed to fetch prediction status and output.{response.status_code}{response.text}"
            )


# Function to extract version ID from model string
def model_to_version_id(model):
    if ":" in model:
        split_model = model.split(":")
        return split_model[1]
    return model


def process_response(
    model_response: ModelResponse,
    result: str,
    model: str,
    encoding: Any,
    prompt: str,
) -> ModelResponse:
    if len(result) == 0:  # edge case, where result from replicate is empty
        result = " "

    ## Building RESPONSE OBJECT
    if len(result) > 1:
        model_response["choices"][0]["message"]["content"] = result

    # Calculate usage
    prompt_tokens = len(encoding.encode(prompt, disallowed_special=()))
    completion_tokens = len(
        encoding.encode(
            model_response["choices"][0]["message"].get("content", ""),
            disallowed_special=(),
        )
    )
    model_response["model"] = "replicate/" + model
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    setattr(model_response, "usage", usage)

    return model_response


# Main function for prediction completion
def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    optional_params: dict,
    logging_obj,
    api_key,
    encoding,
    custom_prompt_dict={},
    litellm_params=None,
    logger_fn=None,
    acompletion=None,
) -> Union[ModelResponse, CustomStreamWrapper]:
    # Start a prediction and get the prediction URL
    version_id = model_to_version_id(model)
    ## Load Config
    config = litellm.ReplicateConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > replicate_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    system_prompt = None
    if optional_params is not None and "supports_system_prompt" in optional_params:
        supports_sys_prompt = optional_params.pop("supports_system_prompt")
    else:
        supports_sys_prompt = False

    if supports_sys_prompt:
        for i in range(len(messages)):
            if messages[i]["role"] == "system":
                first_sys_message = messages.pop(i)
                system_prompt = first_sys_message["content"]
                break

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
        prompt = prompt_factory(model=model, messages=messages)

    if prompt is None or not isinstance(prompt, str):
        raise ReplicateError(
            status_code=400,
            message="LiteLLM Error - prompt is not a string - {}".format(prompt),
        )

    # If system prompt is supported, and a system prompt is provided, use it
    if system_prompt is not None:
        input_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            **optional_params,
        }
    # Otherwise, use the prompt as is
    else:
        input_data = {"prompt": prompt, **optional_params}

    if acompletion is not None and acompletion == True:
        return async_completion(
            model_response=model_response,
            model=model,
            prompt=prompt,
            encoding=encoding,
            optional_params=optional_params,
            version_id=version_id,
            input_data=input_data,
            api_key=api_key,
            api_base=api_base,
            logging_obj=logging_obj,
            print_verbose=print_verbose,
        )  # type: ignore
    ## COMPLETION CALL
    ## Replicate Compeltion calls have 2 steps
    ## Step1: Start Prediction: gets a prediction url
    ## Step2: Poll prediction url for response
    ## Step2: is handled with and without streaming
    model_response["created"] = int(
        time.time()
    )  # for pricing this must remain right before calling api

    prediction_url = start_prediction(
        version_id,
        input_data,
        api_key,
        api_base,
        logging_obj=logging_obj,
        print_verbose=print_verbose,
    )
    print_verbose(prediction_url)

    # Handle the prediction response (streaming or non-streaming)
    if "stream" in optional_params and optional_params["stream"] == True:
        print_verbose("streaming request")
        _response = handle_prediction_response_streaming(
            prediction_url, api_key, print_verbose
        )
        return CustomStreamWrapper(_response, model, logging_obj=logging_obj, custom_llm_provider="replicate")  # type: ignore
    else:
        result, logs = handle_prediction_response(
            prediction_url, api_key, print_verbose
        )

        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=result,
            additional_args={
                "complete_input_dict": input_data,
                "logs": logs,
                "api_base": prediction_url,
            },
        )

        print_verbose(f"raw model_response: {result}")

        return process_response(
            model_response=model_response,
            result=result,
            model=model,
            encoding=encoding,
            prompt=prompt,
        )


async def async_completion(
    model_response: ModelResponse,
    model: str,
    prompt: str,
    encoding,
    optional_params: dict,
    version_id,
    input_data,
    api_key,
    api_base,
    logging_obj,
    print_verbose,
) -> Union[ModelResponse, CustomStreamWrapper]:
    http_handler = AsyncHTTPHandler(concurrent_limit=1)
    prediction_url = await async_start_prediction(
        version_id,
        input_data,
        api_key,
        api_base,
        logging_obj=logging_obj,
        print_verbose=print_verbose,
        http_handler=http_handler,
    )

    if "stream" in optional_params and optional_params["stream"] == True:
        _response = async_handle_prediction_response_streaming(
            prediction_url, api_key, print_verbose
        )
        return CustomStreamWrapper(_response, model, logging_obj=logging_obj, custom_llm_provider="replicate")  # type: ignore

    result, logs = await async_handle_prediction_response(
        prediction_url, api_key, print_verbose, http_handler=http_handler
    )

    return process_response(
        model_response=model_response,
        result=result,
        model=model,
        encoding=encoding,
        prompt=prompt,
    )


# # Example usage:
# response = completion(
#     api_key="",
#     messages=[{"content": "good morning"}],
#     model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
#     model_response=ModelResponse(),
#     print_verbose=print,
#     logging_obj=print, # stub logging_obj
#     optional_params={"stream": False}
# )

# print(response)
