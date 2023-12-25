import os
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse, Usage


class BasetenError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


def validate_environment(api_key):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Api-Key {api_key}"
    return headers


def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    completion_url_fragment_1 = "https://app.baseten.co/models/"
    completion_url_fragment_2 = "/predict"
    model = model
    prompt = ""
    for message in messages:
        if "role" in message:
            if message["role"] == "user":
                prompt += f"{message['content']}"
            else:
                prompt += f"{message['content']}"
        else:
            prompt += f"{message['content']}"
    data = {
        "inputs": prompt,
        "prompt": prompt,
        "parameters": optional_params,
        "stream": True
        if "stream" in optional_params and optional_params["stream"] == True
        else False,
    }

    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
    )
    ## COMPLETION CALL
    response = requests.post(
        completion_url_fragment_1 + model + completion_url_fragment_2,
        headers=headers,
        data=json.dumps(data),
        stream=True
        if "stream" in optional_params and optional_params["stream"] == True
        else False,
    )
    if "text/event-stream" in response.headers["Content-Type"] or (
        "stream" in optional_params and optional_params["stream"] == True
    ):
        return response.iter_lines()
    else:
        ## LOGGING
        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        completion_response = response.json()
        if "error" in completion_response:
            raise BasetenError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            if "model_output" in completion_response:
                if (
                    isinstance(completion_response["model_output"], dict)
                    and "data" in completion_response["model_output"]
                    and isinstance(completion_response["model_output"]["data"], list)
                ):
                    model_response["choices"][0]["message"][
                        "content"
                    ] = completion_response["model_output"]["data"][0]
                elif isinstance(completion_response["model_output"], str):
                    model_response["choices"][0]["message"][
                        "content"
                    ] = completion_response["model_output"]
            elif "completion" in completion_response and isinstance(
                completion_response["completion"], str
            ):
                model_response["choices"][0]["message"][
                    "content"
                ] = completion_response["completion"]
            elif isinstance(completion_response, list) and len(completion_response) > 0:
                if "generated_text" not in completion_response:
                    raise BasetenError(
                        message=f"Unable to parse response. Original response: {response.text}",
                        status_code=response.status_code,
                    )
                model_response["choices"][0]["message"][
                    "content"
                ] = completion_response[0]["generated_text"]
                ## GETTING LOGPROBS
                if (
                    "details" in completion_response[0]
                    and "tokens" in completion_response[0]["details"]
                ):
                    model_response.choices[0].finish_reason = completion_response[0][
                        "details"
                    ]["finish_reason"]
                    sum_logprob = 0
                    for token in completion_response[0]["details"]["tokens"]:
                        sum_logprob += token["logprob"]
                    model_response["choices"][0]["message"]._logprobs = sum_logprob
            else:
                raise BasetenError(
                    message=f"Unable to parse response. Original response: {response.text}",
                    status_code=response.status_code,
                )

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here.
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"]["content"])
        )

        model_response["created"] = int(time.time())
        model_response["model"] = model
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
