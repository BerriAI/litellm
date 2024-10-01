import json
import os
import time
from enum import Enum
from typing import Callable, Optional

import requests  # type: ignore

from litellm.utils import EmbeddingResponse, ModelResponse, Usage

from .prompt_templates.factory import custom_prompt, prompt_factory


class OobaboogaError(Exception):
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
        headers["Authorization"] = f"Token {api_key}"
    return headers


def completion(
    model: str,
    messages: list,
    api_base: Optional[str],
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params: dict,
    custom_prompt_dict={},
    litellm_params=None,
    logger_fn=None,
    default_max_tokens_to_sample=None,
):
    headers = validate_environment(api_key)
    if "https" in model:
        completion_url = model
    elif api_base:
        completion_url = api_base
    else:
        raise OobaboogaError(
            status_code=404,
            message="API Base not set. Set one via completion(..,api_base='your-api-url')",
        )
    model = model

    completion_url = completion_url + "/v1/chat/completions"
    data = {
        "messages": messages,
        **optional_params,
    }
    ## LOGGING

    logging_obj.pre_call(
        input=messages,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
    )
    ## COMPLETION CALL

    response = requests.post(
        completion_url,
        headers=headers,
        data=json.dumps(data),
        stream=optional_params["stream"] if "stream" in optional_params else False,
    )
    if "stream" in optional_params and optional_params["stream"] is True:
        return response.iter_lines()
    else:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except Exception:
            raise OobaboogaError(
                message=response.text, status_code=response.status_code
            )
        if "error" in completion_response:
            raise OobaboogaError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            try:
                model_response.choices[0].message.content = completion_response["choices"][0]["message"]["content"]  # type: ignore
            except Exception:
                raise OobaboogaError(
                    message=json.dumps(completion_response),
                    status_code=response.status_code,
                )

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=completion_response["usage"]["prompt_tokens"],
            completion_tokens=completion_response["usage"]["completion_tokens"],
            total_tokens=completion_response["usage"]["total_tokens"],
        )
        setattr(model_response, "usage", usage)
        return model_response


def embedding(
    model: str,
    input: list,
    model_response: EmbeddingResponse,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    logging_obj=None,
    optional_params=None,
    encoding=None,
):
    # Create completion URL
    if "https" in model:
        embeddings_url = model
    elif api_base:
        embeddings_url = f"{api_base}/v1/embeddings"
    else:
        raise OobaboogaError(
            status_code=404,
            message="API Base not set. Set one via completion(..,api_base='your-api-url')",
        )

    # Prepare request data
    data = {"input": input}
    if optional_params:
        data.update(optional_params)

    # Logging before API call
    if logging_obj:
        logging_obj.pre_call(
            input=input, api_key=api_key, additional_args={"complete_input_dict": data}
        )

    # Send POST request
    headers = validate_environment(api_key)
    response = requests.post(embeddings_url, headers=headers, json=data)
    if not response.ok:
        raise OobaboogaError(message=response.text, status_code=response.status_code)
    completion_response = response.json()

    # Check for errors in response
    if "error" in completion_response:
        raise OobaboogaError(
            message=completion_response["error"],
            status_code=completion_response.get("status_code", 500),
        )

    # Process response data
    model_response.data = [
        {
            "embedding": completion_response["data"][0]["embedding"],
            "index": 0,
            "object": "embedding",
        }
    ]

    num_tokens = len(completion_response["data"][0]["embedding"])
    # Adding metadata to response
    setattr(
        model_response,
        "usage",
        Usage(prompt_tokens=num_tokens, total_tokens=num_tokens),
    )
    model_response.object = "list"
    model_response.model = model

    return model_response
