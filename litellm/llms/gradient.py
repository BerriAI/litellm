import os
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse


class GradientAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class GradientMissingSecretError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def validate_environment(api_key: str, gradient_workspace_id: str):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if api_key and len(api_key) > 8:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise GradientMissingSecretError(
            "Missing valid Gradient API Key -"
            "set it via `GRADIENT_ACCESS_TOKEN`"
            "or via `api_key` params"
            "or via litellm.gradient_key"
        )

    if gradient_workspace_id and len(gradient_workspace_id) > 3:
        headers["x-gradient-workspace-id"] = f"{gradient_workspace_id}"
    else:
        raise GradientMissingSecretError(
            "Missing GRADIENT_WORKSPACE_ID - no workspace set via "
            " environment variable for `GRADIENT_WORKSPACE_ID`"
            " or litllm.gradient_workspace_id"
        )

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
    gradient_workspace_id: str = None,
):
    """Completion to gradient.ai

    Args:
        model (str): fine-tuned or base model `ID` of gradient.ai
        messages (list): list of messages
        model_response (ModelResponse): _description_
        print_verbose (Callable): print function
        encoding (_type_): Callable that implments encoding.encode(str)
        api_key (_type_): gradient.ai access token
        logging_obj (_type_): logging
        optional_params (_type_, optional): _description_. Defaults to None.
        litellm_params (_type_, optional): _description_. Defaults to None.
        logger_fn (_type_, optional): _description_. Defaults to None.
        gradient_workspace_id (str): workspace `ID` of gradient.ai

    Raises:
        Exception: _description_
        GradientAIError: _description_


    Returns:
        _type_: _description_
    """
    # HEADER
    headers = validate_environment(api_key, gradient_workspace_id)
    # URL
    base_url = os.getenv("GRADIENT_API_URL", "https://api.gradient.ai/api")
    completion_url = f"{base_url}/models/{model}/complete"
    # BODY
    prompt = " ".join(message["content"] for message in messages)

    data = {
        "query": prompt,
        **optional_params,
    }
    data.pop("stream", None)  # remove stream

    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
    )
    ## COMPLETION CALL
    response = requests.post(
        completion_url,
        headers=headers,
        data=json.dumps(data),
    )
    if "stream" in optional_params and optional_params["stream"] == True:
        raise GradientAIError(
            message="`stream=True` is NotImplemented.", status_code=response.status_code
        )
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
        try:
            completion_response = response.json()
        except:
            raise GradientAIError(
                message=response.text, status_code=response.status_code
            )
        print_verbose(f"json model_response: {completion_response}")
        try:
            model_response["choices"][0]["message"]["content"] = completion_response[
                "generatedOutput"
            ]
        except:
            raise GradientAIError(
                message=json.dumps(completion_response),
                status_code=response.status_code,
            )

        ## CALCULATING approx USAGE
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"]["content"])
        )

        model_response["created"] = time.time()
        model_response["model"] = model
        model_response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return model_response


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
