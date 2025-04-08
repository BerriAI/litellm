import json
from typing import Any, Callable, Optional

import litellm
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm.utils import EmbeddingResponse, ModelResponse, Usage

from ..common_utils import OobaboogaError
from .transformation import OobaboogaConfig

oobabooga_config = OobaboogaConfig()


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
    litellm_params: dict,
    custom_prompt_dict={},
    logger_fn=None,
    default_max_tokens_to_sample=None,
):
    headers = oobabooga_config.validate_environment(
        api_key=api_key,
        headers={},
        model=model,
        messages=messages,
        optional_params=optional_params,
    )
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
    data = oobabooga_config.transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
    )
    ## LOGGING

    logging_obj.pre_call(
        input=messages,
        api_key=api_key,
        additional_args={"complete_input_dict": data},
    )
    ## COMPLETION CALL
    client = _get_httpx_client()
    response = client.post(
        completion_url,
        headers=headers,
        data=json.dumps(data),
        stream=optional_params["stream"] if "stream" in optional_params else False,
    )
    if "stream" in optional_params and optional_params["stream"] is True:
        return response.iter_lines()
    else:
        return oobabooga_config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )


def embedding(
    model: str,
    input: list,
    model_response: EmbeddingResponse,
    api_key: Optional[str],
    api_base: Optional[str],
    logging_obj: Any,
    optional_params: dict,
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
    headers = oobabooga_config.validate_environment(
        api_key=api_key,
        headers={},
        model=model,
        messages=[],
        optional_params=optional_params,
    )
    response = litellm.module_level_client.post(
        embeddings_url, headers=headers, json=data
    )
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
