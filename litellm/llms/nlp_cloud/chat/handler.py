import json
from typing import Callable, Optional, Union

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
)
from litellm.utils import ModelResponse

from .transformation import NLPCloudConfig

nlp_config = NLPCloudConfig()


def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params: dict,
    litellm_params: dict,
    logger_fn=None,
    default_max_tokens_to_sample=None,
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    headers={},
):
    headers = nlp_config.validate_environment(
        api_key=api_key,
        headers=headers,
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
    )

    ## Load Config
    config = litellm.NLPCloudConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > togetherai_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    completion_url_fragment_1 = api_base
    completion_url_fragment_2 = "/generation"
    model = model

    completion_url = completion_url_fragment_1 + model + completion_url_fragment_2
    data = nlp_config.transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    ## LOGGING
    logging_obj.pre_call(
        input=None,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": completion_url,
        },
    )
    ## COMPLETION CALL
    if client is None or not isinstance(client, HTTPHandler):
        client = _get_httpx_client()

    response = client.post(
        completion_url,
        headers=headers,
        data=json.dumps(data),
        stream=optional_params["stream"] if "stream" in optional_params else False,
    )
    if "stream" in optional_params and optional_params["stream"] is True:
        return clean_and_iterate_chunks(response)
    else:
        return nlp_config.transform_response(
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


# def clean_and_iterate_chunks(response):
#     def process_chunk(chunk):
#         print(f"received chunk: {chunk}")
#         cleaned_chunk = chunk.decode("utf-8")
#         # Perform further processing based on your needs
#         return cleaned_chunk


#     for line in response.iter_lines():
#         if line:
#             yield process_chunk(line)
def clean_and_iterate_chunks(response):
    buffer = b""

    for chunk in response.iter_content(chunk_size=1024):
        if not chunk:
            break

        buffer += chunk
        while b"\x00" in buffer:
            buffer = buffer.replace(b"\x00", b"")
            yield buffer.decode("utf-8")
            buffer = b""

    # No more data expected, yield any remaining data in the buffer
    if buffer:
        yield buffer.decode("utf-8")


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
