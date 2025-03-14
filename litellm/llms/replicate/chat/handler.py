import asyncio
import json
import time
from typing import Callable, List, Union

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import CustomStreamWrapper, ModelResponse

from ..common_utils import ReplicateError
from .transformation import ReplicateConfig

replicate_config = ReplicateConfig()


# Function to handle prediction response (streaming)
def handle_prediction_response_streaming(
    prediction_url, api_token, print_verbose, headers: dict, http_client: HTTPHandler
):
    previous_output = ""
    output_string = ""

    status = ""
    while True and (status not in ["succeeded", "failed", "canceled"]):
        time.sleep(0.5)  # prevent being rate limited by replicate
        print_verbose(f"replicate: polling endpoint: {prediction_url}")
        response = http_client.get(prediction_url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            status = response_data["status"]
            if "output" in response_data:
                try:
                    output_string = "".join(response_data["output"])
                except Exception:
                    raise ReplicateError(
                        status_code=422,
                        message="Unable to parse response. Got={}".format(
                            response_data["output"]
                        ),
                        headers=response.headers,
                    )
                new_output = output_string[len(previous_output) :]
                print_verbose(f"New chunk: {new_output}")
                yield {"output": new_output, "status": status}
                previous_output = output_string
            status = response_data["status"]
            if status == "failed":
                replicate_error = response_data.get("error", "")
                raise ReplicateError(
                    status_code=400,
                    message=f"Error: {replicate_error}",
                    headers=response.headers,
                )
        else:
            # this can fail temporarily but it does not mean the replicate request failed, replicate request fails when status=="failed"
            print_verbose(
                f"Replicate: Failed to fetch prediction status and output.{response.status_code}{response.text}"
            )


# Function to handle prediction response (streaming)
async def async_handle_prediction_response_streaming(
    prediction_url,
    api_token,
    print_verbose,
    headers: dict,
    http_client: AsyncHTTPHandler,
):
    previous_output = ""
    output_string = ""

    status = ""
    while True and (status not in ["succeeded", "failed", "canceled"]):
        await asyncio.sleep(0.5)  # prevent being rate limited by replicate
        print_verbose(f"replicate: polling endpoint: {prediction_url}")
        response = await http_client.get(prediction_url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            status = response_data["status"]
            if "output" in response_data:
                try:
                    output_string = "".join(response_data["output"])
                except Exception:
                    raise ReplicateError(
                        status_code=422,
                        message="Unable to parse response. Got={}".format(
                            response_data["output"]
                        ),
                        headers=response.headers,
                    )
                new_output = output_string[len(previous_output) :]
                print_verbose(f"New chunk: {new_output}")
                yield {"output": new_output, "status": status}
                previous_output = output_string
            status = response_data["status"]
            if status == "failed":
                replicate_error = response_data.get("error", "")
                raise ReplicateError(
                    status_code=400,
                    message=f"Error: {replicate_error}",
                    headers=response.headers,
                )
        else:
            # this can fail temporarily but it does not mean the replicate request failed, replicate request fails when status=="failed"
            print_verbose(
                f"Replicate: Failed to fetch prediction status and output.{response.status_code}{response.text}"
            )


# Main function for prediction completion
def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    optional_params: dict,
    litellm_params: dict,
    logging_obj,
    api_key,
    encoding,
    custom_prompt_dict={},
    logger_fn=None,
    acompletion=None,
    headers={},
) -> Union[ModelResponse, CustomStreamWrapper]:
    headers = replicate_config.validate_environment(
        api_key=api_key,
        headers=headers,
        model=model,
        messages=messages,
        optional_params=optional_params,
    )
    # Start a prediction and get the prediction URL
    version_id = replicate_config.model_to_version_id(model)
    input_data = replicate_config.transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    if acompletion is not None and acompletion is True:
        return async_completion(
            model_response=model_response,
            model=model,
            encoding=encoding,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            version_id=version_id,
            input_data=input_data,
            api_key=api_key,
            api_base=api_base,
            logging_obj=logging_obj,
            print_verbose=print_verbose,
            headers=headers,
        )  # type: ignore
    ## COMPLETION CALL
    model_response.created = int(
        time.time()
    )  # for pricing this must remain right before calling api

    prediction_url = replicate_config.get_complete_url(
        api_base=api_base,
        model=model,
        optional_params=optional_params,
        litellm_params=litellm_params,
    )

    ## COMPLETION CALL
    httpx_client = _get_httpx_client(
        params={"timeout": 600.0},
    )
    response = httpx_client.post(
        url=prediction_url,
        headers=headers,
        data=json.dumps(input_data),
    )

    prediction_url = replicate_config.get_prediction_url(response)

    # Handle the prediction response (streaming or non-streaming)
    if "stream" in optional_params and optional_params["stream"] is True:
        print_verbose("streaming request")
        _response = handle_prediction_response_streaming(
            prediction_url,
            api_key,
            print_verbose,
            headers=headers,
            http_client=httpx_client,
        )
        return CustomStreamWrapper(_response, model, logging_obj=logging_obj, custom_llm_provider="replicate")  # type: ignore
    else:
        for retry in range(litellm.DEFAULT_REPLICATE_POLLING_RETRIES):
            time.sleep(
                litellm.DEFAULT_REPLICATE_POLLING_DELAY_SECONDS + 2 * retry
            )  # wait to allow response to be generated by replicate - else partial output is generated with status=="processing"
            response = httpx_client.get(url=prediction_url, headers=headers)
            if (
                response.status_code == 200
                and response.json().get("status") == "processing"
            ):
                continue
            return litellm.ReplicateConfig().transform_response(
                model=model,
                raw_response=response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                request_data=input_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
            )

    raise ReplicateError(
        status_code=500,
        message="No response received from Replicate API after max retries",
        headers=None,
    )


async def async_completion(
    model_response: ModelResponse,
    model: str,
    messages: List[AllMessageValues],
    encoding,
    optional_params: dict,
    litellm_params: dict,
    version_id,
    input_data,
    api_key,
    api_base,
    logging_obj,
    print_verbose,
    headers: dict,
) -> Union[ModelResponse, CustomStreamWrapper]:

    prediction_url = replicate_config.get_complete_url(
        api_base=api_base,
        model=model,
        optional_params=optional_params,
        litellm_params=litellm_params,
    )
    async_handler = get_async_httpx_client(
        llm_provider=litellm.LlmProviders.REPLICATE,
        params={"timeout": 600.0},
    )
    response = await async_handler.post(
        url=prediction_url, headers=headers, data=json.dumps(input_data)
    )
    prediction_url = replicate_config.get_prediction_url(response)

    if "stream" in optional_params and optional_params["stream"] is True:
        _response = async_handle_prediction_response_streaming(
            prediction_url,
            api_key,
            print_verbose,
            headers=headers,
            http_client=async_handler,
        )
        return CustomStreamWrapper(_response, model, logging_obj=logging_obj, custom_llm_provider="replicate")  # type: ignore

    for retry in range(litellm.DEFAULT_REPLICATE_POLLING_RETRIES):
        await asyncio.sleep(
            litellm.DEFAULT_REPLICATE_POLLING_DELAY_SECONDS + 2 * retry
        )  # wait to allow response to be generated by replicate - else partial output is generated with status=="processing"
        response = await async_handler.get(url=prediction_url, headers=headers)
        if (
            response.status_code == 200
            and response.json().get("status") == "processing"
        ):
            continue
        return litellm.ReplicateConfig().transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=input_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )
    # Add a fallback return if no response is received after max retries
    raise ReplicateError(
        status_code=500,
        message="No response received from Replicate API after max retries",
        headers=None,
    )
