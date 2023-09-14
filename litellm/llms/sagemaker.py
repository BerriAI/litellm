import os
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse, get_secret
import sys

class SagemakerError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

"""
SAGEMAKER AUTH Keys/Vars
os.environ['AWS_ACCESS_KEY_ID'] = ""
os.environ['AWS_SECRET_ACCESS_KEY'] = ""
"""

# set os.environ['AWS_REGION_NAME'] = <your-region_name>

def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    import boto3

    region_name = (
        get_secret("AWS_REGION_NAME") or
        "us-west-2" # default to us-west-2
    )

    client = boto3.client(
        "sagemaker-runtime", 
        region_name=region_name
    )

    model = model
    prompt = ""
    for message in messages:
        if "role" in message:
            if message["role"] == "user":
                prompt += (
                    f"{message['content']}"
                )
            else:
                prompt += (
                    f"{message['content']}"
                )
        else:
            prompt += f"{message['content']}"
    data = {
        "inputs": prompt,
        "parameters": optional_params
    }

    ## LOGGING
    logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={"complete_input_dict": data},
        )
    ## COMPLETION CALL
    response = client.invoke_endpoint(
        EndpointName=model,
        ContentType="application/json",
        Body=json.dumps(data),
        CustomAttributes="accept_eula=true",
    )
    response = response["Body"].read().decode("utf8")
    if "stream" in optional_params and optional_params["stream"] == True:
        return response.iter_lines()
    else:
        ## LOGGING
        logging_obj.post_call(
                input=prompt,
                api_key="",
                original_response=response,
                additional_args={"complete_input_dict": data},
            )
        print_verbose(f"raw model_response: {response}")
        ## RESPONSE OBJECT
        completion_response = json.loads(response)
        if "error" in completion_response:
            raise SagemakerError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            try:
                model_response["choices"][0]["message"]["content"] = completion_response[0]["generation"]
            except:
                raise SagemakerError(message=json.dumps(completion_response), status_code=response.status_code)

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here. 
        prompt_tokens = len(
            encoding.encode(prompt)
        ) 
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
