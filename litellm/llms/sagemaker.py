import os, types
from enum import Enum
import json
import requests
import time
from typing import Callable, Optional
import litellm
from litellm.utils import ModelResponse, get_secret, Usage
import sys
from copy import deepcopy
import httpx

class SagemakerError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://us-west-2.console.aws.amazon.com/sagemaker")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class SagemakerConfig(): 
    """
    Reference: https://d-uuwbxj1u4cnu.studio.us-west-2.sagemaker.aws/jupyter/default/lab/workspaces/auto-q/tree/DemoNotebooks/meta-textgeneration-llama-2-7b-SDK_1.ipynb
    """
    max_new_tokens: Optional[int]=None
    top_p: Optional[float]=None
    temperature: Optional[float]=None
    return_full_text: Optional[bool]=None

    def __init__(self,
                 max_new_tokens: Optional[int]=None,
                 top_p: Optional[float]=None,
                 temperature: Optional[float]=None,
                 return_full_text: Optional[bool]=None) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
   
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}
        
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

    # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
    aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
    aws_access_key_id = optional_params.pop("aws_access_key_id", None)
    aws_region_name = optional_params.pop("aws_region_name", None)

    if aws_access_key_id != None:
        # uses auth params passed to completion
        # aws_access_key_id is not None, assume user is trying to auth using litellm.completion
        client = boto3.client(
            service_name="sagemaker-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
        )
    else:
        # aws_access_key_id is None, assume user is trying to auth using env variables 
        # boto3 automaticaly reads env variables

        # we need to read region name from env
        # I assume majority of users use .env for auth 
        region_name = (
            get_secret("AWS_REGION_NAME") or
            "us-west-2"  # default to us-west-2 if user not specified
        )
        client = boto3.client(
            service_name="sagemaker-runtime",
            region_name=region_name,
        )
    
    # pop streaming if it's in the optional params as 'stream' raises an error with sagemaker
    inference_params = deepcopy(optional_params)
    inference_params.pop("stream", None)

    ## Load Config
    config = litellm.SagemakerConfig.get_config() 
    for k, v in config.items(): 
        if k not in inference_params: # completion(top_k=3) > sagemaker_config(top_k=3) <- allows for dynamic variables to be passed in
            inference_params[k] = v

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
        "parameters": inference_params
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
            if len(completion_response[0]["generation"]) > 0: 
                model_response["choices"][0]["message"]["content"] = completion_response[0]["generation"]
        except:
            raise SagemakerError(message=json.dumps(completion_response), status_code=response.status_code)

    ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here. 
    prompt_tokens = len(
        encoding.encode(prompt)
    ) 
    completion_tokens = len(
        encoding.encode(model_response["choices"][0]["message"].get("content", ""))
    )

    model_response["created"] = time.time()
    model_response["model"] = model
    usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    model_response.usage = usage
    return model_response

def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
