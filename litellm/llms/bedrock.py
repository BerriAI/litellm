import os
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse, get_secret
import sys

class BedrockError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

def init_bedrock_client(region_name):
    import sys
    import boto3
    import subprocess
    try:
        client = boto3.client(
            service_name="bedrock",
            region_name=region_name,
            endpoint_url=f'https://bedrock.{region_name}.amazonaws.com'
        )
    except Exception as e:
        try:
            command1 = "python3 -m pip install https://github.com/BerriAI/litellm/raw/main/cookbook/bedrock_resources/boto3-1.28.21-py3-none-any.whl"
            subprocess.run(command1, shell=True, check=True)
            # Command 2: Install boto3 from URL
            command2 = "python3 -m pip install https://github.com/BerriAI/litellm/raw/main/cookbook/bedrock_resources/botocore-1.31.21-py3-none-any.whl"
            subprocess.run(command2, shell=True, check=True)

            import boto3
            client = boto3.client(
                service_name="bedrock",
                region_name=region_name,
                endpoint_url=f'https://bedrock.{region_name}.amazonaws.com'
            )
        except Exception as e:
            raise e
    return client

"""
BEDROCK AUTH Keys/Vars
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
    stream=False,
    litellm_params=None,
    logger_fn=None,
):

    region_name = (
        get_secret("AWS_REGION_NAME") or
        "us-west-2" # default to us-west-2
    )

    client = init_bedrock_client(region_name)

    model = model
    provider = model.split(".")[0]
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
    
    if provider == "ai21":
        data = json.dumps({
            "prompt": prompt,
        }) 

    else: # amazon titan
        data = json.dumps({
            "inputText": prompt, 
            "textGenerationConfig": optional_params,
            }) 
    ## LOGGING
    logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={"complete_input_dict": data},
        )
    
    ## COMPLETION CALL
    accept = 'application/json'
    contentType = 'application/json'
    if stream == True:
        response = client.invoke_model_with_response_stream(
            body=data, 
            modelId=model, 
            accept=accept, 
            contentType=contentType
        )
        response = response.get('body')
        return response

    response = client.invoke_model(
        body=data, 
        modelId=model, 
        accept=accept, 
        contentType=contentType
    )
    response_body = json.loads(response.get('body').read())

    ## LOGGING
    logging_obj.post_call(
            input=prompt,
            api_key="",
            original_response=response,
            additional_args={"complete_input_dict": data},
        )
    print_verbose(f"raw model_response: {response}")
    ## RESPONSE OBJECT
    outputText = "default"
    if provider == "ai21":
        outputText = response_body.get('completions')[0].get('data').get('text')
    else: # amazon titan
        outputText = response_body.get('results')[0].get('outputText')
    if "error" in outputText:
        raise BedrockError(
            message=outputText,
            status_code=response.status_code,
        )
    else:
        try:
            model_response["choices"][0]["message"]["content"] = outputText
        except:
            raise BedrockError(message=json.dumps(outputText), status_code=response.status_code)

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
