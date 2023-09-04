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
    litellm_params=None,
    logger_fn=None,
):
    import sys
    if 'boto3' not in sys.modules:
        import boto3

    region_name = (
        get_secret("AWS_REGION_NAME") or
        "us-west-2" # default to us-west-2
    )

    client = boto3.client(
        service_name="bedrock",
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


    data = json.dumps({
        "inputText": prompt, 
        "textGenerationConfig":{
            "maxTokenCount":4096,
            "stopSequences":[],
            "temperature":0,
            "topP":0.9
            }
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

    response = client.invoke_model(
        body=data, 
        modelId=model, 
        accept=accept, 
        contentType=contentType
    )
    response_body = json.loads(response.get('body').read())
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
        outputText = response_body.get('results')[0].get('outputText')
        print(outputText)
        if "error" in outputText:
            raise BedrockError(
                message=outputText["error"],
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
