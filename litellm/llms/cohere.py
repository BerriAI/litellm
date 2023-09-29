import os
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse

class CohereError(Exception):
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
        headers["Authorization"] = f"Bearer {api_key}"
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
    completion_url = "https://api.cohere.ai/v1/generate"
    model = model
    prompt = " ".join(message["content"] for message in messages)
    data = {
        "model": model,
        "prompt": prompt,
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
        )
    ## COMPLETION CALL
    response = requests.post(
        completion_url, headers=headers, data=json.dumps(data), stream=optional_params["stream"] if "stream" in optional_params else False
    )
    if "stream" in optional_params and optional_params["stream"] == True:
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
            raise CohereError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            try:
                model_response["choices"][0]["message"]["content"] = completion_response["generations"][0]["text"]
            except:
                raise CohereError(message=json.dumps(completion_response), status_code=response.status_code)

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

def embedding(
    model: str,
    input: list,
    api_key: str,
    logging_obj=None,
    model_response=None,
    encoding=None,
):
    headers = validate_environment(api_key)
    embed_url = "https://api.cohere.ai/v1/embed"
    model = model
    data = {
        "model": model,
        "texts": input,
    }

    ## LOGGING
    logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
        )
    ## COMPLETION CALL
    response = requests.post(
        embed_url, headers=headers, data=json.dumps(data)
    )

    ## LOGGING
    logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response,
        )
    # print(response.json())
    """
        response 
        {
            'object': "list",
            'data': [
            
            ]
            'model', 
            'usage'
        }
    """
    embeddings = response.json()['embeddings']
    output_data = []
    for idx, embedding in enumerate(embeddings):
        output_data.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": embedding
            }
        )
    model_response["object"] = "list"
    model_response["data"] = output_data
    model_response["model"] = model
    input_tokens = 0
    for text in input:
        input_tokens+=len(encoding.encode(text)) 

    model_response["usage"] = { 
        "prompt_tokens": input_tokens, 
        "total_tokens": input_tokens,
    }
    return model_response


    