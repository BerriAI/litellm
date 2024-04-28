import os, types
import json
import random
import requests
import time
from typing import Callable, Optional
from litellm.utils import ModelResponse, Usage
import litellm
import httpx
from litellm.asyncsseclient import asyncsseclient
from .prompt_templates.factory import prompt_factory, custom_prompt

APP_ID = "heurist-llm-gateway"
end_of_stream = "[DONE]"

# TODO: support these
default_priority = 1
default_deadline = 60

def normalize_model_id(model_id):
    if model_id == "mistralai/mixtral-8x7b-instruct-v0.1":
        return "mistralai/mixtral-8x7b-instruct"
    if model_id == "mistralai/mistral-7b-instruct-v0.2":
        return "mistralai/mistral-7b-instruct"
    return model_id

def get_random_job_id():
    # get 10 random letters and numbers
    return APP_ID + "".join([chr(random.randint(97, 122)) for _ in range(10)])

def submit_job(api_base, job_id, model_input, model_id, api_key, temperature, max_tokens, use_stream=True):
    url = api_base + "/submit_job"
    job = {
        "job_id": job_id,
        "model_input": {
            "LLM": {
                "prompt": model_input,
                "use_stream": use_stream,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        },
        "model_type": "LLM",
        "model_id": normalize_model_id(model_id),
        "deadline": default_deadline,
        "priority": default_priority,
    }
    headers = {
        "Authorization" : f"Bearer {api_key}"
    }
    print(f"Submitting job with headers {headers}")
    print(f"Job: {job}")
    response = requests.post(url, json=job, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error submitting job {job_id}: {response.text}")

    if use_stream:
        return api_base + f"/stream/{job_id}"
    else:
        return response.text
    
async def handle_stream(stream_url):
    print("[handle_stream] stream_url: ", stream_url)
    client = asyncsseclient(stream_url)
    async for event in client:
        if end_of_stream in event.data:
            print("[handle_stream] Received EOS from server. Exiting...")
            break
        if event.data:
            yield event.data

def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    logging_obj,
    api_key,
    encoding,
    custom_prompt_dict={},
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    prompt = prompt_factory(model=model, messages=messages)

    user_api_key = litellm_params['metadata'].get('user_api_key', None)

    ## COMPLETION CALL
    model_response["created"] = int(
        time.time()
    )
    temperature = optional_params.get("temperature", 0.75)
    if ("mistral" in model or "mixtral" in model) and temperature < 0.01:
        temperature = 0.01
    max_tokens = optional_params.get("max_tokens", 500)
    if "stream" in optional_params and optional_params["stream"] == True:
        return handle_stream(submit_job(api_base, get_random_job_id(), prompt, model, user_api_key, temperature, max_tokens, use_stream=True))
    else:
        result = submit_job(api_base, get_random_job_id(), prompt, model, user_api_key, temperature, max_tokens, use_stream=False)
        model_response["ended"] = int(
            time.time()
        )

        print_verbose(f"raw model_response: {result}")

        if len(result) == 0:  # edge case, where result is empty
            result = " "

        ## Building RESPONSE OBJECT
        if len(result) > 1:
            model_response["choices"][0]["message"]["content"] = result

        model_response["model"] = "heurist/" + model
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        model_response.usage = usage
        return model_response


# # Example usage:
# response = completion(
#     api_key="",
#     messages=[{"content": "good morning"}],
#     model="",
#     model_response=ModelResponse(),
#     print_verbose=print,
#     logging_obj=print, # stub logging_obj
#     optional_params={"stream": False}
# )

# print(response)
