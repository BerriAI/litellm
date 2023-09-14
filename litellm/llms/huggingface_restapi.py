## Uses the huggingface text generation inference API
import os, copy
import json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse
from typing import Optional
from .prompt_templates.factory import prompt_factory, custom_prompt

class HuggingfaceError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

def validate_environment(api_key):
    headers = {
        "content-type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def completion(
    model: str,
    messages: list,
    api_base: Optional[str],
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    custom_prompt_dict={},
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    task = optional_params.pop("task")
    completion_url = ""
    input_text = None
    if "https" in model:
        completion_url = model
    elif api_base:
        completion_url = api_base
    elif "HF_API_BASE" in os.environ:
        completion_url = os.getenv("HF_API_BASE", "")
    else:
        completion_url = f"https://api-inference.huggingface.co/models/{model}"

    ### MAP INPUT PARAMS
    if task == "conversational":
        inference_params = copy.deepcopy(optional_params)
        inference_params.pop("details")
        past_user_inputs = []
        generated_responses = []
        text = ""
        for message in messages:
            if message["role"] == "user":
                if text != "":
                    past_user_inputs.append(text)
                text = message["content"]
            elif message["role"] == "assistant" or message["role"] == "system":
                generated_responses.append(message["content"])
        data = {
            "inputs": {
                "text": text, 
                "past_user_inputs": past_user_inputs, 
                "generated_responses": generated_responses
            },
            "parameters": inference_params
        }
        input_text = "".join(message["content"] for message in messages)
    elif task == "text-generation-inference":
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"], 
                initial_prompt_value=model_prompt_details["initial_prompt_value"],  
                final_prompt_value=model_prompt_details["final_prompt_value"], 
                messages=messages
            )
        else:
            prompt = prompt_factory(model=model, messages=messages)
        if "https://api-inference.huggingface.co/models" in completion_url: 
            inference_params = copy.deepcopy(optional_params)
            inference_params.pop("details")
            data = {
                "inputs": prompt,
                "parameters": inference_params,
                "stream": True if "stream" in inference_params and inference_params["stream"] == True else False,
            }
        else:
            data = {
                "inputs": prompt,
                "parameters": optional_params,
                "stream": True if "stream" in optional_params and optional_params["stream"] == True else False,
            }
        input_text = prompt
    ## LOGGING
    logging_obj.pre_call(
            input=input_text,
            api_key=api_key,
            additional_args={"complete_input_dict": data, "task": task},
        )
    ## COMPLETION CALL
    if "stream" in optional_params and optional_params["stream"] == True:
        response = requests.post(
            completion_url, 
            headers=headers, 
            data=json.dumps(data), 
            stream=optional_params["stream"]
        )
        return response.iter_lines()
    else:
        response = requests.post(
            completion_url, 
            headers=headers, 
            data=json.dumps(data)
        )
        ## LOGGING
        logging_obj.post_call(
            input=input_text,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data, "task": task},
        )
        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except:
            raise HuggingfaceError(
                message=response.text, status_code=response.status_code
            )
        print_verbose(f"response: {completion_response}")
        if isinstance(completion_response, dict) and "error" in completion_response:
            print_verbose(f"completion error: {completion_response['error']}")
            print_verbose(f"response.status_code: {response.status_code}")
            raise HuggingfaceError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            if task == "conversational": 
                model_response["choices"][0]["message"][
                    "content"
                ] = completion_response["generated_text"]
            elif task == "text-generation-inference": 
                model_response["choices"][0]["message"][
                    "content"
                ] = completion_response[0]["generated_text"]   
                ## GETTING LOGPROBS + FINISH REASON 
                if "details" in completion_response[0] and "tokens" in completion_response[0]["details"]:
                    model_response.choices[0].finish_reason = completion_response[0]["details"]["finish_reason"]
                    sum_logprob = 0
                    for token in completion_response[0]["details"]["tokens"]:
                        sum_logprob += token["logprob"]
                    model_response["choices"][0]["message"]["logprobs"] = sum_logprob
        ## CALCULATING USAGE
        prompt_tokens = len(
            encoding.encode(input_text)
        )  ##[TODO] use the llama2 tokenizer here
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"]["content"])
        )  ##[TODO] use the llama2 tokenizer here

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
