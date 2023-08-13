import os, json
from enum import Enum
import requests
from litellm import logging
import time 
from typing import Callable

class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman:"
    AI_PROMPT = "\n\nAssistant:"

class AnthropicError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message

class AnthropicLLM: 
    
    def __init__(self, encoding, default_max_tokens_to_sample, api_key=None):
        self.encoding = encoding
        self.default_max_tokens_to_sample = default_max_tokens_to_sample
        self.completion_url = "https://api.anthropic.com/v1/complete"
        self.validate_environment(api_key=api_key)
    
    def validate_environment(self, api_key): # set up the environment required to run the model 
        # set the api key 
        try:
            self.api_key = os.getenv("ANTHROPIC_API_KEY") if "ANTHROPIC_API_KEY" in os.environ else api_key
            if self.api_key == None:
                raise Exception
            
            self.headers = {
                "accept": "application/json",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "x-api-key": self.api_key 
            }

        except:
            raise ValueError("Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params")
        pass  

    def completion(self, model: str, messages: list, model_response: dict, print_verbose: Callable, optional_params=None, litellm_params=None, logger_fn=None): # logic for parsing in - calling - parsing out model completion calls
        model = model
        prompt = f"{AnthropicConstants.HUMAN_PROMPT.value}"
        for message in messages:
            if "role" in message:
                if message["role"] == "user":
                    prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
                else:
                    prompt += f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
            else:
                prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
        prompt += f"{AnthropicConstants.AI_PROMPT.value}"
        if "max_tokens" in optional_params and optional_params["max_tokens"] != float('inf'): 
            max_tokens = optional_params["max_tokens"]
        else:
            max_tokens = self.default_max_tokens_to_sample
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens,
            **optional_params
        }

        ## LOGGING
        logging(model=model, input=prompt, additional_args={"litellm_params": litellm_params, "optional_params": optional_params}, logger_fn=logger_fn)
        ## COMPLETION CALL
        response = requests.post(self.completion_url, headers=self.headers, data=json.dumps(data))
        if "stream" in optional_params and optional_params["stream"] == True:
            return response.iter_lines()
        else:
            ## LOGGING
            logging(model=model, input=prompt, additional_args={"litellm_params": litellm_params, "optional_params": optional_params, "original_response": response.text}, logger_fn=logger_fn)
            print_verbose(f"raw model_response: {response.text}")
            ## RESPONSE OBJECT
            completion_response = response.json()
            print(f"completion_response: {completion_response}")
            if "error" in completion_response:
                raise AnthropicError(message=completion_response["error"], status_code=response.status_code)
            else:
                model_response["choices"][0]["message"]["content"] = completion_response["completion"]    
            
            ## CALCULATING USAGE
            prompt_tokens = len(self.encoding.encode(prompt)) ##[TODO] use the anthropic tokenizer here
            completion_tokens = len(self.encoding.encode(model_response["choices"][0]["message"]["content"])) ##[TODO] use the anthropic tokenizer here
            
            
            model_response["created"] = time.time()
            model_response["model"] = model
            model_response["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
                }
            return model_response
    
    def embedding(): # logic for parsing in - calling - parsing out model embedding calls
        pass 