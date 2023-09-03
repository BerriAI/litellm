## Uses the huggingface text generation inference API
import os, json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse
from typing import Optional


class HuggingfaceError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class HuggingfaceRestAPILLM:
    def __init__(self, encoding, logging_obj, api_key=None) -> None:
        self.encoding = encoding
        self.logging_obj = logging_obj
        self.validate_environment(api_key=api_key)

    def validate_environment(
        self, api_key
    ):  # set up the environment required to run the model
        self.headers = {
            "content-type": "application/json",
        }
        # get the api key if it exists in the environment or is passed in, but don't require it
        self.api_key = api_key
        if self.api_key != None:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
    ):  # logic for parsing in - calling - parsing out model completion calls
        completion_url: str = ""
        if "https" in model:
            completion_url = model
        elif api_base:
            completion_url = api_base
        elif "HF_API_BASE" in os.environ:
            completion_url = os.getenv("HF_API_BASE", "")
        else:
            completion_url = f"https://api-inference.huggingface.co/models/{model}"
        prompt = ""
        if (
            "meta-llama" in model and "chat" in model
        ):  # use the required special tokens for meta-llama - https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            prompt = "<s>"
            for message in messages:
                if message["role"] == "system":
                    prompt += "[INST] <<SYS>>" + message["content"]
                elif message["role"] == "assistant":
                    prompt += message["content"] + "</s><s>[INST]"
                elif message["role"] == "user":
                    prompt += message["content"] + "[/INST]"
        else:
            for message in messages:
                prompt += f"{message['content']}"
        ### MAP INPUT PARAMS
        data = {"inputs": prompt, "parameters": optional_params, "stream": True if "stream" in optional_params and optional_params["stream"] == True else False}
        ## LOGGING
        self.logging_obj.pre_call(
            input=prompt,
            api_key=self.api_key,
            additional_args={"complete_input_dict": data},
        )
        ## COMPLETION CALL
        if "stream" in optional_params and optional_params["stream"] == True:
            response = requests.post(
                completion_url, headers=self.headers, data=json.dumps(data), stream=optional_params["stream"]
            )
            return response.iter_lines()
        else:
            response = requests.post(
                completion_url, headers=self.headers, data=json.dumps(data)
            )
            ## LOGGING
            self.logging_obj.post_call(
                input=prompt,
                api_key=self.api_key,
                original_response=response.text,
                additional_args={"complete_input_dict": data},
            )
            ## RESPONSE OBJECT
            try:
                completion_response = response.json()
            except:
                raise HuggingfaceError(message=response.text, status_code=response.status_code)
            print_verbose(f"response: {completion_response}")
            if isinstance(completion_response, dict) and "error" in completion_response:
                print_verbose(f"completion error: {completion_response['error']}")
                print_verbose(f"response.status_code: {response.status_code}")
                raise HuggingfaceError(
                    message=completion_response["error"],
                    status_code=response.status_code,
                )
            else:
                model_response["choices"][0]["message"][
                    "content"
                ] = completion_response[0]["generated_text"]
            
            ## GETTING LOGPROBS 
            if "details" in completion_response[0] and "tokens" in completion_response[0]["details"]:
                sum_logprob = 0
                for token in completion_response[0]["details"]["tokens"]:
                    sum_logprob += token["logprob"]
                model_response["choices"][0]["message"]["logprobs"] = sum_logprob
            ## CALCULATING USAGE
            prompt_tokens = len(
                self.encoding.encode(prompt)
            )  ##[TODO] use the llama2 tokenizer here
            completion_tokens = len(
                self.encoding.encode(model_response["choices"][0]["message"]["content"])
            )  ##[TODO] use the llama2 tokenizer here

            model_response["created"] = time.time()
            model_response["model"] = model
            model_response["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            return model_response
        pass

    def embedding(
        self,
    ):  # logic for parsing in - calling - parsing out model embedding calls
        pass
