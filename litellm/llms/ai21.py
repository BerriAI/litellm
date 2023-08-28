import os, json
from enum import Enum
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse

class AI21Error(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class AI21LLM:
    def __init__(
        self, encoding, logging_obj, api_key=None
    ):
        self.encoding = encoding
        self.completion_url_fragment_1 = "https://api.ai21.com/studio/v1/"
        self.completion_url_fragment_2 = "/complete"
        self.api_key = api_key
        self.logging_obj = logging_obj
        self.validate_environment(api_key=api_key)

    def validate_environment(
        self, api_key
    ):  # set up the environment required to run the model
        # set the api key
        if self.api_key == None:
            raise ValueError(
                "Missing AI21 API Key - A call is being made to ai21 but no key is set either in the environment variables or via params"
            )
        self.api_key = api_key
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
    ):  # logic for parsing in - calling - parsing out model completion calls
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
            "prompt": prompt,
            # "instruction": prompt, # some baseten models require the prompt to be passed in via the 'instruction' kwarg
            **optional_params,
        }

        ## LOGGING
        self.logging_obj.pre_call(
            input=prompt,
            api_key=self.api_key,
            additional_args={"complete_input_dict": data},
        )
        ## COMPLETION CALL
        response = requests.post(
            self.completion_url_fragment_1 + model + self.completion_url_fragment_2, headers=self.headers, data=json.dumps(data)
        )
        if "stream" in optional_params and optional_params["stream"] == True:
            return response.iter_lines()
        else:
            ## LOGGING
            self.logging_obj.post_call(
                input=prompt,
                api_key=self.api_key,
                original_response=response.text,
                additional_args={"complete_input_dict": data},
            )
            print_verbose(f"raw model_response: {response.text}")
            ## RESPONSE OBJECT
            completion_response = response.json()
            if "error" in completion_response:
                raise AI21Error(
                    message=completion_response["error"],
                    status_code=response.status_code,
                )
            else:
                try:
                    model_response["choices"][0]["message"]["content"] = completion_response["completions"][0]["data"]["text"]
                except:
                    raise ValueError(f"Unable to parse response. Original response: {response.text}")

            ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here. 
            prompt_tokens = len(
                self.encoding.encode(prompt)
            ) 
            completion_tokens = len(
                self.encoding.encode(model_response["choices"][0]["message"]["content"])
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
        self,
    ):  # logic for parsing in - calling - parsing out model embedding calls
        pass
