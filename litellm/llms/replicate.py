import os, types
import json
import requests
import time
from typing import Callable, Optional
from litellm.utils import ModelResponse
import litellm 

class ReplicateError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class ReplicateConfig(): 
    """
    Reference: https://replicate.com/meta/llama-2-70b-chat/api
    - `prompt` (string): The prompt to send to the model.
        
    - `system_prompt` (string): The system prompt to send to the model. This is prepended to the prompt and helps guide system behavior. Default value: `You are a helpful assistant`.
        
    - `max_new_tokens` (integer): Maximum number of tokens to generate. Typically, a word is made up of 2-3 tokens. Default value: `128`.
        
    - `min_new_tokens` (integer): Minimum number of tokens to generate. To disable, set to `-1`. A word is usually 2-3 tokens. Default value: `-1`.
        
    - `temperature` (number): Adjusts the randomness of outputs. Values greater than 1 increase randomness, 0 is deterministic, and 0.75 is a reasonable starting value. Default value: `0.75`.
        
    - `top_p` (number): During text decoding, it samples from the top `p` percentage of most likely tokens. Reduce this to ignore less probable tokens. Default value: `0.9`.
        
    - `top_k` (integer): During text decoding, samples from the top `k` most likely tokens. Reduce this to ignore less probable tokens. Default value: `50`.
    
    - `stop_sequences` (string): A comma-separated list of sequences to stop generation at. For example, inputting '<end>,<stop>' will cease generation at the first occurrence of either 'end' or '<stop>'.
        
    - `seed` (integer): This is the seed for the random generator. Leave it blank to randomize the seed.
        
    - `debug` (boolean): If set to `True`, it provides debugging output in logs.

    Please note that Replicate's mapping of these parameters can be inconsistent across different models, indicating that not all of these parameters may be available for use with all models.
    """
    system_prompt: Optional[str]=None
    max_new_tokens: Optional[int]=None
    min_new_tokens: Optional[int]=None
    temperature: Optional[int]=None
    top_p: Optional[int]=None
    top_k: Optional[int]=None
    stop_sequences: Optional[str]=None
    seed: Optional[int]=None
    debug: Optional[bool]=None

    def __init__(self,
                 system_prompt: Optional[str]=None,
                 max_new_tokens: Optional[int]=None,
                 min_new_tokens: Optional[int]=None,
                 temperature: Optional[int]=None,
                 top_p: Optional[int]=None,
                 top_k: Optional[int]=None,
                 stop_sequences: Optional[str]=None,
                 seed: Optional[int]=None,
                 debug: Optional[bool]=None) -> None:
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



# Function to start a prediction and get the prediction URL
def start_prediction(version_id, input_data, api_token, logging_obj):
    base_url = "https://api.replicate.com/v1"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    initial_prediction_data = {
        "version": version_id,
        "input": input_data,
    }

        ## LOGGING
    logging_obj.pre_call(
            input=input_data["prompt"],
            api_key="",
            additional_args={"complete_input_dict": initial_prediction_data, "headers": headers},
    )

    response = requests.post(f"{base_url}/predictions", json=initial_prediction_data, headers=headers)
    if response.status_code == 201:
        response_data = response.json()
        return response_data.get("urls", {}).get("get")
    else:
        raise ReplicateError(response.status_code, f"Failed to start prediction {response.text}")

# Function to handle prediction response (non-streaming)
def handle_prediction_response(prediction_url, api_token, print_verbose):
    output_string = ""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    status = ""
    logs = ""
    while True and (status not in ["succeeded", "failed", "canceled"]):
        print_verbose("making request")
        time.sleep(0.5)
        response = requests.get(prediction_url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            if "output" in response_data:
                output_string = "".join(response_data['output'])
                print_verbose(f"Non-streamed output:{output_string}")
            status = response_data['status']
            logs = response_data.get("logs", "")
        else:
            print_verbose("Failed to fetch prediction status and output.")
    return output_string, logs

# Function to handle prediction response (streaming)
def handle_prediction_response_streaming(prediction_url, api_token, print_verbose):
    previous_output = ""
    output_string = ""

    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }
    status = ""
    while True and (status not in ["succeeded", "failed", "canceled"]):
        time.sleep(0.5) # prevent being rate limited by replicate
        response = requests.get(prediction_url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            status = response_data['status']
            if "output" in response_data:
                output_string = "".join(response_data['output'])
                new_output = output_string[len(previous_output):]
                yield {"output": new_output, "status": status}
                previous_output = output_string
            status = response_data['status']

# Function to extract version ID from model string
def model_to_version_id(model):
    if ":" in model:
        split_model = model.split(":")
        return split_model[1]
    return model

# Main function for prediction completion
def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    logging_obj,
    api_key,
    encoding,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    # Start a prediction and get the prediction URL
    version_id = model_to_version_id(model)

    ## Load Config
    config = litellm.ReplicateConfig.get_config() 
    for k, v in config.items(): 
        if k not in optional_params: # completion(top_k=3) > replicate_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    if "meta/llama-2-13b-chat" in model: 
        system_prompt = ""
        prompt = "" 
        for message in messages: 
            if message["role"] == "system":
                system_prompt = message["content"]
            else: 
                prompt += message["content"]
        input_data = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            **optional_params
        }
    else:
        # Convert messages to prompt
        prompt = ""
        for message in messages:
            prompt += message["content"]

        input_data = {
            "prompt": prompt,
            **optional_params
        }


    ## COMPLETION CALL
    ## Replicate Compeltion calls have 2 steps
    ## Step1: Start Prediction: gets a prediction url
    ## Step2: Poll prediction url for response
    ## Step2: is handled with and without streaming
    model_response["created"] = time.time() # for pricing this must remain right before calling api
    prediction_url = start_prediction(version_id, input_data, api_key, logging_obj=logging_obj)
    print_verbose(prediction_url)

    # Handle the prediction response (streaming or non-streaming)
    if "stream" in optional_params and optional_params["stream"] == True:
        print_verbose("streaming request")
        return handle_prediction_response_streaming(prediction_url, api_key, print_verbose)
    else:
        result, logs = handle_prediction_response(prediction_url, api_key, print_verbose)
        model_response["ended"] = time.time() # for pricing this must remain right after calling api
        ## LOGGING
        logging_obj.post_call(
                input=prompt,
                api_key="",
                original_response=result,
                additional_args={"complete_input_dict": input_data,"logs": logs},
        )

        print_verbose(f"raw model_response: {result}")

        if len(result) == 0: # edge case, where result from replicate is empty
            result = " "
        
        ## Building RESPONSE OBJECT
        model_response["choices"][0]["message"]["content"] = result

        # Calculate usage
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(encoding.encode(model_response["choices"][0]["message"]["content"]))
        model_response["model"] = "replicate/" + model
        model_response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return model_response


# # Example usage:
# response = completion(
#     api_key="",
#     messages=[{"content": "good morning"}],
#     model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
#     model_response=ModelResponse(),
#     print_verbose=print,
#     logging_obj=print, # stub logging_obj
#     optional_params={"stream": False}
# )

# print(response)
