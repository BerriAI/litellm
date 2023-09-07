import os
import json
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse

class ReplicateError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

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
        "max_new_tokens": 500,
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
        time.sleep(0.0001)
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
        time.sleep(0.0001)
        response = requests.get(prediction_url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            if "output" in response_data:
                output_string = "".join(response_data['output'])
                new_output = output_string[len(previous_output):]
                yield new_output
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
    # Convert messages to prompt
    prompt = ""
    for message in messages:
        prompt += message["content"]

    # Start a prediction and get the prediction URL
    version_id = model_to_version_id(model)
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
