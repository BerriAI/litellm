import os
import json
import requests
import time
from typing import Callable
from litellm.utils import ModelResponse
import tiktoken

# Function to start a prediction and get the prediction URL
def start_prediction(version_id, input_data, api_token):
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

    response = requests.post(f"{base_url}/predictions", json=initial_prediction_data, headers=headers)
    if response.status_code == 201:
        response_data = response.json()
        return response_data.get("urls", {}).get("get")
    else:
        raise ValueError(response.status_code, "Failed to start prediction.")

# Function to handle prediction response (non-streaming)
def handle_prediction_response(prediction_url, api_token, print_verbose):
    output_string = ""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    status = ""
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
        else:
            print_verbose("Failed to fetch prediction status and output.")
    return output_string

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
    encoding=tiktoken.get_encoding("cl100k_base"),
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
        "max_new_tokens": 50,
    }

    prediction_url = start_prediction(version_id, input_data, api_key)
    print_verbose(prediction_url)

    # Handle the prediction response (streaming or non-streaming)
    if "stream" in optional_params and optional_params["stream"] == True:
        return handle_prediction_response_streaming(prediction_url, api_key, print_verbose)
    else:
        result = handle_prediction_response(prediction_url, api_key, print_verbose)
        model_response["choices"][0]["message"]["content"] = result

        # Calculate usage
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(encoding.encode(model_response["choices"][0]["message"]["content"]))
        model_response["created"] = time.time()
        model_response["model"] = model
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
