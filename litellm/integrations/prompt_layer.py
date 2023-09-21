#### What this does ####
#    On success, logs events to Promptlayer
import dotenv, os
import requests
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback


class PromptLayerLogger:
    # Class variables or attributes
    def __init__(self):
        # Instance variables
        self.key = os.getenv("PROMPTLAYER_API_KEY")

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        # Method definition
        try:
            if 'litellm_logging_obj' in kwargs:
                kwargs.pop('litellm_logging_obj')

            print_verbose(
                f"Prompt Layer Logging - Enters logging function for model kwargs: {kwargs}\n, response: {response_obj}"
            )

            request_response = requests.post(
                "https://api.promptlayer.com/rest/track-request",
                json={
                    "function_name": "openai.ChatCompletion.create",
                    "kwargs": kwargs,
                    "tags": ["hello", "world"],
                    "request_response": dict(response_obj),
                    "request_start_time": int(start_time.timestamp()),
                    "request_end_time": int(end_time.timestamp()),
                    "api_key": self.key,
                    # Optional params for PromptLayer
                    # "prompt_id": "<PROMPT ID>",
                    # "prompt_input_variables": "<Dictionary of variables for prompt>",
                    # "prompt_version":1,
                },
            )
            print_verbose(
                f"Prompt Layer Logging: success - final response object: {request_response.text}"
            )
            response_json = request_response.json()
            if "success" not in request_response.json(): 
                raise Exception("Promptlayer did not successfully log the response!") 

            if "request_id" in response_json:
                print(kwargs["litellm_params"]["metadata"])
                if kwargs["litellm_params"]["metadata"] is not None:
                    response = requests.post(
                        "https://api.promptlayer.com/rest/track-metadata",
                        json={
                            "request_id": response_json["request_id"],
                            "api_key": self.key,
                            "metadata": kwargs["litellm_params"]["metadata"]
                        },
                    )
                    print_verbose(f"Prompt Layer Logging: success - metadata post response object: {response.text}")

        except:
            print_verbose(f"error: Prompt Layer Error - {traceback.format_exc()}")
            pass
