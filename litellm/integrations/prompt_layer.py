#### What this does ####
#    On success, logs events to Helicone
import dotenv, os
import requests
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback

class PromptLayer:
    # Class variables or attributes
    def __init__(self):
        # Instance variables
        self.key = os.getenv("PROMPTLAYER_API_KEY")

    def log_event(self, model, response_obj, start_time, end_time, print_verbose):
        # Method definition
        try:
            print_verbose(
                f"Prompt Layer Logging - Enters logging function for model {model}"
            )

            request_response = requests.post(
            "https://api.promptlayer.com/rest/track-request",
                json={
                    "function_name": "openai.Completion.create",
                    "kwargs": {"engine": "text-ada-001", "prompt": "My name is"},
                    "tags": ["hello", "world"],
                    "request_response": response_obj,
                    "request_start_time": start_time,
                    "request_end_time": end_time,
                    "prompt_id": "<PROMPT ID>",
                    "prompt_input_variables": "<Dictionary of variables for prompt>",
                    "prompt_version":1,
                    "api_key": self.key
                },
            )

            print_verbose(f"Prompt Layer Logging - final response object: {request_response}")
        except:
            # traceback.print_exc()
            print_verbose(f"Prompt Layer Error - {traceback.format_exc()}")
            pass
