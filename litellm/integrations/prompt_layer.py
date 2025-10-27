#### What this does ####
#    On success, logs events to Promptlayer
import os
import traceback

from pydantic import BaseModel

import litellm


class PromptLayerLogger:
    # Class variables or attributes
    def __init__(self):
        # Instance variables
        self.key = os.getenv("PROMPTLAYER_API_KEY")

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        # Method definition
        try:
            new_kwargs = {}
            new_kwargs["model"] = kwargs["model"]
            new_kwargs["messages"] = kwargs["messages"]

            # add kwargs["optional_params"] to new_kwargs
            for optional_param in kwargs["optional_params"]:
                new_kwargs[optional_param] = kwargs["optional_params"][optional_param]

            # Extract PromptLayer tags from metadata, if such exists
            tags = []
            metadata = {}
            if "metadata" in kwargs["litellm_params"]:
                if "pl_tags" in kwargs["litellm_params"]["metadata"]:
                    tags = kwargs["litellm_params"]["metadata"]["pl_tags"]

                # Remove "pl_tags" from metadata
                metadata = {
                    k: v
                    for k, v in kwargs["litellm_params"]["metadata"].items()
                    if k != "pl_tags"
                }

            print_verbose(
                f"Prompt Layer Logging - Enters logging function for model kwargs: {new_kwargs}\n, response: {response_obj}"
            )

            # python-openai >= 1.0.0 returns Pydantic objects instead of jsons
            if isinstance(response_obj, BaseModel):
                response_obj = response_obj.model_dump()

            request_response = litellm.module_level_client.post(
                "https://api.promptlayer.com/rest/track-request",
                json={
                    "function_name": "openai.ChatCompletion.create",
                    "kwargs": new_kwargs,
                    "tags": tags,
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

            response_json = request_response.json()
            if not request_response.json().get("success", False):
                raise Exception("Promptlayer did not successfully log the response!")

            print_verbose(
                f"Prompt Layer Logging: success - final response object: {request_response.text}"
            )

            if "request_id" in response_json:
                if metadata:
                    response = litellm.module_level_client.post(
                        "https://api.promptlayer.com/rest/track-metadata",
                        json={
                            "request_id": response_json["request_id"],
                            "api_key": self.key,
                            "metadata": metadata,
                        },
                    )
                    print_verbose(
                        f"Prompt Layer Logging: success - metadata post response object: {response.text}"
                    )

        except Exception:
            print_verbose(f"error: Prompt Layer Error - {traceback.format_exc()}")
            pass
