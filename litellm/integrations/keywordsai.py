from typing import Optional
import os
import json
import datetime
import requests
from litellm.utils import ModelResponse

class KeywordsAILogger:
    def __init__(self):
        self.api_key = os.getenv("KEYWORDSAI_API_KEY")
        self.api_base = "https://api.keywordsai.co/api/request-logs/create/"

    def log_success(self, model, messages, response_obj, start_time, end_time, print_verbose, kwargs):
        try:
            print_verbose(f"Keywords AI Logging - Enters logging function for model {model}")
            
            # Get metadata
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or {}
            keywordsai_params = metadata.get("keywordsai_params", {})

            # Convert response object if needed
            if isinstance(response_obj, ModelResponse):
                response_obj = response_obj.json()

            # Prepare payload
            payload = {
                "model": model,
                "prompt_messages": messages,
                "tool_choice": kwargs.get("tool_choice"),
                "tools": kwargs.get("tools"),
                "completion_message": response_obj["choices"][0]["message"],
                "tool_calls": response_obj["choices"][0].get("tool_calls"),
                "latency": (end_time - start_time).total_seconds(),
                "status_code": 200,
                "stream": kwargs.get("stream", False),
                **keywordsai_params
            }

            # Make request to Keywords AI
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                url=self.api_base,
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                print_verbose("Keywords AI Logging - Success!")

            else:
                print_verbose(f"Keywords AI Logging - Error: {response.status_code}, {response.text}")


        except Exception as e:
            print_verbose(f"Keywords AI Logging Error: {str(e)}")
            pass
