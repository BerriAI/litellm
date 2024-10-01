#### What this does ####
#    On success, logs events to Helicone
import os
import traceback

import dotenv
import requests  # type: ignore

import litellm
from litellm._logging import verbose_logger


class HeliconeLogger:
    # Class variables or attributes
    helicone_model_list = [
        "gpt",
        "claude",
        "command-r",
        "command-r-plus",
        "command-light",
        "command-medium",
        "command-medium-beta",
        "command-xlarge-nightly",
        "command-nightly",
    ]

    def __init__(self):
        # Instance variables
        self.provider_url = "https://api.openai.com/v1"
        self.key = os.getenv("HELICONE_API_KEY")

    def claude_mapping(self, model, messages, response_obj):
        from anthropic import AI_PROMPT, HUMAN_PROMPT

        prompt = f"{HUMAN_PROMPT}"
        for message in messages:
            if "role" in message:
                if message["role"] == "user":
                    prompt += f"{HUMAN_PROMPT}{message['content']}"
                else:
                    prompt += f"{AI_PROMPT}{message['content']}"
            else:
                prompt += f"{HUMAN_PROMPT}{message['content']}"
        prompt += f"{AI_PROMPT}"

        choice = response_obj["choices"][0]
        message = choice["message"]

        content = []
        if "tool_calls" in message and message["tool_calls"]:
            for tool_call in message["tool_calls"]:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": tool_call["function"]["arguments"],
                    }
                )
        elif "content" in message and message["content"]:
            content = [{"type": "text", "text": message["content"]}]

        claude_response_obj = {
            "id": response_obj["id"],
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": choice["finish_reason"],
            "stop_sequence": None,
            "usage": {
                "input_tokens": response_obj["usage"]["prompt_tokens"],
                "output_tokens": response_obj["usage"]["completion_tokens"],
            },
        }

        return claude_response_obj

    @staticmethod
    def add_metadata_from_header(litellm_params: dict, metadata: dict) -> dict:
        """
        Adds metadata from proxy request headers to Helicone logging if keys start with "helicone_"
        and overwrites litellm_params.metadata if already included.

        For example if you want to add custom property to your request, send
        `headers: { ..., helicone-property-something: 1234 }` via proxy request.
        """
        if litellm_params is None:
            return metadata

        if litellm_params.get("proxy_server_request") is None:
            return metadata

        if metadata is None:
            metadata = {}

        proxy_headers = (
            litellm_params.get("proxy_server_request", {}).get("headers", {}) or {}
        )

        for header_key in proxy_headers:
            if header_key.startswith("helicone_"):
                metadata[header_key] = proxy_headers.get(header_key)

        return metadata

    def log_success(
        self, model, messages, response_obj, start_time, end_time, print_verbose, kwargs
    ):
        # Method definition
        try:
            print_verbose(
                f"Helicone Logging - Enters logging function for model {model}"
            )
            litellm_params = kwargs.get("litellm_params", {})
            kwargs.get("litellm_call_id", None)
            metadata = litellm_params.get("metadata", {}) or {}
            metadata = self.add_metadata_from_header(litellm_params, metadata)
            model = (
                model
                if any(
                    accepted_model in model
                    for accepted_model in self.helicone_model_list
                )
                else "gpt-3.5-turbo"
            )
            provider_request = {"model": model, "messages": messages}
            if isinstance(response_obj, litellm.EmbeddingResponse) or isinstance(
                response_obj, litellm.ModelResponse
            ):
                response_obj = response_obj.json()

            if "claude" in model:
                response_obj = self.claude_mapping(
                    model=model, messages=messages, response_obj=response_obj
                )

            providerResponse = {
                "json": response_obj,
                "headers": {"openai-version": "2020-10-01"},
                "status": 200,
            }

            # Code to be executed
            provider_url = self.provider_url
            url = "https://api.hconeai.com/oai/v1/log"
            if "claude" in model:
                url = "https://api.hconeai.com/anthropic/v1/log"
                provider_url = "https://api.anthropic.com/v1/messages"
            headers = {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            }
            start_time_seconds = int(start_time.timestamp())
            start_time_milliseconds = int(
                (start_time.timestamp() - start_time_seconds) * 1000
            )
            end_time_seconds = int(end_time.timestamp())
            end_time_milliseconds = int(
                (end_time.timestamp() - end_time_seconds) * 1000
            )
            meta = {"Helicone-Auth": f"Bearer {self.key}"}
            meta.update(metadata)
            data = {
                "providerRequest": {
                    "url": provider_url,
                    "json": provider_request,
                    "meta": meta,
                },
                "providerResponse": providerResponse,
                "timing": {
                    "startTime": {
                        "seconds": start_time_seconds,
                        "milliseconds": start_time_milliseconds,
                    },
                    "endTime": {
                        "seconds": end_time_seconds,
                        "milliseconds": end_time_milliseconds,
                    },
                },  # {"seconds": .., "milliseconds": ..}
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                print_verbose("Helicone Logging - Success!")
            else:
                print_verbose(
                    f"Helicone Logging - Error Request was not successful. Status Code: {response.status_code}"
                )
                print_verbose(f"Helicone Logging - Error {response.text}")
        except Exception:
            print_verbose(f"Helicone Logging Error - {traceback.format_exc()}")
            pass
