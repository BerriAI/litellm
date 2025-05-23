#### What this does ####
#    On success, logs events to Promptlayer
import json
import os
import traceback

from pydantic import BaseModel

import litellm


class PromptLayerLogger:
    # Class variables or attributes
    def __init__(self):
        # Instance variables
        self.key = os.getenv("PROMPTLAYER_API_KEY")
        self._streaming_content = []

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        try:
            # Convert pydantic object to dict if necessary
            if isinstance(response_obj, BaseModel):
                response_obj = response_obj.model_dump()

            # Handle metadata and tags
            tags = []
            metadata = {}
            total_cost = 0

            if kwargs.get("litellm_params"):
                metadata_dict = kwargs["litellm_params"].get("metadata", {})
                if isinstance(metadata_dict, dict):
                    if "pl_tags" in metadata_dict:
                        tags = metadata_dict["pl_tags"]
                    metadata = {
                        k: v for k, v in metadata_dict.items() if k != "pl_tags"
                    }
                    # Get cost from hidden_params if it exists
                    if "hidden_params" in metadata:
                        total_cost = metadata["hidden_params"].get("response_cost", 0)
                        metadata["hidden_params"] = json.dumps(
                            metadata["hidden_params"]
                        )

            # Handle streaming vs non-streaming responses
            if kwargs.get("stream", False):
                for choice in response_obj.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        self._streaming_content.append(content)

                is_final_chunk = (
                    response_obj.get("choices")
                    and response_obj.get("choices")[0].get("finish_reason") == "stop"
                )

                if not is_final_chunk:
                    return None

                full_content = "".join(self._streaming_content)
                self._streaming_content = []  # Reset for next stream
                output_messages = [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": full_content}],
                    }
                ]
            else:
                output_content = (
                    response_obj.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                output_messages = [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": output_content}],
                    }
                ]

            # Format input messages
            input_messages = [
                {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                }
                for msg in kwargs["messages"]
            ]

            # Construct request payload
            payload = {
                "provider": "openai",
                "model": kwargs["model"],
                "input": {"type": "chat", "messages": input_messages},
                "output": {"type": "chat", "messages": output_messages},
                "request_start_time": start_time.timestamp(),
                "request_end_time": end_time.timestamp(),
                "parameters": kwargs.get("optional_params", {}),
                "prompt_name": kwargs.get("prompt_name", ""),
                "prompt_version_number": kwargs.get("prompt_version_number", 1),
                "prompt_input_variables": kwargs.get("prompt_input_variables", {}),
                "input_tokens": response_obj.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": response_obj.get("usage", {}).get(
                    "completion_tokens", 0
                ),
                "function_name": "openai.chat.completions.create",
                "tags": tags,
                "metadata": metadata,
                "price": total_cost,
                "score": 0,
            }

            print_verbose(
                f"Prompt Layer Logging - Sending payload: {json.dumps(payload, indent=2)}"
            )

            request_response = litellm.module_level_client.post(
                "https://api.promptlayer.com/log-request",
                json=payload,
                headers={"X-API-KEY": self.key, "Content-Type": "application/json"},
            )

            request_response.raise_for_status()
            response_json = request_response.json()

            if "id" in response_json:
                print_verbose(
                    f"Prompt Layer Logging: success - request ID: {response_json['id']}"
                )
                return response_json

            print_verbose(
                f"PromptLayer API response missing 'id' field: {response_json}"
            )
            return None

        except Exception as e:
            print_verbose(f"error: Prompt Layer Error - {traceback.format_exc()}")
            return None
