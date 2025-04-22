import json
import traceback
from datetime import datetime, timezone

import litellm


class GreenscaleLogger:
    def __init__(self):
        import os

        self.greenscale_api_key = os.getenv("GREENSCALE_API_KEY")
        self.headers = {
            "api-key": self.greenscale_api_key,
            "Content-Type": "application/json",
        }
        self.greenscale_logging_url = os.getenv("GREENSCALE_ENDPOINT")

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        try:
            response_json = response_obj.model_dump() if response_obj else {}
            data = {
                "modelId": kwargs.get("model"),
                "inputTokenCount": response_json.get("usage", {}).get("prompt_tokens"),
                "outputTokenCount": response_json.get("usage", {}).get(
                    "completion_tokens"
                ),
            }
            data["timestamp"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

            if type(end_time) is datetime and type(start_time) is datetime:
                data["invocationLatency"] = int(
                    (end_time - start_time).total_seconds() * 1000
                )

            # Add additional metadata keys to tags
            tags = []
            metadata = kwargs.get("litellm_params", {}).get("metadata", {})
            for key, value in metadata.items():
                if key.startswith("greenscale"):
                    if key == "greenscale_project":
                        data["project"] = value
                    elif key == "greenscale_application":
                        data["application"] = value
                    else:
                        tags.append(
                            {"key": key.replace("greenscale_", ""), "value": str(value)}
                        )

            data["tags"] = tags

            if self.greenscale_logging_url is None:
                raise Exception("Greenscale Logger Error - No logging URL found")

            response = litellm.module_level_client.post(
                self.greenscale_logging_url,
                headers=self.headers,
                data=json.dumps(data, default=str),
            )
            if response.status_code != 200:
                print_verbose(
                    f"Greenscale Logger Error - {response.text}, {response.status_code}"
                )
            else:
                print_verbose(f"Greenscale Logger Succeeded - {response.text}")
        except Exception as e:
            print_verbose(
                f"Greenscale Logger Error - {e}, Stack trace: {traceback.format_exc()}"
            )
            pass
