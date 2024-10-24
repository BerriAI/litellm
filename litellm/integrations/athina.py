import datetime


class AthinaLogger:
    def __init__(self):
        import os

        self.athina_api_key = os.getenv("ATHINA_API_KEY")
        self.headers = {
            "athina-api-key": self.athina_api_key,
            "Content-Type": "application/json",
        }
        self.athina_logging_url = "https://log.athina.ai/api/v1/log/inference"
        self.additional_keys = [
            "environment",
            "prompt_slug",
            "customer_id",
            "customer_user_id",
            "session_id",
            "external_reference_id",
            "context",
            "expected_response",
            "user_query",
        ]

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        import json
        import traceback

        import requests  # type: ignore

        try:
            is_stream = kwargs.get("stream", False)
            if is_stream:
                if "complete_streaming_response" in kwargs:
                    # Log the completion response in streaming mode
                    completion_response = kwargs["complete_streaming_response"]
                    response_json = (
                        completion_response.model_dump() if completion_response else {}
                    )
                else:
                    # Skip logging if the completion response is not available
                    return
            else:
                # Log the completion response in non streaming mode
                response_json = response_obj.model_dump() if response_obj else {}
            data = {
                "language_model_id": kwargs.get("model"),
                "request": kwargs,
                "response": response_json,
                "prompt_tokens": response_json.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": response_json.get("usage", {}).get(
                    "completion_tokens"
                ),
                "total_tokens": response_json.get("usage", {}).get("total_tokens"),
            }

            if (
                type(end_time) is datetime.datetime
                and type(start_time) is datetime.datetime
            ):
                data["response_time"] = int(
                    (end_time - start_time).total_seconds() * 1000
                )

            if "messages" in kwargs:
                data["prompt"] = kwargs.get("messages", None)

            # Directly add tools or functions if present
            optional_params = kwargs.get("optional_params", {})
            data.update(
                (k, v)
                for k, v in optional_params.items()
                if k in ["tools", "functions"]
            )

            # Add additional metadata keys
            metadata = kwargs.get("litellm_params", {}).get("metadata", {})
            if metadata:
                for key in self.additional_keys:
                    if key in metadata:
                        data[key] = metadata[key]

            response = requests.post(
                self.athina_logging_url,
                headers=self.headers,
                data=json.dumps(data, default=str),
            )
            if response.status_code != 200:
                print_verbose(
                    f"Athina Logger Error - {response.text}, {response.status_code}"
                )
            else:
                print_verbose(f"Athina Logger Succeeded - {response.text}")
        except Exception as e:
            print_verbose(
                f"Athina Logger Error - {e}, Stack trace: {traceback.format_exc()}"
            )
            pass
