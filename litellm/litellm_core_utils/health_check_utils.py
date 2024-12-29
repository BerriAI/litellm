"""
Utils used for litellm.ahealth_check()
"""


def _filter_model_params(model_params: dict) -> dict:
    """Remove 'messages' param from model params."""
    return {k: v for k, v in model_params.items() if k != "messages"}


def _create_health_check_response(response_headers: dict) -> dict:
    response = {}

    if (
        response_headers.get("x-ratelimit-remaining-requests", None) is not None
    ):  # not provided for dall-e requests
        response["x-ratelimit-remaining-requests"] = response_headers[
            "x-ratelimit-remaining-requests"
        ]

    if response_headers.get("x-ratelimit-remaining-tokens", None) is not None:
        response["x-ratelimit-remaining-tokens"] = response_headers[
            "x-ratelimit-remaining-tokens"
        ]

    if response_headers.get("x-ms-region", None) is not None:
        response["x-ms-region"] = response_headers["x-ms-region"]
    return response
