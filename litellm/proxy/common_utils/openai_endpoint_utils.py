"""
Contains utils used by OpenAI compatible endpoints 
"""

from typing import Optional

from fastapi import Request

from litellm.proxy.common_utils.http_parsing_utils import _read_request_body


def remove_sensitive_info_from_deployment(deployment_dict: dict) -> dict:
    """
    Removes sensitive information from a deployment dictionary.

    Args:
        deployment_dict (dict): The deployment dictionary to remove sensitive information from.

    Returns:
        dict: The modified deployment dictionary with sensitive information removed.
    """
    deployment_dict["litellm_params"].pop("api_key", None)
    deployment_dict["litellm_params"].pop("client_secret", None)
    deployment_dict["litellm_params"].pop("vertex_credentials", None)
    deployment_dict["litellm_params"].pop("aws_access_key_id", None)
    deployment_dict["litellm_params"].pop("aws_secret_access_key", None)

    return deployment_dict


async def get_custom_llm_provider_from_request_body(request: Request) -> Optional[str]:
    """
    Get the `custom_llm_provider` from the request body

    Safely reads the request body
    """
    request_body: dict = await _read_request_body(request=request) or {}
    if "custom_llm_provider" in request_body:
        return request_body["custom_llm_provider"]
    return None
