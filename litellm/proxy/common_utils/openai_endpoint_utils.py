"""
Contains utils used by OpenAI compatible endpoints 
"""

from typing import Optional

from fastapi import Depends, Request

from litellm.proxy.common_utils.http_parsing_utils import read_request_body


def remove_sensitive_info_from_deployment(deployment_dict: dict) -> dict:
    """
    Removes sensitive information from a deployment dictionary.

    Args:
        deployment_dict (dict): The deployment dictionary to remove sensitive information from.

    Returns:
        dict: The modified deployment dictionary with sensitive information removed.
    """
    deployment_dict["litellm_params"].pop("api_key", None)
    deployment_dict["litellm_params"].pop("vertex_credentials", None)
    deployment_dict["litellm_params"].pop("aws_access_key_id", None)
    deployment_dict["litellm_params"].pop("aws_secret_access_key", None)

    return deployment_dict


async def get_custom_llm_provider_from_request_body(
    request: Request,
    data: dict = Depends(read_request_body),
) -> Optional[str]:
    """
    Get the `custom_llm_provider` from the request body

    Safely reads the request body
    """
    if "custom_llm_provider" in data:
        return data["custom_llm_provider"]
    return None
