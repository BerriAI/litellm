"""
Contains utils used by OpenAI compatible endpoints 
"""


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
