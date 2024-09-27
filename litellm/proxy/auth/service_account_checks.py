"""
Checks for LiteLLM service account keys

"""

from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth


def check_if_token_is_service_account(valid_token: UserAPIKeyAuth) -> bool:
    """
    Checks if the token is a service account

    Returns:
        bool: True if token is a service account

    """
    if valid_token.metadata:
        if "service_account_id" in valid_token.metadata:
            return True
    return False


async def service_account_checks(
    valid_token: UserAPIKeyAuth, request_data: dict
) -> bool:
    """
    If a virtual key is a service account, checks it's a valid service account

    A token is a service account if it has a service_account_id in its metadata

    Service Account Specific Checks:
        - Check if required_params is set
    """

    if check_if_token_is_service_account(valid_token) is not True:
        return True

    from litellm.proxy.proxy_server import general_settings

    if "service_account_settings" in general_settings:
        service_account_settings = general_settings["service_account_settings"]
        if "enforced_params" in service_account_settings:
            _enforced_params = service_account_settings["enforced_params"]
            for param in _enforced_params:
                if param not in request_data:
                    raise ProxyException(
                        type=ProxyErrorTypes.bad_request_error.value,
                        code=400,
                        param=param,
                        message=f"BadRequest please pass param={param} in request body. This is a required param for service account",
                    )

    return True
