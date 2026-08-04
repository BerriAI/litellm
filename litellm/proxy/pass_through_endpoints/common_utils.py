from dataclasses import dataclass

from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")


@dataclass(frozen=True, slots=True)
class PassThroughDynamicLoggingParams:
    """Key/team-scoped logging settings, shaped for the ``Logging`` constructor"""

    callback_vars: dict[str, str] | None
    success_callbacks: list[str] | None
    failure_callbacks: list[str] | None


NO_PASS_THROUGH_DYNAMIC_LOGGING = PassThroughDynamicLoggingParams(
    callback_vars=None, success_callbacks=None, failure_callbacks=None
)


def get_pass_through_dynamic_logging_params(
    user_api_key_dict: UserAPIKeyAuth,
) -> PassThroughDynamicLoggingParams:
    """
    Resolve the key-level or team-level logging settings for a pass-through request.

    ``/chat/completions`` and friends get these through
    ``add_litellm_data_to_request``, which pass-through routes never call; without
    this, a team's logging credentials are ignored and its traces land in whichever
    project the global callback points at.
    """
    from litellm.proxy.litellm_pre_call_utils import _get_dynamic_logging_metadata
    from litellm.proxy.proxy_server import proxy_config

    callback_settings = _get_dynamic_logging_metadata(user_api_key_dict=user_api_key_dict, proxy_config=proxy_config)
    if callback_settings is None:
        return NO_PASS_THROUGH_DYNAMIC_LOGGING

    return PassThroughDynamicLoggingParams(
        callback_vars=callback_settings.callback_vars,
        success_callbacks=callback_settings.success_callback,
        failure_callbacks=callback_settings.failure_callback,
    )
