from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

_NATIVE_REQUEST_EXCLUDED_KEYS: Final = frozenset(
    {
        "api_base",
        "api_key",
        "client",
        "custom_llm_provider",
        "is_async",
        "kwargs",
        "litellm_logging_obj",
        "litellm_call_id",
        "model",
        "messages",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_region_name",
        "aws_session_name",
        "aws_profile_name",
        "aws_role_name",
        "aws_web_identity_token",
        "aws_sts_endpoint",
        "aws_bedrock_runtime_endpoint",
        "aws_external_id",
        "aws_session_tags",
    }
)


def get_native_messages_request_params(
    optional_params: Mapping[str, object],
    kwargs: Mapping[str, object],
    preserves_request_body: bool,
) -> Mapping[str, object]:
    if not preserves_request_body:
        return optional_params
    return MappingProxyType(
        {
            **optional_params,
            **MappingProxyType(
                {
                    key: value
                    for key, value in kwargs.items()
                    if key not in _NATIVE_REQUEST_EXCLUDED_KEYS and value is not None
                }
            ),
        }
    )
