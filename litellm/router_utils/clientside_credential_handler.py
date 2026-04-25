"""
Utils for handling clientside credentials

Supported clientside credentials:
- api_key
- api_base
- base_url

If given, generate a unique model_id for the deployment.

Ensures cooldowns are applied correctly.
"""

clientside_credential_keys = ["api_key", "api_base", "base_url"]

# Admin-configured fields that carry secrets or environment-specific config
# meant for the *original* upstream. When the caller redirects ``api_base`` /
# ``base_url`` to their own server, these MUST NOT flow through unchanged or
# the admin's ``OpenAI-Organization`` header, ``extra_body`` payloads, AWS /
# Vertex / Azure credentials, etc. would be sent to the attacker. Only carry
# them through when the caller explicitly re-supplies the field.
_ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE = [
    "organization",
    "extra_body",
    "extra_headers",
    "default_headers",
    "api_version",
    "api_type",
    "azure_ad_token",
    "azure_ad_token_provider",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_sts_endpoint",
    "aws_web_identity_token",
    "aws_role_name",
    "vertex_credentials",
    "vertex_project",
    "vertex_location",
]


def is_clientside_credential(request_kwargs: dict) -> bool:
    """
    Check if the credential is a clientside credential.
    """
    return any(key in request_kwargs for key in clientside_credential_keys)


def get_dynamic_litellm_params(litellm_params: dict, request_kwargs: dict) -> dict:
    """
    Generate a unique model_id for the deployment.

    Returns
    - litellm_params: dict

    for generating a unique model_id.
    """
    # update litellm_params with clientside credentials
    for key in clientside_credential_keys:
        if key in request_kwargs:
            litellm_params[key] = request_kwargs[key]

    # If the caller redirected api_base/base_url to a client-controlled value,
    # don't forward the admin's organization / extra_body / region / token /
    # vertex / aws fields — those were meant for the original upstream.
    if "api_base" in request_kwargs or "base_url" in request_kwargs:
        for field in _ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE:
            if field not in request_kwargs:
                litellm_params.pop(field, None)

    return litellm_params
