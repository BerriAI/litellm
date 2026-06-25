"""OAuth client-credentials deployment config must not be forwarded to a
client-redirected api_base.

Regression for the security finding on PR #31026: when a deployment permits
clientside api_base override, the router must clear the admin's OAuth flag and
fields so the bearer token litellm mints with them is never sent to the caller's
endpoint.
"""

from litellm.router_utils.clientside_credential_handler import (
    _ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE,
    get_dynamic_litellm_params,
)

_OAUTH_FIELDS = (
    "oauth_client_credentials",
    "oauth_token_url",
    "oauth_client_id",
    "oauth_client_secret",
    "oauth_scope",
)


def _deployment_params() -> dict:
    return {
        "model": "openai/gpt-4o",
        "api_base": "https://gateway.internal/v1",
        "oauth_client_credentials": True,
        "oauth_token_url": "https://idp.internal/oauth/token",
        "oauth_client_id": "admin-id",
        "oauth_client_secret": "admin-secret",
        "oauth_scope": "admin-scope",
    }


def test_oauth_fields_in_base_override_clear_list():
    for field in _OAUTH_FIELDS:
        assert field in _ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE


def test_oauth_creds_cleared_when_client_overrides_api_base():
    params = get_dynamic_litellm_params(
        _deployment_params(), {"api_base": "https://client.example/v1"}
    )
    assert params["api_base"] == "https://client.example/v1"
    for field in _OAUTH_FIELDS:
        assert field not in params


def test_oauth_creds_preserved_without_base_override():
    params = get_dynamic_litellm_params(_deployment_params(), {"temperature": 0.5})
    for field in _OAUTH_FIELDS:
        assert field in params
