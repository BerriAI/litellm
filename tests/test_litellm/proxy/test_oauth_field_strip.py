"""Proxy strips client-supplied custom_oauth credential fields from request bodies.

`litellm.completion` accepts `oauth_token_url` / `oauth_client_id` /
`oauth_client_secret` / `oauth_scope` as kwargs (intentional on direct SDK use).
On the proxy those same fields would let a caller redirect a configured
deployment's client_secret to an arbitrary token URL, so the proxy strips them
at the boundary; OAuth credentials are deployment configuration only.

Regression coverage for the security finding on PR #31026.
"""

from litellm.proxy.litellm_pre_call_utils import (
    _CLIENT_OAUTH_CONTROL_FIELDS,
    _strip_client_oauth_overrides,
)


class TestStripClientOAuthOverrides:
    def test_control_fields_cover_oauth_params(self):
        assert _CLIENT_OAUTH_CONTROL_FIELDS == frozenset(
            {
                "oauth_token_url",
                "oauth_client_id",
                "oauth_client_secret",
                "oauth_scope",
            }
        )

    def test_root_oauth_fields_dropped(self):
        data = {
            "model": "custom_oauth/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "oauth_token_url": "https://attacker.example/token",
            "oauth_client_id": "evil",
            "oauth_client_secret": "evil",
            "oauth_scope": "evil",
        }
        _strip_client_oauth_overrides(data)
        assert data == {
            "model": "custom_oauth/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_metadata_oauth_fields_dropped(self):
        data = {
            "model": "custom_oauth/gpt-4o",
            "metadata": {
                "user_session": "keep-me",
                "oauth_token_url": "https://attacker.example/token",
            },
            "litellm_metadata": {"oauth_client_secret": "evil"},
        }
        _strip_client_oauth_overrides(data)
        assert data["metadata"] == {"user_session": "keep-me"}
        assert data["litellm_metadata"] == {}

    def test_non_oauth_fields_untouched(self):
        data = {
            "model": "custom_oauth/gpt-4o",
            "temperature": 0.7,
            "max_tokens": 100,
            "api_base": "https://gateway.internal/v1",
        }
        before = dict(data)
        _strip_client_oauth_overrides(data)
        assert data == before
