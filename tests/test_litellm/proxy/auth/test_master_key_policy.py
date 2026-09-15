from typing import Final

import pytest

from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.proxy.auth.master_key_policy import (
    alternative_auth_enabled,
    insecure_master_key_reason,
    insecure_master_key_warning,
    master_key_lockout_action,
)


def test_insecure_master_key_reason_for_example_key():
    assert insecure_master_key_reason("sk-1234", alternative_auth_enabled=False) == "example_key"


def test_insecure_master_key_reason_for_missing_key():
    assert insecure_master_key_reason(None, alternative_auth_enabled=False) == "missing"
    assert insecure_master_key_reason("", alternative_auth_enabled=False) == "missing"


def test_insecure_master_key_reason_none_for_strong_key():
    assert insecure_master_key_reason("sk-strong-random-key", alternative_auth_enabled=False) is None


def test_insecure_master_key_reason_none_with_alternative_auth():
    assert insecure_master_key_reason(None, alternative_auth_enabled=True) is None


def test_insecure_master_key_reason_none_with_custom_auth():
    general_settings: Final = {"custom_auth": "my_package.custom_auth_handler"}

    assert insecure_master_key_reason(None, alternative_auth_enabled=alternative_auth_enabled(general_settings)) is None


def test_insecure_master_key_warning_returned_for_example_key():
    warning = insecure_master_key_warning("sk-1234", alternative_auth_enabled=False)

    assert warning is not None
    assert "sk-1234" in warning


def test_insecure_master_key_warning_for_missing_key():
    warning = insecure_master_key_warning(None, alternative_auth_enabled=False)

    assert warning is not None
    assert "No master key" in warning


def test_insecure_master_key_warning_for_empty_key():
    warning = insecure_master_key_warning("", alternative_auth_enabled=False)

    assert warning is not None
    assert "No master key" in warning


def test_insecure_master_key_warning_none_for_missing_key_with_alt_auth():
    assert insecure_master_key_warning(None, alternative_auth_enabled=True) is None


def test_insecure_master_key_warning_none_for_strong_key():
    assert insecure_master_key_warning("sk-strong-random-key", alternative_auth_enabled=False) is None


def test_insecure_master_key_warning_survives_redaction():
    warning = insecure_master_key_warning("sk-1234", alternative_auth_enabled=False)

    assert warning is not None
    assert "secrets.token_urlsafe" in redact_string(warning)


@pytest.mark.parametrize(
    "route,method,expected",
    [
        ("/credentials", "POST", "store_credentials"),
        ("/credentials/my_creds", "PATCH", "store_credentials"),
        ("/credentials/my_creds", "DELETE", "store_credentials"),
        ("/model/new", "POST", "store_credentials"),
        ("/model/update", "POST", "store_credentials"),
        ("/model/abc-123/update", "PATCH", "store_credentials"),
        ("/config/update", "POST", "store_credentials"),
        ("/model/delete", "POST", None),
        ("/model/block", "POST", None),
        ("/model/unblock", "POST", None),
        ("/credentials", "GET", "access_credentials"),
        ("/credentials/by_name/my_creds", "GET", "access_credentials"),
        ("/credentials/by_model/abc123", "GET", "access_credentials"),
        ("/model/info", "GET", "access_credentials"),
        ("/v1/model/info", "GET", "access_credentials"),
        ("/v2/model/info", "GET", "access_credentials"),
        ("/get/config/callbacks", "GET", "access_credentials"),
        ("/config/list", "GET", "access_credentials"),
        ("/config/field/info", "GET", "access_credentials"),
        ("/chat/completions", "POST", "use_credentials"),
        ("/v1/chat/completions", "POST", "use_credentials"),
        ("/v1/embeddings", "POST", "use_credentials"),
        ("/v1/messages", "POST", "use_credentials"),
        ("/key/generate", "POST", "manage_virtual_keys"),
        ("/key/update", "POST", "manage_virtual_keys"),
        ("/key/delete", "POST", "manage_virtual_keys"),
        ("/key/abc-def/regenerate", "POST", "manage_virtual_keys"),
        ("/key/service-account/generate", "POST", "manage_virtual_keys"),
        ("/key/block", "POST", "manage_virtual_keys"),
        ("/key/info", "GET", None),
        ("/key/list", "GET", None),
        ("/login", "POST", None),
        ("/health/readiness", "GET", None),
        ("/health/readiness/details", "GET", None),
        ("/models", "GET", None),
        ("/v1/models", "GET", None),
    ],
)
@pytest.mark.parametrize("reason", ["example_key", "missing"])
def test_master_key_lockout_action(route, method, expected, reason):
    stored: Final = expected != "store_credentials"
    assert master_key_lockout_action(route, method, reason, stored_credentials_present=stored) == expected


@pytest.mark.parametrize(
    "route,method",
    [
        ("/credentials", "POST"),
        ("/model/new", "POST"),
        ("/credentials", "GET"),
        ("/model/info", "GET"),
        ("/chat/completions", "POST"),
        ("/key/generate", "POST"),
    ],
)
def test_master_key_lockout_action_none_when_key_secure(route, method):
    assert master_key_lockout_action(route, method, None, stored_credentials_present=True) is None


@pytest.mark.parametrize(
    "route,method",
    [
        ("/credentials", "GET"),
        ("/model/info", "GET"),
        ("/chat/completions", "POST"),
        ("/key/generate", "POST"),
        ("/key/abc/regenerate", "POST"),
    ],
)
@pytest.mark.parametrize("reason", ["example_key", "missing"])
def test_master_key_lockout_action_not_blocked_without_stored_credentials(route, method, reason):
    assert master_key_lockout_action(route, method, reason, stored_credentials_present=False) is None
