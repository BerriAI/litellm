from typing import Final

from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.proxy.auth.master_key_policy import (
    alternative_auth_enabled,
    insecure_master_key_reason,
    insecure_master_key_warning,
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
