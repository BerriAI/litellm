from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.proxy.auth.master_key_policy import insecure_master_key_warning


def test_insecure_master_key_warning_returned_for_example_key():
    warning = insecure_master_key_warning("sk-1234")

    assert warning is not None
    assert "sk-1234" in warning


def test_insecure_master_key_warning_none_for_strong_key():
    assert insecure_master_key_warning("sk-strong-random-key") is None


def test_insecure_master_key_warning_none_for_none():
    assert insecure_master_key_warning(None) is None


def test_insecure_master_key_warning_survives_redaction():
    warning = insecure_master_key_warning("sk-1234")

    assert warning is not None
    assert "secrets.token_urlsafe" in redact_string(warning)
