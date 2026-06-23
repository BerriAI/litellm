import logging

from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_and_resolve_litellm_params,
)


def test_decrypt_and_resolve_litellm_params_resolves_os_environ(monkeypatch):
    monkeypatch.setenv("MY_RESOLVE_KEY", "sk-resolved")
    out = decrypt_and_resolve_litellm_params(
        {"api_key": "os.environ/MY_RESOLVE_KEY", "model": "gpt-4o"}
    )
    assert out == {"api_key": "sk-resolved", "model": "gpt-4o"}


def test_decrypt_and_resolve_litellm_params_passes_plain_values_through():
    out = decrypt_and_resolve_litellm_params(
        {"api_key": "sk-literal", "model": "gpt-4o"}
    )
    assert out == {"api_key": "sk-literal", "model": "gpt-4o"}


def test_decrypt_and_resolve_litellm_params_leaves_non_string_values_untouched():
    out = decrypt_and_resolve_litellm_params({"timeout": 30, "stream": True})
    assert out == {"timeout": 30, "stream": True}


def test_decrypt_and_resolve_litellm_params_warns_on_missing_env_var(
    monkeypatch, caplog
):
    monkeypatch.delenv("MY_MISSING_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        out = decrypt_and_resolve_litellm_params(
            {"api_key": "os.environ/MY_MISSING_KEY", "model": "gpt-4o"}
        )
    assert out == {"api_key": None, "model": "gpt-4o"}
    assert "MY_MISSING_KEY" in caplog.text
