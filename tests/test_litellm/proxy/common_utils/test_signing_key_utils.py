import sys
from types import SimpleNamespace

from litellm.proxy.common_utils.signing_key_utils import get_proxy_signing_key


def test_get_proxy_signing_key_prefers_salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "salt-key")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "env-master-key")
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(master_key="proxy-master-key"),
    )

    assert get_proxy_signing_key() == "salt-key"


def test_get_proxy_signing_key_uses_loaded_proxy_server_master_key(monkeypatch):
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(master_key="proxy-master-key"),
    )

    assert get_proxy_signing_key() == "proxy-master-key"


def test_get_proxy_signing_key_falls_back_to_env_master_key(monkeypatch):
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "env-master-key")
    monkeypatch.delitem(sys.modules, "litellm.proxy.proxy_server", raising=False)

    assert get_proxy_signing_key() == "env-master-key"
