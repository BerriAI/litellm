import _hashlib
from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from cryptography.hazmat.bindings._rust import openssl as rust_openssl

from litellm.proxy.common_utils.fips import enforce_fips_mode
from litellm.proxy.mcp_server.outbound_credentials.bridge_credentials import envelope_keys_from_master_key
from litellm.proxy.mcp_server.outbound_credentials.session_credentials import session_keys_from_master_key


@pytest.fixture
def openssl_enforcing_fips(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(_hashlib, "get_fips_mode", lambda: 1)
    monkeypatch.setattr(_hashlib, "scrypt", Mock(side_effect=ValueError("[digital envelope routines] unsupported")))
    envelope_keys_from_master_key.cache_clear()
    session_keys_from_master_key.cache_clear()
    yield
    envelope_keys_from_master_key.cache_clear()
    session_keys_from_master_key.cache_clear()


@pytest.fixture
def fips_mode_on(openssl_enforcing_fips: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(rust_openssl, "is_fips_enabled", lambda: True)
    enforce_fips_mode({"fips_mode": True})
    yield
    enforce_fips_mode({})
