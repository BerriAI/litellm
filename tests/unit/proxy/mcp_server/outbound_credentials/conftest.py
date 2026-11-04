import _hashlib
from collections.abc import Iterator
from typing import Final

import pytest

from litellm.proxy.mcp_server.outbound_credentials.bridge_credentials import envelope_keys_from_master_key
from litellm.proxy.mcp_server.outbound_credentials.session_credentials import session_keys_from_master_key


@pytest.fixture
def openssl_enforcing_fips(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    real_new: Final = _hashlib.new

    def new(name: str, data: bytes = b"", *, usedforsecurity: bool = True) -> _hashlib.HASH:
        if name == "md5":
            raise _hashlib.UnsupportedDigestmodError("unsupported hash type")
        return real_new(name, data, usedforsecurity=usedforsecurity)

    monkeypatch.setattr(_hashlib, "get_fips_mode", lambda: 1)
    monkeypatch.setattr(_hashlib, "new", new)
    envelope_keys_from_master_key.cache_clear()
    session_keys_from_master_key.cache_clear()
    yield
    envelope_keys_from_master_key.cache_clear()
    session_keys_from_master_key.cache_clear()
