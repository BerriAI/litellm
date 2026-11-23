from collections.abc import Iterator

import pytest

from litellm.proxy.mcp_server.outbound_credentials.bridge_credentials import envelope_keys_from_master_key
from litellm.proxy.mcp_server.outbound_credentials.session_credentials import session_keys_from_master_key


@pytest.fixture
def openssl_enforcing_fips(fips_openssl: None) -> Iterator[None]:
    envelope_keys_from_master_key.cache_clear()
    session_keys_from_master_key.cache_clear()
    yield
    envelope_keys_from_master_key.cache_clear()
    session_keys_from_master_key.cache_clear()


@pytest.fixture
def fips_mode_on(openssl_enforcing_fips: None, fips_mode: None) -> None:
    return None
