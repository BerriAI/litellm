from __future__ import annotations

from pathlib import Path
from typing import Final

from integration._support.client import Gateway
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy

from litellm.proxy.utils import hash_token


def test_master_key_not_inserted(gateway: Gateway, tmp_path: Path) -> None:
    with scratch_database() as database:
        with owned_proxy(
            gateway, tmp_path, {"DATABASE_URL": database}, remove_environment=("DIRECT_URL", "DATABASE_URL_READ_REPLICA")
        ) as proxy:
            assert proxy.request("GET", "/health/liveliness").status_code == 200
            stored_tokens: Final = frozenset(
                row["token"]
                for row in read_rows('SELECT token FROM "LiteLLM_VerificationToken"', (), database_url=database)
            )
    for leaked in (hash_token(gateway.key), gateway.key):
        assert leaked not in stored_tokens, "the master key was found in LiteLLM_VerificationToken"
    assert stored_tokens == frozenset(), f"startup unexpectedly wrote tokens to LiteLLM_VerificationToken: {stored_tokens}"
