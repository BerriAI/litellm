import hashlib
from typing import Final, NoReturn

import pytest

from litellm.proxy.mcp_server.db import _mcp_identifier_lock_keys, merge_user_env_vars

from .test_mcp_env_vars import _transactional_env_vars_prisma, env_vars_salt_key  # noqa: F401


@pytest.fixture
def blake2b_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """hashlib.blake2b as FIPS Python builds ship it: the call raises."""

    def blake2b(*_args: object, **_kwargs: object) -> NoReturn:
        raise ValueError("blake2b is blocked when usedforsecurity=True")

    monkeypatch.setattr(hashlib, "blake2b", blake2b)


def test_identifier_lock_keys_are_unchanged_without_fips_mode() -> None:
    assert _mcp_identifier_lock_keys("GitHub", None, "gh") == (-1720930101670657375, -1498181317228972674)


@pytest.mark.usefixtures("fips_mode", "blake2b_refused")
def test_fips_mode_identifier_lock_keys_are_truncated_sha256() -> None:
    assert _mcp_identifier_lock_keys("GitHub", None, "gh") == tuple(
        int.from_bytes(hashlib.sha256(name).digest()[:8], "big", signed=True)
        for name in (b"mcp_identifier:gh", b"mcp_identifier:github")
    )
    assert _mcp_identifier_lock_keys("GitHub") == (-7731150984991732978,)


@pytest.mark.asyncio
@pytest.mark.usefixtures("fips_mode", "blake2b_refused", "env_vars_salt_key")
async def test_fips_mode_merge_user_env_vars_takes_its_lock_without_blake2b() -> None:
    merged: Final = await merge_user_env_vars(
        _transactional_env_vars_prisma(), "alice", "srv-1", {"TOKEN": "abc"}, allowed_names=["TOKEN"]
    )

    assert merged == {"TOKEN": "abc"}
