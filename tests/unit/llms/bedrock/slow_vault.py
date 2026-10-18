"""A Vault that holds none of its hosted secrets and takes half a second per read.

get_secret pays that round trip for each hosted name and then falls back to the environment.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable
from typing import Final, TypeVar

import pytest

import litellm
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem

VAULT_READ_SECONDS: Final = 0.5
LOOP_STALL_LIMIT_SECONDS: Final = 0.4

_Result = TypeVar("_Result")


class SlowVault:
    def __init__(self) -> None:
        self.read: Final = threading.Event()

    def sync_read_secret(self, secret_name: str) -> None:
        self.read.set()
        time.sleep(VAULT_READ_SECONDS)


def host_in_slow_vault(monkeypatch: pytest.MonkeyPatch, *secret_names: str) -> SlowVault:
    vault: Final = SlowVault()
    monkeypatch.setattr(litellm, "secret_manager_client", vault)
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.HASHICORP_VAULT)
    monkeypatch.setattr(
        litellm,
        "_key_management_settings",
        KeyManagementSettings(access_mode="read_only", hosted_keys=list(secret_names)),
    )
    return vault


async def _longest_stall(done: asyncio.Event) -> float:
    if done.is_set():
        return 0.0
    started: Final = time.monotonic()
    await asyncio.sleep(0.01)
    return max(time.monotonic() - started, await _longest_stall(done))


async def longest_loop_stall_while(call: Awaitable[_Result]) -> tuple[float, _Result]:
    done: Final = asyncio.Event()

    async def run() -> _Result:
        try:
            return await call
        finally:
            done.set()

    return await asyncio.gather(_longest_stall(done), run())
