"""Credentials from a redis-py ``CredentialProvider`` an SDK caller passed in ``cache_kwargs``, asked the way
redis-py's async connection asks it."""

from __future__ import annotations

import asyncio
from typing import Protocol


class CredentialProvider(Protocol):
    async def get_credentials_async(self) -> tuple[str] | tuple[str, str]: ...


def credentials(provider: CredentialProvider) -> tuple[str | None, str]:
    match asyncio.run(provider.get_credentials_async()):
        case (password,):
            return None, password
        case (username, password):
            return username, password
