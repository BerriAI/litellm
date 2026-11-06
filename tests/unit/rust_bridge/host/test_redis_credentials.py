import asyncio
from typing import Final

import pytest
from redis.credentials import CredentialProvider

from litellm.rust_bridge.host.redis_credentials import credentials


class _SyncOnly(CredentialProvider):
    def __init__(self, answer: tuple[str] | tuple[str, str]) -> None:
        self.answer: Final = answer

    def get_credentials(self) -> tuple[str] | tuple[str, str]:
        return self.answer


class _AsyncOnly(CredentialProvider):
    def get_credentials(self) -> tuple[str, str]:
        raise AssertionError("redis-py's async connection asks get_credentials_async")

    async def get_credentials_async(self) -> tuple[str, str]:
        await asyncio.sleep(0)
        return "async-user", "async-password"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (_SyncOnly(("token",)), (None, "token")),
        (_SyncOnly(("user", "password")), ("user", "password")),
        (_AsyncOnly(), ("async-user", "async-password")),
    ],
)
def test_credentials_are_what_redis_py_would_authenticate_with(
    provider: CredentialProvider, expected: tuple[str | None, str]
) -> None:
    assert credentials(provider) == expected
