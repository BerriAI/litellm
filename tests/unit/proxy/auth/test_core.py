import dataclasses
from collections.abc import Callable
from typing import Final

import pytest

from litellm.proxy._types import (
    LiteLLM_JWTAuth,
)
from litellm.proxy.auth import core
from litellm.proxy.auth.host.python import PythonAuthHost
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.rust_bridge._native import RustAuth


@pytest.mark.asyncio
async def test_a_host_failure_the_core_cannot_skip_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch, auth_host: Callable[..., PythonAuthHost]
) -> None:
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://idp.example/jwks")
    settings: Final = core.build_settings(
        {"enable_jwt_auth": True},
        LiteLLM_JWTAuth(),
        master_key=None,
        premium_user=True,
        has_db=False,
    )

    class _Unreadable(UserApiKeyCache):
        async def async_get_cache(self, *args: object, **kwargs: object) -> object:
            raise ConnectionError("cache down")

    failing: Final = dataclasses.replace(auth_host(), user_api_key_cache=_Unreadable())
    token: Final = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsifQ.eyJzdWIiOiJ1In0.c2ln"

    with pytest.raises(ConnectionError, match="cache down"):
        await core.drive(RustAuth.verify_jwt(token, 0.0, settings), failing)
