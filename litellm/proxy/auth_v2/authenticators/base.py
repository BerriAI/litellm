from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from fastapi import Request

from litellm.proxy.auth_v2.models import Credential


@runtime_checkable
class Authenticator(Protocol):
    async def authenticate(self, request: Request) -> Optional[Credential]: ...

    def challenge(self) -> str: ...
