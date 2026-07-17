from __future__ import annotations

from typing import Callable, Dict, Mapping, Protocol, runtime_checkable

Claims = Dict[str, object]


@runtime_checkable
class BasicAuthVerifier(Protocol):
    def verify(self, username: str, password: str) -> bool: ...


class IntrospectionResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class IntrospectionClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        data: Mapping[str, str],
        headers: Mapping[str, str],
        timeout: float,
    ) -> IntrospectionResponse: ...


IntrospectionClientFactory = Callable[[], IntrospectionClient]
