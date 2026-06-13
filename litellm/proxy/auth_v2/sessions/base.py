from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, TypeVar, runtime_checkable

SessionValue = TypeVar("SessionValue", bound=Mapping[str, Any])


@runtime_checkable
class SessionStore(Protocol[SessionValue]):
    """Async key/value store with per-key TTL, generic over its value schema.

    Backs short-lived auth state. Each store is parameterized by the typed
    payload it holds (see ``schemas``) and namespaced by the backend that hands
    it out, so several stores can share one Redis instance without colliding.
    """

    async def get(self, key: str) -> Optional[SessionValue]: ...

    async def set(
        self, key: str, value: SessionValue, ttl_seconds: Optional[int] = None
    ) -> None: ...

    async def pop(self, key: str) -> Optional[SessionValue]: ...

    async def delete(self, key: str) -> None: ...

    async def add_if_absent(self, key: str, ttl_seconds: Optional[int] = None) -> bool:
        """Set a marker only if the key is absent; return True iff newly set.

        Atomic and value-free. Use for one-time guards like SAML assertion
        replay detection, where only key presence matters.
        """
        ...
