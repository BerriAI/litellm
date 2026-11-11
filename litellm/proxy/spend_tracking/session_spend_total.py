import hashlib
import logging
from collections.abc import Callable
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import log_redis_failure
from litellm.constants import SESSION_SPEND_TOTAL_TTL_SECONDS

_STORED_SPEND: Final[TypeAdapter[float | str | bytes | None]] = TypeAdapter(float | str | bytes | None)


class SharedCounterStore(Protocol):
    async def async_increment(
        self, key: str, value: float, *, ttl: int | None = None, refresh_ttl: bool = False
    ) -> float: ...

    async def async_get_cache(self, key: str) -> object: ...


def _key(api_key: str, session_id: str) -> str:
    session_digest: Final = hashlib.sha256(session_id.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"session_spend_total:{api_key}:{session_digest}"


class SessionSpendTotals:
    """Spend billed to one key's caller-supplied session, kept in the shared Redis on a TTL that every write slides."""

    def __init__(self, shared_store: Callable[[], SharedCounterStore | None]) -> None:
        self.shared_store: Final = shared_store

    @with_service_target("session_spend_totals")
    async def record(self, api_key: str, session_id: str, spend: float) -> None:
        store: Final = self.shared_store()
        if store is None:
            return
        try:
            await store.async_increment(
                _key(api_key, session_id), spend, ttl=SESSION_SPEND_TOTAL_TTL_SECONDS, refresh_ttl=True
            )
        except Exception as record_error:  # noqa: BLE001  # a lost session total must never fail the spend write
            log_redis_failure(verbose_proxy_logger, logging.WARNING, "Session spend total write failed", record_error)

    @with_service_target("session_spend_totals")
    async def read(self, api_key: str, session_id: str) -> float | None:
        store: Final = self.shared_store()
        if store is None:
            return None
        try:
            stored: Final = _STORED_SPEND.validate_python(await store.async_get_cache(_key(api_key, session_id)))
        except Exception as read_error:  # noqa: BLE001  # an unreadable total is reported as no total
            log_redis_failure(verbose_proxy_logger, logging.WARNING, "Session spend total read failed", read_error)
            return None
        if stored is None:
            return None
        return float(stored.decode() if isinstance(stored, bytes) else stored)
