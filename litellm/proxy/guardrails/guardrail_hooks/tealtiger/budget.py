import threading
from datetime import date
from typing import Final


def _today() -> str:
    return date.today().isoformat()  # noqa: DTZ011  # daily boundary is intentionally local-server-time, not UTC


class BudgetLedger:
    """In-memory spend per user (daily) and per session. Per-process, and reset when the local day rolls over."""

    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._day = _today()
        self._by_user: dict[str, float] = {}  # mutable-ok: live counters guarded by the lock
        self._by_session: dict[str, float] = {}  # mutable-ok: live counters guarded by the lock

    def _roll_day_locked(self) -> None:
        today: Final = _today()
        if today == self._day:
            return
        self._day = today
        self._by_user = {}
        self._by_session = {}

    def spent(self, user: str, session: str | None) -> tuple[float, float]:
        """Return (spent today by this user, spent by this session)."""
        with self._lock:
            self._roll_day_locked()
            return self._by_user.get(user, 0.0), self._by_session.get(session, 0.0) if session else 0.0

    def record(self, user: str, session: str | None, cost_usd: float) -> None:
        with self._lock:
            self._roll_day_locked()
            self._by_user[user] = self._by_user.get(user, 0.0) + cost_usd
            if session:
                self._by_session[session] = self._by_session.get(session, 0.0) + cost_usd
