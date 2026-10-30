import ast
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_VALUE: Final = TypeAdapter(object)


@dataclass(frozen=True, slots=True)
class BudgetAlertMarker:
    event: Literal[
        "failed_tracking", "budget_crossed", "threshold_crossed", "projected_limit_exceeded", "soft_budget_crossed"
    ]
    identity: str | None


def budget_marker_key(marker: BudgetAlertMarker) -> str:
    return f"budget_alerts:{marker.event}:{marker.identity}"


class AlertBudgetMarkersStore(Protocol):
    async def was_budget_alert_sent(self, marker: BudgetAlertMarker) -> bool | RedisFailure: ...

    async def mark_budget_alert_sent(
        self, marker: BudgetAlertMarker, *, ttl: timedelta | None
    ) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportAlertBudgetMarkersStore:
    transport: RedisTransport

    async def was_budget_alert_sent(self, marker: BudgetAlertMarker) -> bool | RedisFailure:
        raw: Final = await self.transport.get(budget_marker_key(marker))
        if isinstance(raw, RedisFailure):
            return raw
        if raw is None:
            return False
        try:
            return _VALUE.validate_json(raw) is not None
        except ValidationError:
            try:
                return _VALUE.validate_python(ast.literal_eval(raw.decode("utf-8"))) is not None
            except (UnicodeError, ValueError, SyntaxError, ValidationError):
                return RedisFailure(kind="decode", message="budget marker is not a literal")

    async def mark_budget_alert_sent(self, marker: BudgetAlertMarker, *, ttl: timedelta | None) -> None | RedisFailure:
        if ttl is not None and ttl <= timedelta(0):
            return RedisFailure(kind="server", message="invalid expiry for budget marker")
        reply: Final = await self.transport.set(budget_marker_key(marker), b'"SENT"', ttl=ttl)
        return reply if isinstance(reply, RedisFailure) else None
