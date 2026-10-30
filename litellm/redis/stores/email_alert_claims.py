from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol, TypeAlias

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_EXPIRE_SECONDS: Final = 'return redis.call("EXPIRE", KEYS[1], ARGV[1])'


@dataclass(frozen=True, slots=True)
class SoftBudgetAlert:
    identifier: str


@dataclass(frozen=True, slots=True)
class MaxBudgetAlert:
    identifier: str


@dataclass(frozen=True, slots=True)
class TeamMemberAlert:
    user_id: str | None
    team_id: str | None


@dataclass(frozen=True, slots=True)
class ThresholdBudgetAlert:
    identifier: str | TeamMemberAlert
    threshold_pct: int


EmailBudgetAlert: TypeAlias = SoftBudgetAlert | MaxBudgetAlert | ThresholdBudgetAlert


def alert_key(alert: EmailBudgetAlert) -> str:
    match alert:
        case SoftBudgetAlert(identifier):
            return f"email_budget_alerts:soft_budget_crossed:{identifier}"
        case MaxBudgetAlert(identifier):
            return f"email_budget_alerts:max_budget_alert:{identifier}"
        case ThresholdBudgetAlert(identifier, threshold_pct):
            subject: Final = (
                f"team_member:{identifier.user_id}:{identifier.team_id}"
                if isinstance(identifier, TeamMemberAlert)
                else identifier
            )
            return f"email_budget_alerts:max_budget_alert:{threshold_pct}:{subject}"


class EmailAlertClaimsStore(Protocol):
    async def claim(self, alert: EmailBudgetAlert, *, ttl: timedelta) -> float | RedisFailure: ...

    async def release_failed_send(self, alert: EmailBudgetAlert) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportEmailAlertClaimsStore:
    transport: RedisTransport

    async def claim(self, alert: EmailBudgetAlert, *, ttl: timedelta) -> float | RedisFailure:
        key: Final = alert_key(alert)
        batch: Final = self.transport.batch(transaction=False)
        increment: Final = batch.incrbyfloat(key, 1)
        expiry: Final = batch.ttl(key)
        count: Final = await increment
        current_ttl: Final = await expiry
        if isinstance(count, RedisFailure):
            return count
        if isinstance(current_ttl, RedisFailure):
            return current_ttl
        if current_ttl == -1:
            expired: Final = await self.transport.eval(_EXPIRE_SECONDS, (key,), (int(ttl.total_seconds()),))
            if isinstance(expired, RedisFailure):
                return expired
            if type(expired) is not int or expired not in (0, 1):
                return RedisFailure(kind="decode", message="email alert expiry reply is not 0 or 1")
        return count

    async def release_failed_send(self, alert: EmailBudgetAlert) -> bool | RedisFailure:
        deleted: Final = await self.transport.delete(alert_key(alert))
        return deleted if isinstance(deleted, RedisFailure) else deleted > 0
