from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

CHARGE_OUTCOME_KEY: Final = "_litellm_required_charge_outcome"


@dataclass(frozen=True, slots=True)
class CommittedCharge:
    event_id: str
    key_hash: str
    amount_usd: str
    duplicate: bool = False
    key_counter_updated: bool = True


@dataclass(frozen=True, slots=True)
class FailedCharge:
    key_hash: str
    reason: str


class RequiredChargeCapture(Protocol):
    async def capture_success(
        self,
        details: Mapping[str, object],
        response: object,
        start_time: datetime,
        end_time: datetime,
    ) -> CommittedCharge | FailedCharge: ...

    async def wait(self) -> bool: ...

    def fork(self) -> "RequiredChargeCapture": ...


def charge_outcome(details: Mapping[str, object]) -> CommittedCharge | FailedCharge | None:
    outcome: Final = details.get(CHARGE_OUTCOME_KEY)
    return outcome if isinstance(outcome, (CommittedCharge, FailedCharge)) else None
