from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY

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


@runtime_checkable
class ChargeCaptureSource(Protocol):
    def new_charge_capture(self) -> RequiredChargeCapture | None: ...


def inherited_charge_capture(params: Mapping[str, object]) -> RequiredChargeCapture | None:
    from litellm.litellm_core_utils.core_helpers import as_str_mapping

    for metadata in (as_str_mapping(params.get("litellm_metadata")), as_str_mapping(params.get("metadata"))):
        if metadata is None or not (
            metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY) or metadata.get("semantic-cache-embedding")
        ):
            continue
        if (
            isinstance(source := metadata.get("user_api_key_auth"), ChargeCaptureSource)
            and (capture := source.new_charge_capture()) is not None
        ):
            return capture
    return None


def charge_outcome(details: Mapping[str, object]) -> CommittedCharge | FailedCharge | None:
    outcome: Final = details.get(CHARGE_OUTCOME_KEY)
    return outcome if isinstance(outcome, (CommittedCharge, FailedCharge)) else None
