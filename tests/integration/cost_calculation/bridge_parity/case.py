from dataclasses import dataclass
from typing import Final

from integration.cost_calculation.cost_tracking_case import CostTrackingTestCase, ExactExpected

ROW_FIELDS: Final = frozenset({"spend", "prompt_tokens", "completion_tokens"})
_EXPECTED_FIELDS: Final = frozenset(ExactExpected.model_fields)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderQuirk:
    """A provider-level reason the bridged leg legitimately bills a different row: the `ExactExpected` fields that
    differ, why, and the provider document that says so."""

    fields: frozenset[str]
    reason: str
    source: str

    def __post_init__(self) -> None:
        unknown: Final = self.fields - _EXPECTED_FIELDS
        if not self.fields or unknown:
            raise ValueError(f"quirk fields must be non-empty ExactExpected fields, got {sorted(unknown)}")
        if not self.source.startswith("https://"):
            raise ValueError("a quirk cites the provider document that explains it")


@dataclass(frozen=True, slots=True, kw_only=True)
class BridgeParityTestCase:
    """One stored provider response billed through the provider's own API (`native`) and through a client API that
    LiteLLM bridges into it (`bridged`). Both legs bill the one row unless `quirk` names the fields that differ."""

    native: CostTrackingTestCase
    bridged: CostTrackingTestCase
    quirk: ProviderQuirk | None = None

    def __post_init__(self) -> None:
        if self.native.endpoint == self.bridged.endpoint:
            raise ValueError(f"{self.id}: native and bridged cases must use different client endpoints")
        if self.native.response != self.bridged.response:
            raise ValueError(f"{self.id}: native and bridged cases must share one stored provider response")
        if (self.native.model, self.native.deployment) != (self.bridged.model, self.bridged.deployment):
            raise ValueError(f"{self.id}: native and bridged cases must share model and deployment")
        if not isinstance(self.native.expected, ExactExpected) or not isinstance(self.bridged.expected, ExactExpected):
            raise ValueError(f"{self.id}: both legs need an ExactExpected row")
        native: Final = self.native.expected.model_dump()
        bridged: Final = self.bridged.expected.model_dump()
        differing: Final = frozenset(field for field in _EXPECTED_FIELDS if native[field] != bridged[field])
        if differing != self.quirk_fields:
            raise ValueError(
                f"{self.id}: expected rows differ in {sorted(differing)} but the quirk covers {sorted(self.quirk_fields)}"
            )

    @property
    def quirk_fields(self) -> frozenset[str]:
        return self.quirk.fields if self.quirk is not None else frozenset()

    @property
    def compared_row_fields(self) -> frozenset[str]:
        return ROW_FIELDS - self.quirk_fields

    @property
    def id(self) -> str:
        return self.bridged.name
