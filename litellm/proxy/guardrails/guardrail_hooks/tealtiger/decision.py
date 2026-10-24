import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Final


class Action(str, Enum):
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    BLOCK = "BLOCK"


class RiskLevel(str, Enum):
    NONE = "NONE"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of one governance check. Every check returns one, and each is logged as a receipt."""

    action: Action
    reason_code: str
    risk_level: RiskLevel
    correlation_id: str
    redacted_text: str | None = None
    details: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def receipt_json(self) -> str:
        """Typed, JSON-serializable decision receipt for audit trails."""
        return json.dumps(
            {
                "correlation_id": self.correlation_id,
                "action": self.action.value,
                "reason_code": self.reason_code,
                "risk_level": self.risk_level.value,
                "details": dict(self.details),
            },
            sort_keys=True,
        )


NO_VIOLATIONS: Final = "NO_VIOLATIONS"
