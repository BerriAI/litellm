"""Audit event schema. Append-only by contract; sinks never update or delete."""

import enum
from dataclasses import dataclass
from datetime import datetime

from litellm.integrations.governor.model.counters import Outcome
from litellm.integrations.governor.model.decisions import Correlation, Decision
from litellm.integrations.governor.model.subjects import SubjectRef


class AuditSeverity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ADMIN_RESET = "admin_reset"


@dataclass(frozen=True)
class AuditEvent:
    request_id: str
    decided_at: datetime
    latency_ms: float
    subjects: SubjectRef
    decision: Decision
    correlation: Correlation
    severity: AuditSeverity = AuditSeverity.INFO
    outcome: Outcome | None = None
