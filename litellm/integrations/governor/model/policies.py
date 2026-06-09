"""Per-policy configuration dataclasses.

Pure data; the policy classes in ``governor.policies`` consume these. ``fail_mode``
lives here so a config alone fully determines a policy's degradation behavior.
"""

from dataclasses import dataclass

from litellm.integrations.governor.model.counters import WindowKind
from litellm.integrations.governor.model.decisions import FailMode
from litellm.integrations.governor.model.subjects import SubjectKind


@dataclass(frozen=True)
class PolicyConfig:
    policy_id: str
    subject_kind: SubjectKind
    fail_mode: FailMode
    enabled: bool = True


@dataclass(frozen=True)
class BudgetPolicyConfig(PolicyConfig):
    max_budget: float = 0.0
    window: WindowKind = "monthly"
    fail_mode: FailMode = "closed"


@dataclass(frozen=True)
class RpmPolicyConfig(PolicyConfig):
    capacity: int = 0
    period_seconds: int = 60
    burst: int = 0
    fail_mode: FailMode = "open"


@dataclass(frozen=True)
class TpmPolicyConfig(PolicyConfig):
    capacity: int = 0
    period_seconds: int = 60
    fail_mode: FailMode = "open"


@dataclass(frozen=True)
class ConcurrentRequestPolicyConfig(PolicyConfig):
    max_inflight: int = 0
    slot_ttl_seconds: int = 600
    fail_mode: FailMode = "closed"


@dataclass(frozen=True)
class CostThrottlePolicyConfig(PolicyConfig):
    budget_policy_id: str = ""
    reject_at_fraction: float = 0.95
    fail_mode: FailMode = "closed"
