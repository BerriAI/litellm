"""Built-in policy registry.

Per R7 a policy is described once by a :class:`PolicyDescriptor`; the engine,
adapter, and audit all derive their behavior from it. Adding a policy is one
descriptor plus one Policy class, never five independent enum edits.
"""

from dataclasses import dataclass
from typing import Type

from litellm.integrations.governor.model.counters import CounterKind
from litellm.integrations.governor.model.decisions import FailMode
from litellm.integrations.governor.model.policies import (
    BudgetPolicyConfig,
    ConcurrentRequestPolicyConfig,
    CostThrottlePolicyConfig,
    PolicyConfig,
    RpmPolicyConfig,
    TpmPolicyConfig,
)
from litellm.integrations.governor.policies.base import Policy
from litellm.integrations.governor.policies.budget import BudgetPolicy
from litellm.integrations.governor.policies.concurrent import ConcurrentRequestPolicy
from litellm.integrations.governor.policies.cost_throttle import CostThrottlePolicy
from litellm.integrations.governor.policies.rpm import RpmPolicy
from litellm.integrations.governor.policies.tpm import TpmPolicy


@dataclass(frozen=True)
class PolicyDescriptor:
    name: str
    config_cls: Type[PolicyConfig]
    policy_cls: Type[Policy]
    counter_kind: CounterKind
    reject_code: str
    http_status: int
    default_fail_mode: FailMode


BUILTIN_POLICIES: dict[str, PolicyDescriptor] = {
    "budget": PolicyDescriptor(
        name="budget",
        config_cls=BudgetPolicyConfig,
        policy_cls=BudgetPolicy,
        counter_kind="spend",
        reject_code="BUDGET_EXCEEDED",
        http_status=403,
        default_fail_mode="closed",
    ),
    "rpm": PolicyDescriptor(
        name="rpm",
        config_cls=RpmPolicyConfig,
        policy_cls=RpmPolicy,
        counter_kind="rpm",
        reject_code="RPM_EXCEEDED",
        http_status=429,
        default_fail_mode="open",
    ),
    "tpm": PolicyDescriptor(
        name="tpm",
        config_cls=TpmPolicyConfig,
        policy_cls=TpmPolicy,
        counter_kind="tpm",
        reject_code="TPM_EXCEEDED",
        http_status=429,
        default_fail_mode="open",
    ),
    "concurrent": PolicyDescriptor(
        name="concurrent",
        config_cls=ConcurrentRequestPolicyConfig,
        policy_cls=ConcurrentRequestPolicy,
        counter_kind="inflight",
        reject_code="CONCURRENT_REQUESTS_EXCEEDED",
        http_status=429,
        default_fail_mode="closed",
    ),
    "cost_throttle": PolicyDescriptor(
        name="cost_throttle",
        config_cls=CostThrottlePolicyConfig,
        policy_cls=CostThrottlePolicy,
        counter_kind="cost_throttle",
        reject_code="BUDGET_EXCEEDED",
        http_status=403,
        default_fail_mode="closed",
    ),
}
