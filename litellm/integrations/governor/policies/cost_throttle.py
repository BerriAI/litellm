"""Reject-early as spend approaches a budget threshold. Fail-closed.

v1 rejects at the configured fraction of budget (e.g. >=95%); probabilistic
admit and graduated delays are a follow-up once telemetry justifies them (R10)."""

from litellm.integrations.governor.model.counters import Outcome
from litellm.integrations.governor.model.decisions import Verdict
from litellm.integrations.governor.model.policies import CostThrottlePolicyConfig
from litellm.integrations.governor.plumbing.cache import CounterStore
from litellm.integrations.governor.policies.base import AdmitContext


class CostThrottlePolicy:
    def __init__(self, config: CostThrottlePolicyConfig) -> None:
        self._config = config
        self.policy_id = config.policy_id
        self.fail_mode = config.fail_mode
        self.enabled = config.enabled

    async def admit(self, ctx: AdmitContext, store: CounterStore) -> Verdict:
        raise NotImplementedError("CostThrottlePolicy.admit lands in the adapter phase")

    async def reconcile(
        self, ctx: AdmitContext, outcome: Outcome, store: CounterStore
    ) -> None:
        raise NotImplementedError(
            "CostThrottlePolicy.reconcile lands in the adapter phase"
        )
