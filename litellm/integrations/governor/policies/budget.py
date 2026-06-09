"""Window-bucketed cost budget. Fail-closed: an outage rejects (R3)."""

from litellm.integrations.governor.model.counters import Outcome
from litellm.integrations.governor.model.decisions import Verdict
from litellm.integrations.governor.model.policies import BudgetPolicyConfig
from litellm.integrations.governor.plumbing.cache import CounterStore
from litellm.integrations.governor.policies.base import AdmitContext


class BudgetPolicy:
    def __init__(self, config: BudgetPolicyConfig) -> None:
        self._config = config
        self.policy_id = config.policy_id
        self.fail_mode = config.fail_mode
        self.enabled = config.enabled

    async def admit(self, ctx: AdmitContext, store: CounterStore) -> Verdict:
        raise NotImplementedError("BudgetPolicy.admit lands in the adapter phase")

    async def reconcile(
        self, ctx: AdmitContext, outcome: Outcome, store: CounterStore
    ) -> None:
        raise NotImplementedError("BudgetPolicy.reconcile lands in the adapter phase")
