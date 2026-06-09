"""Requests-per-minute via GCRA. Fail-open: an outage admits-degraded."""

from litellm.integrations.governor.model.counters import Outcome
from litellm.integrations.governor.model.decisions import Verdict
from litellm.integrations.governor.model.policies import RpmPolicyConfig
from litellm.integrations.governor.plumbing.cache import CounterStore
from litellm.integrations.governor.policies.base import AdmitContext


class RpmPolicy:
    def __init__(self, config: RpmPolicyConfig) -> None:
        self._config = config
        self.policy_id = config.policy_id
        self.fail_mode = config.fail_mode
        self.enabled = config.enabled

    async def admit(self, ctx: AdmitContext, store: CounterStore) -> Verdict:
        raise NotImplementedError("RpmPolicy.admit lands in the adapter phase")

    async def reconcile(
        self, ctx: AdmitContext, outcome: Outcome, store: CounterStore
    ) -> None:
        raise NotImplementedError("RpmPolicy.reconcile lands in the adapter phase")
