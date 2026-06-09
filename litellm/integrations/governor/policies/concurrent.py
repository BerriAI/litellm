"""Concurrent in-flight request gauge. Fail-closed.

A concurrency gauge, not a rate: reserve on admit, release on completion, with a
per-slot TTL safety net so a crashed pod's leaked slots auto-expire. Never DELs a
key on error."""

from litellm.integrations.governor.model.counters import Outcome
from litellm.integrations.governor.model.decisions import Verdict
from litellm.integrations.governor.model.policies import ConcurrentRequestPolicyConfig
from litellm.integrations.governor.plumbing.cache import CounterStore
from litellm.integrations.governor.policies.base import AdmitContext


class ConcurrentRequestPolicy:
    def __init__(self, config: ConcurrentRequestPolicyConfig) -> None:
        self._config = config
        self.policy_id = config.policy_id
        self.fail_mode = config.fail_mode
        self.enabled = config.enabled

    async def admit(self, ctx: AdmitContext, store: CounterStore) -> Verdict:
        raise NotImplementedError(
            "ConcurrentRequestPolicy.admit lands in the adapter phase"
        )

    async def reconcile(
        self, ctx: AdmitContext, outcome: Outcome, store: CounterStore
    ) -> None:
        raise NotImplementedError(
            "ConcurrentRequestPolicy.reconcile lands in the adapter phase"
        )
