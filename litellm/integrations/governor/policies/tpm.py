"""Tokens-per-minute via sliding-window-counter. Fail-open: outage admits-degraded.

GCRA does not fit TPM: the output-token cost is unknown at admission, so a
sliding-window counter is cheaper than reserving the upper bound and refunding."""

from litellm.integrations.governor.model.counters import Outcome
from litellm.integrations.governor.model.decisions import Verdict
from litellm.integrations.governor.model.policies import TpmPolicyConfig
from litellm.integrations.governor.plumbing.cache import CounterStore
from litellm.integrations.governor.policies.base import AdmitContext


class TpmPolicy:
    def __init__(self, config: TpmPolicyConfig) -> None:
        self._config = config
        self.policy_id = config.policy_id
        self.fail_mode = config.fail_mode
        self.enabled = config.enabled

    async def admit(self, ctx: AdmitContext, store: CounterStore) -> Verdict:
        raise NotImplementedError("TpmPolicy.admit lands in the adapter phase")

    async def reconcile(
        self, ctx: AdmitContext, outcome: Outcome, store: CounterStore
    ) -> None:
        raise NotImplementedError("TpmPolicy.reconcile lands in the adapter phase")
