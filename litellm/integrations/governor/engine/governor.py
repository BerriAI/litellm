"""The engine: owns policy iteration, degradation handling, and startup safety.

A policy writes only its happy path; the engine wraps every ``admit`` in the
TierDegraded -> verdict conversion and combines the results worst-wins. It never
lets a policy decide what an outage means.
"""

from typing import Dict, List, Sequence, Type

from litellm._logging import verbose_proxy_logger
from litellm.integrations.governor.engine.admission import combine
from litellm.integrations.governor.engine.degradation import verdict_for_degradation
from litellm.integrations.governor.model.config import GovernorV2Config
from litellm.integrations.governor.model.counters import Outcome
from litellm.integrations.governor.model.decisions import Decision
from litellm.integrations.governor.model.errors import TierDegraded
from litellm.integrations.governor.model.policies import PolicyConfig
from litellm.integrations.governor.plumbing.cache import ThreeTierCache
from litellm.integrations.governor.plumbing.clock import Clock
from litellm.integrations.governor.policies.base import AdmitContext, Policy
from litellm.integrations.governor.policies.registry import BUILTIN_POLICIES


def build_policies(configs: Sequence[PolicyConfig]) -> List[Policy]:
    by_config: Dict[Type[PolicyConfig], Type[Policy]] = {
        descriptor.config_cls: descriptor.policy_cls
        for descriptor in BUILTIN_POLICIES.values()
    }
    policies: List[Policy] = []
    for config in configs:
        policy_cls = by_config.get(type(config))
        if policy_cls is None:
            raise ValueError(f"no registered policy for config {type(config).__name__}")
        policies.append(policy_cls(config))  # type: ignore[call-arg]
    return policies


class Engine:
    def __init__(
        self,
        *,
        policies: Sequence[Policy],
        cache: ThreeTierCache,
        clock: Clock,
        config: GovernorV2Config,
    ) -> None:
        self._policies = list(policies)
        self._cache = cache
        self._clock = clock
        self._config = config

    @classmethod
    def load_policies(
        cls,
        *,
        configs: Sequence[PolicyConfig],
        cache: ThreeTierCache,
        clock: Clock,
        config: GovernorV2Config,
    ) -> "Engine":
        return cls(
            policies=build_policies(configs), cache=cache, clock=clock, config=config
        )

    @property
    def has_fail_closed(self) -> bool:
        return any(p.fail_mode == "closed" for p in self._enabled())

    def _enabled(self) -> List[Policy]:
        return [p for p in self._policies if p.enabled]

    async def verify_runtime_safety(self) -> None:
        if not self._config.require_noeviction_for_fail_closed:
            return
        has_fail_closed = self.has_fail_closed
        try:
            await self._cache.assert_safe_eviction_policy(
                has_fail_closed=has_fail_closed
            )
        except TierDegraded as td:
            if has_fail_closed:
                verbose_proxy_logger.critical(
                    "governor refusing to start: cannot confirm a safe Redis "
                    "maxmemory-policy for fail-closed budgets (%s)",
                    td.reason,
                )
                raise RuntimeError(f"governor unsafe to start: {td.reason}") from td

    async def admit(self, ctx: AdmitContext) -> Decision:
        started = self._clock.monotonic_s()
        verdicts = []
        for policy in self._enabled():
            try:
                verdicts.append(await policy.admit(ctx, self._cache))
            except TierDegraded as td:
                verdicts.append(verdict_for_degradation(policy, td))
        latency_ms = (self._clock.monotonic_s() - started) * 1000
        return combine(verdicts, request_id=ctx.request_id, latency_ms=latency_ms)

    async def reconcile(self, ctx: AdmitContext, outcome: Outcome) -> None:
        for policy in self._enabled():
            try:
                await policy.reconcile(ctx, outcome, self._cache)
            except TierDegraded as td:
                verbose_proxy_logger.error(
                    "governor reconcile degraded policy=%s reason=%s",
                    policy.policy_id,
                    td.reason,
                )
