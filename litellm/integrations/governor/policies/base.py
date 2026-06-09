"""Policy contract and the admission context passed to every policy."""

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from litellm.integrations.governor.model.counters import Outcome
from litellm.integrations.governor.model.decisions import Correlation, FailMode, Verdict
from litellm.integrations.governor.model.subjects import SubjectRef
from litellm.integrations.governor.plumbing.cache import CounterStore


@dataclass(frozen=True)
class AdmitContext:
    request_id: str
    subjects: SubjectRef
    model: str
    estimated_cost: float | None
    estimated_tokens: int | None
    correlation: Correlation
    metadata: Mapping[str, str] = field(default_factory=dict)


class Policy(Protocol):
    """A policy writes only the happy path. On tier outage it raises
    :class:`TierDegraded`; the engine converts that to a verdict per
    ``fail_mode``. It receives a :class:`CounterStore` Protocol, not the concrete
    cache, so a fake store is enough to test it (R9)."""

    policy_id: str
    fail_mode: FailMode
    enabled: bool

    async def admit(self, ctx: AdmitContext, store: CounterStore) -> Verdict: ...

    async def reconcile(
        self, ctx: AdmitContext, outcome: Outcome, store: CounterStore
    ) -> None: ...
