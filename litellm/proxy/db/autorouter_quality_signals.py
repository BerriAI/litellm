"""Quality signals for the auto-router, read from per-request spend logs.

Cost tells an operator the router is cheaper; it cannot tell them the router is good. These
two signals are the cheapest honest evidence available without new instrumentation, because
both are already recorded on every request:

``escalation``  a turn moved to a costlier model than the turn before it, inside one session.
                When a caller does this they are saying the model they were given could not
                do the job. High precision, low recall: it only sees callers who *can* switch
                and bother to, so an API-only integration can report zero while suffering.

``abandonment`` the caller hung up before the stream finished. Low precision, high recall:
                it catches the giving-up that escalation structurally misses, but a dropped
                connection and "I read enough" look the same from here.

They are reported side by side rather than blended, because the two fail in opposite
directions and one number would hide which of them fired.

Both are measured for auto-routed sessions and for the operator's own directly-addressed
sessions, with the same definitions over the same table, so the two can be read against each
other. That comparison is not an experiment: the cohorts self-select, and a deployment that
pins its hardest prompts to one model and routes only the easy ones will flatter itself. It
is directional evidence, and the surface that renders it says so.

Neither signal can come from LiteLLM_AutoRouterSession. That rollup folds a session down to
per-model last-touch facts and deliberately discards turn order, and escalation is a question
about order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.router import Router

MIN_COHORT_SESSIONS: Final = 20
MIN_SESSION_ID_COVERAGE: Final = 0.8


class Turn:
    """One request, reduced to what the two signals need.

    ``escalation`` needs the model and the order; ``abandonment`` needs the disconnect flag.
    Kept as a plain object rather than a row dict so the signal functions state their inputs.
    """

    __slots__ = (
        "api_key",
        "client_disconnected",
        "has_client_session_id",
        "model",
        "router_name",
        "session_id",
        "started_at",
    )

    def __init__(
        self,
        *,
        session_id: str,
        api_key: str,
        model: str,
        started_at: float,
        client_disconnected: bool,
        router_name: str | None,
        has_client_session_id: bool,
    ) -> None:
        self.session_id = session_id
        self.api_key = api_key
        self.model = model
        self.started_at = started_at
        self.client_disconnected = client_disconnected
        self.router_name = router_name
        self.has_client_session_id = has_client_session_id


class CohortSignals:
    """What one population evidences, and how much of it there was to look at."""

    __slots__ = ("abandonment_rate_pct", "escalation_rate_pct", "sessions")

    def __init__(
        self,
        *,
        sessions: int,
        escalation_rate_pct: float | None,
        abandonment_rate_pct: float | None,
    ) -> None:
        self.sessions = sessions
        self.escalation_rate_pct = escalation_rate_pct
        self.abandonment_rate_pct = abandonment_rate_pct


def rank_models_by_cost(router: Router, models: Iterable[str]) -> Mapping[str, int]:
    """Order models cheapest-first, by what one fixed reference request would cost on each.

    Rank has to come from a request, not from a rate card: a model dearer per output token
    can be cheaper per cached token, so picking one rate to sort on orders cache-heavy
    traffic backwards. Costing a single reference request through the pricing engine leaves
    cache rates and tiered tables to the engine that already knows them. This mirrors how the
    savings baseline picks "most expensive" (litellm/router_strategy/savings_baseline.py).

    Models that cannot be priced are absent from the result rather than sorted to an end,
    because an unknown rate is not a low one; callers must treat a missing model as unrankable
    and decline to judge the move, instead of reading it as cheap.
    """
    from litellm.router_strategy.savings_baseline import Baseline, _priced

    quotes: Final = tuple((model, _priced(router, Baseline(model))) for model in dict.fromkeys(models))
    for model, quote in quotes:
        if quote is None:
            verbose_proxy_logger.debug("quality signals: cannot price %s, leaving it unranked", model)
    priced: Final = sorted((quote[0], model) for model, quote in quotes if quote is not None)
    distinct_costs: Final = tuple(dict.fromkeys(cost for cost, _ in priced))
    return MappingProxyType({model: distinct_costs.index(cost) for cost, model in priced})


def session_escalated(turns: Sequence[Turn], ranks: Mapping[str, int]) -> bool:
    """True when some turn ran on a costlier model than the turn before it.

    Only upward moves count. A move down to a cheaper model is a caller trading quality for
    price or latency deliberately, which is not evidence the router was wrong, and counting it
    as a miss would make every cost-conscious user look like a dissatisfied one.

    A pair is skipped when either side is unrankable: an unpriceable model gives no direction,
    and guessing one would invent the signal this function exists to measure.
    """
    ordered: Final = sorted(turns, key=lambda turn: turn.started_at)
    pairs: Final = (
        (ranks.get(previous.model), ranks.get(current.model)) for previous, current in zip(ordered, ordered[1:])
    )
    return any(from_rank is not None and to_rank is not None and to_rank > from_rank for from_rank, to_rank in pairs)


def could_escalate(turns: Sequence[Turn], ranks: Mapping[str, int], reachable: Iterable[str]) -> bool:
    """True when something strictly costlier than where the session *started* was available.

    Eligibility is judged from the first turn, not from the priciest model the session went on
    to use. Judging it from the maximum excludes the very sessions that escalated -- once a
    session has moved up to the ceiling, "could it have moved up?" answers itself backwards --
    which would drive the measured rate toward zero exactly as the real one rose.

    A session that opened on the most capable model it could reach had nowhere to go, so its
    silence says nothing about quality. Counting it as a satisfied session would let a
    deployment lower its escalation rate by restricting keys, which is the opposite of what
    this number should reward.
    """
    ordered: Final = sorted(turns, key=lambda turn: turn.started_at)
    opening_rank: Final = next((rank for turn in ordered if (rank := ranks.get(turn.model)) is not None), None)
    if opening_rank is None:
        return False
    return any(reachable_rank > opening_rank for model in reachable if (reachable_rank := ranks.get(model)) is not None)


def _group_by_session(turns: Iterable[Turn]) -> Mapping[str, tuple[Turn, ...]]:
    ordered: Final = sorted(turns, key=lambda turn: (turn.api_key, turn.session_id))
    return MappingProxyType(
        {key: tuple(group) for key, group in groupby(ordered, key=lambda turn: (turn.api_key, turn.session_id))}
    )


def signals_for_cohort(
    turns: Sequence[Turn],
    ranks: Mapping[str, int],
    reachable_by_key: Mapping[str, Iterable[str]],
) -> CohortSignals:
    """Both rates over the sessions that could have escalated.

    Abandonment shares escalation's denominator on purpose. The two numbers sit next to each
    other in the UI and get read as one population; computing them over different sets would
    make that reading wrong in a way nothing on screen would reveal.

    Reachability is looked up per session's own api_key, not pooled across every key in the
    cohort: keys can carry different model lists, and a shared pool would make a key that
    cannot reach a costlier model borrow one it never had, inventing escalation opportunities
    that were never actually available to it.
    """
    eligible: Final = tuple(
        session_turns
        for session_turns in _group_by_session(turns).values()
        if could_escalate(session_turns, ranks, reachable_by_key.get(session_turns[0].api_key, ()))
    )
    if not eligible:
        return CohortSignals(sessions=0, escalation_rate_pct=None, abandonment_rate_pct=None)

    escalated: Final = sum(1 for session_turns in eligible if session_escalated(session_turns, ranks))
    eligible_turns: Final = tuple(turn for session_turns in eligible for turn in session_turns)
    abandoned: Final = sum(1 for turn in eligible_turns if turn.client_disconnected)
    return CohortSignals(
        sessions=len(eligible),
        escalation_rate_pct=round(100.0 * escalated / len(eligible), 1),
        abandonment_rate_pct=round(100.0 * abandoned / len(eligible_turns), 1) if eligible_turns else None,
    )


def baseline_unavailable_reason(turns: Sequence[Turn], cohort: CohortSignals) -> str | None:
    """Why the non-routed cohort cannot be shown, or None when it can.

    Session-id coverage is checked before size because the two failures need different words:
    a deployment that never sends session ids has plenty of rows and no sessions, and telling
    it "not enough traffic" would send it looking for volume it already has. Below
    ``MIN_SESSION_ID_COVERAGE``, "sessions" in the non-routed cohort are mostly one-request
    artefacts of the fallback uuid, and any within-session signal computed over them is
    measuring the fallback, not the traffic. Below ``MIN_COHORT_SESSIONS``, one unlucky session
    moves the rate by whole points and a comparison drawn from it would read as fact.
    """
    if turns:
        with_client_id: Final = sum(1 for turn in turns if turn.has_client_session_id)
        if with_client_id / len(turns) < MIN_SESSION_ID_COVERAGE:
            return "no_session_ids"
    if cohort.sessions < MIN_COHORT_SESSIONS:
        return "insufficient_sessions"
    return None
