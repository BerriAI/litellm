"""Per-session rollup of auto-router traffic, folded one turn at a time.

The benchmarks dashboard used to answer every question by scanning
``LiteLLM_SpendLogs`` at read time, deriving each turn's meaning from window
functions over the per-request rows: which model the previous turn used, how long
a tier had been idle, how big the prefix was last time. Those are sequential
facts, and the request that produces them already knows all of them. This module
computes them once, when the turn happens, and folds the answer into a durable
per-(session, auto-router) row.

The fold is pure. ``fold_turn`` takes the session's prior state and one turn's
facts and returns the increments plus the next state, with no I/O and no clock,
so every rate and dollar formula is testable in isolation.

Dollars are computed here rather than at read time because the rates belong to
the model that served the turn, and a rollup row has already summed across
models. This is the same reason ``savings.py`` prices in the spend writer.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.router_utils.auto_router_model_naming import classify_strategy_router_model

if TYPE_CHECKING:
    from litellm.router import Router

PROMPT_CACHE_TTL_SECONDS: Mapping[str, int] = MappingProxyType(
    {"5m": 300, "1h": 3600}  # mutable-ok: a JSON object is a dict by definition
)  # mutable-ok: frozen by MappingProxyType on this line

TurnBucket = Literal["same_model", "first_visit", "return"]


class RateLookup(Protocol):
    """Per-token ``(cache_read, cache_write)`` prices for a model at a TTL."""

    def __call__(self, model: str, ttl_seconds: int) -> tuple[float, float]: ...


@dataclass(frozen=True, slots=True)
class ModelMark:
    """What a session remembers about a model it has already been served on.

    ``provisioned_replay_spend`` is the replay this model is currently charged
    for on the assumption that the turn which set it was the session's last on
    that model. A refresher fires once per idle window whether or not the caller
    ever comes back, so every use has to carry that charge until the session
    proves it returned inside the TTL, at which point the charge is withdrawn.
    """

    last_used_at: float
    provisioned_replay_spend: float


@dataclass(frozen=True, slots=True)
class SessionState:
    """The prior turns of one session on one auto-router, compressed.

    Everything ``fold_turn`` needs to classify the next turn, and nothing else;
    this is what the ``model_state`` column round-trips.
    """

    last_model: str | None
    last_turn_at: float
    model_marks: Mapping[str, ModelMark]


EMPTY_SESSION_STATE = SessionState(
    last_model=None,
    last_turn_at=0.0,
    model_marks=MappingProxyType({}),  # mutable-ok: frozen by MappingProxyType on this line
)


@dataclass(frozen=True, slots=True)
class TurnFacts:
    """One auto-routed request, as the spend writer sees it.

    ``autorouter_savings`` arrives already computed by
    ``savings.compute_autorouter_savings`` so that the benchmarks tab and the
    usage tab cannot report different savings for the same traffic; the
    counterfactual baseline is reconstructed from it rather than priced again.
    """

    model: str
    started_at: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    ephemeral_5m_tokens: int
    ephemeral_1h_tokens: int
    spend: float
    autorouter_savings: float
    has_usage: bool


@dataclass(frozen=True, slots=True)
class TurnDelta:
    """Increments one turn contributes to its session row, plus the next state.

    Every field but ``state`` is additive, so the flusher can hand them straight
    to an atomic increment upsert; that is what lets two pods writing the same
    session compose rather than overwrite each other.

    The counters are declared here and nowhere else. ``COUNTER_FIELDS`` derives
    from this declaration and the merge, the flush payload and the read query all
    build off it, so a metric added here reaches the database and the dashboard
    without a second edit. Enumerating the same names in five places is how a
    rollup field ends up written but never read.
    """

    state: SessionState
    turns: int = 0
    turns_with_usage: int = 0
    total_tokens: int = 0
    ephemeral_5m_tokens: int = 0
    ephemeral_1h_tokens: int = 0
    spend: float = 0.0
    baseline_spend: float = 0.0
    same_model_turns: int = 0
    same_model_hits: int = 0
    first_visit_turns: int = 0
    first_visit_hits: int = 0
    return_turns: int = 0
    return_hits: int = 0
    stale_return_misses: int = 0
    savable_return_misses: int = 0
    rescued_spend: float = 0.0
    replay_spend: float = 0.0


COUNTER_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(TurnDelta) if f.name != "state")


def counters_of(delta: TurnDelta) -> Mapping[str, float]:
    """The additive part of a delta, keyed the way the rollup columns are named."""
    return {name: getattr(delta, name) for name in COUNTER_FIELDS}  # mutable-ok: a fresh per-call payload


def turn_ttl_seconds(turn: TurnFacts) -> int:
    """The prompt-cache TTL this turn was written under.

    Read from the turn's own ``cache_creation`` split rather than guessed for
    the window, so a deployment mixing both TTLs is scored per request instead of
    having one regime imposed on all of it. Absent any ephemeral breakdown the
    provider default of five minutes applies; treating no evidence as the one
    hour tier would silently move every staleness verdict.
    """
    if turn.ephemeral_1h_tokens > 0 and turn.ephemeral_1h_tokens >= turn.ephemeral_5m_tokens:
        return PROMPT_CACHE_TTL_SECONDS["1h"]
    return PROMPT_CACHE_TTL_SECONDS["5m"]


def cache_rates(model: str, ttl_seconds: int) -> tuple[float, float]:
    """``(cache_read, cache_write)`` per-token costs for a model at a TTL.

    Tries the name as given and then bare, because spend rows carry models
    provider-prefixed while the cost map often keys them bare, and a single
    lookup would silently price the turn at zero. Falls open to zero rates, which
    surfaces as no warming economics rather than a raised error inside the spend
    writer.
    """
    for candidate in _pricing_candidates(model):
        try:
            info = litellm.get_model_info(model=candidate)
        except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
            verbose_proxy_logger.debug("auto_router_sessions: no model info for %s (%s)", candidate, e)
            continue
        read = float(info.get("cache_read_input_token_cost") or 0.0)
        write_5m = float(info.get("cache_creation_input_token_cost") or 0.0)
        write_1h = float(info.get("cache_creation_input_token_cost_above_1hr") or 0.0)
        return read, (write_1h or write_5m) if ttl_seconds >= PROMPT_CACHE_TTL_SECONDS["1h"] else write_5m
    return 0.0, 0.0


def _pricing_candidates(model: str) -> tuple[str, ...]:
    stripped = model.split("/", 1)[1] if "/" in model else model
    return tuple(dict.fromkeys((model, stripped)))


def _bucket(state: SessionState, turn: TurnFacts) -> TurnBucket:
    """Which of three mutually exclusive things the router did on this turn.

    The session's opening turn is a first visit to whatever tier served it, which
    is what makes the three buckets exhaustive: every turn lands in exactly one,
    and they sum to the turn count. The previous split left a session's first
    turn in none of them, so the bucket totals silently disagreed with the
    headline.
    """
    if state.last_model is None:
        return "first_visit"
    if turn.model == state.last_model:
        return "same_model"
    return "return" if turn.model in state.model_marks else "first_visit"


def fold_turn(state: SessionState, turn: TurnFacts, rates: RateLookup = cache_rates) -> TurnDelta:
    """Fold one turn into its session, returning the increments and next state.

    A turn that arrives out of order still contributes its tokens and dollars,
    because those are order-free sums, but it is left out of the classification
    and of the state: reordering it in would rewrite what "the previous model"
    means for turns already folded, and a late arrival is far likelier than a
    genuine reversal of a caller's own sequential turns.

    ``rates`` is injected so the fold can be exercised against fixed prices
    rather than whatever the cost map happens to say today.
    """
    baseline_spend = turn.spend + turn.autorouter_savings
    if turn.started_at < state.last_turn_at:
        return _unclassified(turn, baseline_spend, state)

    ttl = turn_ttl_seconds(turn)
    read_rate, write_rate = rates(turn.model, ttl)
    bucket = _bucket(state, turn)
    hit = turn.cache_read_tokens > 0
    mark = state.model_marks.get(turn.model)
    idle = turn.started_at - mark.last_used_at if mark is not None else 0.0

    stale = bucket == "return" and not hit and idle > ttl
    savable = stale and idle <= 2 * ttl
    rescued_spend = turn.cache_creation_tokens * max(write_rate - read_rate, 0.0) if savable else 0.0

    abandon_spend = (turn.cache_read_tokens + turn.cache_creation_tokens) * read_rate
    withdrawn = mark.provisioned_replay_spend if mark is not None and idle <= ttl else 0.0

    return TurnDelta(
        turns=1,
        turns_with_usage=1 if turn.has_usage else 0,
        total_tokens=turn.total_tokens,
        ephemeral_5m_tokens=turn.ephemeral_5m_tokens,
        ephemeral_1h_tokens=turn.ephemeral_1h_tokens,
        spend=turn.spend,
        baseline_spend=baseline_spend,
        same_model_turns=1 if bucket == "same_model" else 0,
        same_model_hits=1 if bucket == "same_model" and hit else 0,
        first_visit_turns=1 if bucket == "first_visit" else 0,
        first_visit_hits=1 if bucket == "first_visit" and hit else 0,
        return_turns=1 if bucket == "return" else 0,
        return_hits=1 if bucket == "return" and hit else 0,
        stale_return_misses=1 if stale else 0,
        savable_return_misses=1 if savable else 0,
        rescued_spend=rescued_spend,
        replay_spend=abandon_spend - withdrawn,
        state=SessionState(
            last_model=turn.model,
            last_turn_at=turn.started_at,
            model_marks=MappingProxyType(
                {  # mutable-ok: a JSON object is a dict by definition
                    **state.model_marks,
                    turn.model: ModelMark(last_used_at=turn.started_at, provisioned_replay_spend=abandon_spend),
                }
            ),
        ),
    )


def _unclassified(turn: TurnFacts, baseline_spend: float, state: SessionState) -> TurnDelta:
    return TurnDelta(
        state=state,
        turns=1,
        turns_with_usage=1 if turn.has_usage else 0,
        total_tokens=turn.total_tokens,
        ephemeral_5m_tokens=turn.ephemeral_5m_tokens,
        ephemeral_1h_tokens=turn.ephemeral_1h_tokens,
        spend=turn.spend,
        baseline_spend=baseline_spend,
    )


def auto_router_group_kinds(router: "Router") -> Mapping[str, str]:
    """Public alias to router kind, for every auto-router on the proxy.

    ``model_name`` is what a caller sends and what spend rows record, while the
    ``litellm_params.model`` string carries the ``auto_router/...`` discriminator
    that says it is one. Filtering turns by this mapping is the same filter the
    dashboard has always used, and it is load-bearing: the auto-router's own
    classifier sub-calls share the session but carry the judge model's group, so
    keying on the alias yields one entry per routed turn with no classifier noise.

    Derived per call rather than cached because the router gains and loses
    deployments while it runs.
    """
    return MappingProxyType(
        {  # mutable-ok: a JSON object is a dict by definition
            str(entry["model_name"]): kind
            for entry in (router.model_list or [])  # mutable-ok: a JSON object is a dict by definition
            if (model := _entry_model(entry)) is not None
            and (kind := classify_strategy_router_model(model)) is not None
        }
    )


def _entry_model(entry: Mapping[str, object]) -> str | None:
    params = entry.get("litellm_params")
    if not isinstance(params, Mapping):
        return None
    model = params.get("model")
    return model if isinstance(model, str) else None


def _ephemeral_split(usage_object: Mapping[str, object]) -> tuple[int, int]:
    """``(5m, 1h)`` cache-creation tokens, when the provider breaks them out."""
    creation = usage_object.get("cache_creation")
    if not isinstance(creation, Mapping):
        return 0, 0
    return (
        int(creation.get("ephemeral_5m_input_tokens") or 0),
        int(creation.get("ephemeral_1h_input_tokens") or 0),
    )


def turn_from_spend_payload(
    model: str,
    started_at: datetime,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    spend: float,
    autorouter_savings: float,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    usage_object: Mapping[str, object],
) -> TurnFacts:
    """One spend log payload as the fold sees it.

    The cache token counts arrive already extracted because the spend writer owns
    those readers and they have to agree with what the daily rows recorded for the
    same request.
    """
    ephemeral_5m, ephemeral_1h = _ephemeral_split(usage_object)
    return TurnFacts(
        model=model,
        started_at=as_epoch(started_at),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        ephemeral_5m_tokens=ephemeral_5m,
        ephemeral_1h_tokens=ephemeral_1h,
        spend=spend,
        autorouter_savings=autorouter_savings,
        has_usage=_reports_cache_usage(usage_object),
    )


def _reports_cache_usage(usage_object: Mapping[str, object]) -> bool:
    """Whether this turn's usage payload says anything about the prompt cache.

    Coverage answers "can we see cache behaviour here at all", so it counts the
    presence of a cache field rather than a non-zero one; a turn that genuinely
    read nothing is a miss, not a gap in reporting. A low figure means response
    logging is off, which is why it is surfaced next to the hit rate.
    """
    return "cache_read_input_tokens" in usage_object or bool(usage_object.get("prompt_tokens_details"))


def as_epoch(value: datetime) -> float:
    """Seconds since the epoch, treating a naive timestamp as UTC.

    Spend rows are written in UTC but reach here either naive or aware depending
    on the driver, and mixing the two silently shifts every idle-time comparison
    by the local offset.
    """
    return (value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)).timestamp()


class _StoredMark(BaseModel):
    last_used_at: float
    provisioned_replay_spend: float


_STORED_MARKS = TypeAdapter(dict[str, _StoredMark])


def state_to_json(state: SessionState) -> Mapping[str, Mapping[str, float]]:
    """``model_marks`` as the ``model_state`` column stores it."""
    return {  # mutable-ok: a JSON object is a dict by definition
        model: {  # mutable-ok: a JSON object is a dict by definition
            "last_used_at": mark.last_used_at,
            "provisioned_replay_spend": mark.provisioned_replay_spend,
        }
        for model, mark in state.model_marks.items()
    }


def state_column(state: SessionState) -> object:
    """``model_state`` wrapped the way prisma requires for a Json column.

    Model names contain a slash, and prisma-client-py inlines Json into a GraphQL
    document where an unquoted key containing one is a parse error, so a plain
    dict fails the whole write. Every writer goes through here so that cannot be
    rediscovered one call site at a time.
    """
    import prisma

    return prisma.Json(state_to_json(state))


def state_from_row(last_model: str | None, last_turn_at: datetime | None, model_state: object) -> SessionState:
    """Rebuild a session's state from its row.

    A row whose ``model_state`` cannot be parsed is treated as a session with no
    history rather than raising: the counters it already carries stay correct and
    the next turn simply reads as a first visit, which beats failing the spend
    write over a state blob.
    """
    try:
        marks = _STORED_MARKS.validate_python(model_state or {})  # mutable-ok: empty fallback for an absent mapping
    except ValidationError as e:
        verbose_proxy_logger.warning("auto_router_sessions: unreadable model_state, session history reset (%s)", e)
        return EMPTY_SESSION_STATE
    return SessionState(
        last_model=last_model,
        last_turn_at=as_epoch(last_turn_at) if last_turn_at is not None else 0.0,
        model_marks=MappingProxyType(
            {  # mutable-ok: a JSON object is a dict by definition
                model: ModelMark(
                    last_used_at=stored.last_used_at, provisioned_replay_spend=stored.provisioned_replay_spend
                )
                for model, stored in marks.items()
            }
        ),
    )


def merge_deltas(left: TurnDelta, right: TurnDelta) -> TurnDelta:
    """Combine two folds of the same session so a flush writes one row once.

    Counters add and the later state wins, which is what the database would have
    done had the two turns flushed separately; folding them in memory first just
    spares the round trip.
    """
    return TurnDelta(
        state=right.state,
        **{  # mutable-ok: a JSON object is a dict by definition
            name: getattr(left, name) + getattr(right, name) for name in COUNTER_FIELDS
        },  # mutable-ok: a JSON object is a dict by definition
    )
