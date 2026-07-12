"""
Main adaptive router strategy. See README.md for design overview.

One AdaptiveRouter instance per router_name. Holds in-memory caches:
- _cells:           Beta(alpha, beta) bandit posteriors per (request_type, model)
- _owner_cache:     session_key -> (owner_model, expires_at) — the first model
                    picked for a conversation owns its bandit-update slot
- _session_states:  (session_key, model) -> SessionState for incremental signal updates

Owns the AdaptiveRouterUpdateQueue used by the proxy's flusher to persist
state and session snapshots back to Postgres.

Routing is stateless per-turn (Thompson sample fresh on every call). The
owner cache is consulted only at post-call time to decide whether a turn's
signals should fire a bandit update — turns served by a different model than
the conversation's owner are skipped to avoid cross-model misattribution.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_last_user_message,
)
from litellm.router_strategy.adaptive_router.bandit import (
    BanditCell,
    apply_delta,
    initial_cell,
    pick_best,
)
from litellm.router_strategy.adaptive_router.classifier import classify_prompt
from litellm.router_strategy.adaptive_router.config import (
    ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY,
    MIN_TURNS_FOR_CLEAN_CREDIT,
    MIN_QUALITY_TIER_HEADER,
    MIN_QUALITY_TIER_METADATA_KEY,
    OWNER_CACHE_TTL_SECONDS,
)
from litellm.router_strategy.adaptive_router.signals import (
    SessionState,
    SignalDelta,
    Turn,
    advance_session_state,
    apply_signal_delta,
    detect_response_signals,
    detect_user_feedback,
    merge_signal_deltas,
)
from litellm.router_strategy.adaptive_router.update_queue import (
    AdaptiveRouterUpdateQueue,
)

# Sweep session-state cache when it exceeds this many live entries. Expired
# entries are dropped in bulk; amortizes to O(1) per insert.
_SESSION_STATE_SWEEP_THRESHOLD: int = 1024
# Same pattern for the owner cache.
_OWNER_CACHE_SWEEP_THRESHOLD: int = 1024
_FEEDBACK_CONTEXT_MAX_ENTRIES: int = 1024
from litellm.repositories.table_repositories import AdaptiveRouterStateRepository
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import (
    AdaptiveRouterConfig,
    AdaptiveRouterPreferences,
    PreRoutingHookResponse,
    RequestType,
)


def _default_prefs() -> AdaptiveRouterPreferences:
    """Tier-2 prior with no declared strengths; used when a model omits prefs."""
    return AdaptiveRouterPreferences(quality_tier=2, strengths=[])


@dataclass(frozen=True, slots=True)
class _FeedbackContext:
    model_name: str
    request_type: RequestType
    user_content: Optional[str]
    assistant_content: Optional[str]
    turn_count: int
    clean_credit_awarded: bool
    expires_at: float


class AdaptiveRouter:
    """One instance per router_name. Holds in-memory caches + the update queue."""

    def __init__(
        self,
        router_name: str,
        config: AdaptiveRouterConfig,
        model_to_prefs: Dict[str, AdaptiveRouterPreferences],
        model_to_cost: Dict[str, float],
    ) -> None:
        self.router_name = router_name
        self.config = config
        self.model_to_prefs = model_to_prefs
        self.model_to_cost = model_to_cost
        self.queue = AdaptiveRouterUpdateQueue()

        self._cells: Dict[Tuple[RequestType, str], BanditCell] = {}
        self._owner_cache: Dict[str, Tuple[str, float]] = {}
        self._session_states: Dict[Tuple[str, str], SessionState] = {}
        self._feedback_contexts: OrderedDict[str, _FeedbackContext] = OrderedDict()
        # Parallel expiry map for _session_states, same TTL as _owner_cache.
        # Evicted opportunistically in `get_or_create_session_state`.
        self._session_states_expiry: Dict[Tuple[str, str], float] = {}
        self._skipped_updates_total: int = 0
        self._feedback_attributed_total: int = 0
        self._feedback_without_context_total: int = 0
        self._cross_model_feedback_total: int = 0
        self._response_signal_updates_total: int = 0
        # Set to True once the proxy flusher has loaded persisted priors from
        # Postgres. Checked to support lazy-load on hot-reloaded routers.
        self._state_loaded: bool = False
        self._lock = asyncio.Lock()

        self._init_cold_start_cells()

    # ---- Cold-start ------------------------------------------------------

    def _init_cold_start_cells(self) -> None:
        """Populate _cells with cold-start priors for every (rt, model) combination."""
        for rt in RequestType:
            for model in self.config.available_models:
                prefs = self.model_to_prefs.get(model) or _default_prefs()
                self._cells[(rt, model)] = initial_cell(prefs, rt)

    async def load_state_from_db(self, prisma_client: Any) -> None:
        """Override cold-start cells with persisted state. Called once at startup."""
        if prisma_client is None:
            return
        try:
            rows = await AdaptiveRouterStateRepository(prisma_client).table.find_many(
                where={"router_name": self.router_name}
            )
            loaded = 0
            for row in rows:
                try:
                    rt = RequestType(row.request_type)
                except ValueError:
                    # Unknown taxonomy entry from an older/newer version. Skip.
                    continue
                if row.model_name not in self.config.available_models:
                    continue
                self._cells[(rt, row.model_name)] = BanditCell(alpha=row.alpha, beta=row.beta)
                loaded += 1
            verbose_router_logger.info(
                "AdaptiveRouter[%s]: loaded %d cells from DB",
                self.router_name,
                loaded,
            )
        except Exception as e:
            verbose_router_logger.exception(
                "AdaptiveRouter[%s]: failed to load state from DB: %s",
                self.router_name,
                e,
            )

    # ---- Pre-routing hook ------------------------------------------------

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict[str, Any],
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional[PreRoutingHookResponse]:
        """
        Plugin entry point invoked by `Router.async_pre_routing_hook` when the
        inbound `model` matches this adaptive router's `router_name`.

        Classifies the last user message, picks a logical model via the bandit,
        and stashes the chosen model on `request_kwargs["metadata"]` so the
        post-call hook can surface it as a response header.

        Routing is stateless per-turn: every call Thompson-samples fresh,
        regardless of any prior pick for the same session. Cross-turn
        attribution is enforced post-call via the owner cache (see
        `claim_or_check_owner`).
        """
        user_text = get_last_user_message(cast(List[AllMessageValues], messages or [])) or ""

        request_type = classify_prompt(user_text)
        min_quality_tier = self._extract_min_quality_tier(request_kwargs)
        chosen_model = await self.pick_model(request_type=request_type, min_quality_tier=min_quality_tier)
        verbose_router_logger.debug(
            "AdaptiveRouter[%s]: classified=%s -> chose %s",
            self.router_name,
            request_type.value,
            chosen_model,
        )

        # Relay the chosen logical model to the post-call hook, which surfaces
        # it as the `x-litellm-adaptive-router-model` response header. We use
        # `metadata` (not a top-level kwarg) so the value doesn't leak into
        # `litellm.acompletion(**input_kwargs)`.
        kwargs_metadata = request_kwargs.setdefault("metadata", {})
        if isinstance(kwargs_metadata, dict):
            kwargs_metadata[ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY] = chosen_model

        return PreRoutingHookResponse(model=chosen_model, messages=messages)

    # ---- Pick model ------------------------------------------------------

    async def pick_model(
        self,
        request_type: RequestType,
        min_quality_tier: Optional[int] = None,
    ) -> str:
        """Thompson-sample across eligible models. Stateless per-turn."""
        eligible = self._eligible_models(min_quality_tier)
        if not eligible:
            raise ValueError(f"AdaptiveRouter[{self.router_name}]: no models meet min_quality_tier={min_quality_tier}")

        cells = {m: self._cells[(request_type, m)] for m in eligible}
        costs = {m: self.model_to_cost.get(m, 0.0) for m in eligible}
        return pick_best(
            cells,
            costs,
            quality_weight=self.config.weights.quality,
            cost_weight=self.config.weights.cost,
        )

    def claim_or_check_owner(self, session_key: str, current_model: str) -> bool:
        """Resolve attribution for a turn under stateless routing.

        Returns True iff this turn should fire a bandit/state update. The
        first call for a `session_key` claims ownership for `current_model`
        and returns True. Subsequent calls return True only if the owner is
        still live AND matches `current_model`. Mismatches (a different
        model handled this turn) and expired owners both increment
        `_skipped_updates_total` and return False — no attribution.
        """
        now = time.time()
        existing = self._owner_cache.get(session_key)
        if existing is not None and existing[1] > now:
            owner_model, _ = existing
            if owner_model == current_model:
                return True
            self._skipped_updates_total += 1
            return False

        # Opportunistic bulk sweep — sessions that never come back would
        # otherwise pile up here forever. Same threshold pattern as the
        # session-state cache.
        if len(self._owner_cache) >= _OWNER_CACHE_SWEEP_THRESHOLD:
            self._evict_expired_owner_cache(now)

        # No live owner -> claim for current_model.
        self._owner_cache[session_key] = (
            current_model,
            now + OWNER_CACHE_TTL_SECONDS,
        )
        return True

    def _evict_expired_owner_cache(self, now: float) -> None:
        expired = [k for k, (_, exp) in self._owner_cache.items() if exp <= now]
        for k in expired:
            self._owner_cache.pop(k, None)

    async def get_state_snapshot(self) -> Dict[str, Any]:
        """In-memory snapshot for the introspection endpoint. Cheap; no DB hit."""
        cells = []
        for (rt, model), cell in sorted(self._cells.items(), key=lambda kv: (kv[0][0].value, kv[0][1])):
            total = cell.alpha + cell.beta
            cells.append(
                {
                    "request_type": rt.value,
                    "model": model,
                    "alpha": cell.alpha,
                    "beta": cell.beta,
                    # Net observations that have moved the posterior, excluding
                    # the cold-start prior mass. `alpha + beta` would show the
                    # initial COLD_START_MASS (e.g. 10) before any real traffic
                    # arrives, which confuses operators reading the endpoint.
                    "samples": cell.total_samples,
                    "quality_mean": cell.alpha / total if total > 0 else 0.0,
                }
            )
        queue = await self.queue.queue_size()
        now = time.time()
        owner_cache_live = sum(1 for _, exp in self._owner_cache.values() if exp > now)
        feedback_contexts_live = sum(1 for context in self._feedback_contexts.values() if context.expires_at > now)
        return {
            "router_name": self.router_name,
            "available_models": list(self.config.available_models),
            "weights": {
                "quality": self.config.weights.quality,
                "cost": self.config.weights.cost,
            },
            "model_costs": dict(self.model_to_cost),
            "cells": cells,
            "owner_cache_live": owner_cache_live,
            "skipped_updates_total": self._skipped_updates_total,
            "feedback_contexts_live": feedback_contexts_live,
            "feedback_attributed_total": self._feedback_attributed_total,
            "feedback_without_context_total": self._feedback_without_context_total,
            "cross_model_feedback_total": self._cross_model_feedback_total,
            "response_signal_updates_total": self._response_signal_updates_total,
            "queue": queue,
        }

    @staticmethod
    def _extract_min_quality_tier(
        request_kwargs: Dict[str, Any],
    ) -> Optional[int]:
        """Pull `min_quality_tier` from request headers or metadata.

        Precedence: headers (`x-litellm-min-quality-tier`) over metadata
        (`min_quality_tier`). Headers arrive lowercased from the proxy but we
        lookup case-insensitively to be safe. Unparseable values are ignored
        (treated as "not set") rather than raising — a bad header shouldn't
        fail the request.
        """
        headers = request_kwargs.get("headers") or {}
        if isinstance(headers, dict):
            for k, v in headers.items():
                if isinstance(k, str) and k.lower() == MIN_QUALITY_TIER_HEADER:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        return None

        metadata = request_kwargs.get("metadata") or {}
        if isinstance(metadata, dict):
            raw = metadata.get(MIN_QUALITY_TIER_METADATA_KEY)
            if raw is not None:
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    return None
        return None

    def _eligible_models(self, min_quality_tier: Optional[int]) -> List[str]:
        if min_quality_tier is None:
            return list(self.config.available_models)
        return [
            m
            for m in self.config.available_models
            if (self.model_to_prefs.get(m) or _default_prefs()).quality_tier >= min_quality_tier
        ]

    # ---- Session state ---------------------------------------------------

    def get_or_create_session_state(
        self,
        session_id: str,
        model_name: str,
        request_type: RequestType,
    ) -> SessionState:
        key = (session_id, model_name)
        now = time.time()

        # Opportunistic bulk sweep when the cache grows past the threshold.
        # Cheap relative to the alternative of a bounded LRU — conversations
        # naturally become inactive within OWNER_CACHE_TTL_SECONDS.
        if len(self._session_states) >= _SESSION_STATE_SWEEP_THRESHOLD:
            self._evict_expired_session_states(now)

        state = self._session_states.get(key)
        if state is None:
            state = SessionState(
                session_id=session_id,
                router_name=self.router_name,
                model_name=model_name,
                classified_type=request_type.value,
            )
            self._session_states[key] = state
        self._session_states_expiry[key] = now + OWNER_CACHE_TTL_SECONDS
        return state

    def _evict_expired_session_states(self, now: float) -> None:
        """Drop session states whose TTL has passed. O(n) but amortized O(1)
        per insert thanks to `_SESSION_STATE_SWEEP_THRESHOLD`."""
        expired = [k for k, exp in self._session_states_expiry.items() if exp <= now]
        for k in expired:
            self._session_states.pop(k, None)
            self._session_states_expiry.pop(k, None)

    async def record_turn(
        self,
        session_id: str,
        model_name: str,
        request_type: RequestType,
        turn: Turn,
    ) -> SignalDelta:
        """Attribute feedback to the previous response and response signals to the current model."""
        async with self._lock:
            now = time.time()
            while self._feedback_contexts:
                oldest_context = next(iter(self._feedback_contexts.values()))
                if oldest_context.expires_at > now:
                    break
                self._feedback_contexts.popitem(last=False)
            previous = self._feedback_contexts.pop(session_id, None)

            effective_request_type = (
                previous.request_type if previous is not None and request_type == RequestType.GENERAL else request_type
            )
            current_state = self.get_or_create_session_state(
                session_id,
                model_name,
                effective_request_type,
            )
            feedback_delta = detect_user_feedback(
                previous.user_content if previous else None,
                turn.user_content,
                turn.tool_results,
                allow_satisfaction=(
                    previous is not None
                    and not previous.clean_credit_awarded
                    and previous.turn_count + 1 >= MIN_TURNS_FOR_CLEAN_CREDIT
                ),
            )
            previous_assistant = previous.assistant_content if previous else None
            response_delta = detect_response_signals(
                previous_assistant,
                turn.assistant_content,
                current_state.tool_call_history,
                turn.tool_calls,
                turn.response_status,
            )
            states_to_persist: Dict[str, SessionState] = {model_name: current_state}
            bandit_deltas: Dict[Tuple[RequestType, str], SignalDelta] = {}

            if previous is not None:
                feedback_state = self.get_or_create_session_state(
                    session_id,
                    previous.model_name,
                    previous.request_type,
                )
                apply_signal_delta(feedback_state, feedback_delta)
                if feedback_delta.satisfaction:
                    feedback_state.clean_credit_awarded = True
                states_to_persist[previous.model_name] = feedback_state
                if feedback_delta.any_fired():
                    self._feedback_attributed_total += 1
                    if previous.model_name != model_name:
                        self._cross_model_feedback_total += 1
                bandit_deltas[(previous.request_type, previous.model_name)] = feedback_delta
            else:
                if feedback_delta.any_fired():
                    self._feedback_without_context_total += 1
                initial_failure = SignalDelta(failure=feedback_delta.failure)
                apply_signal_delta(current_state, initial_failure)
                bandit_deltas[(effective_request_type, model_name)] = initial_failure

            apply_signal_delta(current_state, response_delta)
            if self._compute_bandit_delta(response_delta) != (0.0, 0.0):
                self._response_signal_updates_total += 1
            current_key = (effective_request_type, model_name)
            bandit_deltas[current_key] = merge_signal_deltas(
                bandit_deltas.get(current_key, SignalDelta()),
                response_delta,
            )
            advance_session_state(current_state, turn)

            next_turn_count = (previous.turn_count if previous else 0) + 1
            clean_credit_awarded = bool((previous and previous.clean_credit_awarded) or feedback_delta.satisfaction)
            if len(self._feedback_contexts) >= _FEEDBACK_CONTEXT_MAX_ENTRIES:
                self._feedback_contexts.popitem(last=False)
            self._feedback_contexts[session_id] = _FeedbackContext(
                model_name=model_name,
                request_type=effective_request_type,
                user_content=turn.user_content,
                assistant_content=turn.assistant_content,
                turn_count=next_turn_count,
                clean_credit_awarded=clean_credit_awarded,
                expires_at=now + OWNER_CACHE_TTL_SECONDS,
            )

            for state_model, state in states_to_persist.items():
                await self.queue.add_session_state(
                    session_id,
                    self.router_name,
                    state_model,
                    self._persistable_session_snapshot(state),
                )

            combined_delta = SignalDelta()
            for (attribution_type, target_model), delta in bandit_deltas.items():
                combined_delta = merge_signal_deltas(combined_delta, delta)
                d_alpha, d_beta = self._compute_bandit_delta(delta)
                if d_alpha == 0 and d_beta == 0:
                    continue
                cell_key = (attribution_type, target_model)
                self._cells[cell_key] = apply_delta(
                    self._cells[cell_key],
                    d_alpha,
                    d_beta,
                )
                await self.queue.add_state_delta(
                    self.router_name,
                    attribution_type.value,
                    target_model,
                    d_alpha,
                    d_beta,
                )

            verbose_router_logger.debug(
                "AdaptiveRouter[%s]: feedback_target=%s current_model=%s delta=%s",
                self.router_name,
                previous.model_name if previous else None,
                model_name,
                combined_delta,
            )
            return combined_delta

    @staticmethod
    def _persistable_session_snapshot(state: SessionState) -> Dict[str, Any]:
        snapshot = asdict(state)
        for sensitive in (
            "last_user_content",
            "last_assistant_content",
            "tool_call_history",
            "pending_tool_calls",
        ):
            snapshot.pop(sensitive, None)
        return snapshot

    @staticmethod
    def _compute_bandit_delta(delta: SignalDelta) -> Tuple[float, float]:
        """
        Translate per-turn signal deltas into bandit-cell deltas.

        v0 mapping (UNVALIDATED — D6):
        - satisfaction               -> +1 alpha
        - misalignment, stagnation,
          disengagement, failure     -> +1 beta each
        - loop                       -> +0.5 beta (weak; could be model OR user)
        - exhaustion                 -> 0 (uptime issue, tracked separately later)
        """
        d_alpha = float(delta.satisfaction)
        d_beta = float(delta.misalignment + delta.stagnation + delta.disengagement + delta.failure) + 0.5 * delta.loop
        return d_alpha, d_beta
