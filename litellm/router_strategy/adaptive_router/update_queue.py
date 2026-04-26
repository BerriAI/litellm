"""
In-memory queues for adaptive router state and session updates.

Pattern follows DailySpendUpdateQueue: hot path is fully in-memory; a background
flusher task drains the aggregator and writes batches to Postgres.

Two logical queues (one class):
  1. STATE updates: increments to (router, request_type, model) bandit cell.
     Aggregator key = (router_name, request_type, model_name)
     Aggregated payload = {"delta_alpha": float, "delta_beta": float, "samples_added": int}
  2. SESSION updates: full snapshot of a session row (last-write-wins per session+router+model).
     Aggregator key = (session_id, router_name, model_name)
     Aggregated payload = the full session state dict.

Hot-path API is non-blocking and synchronous from the caller's POV (it just appends
to the in-memory aggregator). Flush is async and batched.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple

from litellm._logging import verbose_router_logger

StateKey = Tuple[str, str, str]  # (router_name, request_type, model_name)
SessionKey = Tuple[str, str, str]  # (session_id, router_name, model_name)


class AdaptiveRouterUpdateQueue:
    """
    Single class managing both state-update aggregation and session-snapshot aggregation.
    Held by the AdaptiveRouter strategy instance and started by the proxy on boot.
    """

    def __init__(self) -> None:
        self._state_agg: Dict[StateKey, Dict[str, float]] = {}
        self._session_agg: Dict[SessionKey, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._max_state_size_seen = 0
        self._max_session_size_seen = 0

    # ---- Hot-path: state delta -------------------------------------------

    async def add_state_delta(
        self,
        router_name: str,
        request_type: str,
        model_name: str,
        delta_alpha: float,
        delta_beta: float,
    ) -> None:
        """Aggregate a bandit-cell delta. Multiple deltas to the same cell sum."""
        key: StateKey = (router_name, request_type, model_name)
        async with self._lock:
            current = self._state_agg.get(key)
            if current is None:
                self._state_agg[key] = {
                    "delta_alpha": delta_alpha,
                    "delta_beta": delta_beta,
                    "samples_added": 1,
                }
            else:
                current["delta_alpha"] += delta_alpha
                current["delta_beta"] += delta_beta
                current["samples_added"] += 1
            if len(self._state_agg) > self._max_state_size_seen:
                self._max_state_size_seen = len(self._state_agg)

    # ---- Hot-path: session snapshot --------------------------------------

    async def add_session_state(
        self,
        session_id: str,
        router_name: str,
        model_name: str,
        state_dict: Dict[str, Any],
    ) -> None:
        """
        Last-write-wins per session row. The state_dict is a snapshot of the
        SessionState (signals counts + bookkeeping fields). The flusher will
        upsert this into LiteLLM_AdaptiveRouterSession.
        """
        key: SessionKey = (session_id, router_name, model_name)
        async with self._lock:
            self._session_agg[key] = state_dict
            if len(self._session_agg) > self._max_session_size_seen:
                self._max_session_size_seen = len(self._session_agg)

    # ---- Flushers (called by background task) ----------------------------

    async def flush_state_to_db(self, prisma_client: Any) -> int:
        """
        Drain state aggregator and apply to LiteLLM_AdaptiveRouterState.
        Returns number of cells flushed.
        """
        async with self._lock:
            batch = self._state_agg
            self._state_agg = {}

        if not batch:
            return 0

        # Sort keys to give deterministic write order across writers and
        # reduce the chance of cross-row deadlocks when other workers race us.
        for key in sorted(batch.keys()):
            router, rt, model = key
            payload = batch[key]
            try:
                # Atomic increment: push the delta directly into the DB so
                # concurrent flushers from multiple pods don't overwrite each
                # other. The upsert creates the row with the delta as the
                # initial value on first write, then increments on subsequent
                # writes — no read-modify-write race.
                await prisma_client.db.litellm_adaptiverouterstate.upsert(
                    where={
                        "router_name_request_type_model_name": {
                            "router_name": router,
                            "request_type": rt,
                            "model_name": model,
                        }
                    },
                    data={
                        "create": {
                            "router_name": router,
                            "request_type": rt,
                            "model_name": model,
                            "alpha": payload["delta_alpha"],
                            "beta": payload["delta_beta"],
                            "total_samples": int(payload["samples_added"]),
                        },
                        "update": {
                            "alpha": {"increment": payload["delta_alpha"]},
                            "beta": {"increment": payload["delta_beta"]},
                            "total_samples": {
                                "increment": int(payload["samples_added"])
                            },
                        },
                    },
                )
            except Exception as e:
                verbose_router_logger.exception(
                    "AdaptiveRouterUpdateQueue: failed to flush state for %s: %s",
                    key,
                    e,
                )

        return len(batch)

    async def flush_session_to_db(self, prisma_client: Any) -> int:
        """
        Drain session aggregator and upsert into LiteLLM_AdaptiveRouterSession.
        Returns number of session rows flushed.
        """
        async with self._lock:
            batch = self._session_agg
            self._session_agg = {}

        if not batch:
            return 0

        for key in sorted(batch.keys()):
            session_id, router, model = key
            payload = batch[key]
            try:
                # NOTE: Prisma client lower-cases model names, so
                # `LiteLLM_AdaptiveRouterSession` -> `litellm_adaptiveroutersession`
                # (single 's', not 'litellm_adaptiverouterssession').
                # Strip PK fields from the update payload — Prisma rejects
                # writes to fields that are part of the @@id. asdict(state)
                # always carries them, so build a separate update dict.
                update_payload = {
                    k: v
                    for k, v in payload.items()
                    if k not in ("session_id", "router_name", "model_name")
                }
                await prisma_client.db.litellm_adaptiveroutersession.upsert(
                    where={
                        "session_id_router_name_model_name": {
                            "session_id": session_id,
                            "router_name": router,
                            "model_name": model,
                        }
                    },
                    data={
                        "create": {
                            "session_id": session_id,
                            "router_name": router,
                            "model_name": model,
                            **update_payload,
                        },
                        "update": update_payload,
                    },
                )
            except Exception as e:
                verbose_router_logger.exception(
                    "AdaptiveRouterUpdateQueue: failed to flush session for %s: %s",
                    key,
                    e,
                )

        return len(batch)

    # ---- Observability ---------------------------------------------------

    async def queue_size(self) -> Dict[str, int]:
        async with self._lock:
            return {
                "state_pending": len(self._state_agg),
                "session_pending": len(self._session_agg),
                "max_state_seen": self._max_state_size_seen,
                "max_session_seen": self._max_session_size_seen,
            }
