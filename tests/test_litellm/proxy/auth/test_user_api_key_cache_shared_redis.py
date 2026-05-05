"""
UserApiKeyCache (DualCache + Pydantic codec) verified through ``_user_api_key_auth_builder``.

These tests drive the real auth entry-point (``litellm/proxy/auth/user_api_key_auth.py``)
with a FastAPI ``Request`` object, round-robined across replica caches.  The single
metric tracked is ``prisma_client.get_data(table_name="combined_view")``, which is the
only DB trip taken during a virtual-key cold start.

Two topology variants:

* **Shared Redis** — replicas share one ``InMemoryCache`` wired as their ``redis_cache``.
  A cold start on any pod propagates to the shared layer; all peers read from there.
  Expected result: 1 DB call per TTL phase, regardless of replica count.

* **No shared Redis** — each replica carries in-memory only.
  Every cold start misses on every pod independently.
  Expected result: ``NUM_REPLICAS`` DB calls per TTL phase.

A mid-run ``_flush_cluster()`` simulates TTL expiry so the second phase begins cold.

**AuthFlows** is a small catalog of virtual-key shapes — each entry is what Prisma
returns from ``combined_view`` for a given product scenario.  Use the short ``label``
field for compact, greppable identification in failure output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from starlette.datastructures import URL

import litellm.proxy.proxy_server as _proxy_server_mod
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth, hash_token
from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from tests.test_litellm.proxy.auth.fake_redis_cache import FakeRedisCache

_CACHE_SIZE = 100
NUM_REPLICAS = 8
CALLS_PER_PHASE = 50
TOTAL_CALLS = CALLS_PER_PHASE * 2

# --------------------------------------------------------------------------- #
# AuthFlow catalog                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AuthFlow:
    """
    One virtual-key auth scenario: abridged ``label``, raw sk- key string, and the
    ``combined_view`` row Prisma returns on cold load.

    Use the named classmethods to build predefined scenarios:

    - ``AuthFlow.enterprise_virtual_key()`` — internal-user key, no budget
    - ``AuthFlow.budgeted_virtual_key()`` — key with ``max_budget`` set
    """

    label: str
    raw_sk_key: str
    combined_view_row: UserAPIKeyAuth

    def prisma_client(self) -> MagicMock:
        mock = MagicMock()
        mock.get_data = AsyncMock(return_value=self.combined_view_row)
        return mock

    @classmethod
    def enterprise_virtual_key(cls, *, raw: str, label: str) -> "AuthFlow":
        # internal-user virtual key — no budget, no team
        return cls(
            label=label,
            raw_sk_key=raw,
            combined_view_row=UserAPIKeyAuth(
                token=hash_token(raw),
                key_name="k",
                key_alias="a",
                user_role=LitellmUserRoles.INTERNAL_USER,
            ),
        )

    @classmethod
    def monthly_budget_virtual_key(
        cls,
        *,
        raw: str,
        max_budget: float = 50.0,
        budget_reset_at: Optional[datetime] = None,
    ) -> "AuthFlow":
        # virtual key with budget_duration="1mo" — models the monthly-budget path
        # that combined_view returns when a key carries a rolling monthly spend window
        from datetime import timezone

        reset_at = budget_reset_at or datetime(2099, 1, 1, tzinfo=timezone.utc)
        return cls(
            label=f"vk-monthly:{max_budget:g}",
            raw_sk_key=raw,
            combined_view_row=UserAPIKeyAuth(
                token=hash_token(raw),
                key_name="budget-k",
                key_alias="a",
                max_budget=max_budget,
                budget_duration="1mo",
                budget_reset_at=reset_at,
            ),
        )


# --------------------------------------------------------------------------- #
# Pod / cluster helpers                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class MockPod:
    name: str
    user_api_key_cache: UserApiKeyCache


class PodCluster:
    """
    A simulated proxy cluster whose topology is driven by ``litellm_settings``,
    mirroring what ``ProxyConfig._init_cache`` does at startup.

    ``litellm_settings.enable_redis_auth_cache: true``
        A single ``FakeRedisCache`` is created and attached as the ``redis_cache``
        of every pod's ``UserApiKeyCache`` — the same operation that
        ``user_api_key_cache.attach_redis_cache(...)`` performs in production.
        Cold starts on any pod propagate to the shared layer; peers read from there.

    ``enable_redis_auth_cache`` absent / ``false`` (default)
        Each pod gets an in-memory-only ``UserApiKeyCache``.  Cold starts do not
        propagate; every pod hits the DB independently.
    """

    def __init__(
        self,
        num_pods: int = NUM_REPLICAS,
        litellm_settings: Optional[dict] = None,
    ) -> None:
        self.general_settings: dict = litellm_settings or {}
        enable_redis = self.general_settings.get("enable_redis_auth_cache", False)

        shared: Optional[FakeRedisCache] = (
            FakeRedisCache(max_size_in_memory=_CACHE_SIZE) if enable_redis else None
        )
        self.shared_redis = shared
        self.pods: List[MockPod] = [
            MockPod(
                f"pod-{i}",
                UserApiKeyCache(
                    in_memory_cache=InMemoryCache(max_size_in_memory=_CACHE_SIZE),
                    **({"redis_cache": shared} if shared is not None else {}),  # type: ignore[arg-type]
                ),
            )
            for i in range(num_pods)
        ]

    def flush(self) -> None:
        """Simulate TTL expiry: wipe the shared layer and every pod's local memory."""
        if self.shared_redis is not None:
            self.shared_redis.flush_cache()
        for pod in self.pods:
            pod.user_api_key_cache.in_memory_cache.flush_cache()


# --------------------------------------------------------------------------- #
# Auth-builder entry point                                                     #
# --------------------------------------------------------------------------- #

_PROXY_SERVER_FIXED_ATTRS: dict = {
    "master_key": "sk-master-key",
    "general_settings": {},
    "llm_model_list": [],
    "llm_router": None,
    "open_telemetry_logger": None,
    "user_custom_auth": None,
    "litellm_proxy_admin_name": "default-proxy-admin",
}


async def run_proxy_virtual_key_auth_flow(
    *,
    flow: AuthFlow,
    prisma_client: MagicMock,
    user_api_key_cache: UserApiKeyCache,
    general_settings: Optional[dict] = None,
) -> UserAPIKeyAuth:
    """
    Invoke ``_user_api_key_auth_builder`` exactly as the FastAPI dependency does:
    a real ``Request`` object carrying a Bearer virtual key.

    ``general_settings`` should reflect the cluster topology — in particular,
    ``{"enable_redis_auth_cache": True}`` when the cluster was built with
    ``PodCluster.with_shared_redis()``.  This mirrors the proxy startup config
    that caused Redis to be attached to ``user_api_key_cache`` in the first place.

    ``user_api_key_service_logger_obj`` is patched because it is a module-level
    ``ServiceLogging`` instance that records OTEL latency — irrelevant to the
    ``combined_view`` DB call count we are measuring.

    ``get_key_object`` is **not** patched; it exercises the real cache → DB fallback
    path that the test is designed to characterize.
    """
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
    # budget_alerts is called via asyncio.create_task inside _virtual_key_max_budget_check
    proxy_logging_obj.budget_alerts = AsyncMock(return_value=None)

    proxy_attrs = {
        **_PROXY_SERVER_FIXED_ATTRS,
        "general_settings": general_settings if general_settings is not None else {},
        "prisma_client": prisma_client,
        "user_api_key_cache": user_api_key_cache,
        "proxy_logging_obj": proxy_logging_obj,
        "model_max_budget_limiter": MagicMock(),
        "jwt_handler": MagicMock(),
    }
    originals = {k: getattr(_proxy_server_mod, k, None) for k in proxy_attrs}

    request = Request(scope={"type": "http"})
    request._url = URL(url="/v1/chat/completions")

    mock_service_logger = MagicMock()
    mock_service_logger.async_service_success_hook = AsyncMock()
    mock_service_logger.async_service_failure_hook = AsyncMock()

    try:
        for k, v in proxy_attrs.items():
            setattr(_proxy_server_mod, k, v)

        with patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_service_logger_obj",
            new=mock_service_logger,
        ):
            return await _user_api_key_auth_builder(
                request=request,
                api_key=f"Bearer {flow.raw_sk_key}",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={"model": "gpt-4"},
            )
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


async def _run_round_robin_phases(
    call: Callable[[MockPod], Awaitable[object]],
    cluster: PodCluster,
) -> None:
    """
    Distribute ``TOTAL_CALLS`` across the cluster's pods in round-robin order.
    After ``CALLS_PER_PHASE`` requests the cluster is flushed to simulate TTL
    expiry, starting a fresh cold phase.

    ``call`` is a coroutine factory — it receives the target pod and must return
    an awaitable.  The caller owns any mocks and asserts on them after this
    returns.
    """
    for call_idx in range(TOTAL_CALLS):
        if call_idx == CALLS_PER_PHASE:
            cluster.flush()
        pod = cluster.pods[call_idx % len(cluster.pods)]
        await call(pod)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_shared_redis_collapses_cold_starts_to_one_db_call_per_phase():
    """
    Shared Redis: first cold start on any replica writes to the shared layer;
    all peers read from there.  One ``combined_view`` DB trip per TTL phase,
    regardless of replica count.
    """
    flow = AuthFlow.enterprise_virtual_key(raw="sk-shared-ttl-sim", label="vk-ent:shared-ttl")
    cluster = PodCluster(num_pods=NUM_REPLICAS, litellm_settings={"enable_redis_auth_cache": True})
    prisma = flow.prisma_client()

    await _run_round_robin_phases(
        lambda pod: run_proxy_virtual_key_auth_flow(
            flow=flow,
            prisma_client=prisma,
            user_api_key_cache=pod.user_api_key_cache,
            general_settings=cluster.general_settings,
        ),
        cluster,
    )

    assert prisma.get_data.await_count == 2
    for call in prisma.get_data.await_args_list:
        assert call.kwargs["table_name"] == "combined_view"
        assert call.kwargs["token"] == hash_token(flow.raw_sk_key)


@pytest.mark.asyncio
async def test_no_shared_redis_each_cold_replica_hits_db_independently():
    """
    No shared Redis: every replica starts cold on its own.  One
    ``combined_view`` DB trip per replica per TTL phase.
    """
    flow = AuthFlow.enterprise_virtual_key(raw="sk-isolated-ttl-sim", label="vk-ent:isolated-ttl")
    cluster = PodCluster(num_pods=NUM_REPLICAS)
    prisma = flow.prisma_client()

    await _run_round_robin_phases(
        lambda pod: run_proxy_virtual_key_auth_flow(
            flow=flow,
            prisma_client=prisma,
            user_api_key_cache=pod.user_api_key_cache,
            general_settings=cluster.general_settings,
        ),
        cluster,
    )

    assert prisma.get_data.await_count == NUM_REPLICAS * 2
    for call in prisma.get_data.await_args_list:
        assert call.kwargs["table_name"] == "combined_view"
        assert call.kwargs["token"] == hash_token(flow.raw_sk_key)