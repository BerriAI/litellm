import asyncio
from typing import Final

import pytest

from litellm.proxy.common_utils.registry_read_through import RegistryReadThrough


class ResyncSpy:
    def __init__(self, found: bool = True, error: Exception | None = None) -> None:
        self.found = found
        self.error = error
        self.calls: list[str] = []

    async def __call__(self, key: str) -> bool:
        self.calls.append(key)
        if self.error is not None:
            raise self.error
        return self.found


@pytest.mark.asyncio
async def test_attempt_returns_true_when_resync_finds_object():
    spy: Final = ResyncSpy(found=True)
    read_through: Final = RegistryReadThrough(resync=spy)

    assert await read_through.attempt("new-model") is True
    assert spy.calls == ["new-model"]


@pytest.mark.asyncio
async def test_attempt_found_key_is_not_negative_cached():
    spy: Final = ResyncSpy(found=True)
    read_through: Final = RegistryReadThrough(resync=spy)

    assert await read_through.attempt("new-model") is True
    assert await read_through.attempt("new-model") is True
    assert spy.calls == ["new-model", "new-model"]


@pytest.mark.asyncio
async def test_missing_key_is_negative_cached_within_ttl():
    spy: Final = ResyncSpy(found=False)
    read_through: Final = RegistryReadThrough(resync=spy, miss_ttl_seconds=60.0)

    assert await read_through.attempt("ghost-model") is False
    assert await read_through.attempt("ghost-model") is False
    assert spy.calls == ["ghost-model"]


@pytest.mark.asyncio
async def test_negative_cache_expires_and_resync_runs_again():
    spy: Final = ResyncSpy(found=False)
    read_through: Final = RegistryReadThrough(resync=spy, miss_ttl_seconds=0.05)

    assert await read_through.attempt("ghost-model") is False
    await asyncio.sleep(0.1)
    assert await read_through.attempt("ghost-model") is False
    assert spy.calls == ["ghost-model", "ghost-model"]


@pytest.mark.asyncio
async def test_resync_exception_returns_false_without_negative_caching():
    spy: Final = ResyncSpy(error=RuntimeError("db down"))
    read_through: Final = RegistryReadThrough(resync=spy)

    assert await read_through.attempt("new-model") is False
    assert await read_through.attempt("new-model") is False
    assert spy.calls == ["new-model", "new-model"]


@pytest.mark.asyncio
async def test_concurrent_attempts_for_missing_key_resync_once():
    class SlowResyncSpy(ResyncSpy):
        async def __call__(self, key: str) -> bool:
            await asyncio.sleep(0.05)
            return await super().__call__(key)

    spy: Final = SlowResyncSpy(found=False)
    read_through: Final = RegistryReadThrough(resync=spy, miss_ttl_seconds=60.0)

    results: Final = await asyncio.gather(*(read_through.attempt("ghost-model") for _ in range(5)))
    assert results == [False] * 5
    assert spy.calls == ["ghost-model"]


@pytest.mark.asyncio
async def test_distinct_keys_do_not_share_negative_cache():
    spy: Final = ResyncSpy(found=False)
    read_through: Final = RegistryReadThrough(resync=spy, miss_ttl_seconds=60.0)

    assert await read_through.attempt("ghost-a") is False
    assert await read_through.attempt("ghost-b") is False
    assert spy.calls == ["ghost-a", "ghost-b"]


@pytest.mark.asyncio
async def test_resync_budget_exhausted_blocks_resync_without_negative_caching():
    spy: Final = ResyncSpy(found=False)
    read_through: Final = RegistryReadThrough(
        resync=spy, miss_ttl_seconds=60.0, max_resyncs_per_window=2, resync_window_seconds=60.0
    )

    assert await read_through.attempt("ghost-a") is False
    assert await read_through.attempt("ghost-b") is False
    assert await read_through.attempt("ghost-c") is False
    assert spy.calls == ["ghost-a", "ghost-b"]
    assert read_through._recent_misses.get_cache("ghost-c") is None


@pytest.mark.asyncio
async def test_resync_budget_replenishes_after_window():
    spy: Final = ResyncSpy(found=True)
    read_through: Final = RegistryReadThrough(resync=spy, max_resyncs_per_window=1, resync_window_seconds=0.05)

    assert await read_through.attempt("model-a") is True
    assert await read_through.attempt("model-b") is False
    await asyncio.sleep(0.1)
    assert await read_through.attempt("model-b") is True
    assert spy.calls == ["model-a", "model-b"]


class FakeAgentRow:
    def __init__(self, agent_id: str, agent_name: str) -> None:
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.object_permission = None
        self.spend = 0.0

    def model_dump(self):
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_card_params": {"name": self.agent_name, "url": "http://db-agent"},
            "litellm_params": {},
            "object_permission": None,
            "spend": self.spend,
        }


@pytest.fixture
def clean_agent_registry():
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    original_agents: Final = list(global_agent_registry.agent_list)
    original_config_agents: Final = getattr(global_agent_registry, "config_agents", ())
    global_agent_registry.agent_list = []
    global_agent_registry.config_agents = ()
    try:
        yield global_agent_registry
    finally:
        global_agent_registry.agent_list = original_agents
        global_agent_registry.config_agents = original_config_agents


@pytest.mark.asyncio
async def test_get_agent_with_read_through_recovers_agent_created_on_sibling_replica(
    clean_agent_registry, monkeypatch
):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    agent_id: Final = "read-through-db-agent-id"
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_agentstable.find_unique = AsyncMock(
        return_value=FakeAgentRow(agent_id, "read-through-db-agent")
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    assert clean_agent_registry.get_agent_by_id(agent_id=agent_id) is None
    agent: Final = await get_agent_with_read_through(agent_id)

    assert agent is not None
    assert agent.agent_id == agent_id
    prisma_client.db.litellm_agentstable.find_unique.assert_awaited_once_with(
        where={"agent_id": agent_id},
        include={"object_permission": True},
    )


@pytest.mark.asyncio
async def test_get_agent_with_read_through_recovers_agent_by_name(clean_agent_registry, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    agent_name: Final = "read-through-db-agent-by-name"
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_agentstable.find_unique = AsyncMock(
        side_effect=[None, FakeAgentRow("read-through-name-lookup-id", agent_name)]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    agent: Final = await get_agent_with_read_through(agent_name)

    assert agent is not None
    assert agent.agent_name == agent_name
    prisma_client.db.litellm_agentstable.find_unique.assert_awaited_with(
        where={"agent_name": agent_name},
        include={"object_permission": True},
    )


@pytest.mark.asyncio
async def test_get_agent_with_read_through_returns_none_for_unknown_agent(clean_agent_registry, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    assert await get_agent_with_read_through("agent-nobody-created") is None
    assert prisma_client.db.litellm_agentstable.find_unique.await_count == 2


@pytest.mark.asyncio
async def test_resync_agents_already_registered_skips_db(clean_agent_registry, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import _resync_agents

    agent_id: Final = "read-through-dedup-agent-id"
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_agentstable.find_unique = AsyncMock(
        return_value=FakeAgentRow(agent_id, "read-through-dedup-agent")
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    assert await _resync_agents(agent_id) is True
    assert await _resync_agents(agent_id) is True
    assert prisma_client.db.litellm_agentstable.find_unique.await_count == 1
    assert len(clean_agent_registry.agent_list) == 1


class FakeGuardrailRow:
    def __init__(self, guardrail_id: str, guardrail_name: str) -> None:
        self.guardrail_id = guardrail_id
        self.guardrail_name = guardrail_name

    def __iter__(self):
        return iter(
            {
                "guardrail_id": self.guardrail_id,
                "guardrail_name": self.guardrail_name,
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "pre_call",
                    "default_on": True,
                    "blocked_words": [{"keyword": "secret", "action": "BLOCK"}],
                },
                "guardrail_info": {},
                "status": "active",
            }.items()
        )


@pytest.mark.asyncio
async def test_get_guardrail_with_read_through_recovers_guardrail_created_on_sibling_replica(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import (
        get_initialized_guardrail_with_read_through,
    )
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    guardrail_id: Final = "read-through-db-guardrail-id"
    guardrail_name: Final = "read-through-db-guardrail"
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_guardrailstable.find_first = AsyncMock(
        return_value=FakeGuardrailRow(guardrail_id, guardrail_name)
    )
    prisma_client.db.litellm_guardrailstable.find_many = AsyncMock(
        side_effect=AssertionError("full-table guardrail scan on read-through miss")
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    try:
        guardrail: Final = await get_initialized_guardrail_with_read_through(guardrail_name=guardrail_name)
        assert guardrail is not None
        assert guardrail.guardrail_name == guardrail_name
        prisma_client.db.litellm_guardrailstable.find_first.assert_awaited_once_with(
            where={"guardrail_name": guardrail_name, "status": "active"}
        )
    finally:
        IN_MEMORY_GUARDRAIL_HANDLER.delete_in_memory_guardrail(guardrail_id)


@pytest.mark.asyncio
async def test_get_guardrail_with_read_through_returns_none_for_unknown_guardrail(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import (
        get_initialized_guardrail_with_read_through,
    )

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_guardrailstable.find_first = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    assert await get_initialized_guardrail_with_read_through(guardrail_name="guardrail-nobody-created") is None


@pytest.mark.asyncio
async def test_resync_guardrails_never_loads_non_active_rows(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import _resync_guardrails

    pending_name: Final = "pending-review-guardrail"
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_guardrailstable.find_first = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    assert await _resync_guardrails(pending_name) is False
    prisma_client.db.litellm_guardrailstable.find_first.assert_awaited_once_with(
        where={"guardrail_name": pending_name, "status": "active"}
    )


@pytest.mark.asyncio
async def test_resync_guardrails_syncs_under_guardrail_reconcile_lock(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.common_utils.registry_read_through as read_through_module
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import _resync_guardrails
    from litellm.proxy.guardrails.guardrail_registry import (
        GUARDRAIL_RECONCILE_LOCK,
        IN_MEMORY_GUARDRAIL_HANDLER,
    )

    guardrail_name: Final = "lock-scope-guardrail"
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_guardrailstable.find_first = AsyncMock(
        return_value=FakeGuardrailRow("lock-scope-guardrail-id", guardrail_name)
    )
    lock_states: list[bool] = []

    def record_sync(guardrail) -> None:
        lock_states.append(GUARDRAIL_RECONCILE_LOCK.locked())

    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(IN_MEMORY_GUARDRAIL_HANDLER, "sync_guardrail_from_db", record_sync)
    monkeypatch.setattr(read_through_module, "_initialized_guardrail", lambda guardrail_name: MagicMock())

    assert await _resync_guardrails(guardrail_name) is True
    assert lock_states == [True]
    assert not GUARDRAIL_RECONCILE_LOCK.locked()


@pytest.mark.asyncio
async def test_resync_model_deployments_mutates_router_under_model_reconcile_lock(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import _resync_model_deployments

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[MagicMock()])
    router: Final = MagicMock()
    router.get_model_list.return_value = []
    lock_states: list[bool] = []

    def record_add_deployment(db_models) -> None:
        lock_states.append(proxy_server.MODEL_RECONCILE_LOCK.locked())

    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", None)
    monkeypatch.setattr(proxy_server.proxy_config, "_add_deployment", record_add_deployment)

    assert await _resync_model_deployments("lock-scope-model") is True
    assert lock_states == [True]
    assert not proxy_server.MODEL_RECONCILE_LOCK.locked()


@pytest.mark.asyncio
async def test_resync_model_deployments_respects_supported_db_objects(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import _resync_model_deployments

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(
        side_effect=AssertionError("db hit for an object type this replica does not load")
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": ["guardrails"]})

    assert await _resync_model_deployments("gated-out-model") is False


@pytest.mark.asyncio
async def test_resync_guardrails_respects_supported_db_objects(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import _resync_guardrails

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_guardrailstable.find_unique = AsyncMock(
        side_effect=AssertionError("db hit for an object type this replica does not load")
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": ["models"]})

    assert await _resync_guardrails("gated-out-guardrail") is False


@pytest.mark.asyncio
async def test_resync_agents_respects_supported_db_objects(clean_agent_registry, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import _resync_agents

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_agentstable.find_unique = AsyncMock(
        side_effect=AssertionError("db hit for an object type this replica does not load")
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": ["models"]})

    assert await _resync_agents("gated-out-agent") is False


@pytest.mark.asyncio
async def test_resync_agents_waits_for_agent_reload_and_skips_duplicate_registration(clean_agent_registry, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.agent_endpoints.agent_registry import AGENT_RECONCILE_LOCK
    from litellm.proxy.common_utils.registry_read_through import _resync_agents
    from litellm.types.agents import AgentResponse

    agent_id: Final = "reload-race-agent-id"
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_agentstable.find_unique = AsyncMock(
        side_effect=AssertionError("db hit while the agent reload held the reconcile lock")
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    async with AGENT_RECONCILE_LOCK:
        resync_task: Final = asyncio.ensure_future(_resync_agents(agent_id))
        await asyncio.sleep(0.05)
        assert not resync_task.done()
        clean_agent_registry.register_agent(
            agent_config=AgentResponse.model_validate(FakeAgentRow(agent_id, "reload-race-agent").model_dump())
        )

    assert await resync_task is True
    assert len(clean_agent_registry.agent_list) == 1
