import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import pytest

from litellm.proxy.common_utils.registry_read_through import RegistryReadThrough

if TYPE_CHECKING:
    from litellm import Router
    from litellm.proxy._types import LiteLLM_ProxyModelTable
    from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
    from litellm.types.utils import CredentialItem


def nothing_loaded(_key: str) -> bool:
    return False


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
    read_through: Final = RegistryReadThrough(resync=spy, is_loaded=nothing_loaded)

    assert await read_through.attempt("new-model") is True
    assert spy.calls == ["new-model"]


@pytest.mark.asyncio
async def test_attempt_found_key_is_not_negative_cached():
    spy: Final = ResyncSpy(found=True)
    read_through: Final = RegistryReadThrough(resync=spy, is_loaded=nothing_loaded)

    assert await read_through.attempt("new-model") is True
    assert await read_through.attempt("new-model") is True
    assert spy.calls == ["new-model", "new-model"]


@pytest.mark.asyncio
async def test_missing_key_is_negative_cached_within_ttl():
    spy: Final = ResyncSpy(found=False)
    read_through: Final = RegistryReadThrough(resync=spy, is_loaded=nothing_loaded, miss_ttl_seconds=60.0)

    assert await read_through.attempt("ghost-model") is False
    assert await read_through.attempt("ghost-model") is False
    assert spy.calls == ["ghost-model"]


@pytest.mark.asyncio
async def test_negative_cache_expires_and_resync_runs_again():
    spy: Final = ResyncSpy(found=False)
    read_through: Final = RegistryReadThrough(resync=spy, is_loaded=nothing_loaded, miss_ttl_seconds=0.05)

    assert await read_through.attempt("ghost-model") is False
    await asyncio.sleep(0.1)
    assert await read_through.attempt("ghost-model") is False
    assert spy.calls == ["ghost-model", "ghost-model"]


@pytest.mark.asyncio
async def test_resync_exception_returns_false_without_negative_caching():
    spy: Final = ResyncSpy(error=RuntimeError("db down"))
    read_through: Final = RegistryReadThrough(resync=spy, is_loaded=nothing_loaded)

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
    read_through: Final = RegistryReadThrough(resync=spy, is_loaded=nothing_loaded, miss_ttl_seconds=60.0)

    results: Final = await asyncio.gather(*(read_through.attempt("ghost-model") for _ in range(5)))
    assert results == [False] * 5
    assert spy.calls == ["ghost-model"]


@pytest.mark.asyncio
async def test_distinct_keys_do_not_share_negative_cache():
    spy: Final = ResyncSpy(found=False)
    read_through: Final = RegistryReadThrough(resync=spy, is_loaded=nothing_loaded, miss_ttl_seconds=60.0)

    assert await read_through.attempt("ghost-a") is False
    assert await read_through.attempt("ghost-b") is False
    assert spy.calls == ["ghost-a", "ghost-b"]


@pytest.mark.asyncio
async def test_resync_budget_exhausted_blocks_resync_without_negative_caching():
    spy: Final = ResyncSpy(found=False)
    read_through: Final = RegistryReadThrough(
        resync=spy,
        is_loaded=nothing_loaded,
        miss_ttl_seconds=60.0,
        max_resyncs_per_window=2,
        resync_window_seconds=60.0,
    )

    assert await read_through.attempt("ghost-a") is False
    assert await read_through.attempt("ghost-b") is False
    assert await read_through.attempt("ghost-c") is False
    assert spy.calls == ["ghost-a", "ghost-b"]
    assert read_through._recent_misses.get_cache("ghost-c") is None


@pytest.mark.asyncio
async def test_requests_queued_behind_a_successful_resync_spend_no_budget():
    from unittest.mock import AsyncMock, call

    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()
    new_model_loaded: Final = asyncio.Event()

    async def gated_load(key: str) -> bool:
        entered.set()
        await release.wait()
        if key == "new-model":
            new_model_loaded.set()
        return True

    def is_loaded(key: str) -> bool:
        return key == "new-model" and new_model_loaded.is_set()

    resync: Final = AsyncMock(side_effect=gated_load)
    read_through: Final = RegistryReadThrough(
        resync=resync,
        is_loaded=is_loaded,
        max_resyncs_per_window=2,
        resync_window_seconds=60.0,
    )

    burst: Final = asyncio.gather(*(read_through.attempt("new-model") for _ in range(25)))
    await entered.wait()
    release.set()

    assert await burst == [True] * 25
    assert resync.await_args_list == [call("new-model")]
    assert await read_through.attempt("other-model") is True
    assert resync.await_args_list == [call("new-model"), call("other-model")]


@pytest.mark.asyncio
async def test_resync_budget_replenishes_after_window():
    spy: Final = ResyncSpy(found=True)
    read_through: Final = RegistryReadThrough(
        resync=spy, is_loaded=nothing_loaded, max_resyncs_per_window=1, resync_window_seconds=0.05
    )

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
async def test_get_agent_with_read_through_recovers_agent_created_on_sibling_replica(clean_agent_registry, monkeypatch):
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
        include={"object_permission": True, "identity": True},
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
        include={"object_permission": True, "identity": True},
    )


@pytest.mark.asyncio
async def test_get_agent_with_read_through_returns_none_for_unknown_agent(
    clean_agent_registry, fresh_agent_read_through, monkeypatch
):
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
    router.registry.return_value.deployments.return_value = []
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
async def test_resync_model_deployments_loads_db_credentials_before_reconciling_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    import litellm
    import litellm.proxy.proxy_server as proxy_server
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.proxy.common_utils.registry_read_through import _resync_model_deployments
    from litellm.types.utils import CredentialItem

    rows: Final = [MagicMock()]
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=rows)
    router: Final = MagicMock()
    router.registry.return_value.deployments.return_value = []
    installed: Final = MagicMock()

    async def load_credentials_from_db(prisma_client: object) -> None:
        CredentialAccessor.upsert_credentials(
            [
                CredentialItem(
                    credential_name="openai-cred",
                    credential_values={"api_key": "sk-from-db"},
                    credential_info={},
                )
            ]
        )

    def install_models(db_models: object) -> None:
        installed(db_models=db_models, credential=CredentialAccessor.get_credential_values("openai-cred"))

    monkeypatch.setattr(litellm, "credential_list", [])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", None)
    monkeypatch.setattr(proxy_server.proxy_config, "get_credentials", load_credentials_from_db)
    monkeypatch.setattr(proxy_server.proxy_config, "_add_deployment", install_models)

    assert await _resync_model_deployments("model-created-on-a-sibling-replica") is True
    installed.assert_called_once_with(db_models=rows, credential={"api_key": "sk-from-db"})


@pytest.fixture
def stored_params_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-stored-model-rows")


def _stored_model_row(model_id: str, stored_params: Mapping[str, str]) -> "LiteLLM_ProxyModelTable":
    from litellm.proxy._types import LiteLLM_ProxyModelTable
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    return LiteLLM_ProxyModelTable(
        model_id=model_id,
        model_name="gpt-failover-group",
        litellm_params={key: encrypt_value_helper(value) for key, value in stored_params.items()},
        model_info={"id": model_id},
    )


def _backup_row(model_id: str, model: str = "openai/gpt-backup") -> "LiteLLM_ProxyModelTable":
    return _stored_model_row(model_id, {"model": model, "api_key": "sk-backup"})


def _router_serving(*model_ids: str) -> "Router":
    from litellm import Router

    return Router(
        model_list=[
            {
                "model_name": "gpt-failover-group",
                "litellm_params": {"model": "openai/gpt-primary", "api_key": "sk-primary"},
                "model_info": {"id": model_id},
            }
            for model_id in model_ids
        ]
    )


def _served_ids(deployments: "Sequence[Mapping[str, Mapping[str, object]]] | None") -> frozenset[object]:
    return frozenset(deployment["model_info"]["id"] for deployment in deployments or ())


class _ProxyModelTable:
    def __init__(self, rows: "tuple[LiteLLM_ProxyModelTable, ...]") -> None:
        self.rows = rows
        self.reads = 0

    async def find_many(
        self, where: Mapping[str, Mapping[str, Sequence[str]]]
    ) -> "tuple[LiteLLM_ProxyModelTable, ...]":
        self.reads += 1
        return tuple(row for row in self.rows if row.model_id in where["model_id"]["in"])

    async def create(self, data: Mapping[str, object]) -> None:
        raise AssertionError(f"a model load wrote a row: {data}")


class _CredentialsTable:
    def __init__(self, rows: "tuple[CredentialItem, ...]") -> None:
        self.rows = rows

    async def find_many(self) -> "tuple[CredentialItem, ...]":
        return self.rows


class _Db:
    def __init__(
        self, litellm_proxymodeltable: _ProxyModelTable, litellm_credentialstable: _CredentialsTable | None = None
    ) -> None:
        self.litellm_proxymodeltable = litellm_proxymodeltable
        self.litellm_credentialstable = litellm_credentialstable or _CredentialsTable(())
        self.other_tables_read: list[str] = []

    def __getattr__(self, table_name: str) -> object:
        self.other_tables_read.append(table_name)
        raise AttributeError(table_name)


class _PrismaClientWithoutConfigRows:
    def __init__(self, db: object) -> None:
        self.db = db

    async def get_generic_data(self, key: str, value: str, table_name: str) -> None:
        return None


def _serve_db_models(monkeypatch: pytest.MonkeyPatch, db: object, router: "Router | None") -> None:
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", _PrismaClientWithoutConfigRows(db))
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", None)


@pytest.mark.asyncio
@pytest.mark.usefixtures("stored_params_salt")
async def test_resync_model_deployments_by_ids_installs_only_the_published_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import resync_model_deployments_by_ids

    router: Final = _router_serving("primary-id")
    _serve_db_models(
        monkeypatch, _Db(_ProxyModelTable((_backup_row("m-1"), _backup_row("m-2"), _backup_row("m-3")))), router
    )

    await resync_model_deployments_by_ids(("m-1", "m-2"))

    assert _served_ids(router.registry().deployments()) == {"primary-id", "m-1", "m-2"}
    assert _served_ids(proxy_server.llm_model_list) == {"primary-id", "m-1", "m-2"}
    assert not proxy_server.MODEL_RECONCILE_LOCK.locked()


@pytest.mark.asyncio
@pytest.mark.usefixtures("stored_params_salt")
async def test_resync_model_deployments_by_ids_leaves_the_router_alone_for_deleted_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import resync_model_deployments_by_ids

    router: Final = _router_serving("primary-id")
    _serve_db_models(monkeypatch, _Db(_ProxyModelTable(())), router)

    await resync_model_deployments_by_ids(("deleted-id",))

    assert _served_ids(router.registry().deployments()) == {"primary-id"}
    assert proxy_server.llm_model_list is None
    assert not proxy_server.MODEL_RECONCILE_LOCK.locked()


@pytest.mark.asyncio
@pytest.mark.usefixtures("stored_params_salt")
async def test_resync_model_deployments_by_ids_without_a_router_leaves_the_reload_to_the_debounced_resync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import resync_model_deployments_by_ids

    table: Final = _ProxyModelTable((_backup_row("m-1"),))
    db: Final = _Db(table)
    _serve_db_models(monkeypatch, db, None)

    await resync_model_deployments_by_ids(("m-1",))

    assert proxy_server.llm_router is None
    assert table.reads == 1
    assert db.other_tables_read == []


@pytest.mark.asyncio
async def test_resync_model_deployments_by_ids_respects_supported_db_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import resync_model_deployments_by_ids

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(
        side_effect=AssertionError("db hit for an object type this replica does not load")
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": ["guardrails"]})

    await resync_model_deployments_by_ids(("gated-out-id",))

    prisma_client.db.litellm_proxymodeltable.find_many.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_resync_guardrails_syncs_decrypted_litellm_params(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.common_utils.registry_read_through as read_through_module
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import _resync_guardrails
    from litellm.proxy.guardrails.guardrail_registry import (
        IN_MEMORY_GUARDRAIL_HANDLER,
        encrypt_guardrail_litellm_params,
    )

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-guardrail-test")
    encrypted_params: Final = encrypt_guardrail_litellm_params(
        {"guardrail": "generic_guardrail_api", "mode": "pre_call", "api_key": "vendor-key"}
    )
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_guardrailstable.find_first = AsyncMock(
        return_value={
            "guardrail_id": "enc-id",
            "guardrail_name": "enc-guardrail",
            "litellm_params": encrypted_params,
            "guardrail_info": {},
            "status": "active",
        }
    )
    synced: list[dict] = []
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(
        IN_MEMORY_GUARDRAIL_HANDLER, "sync_guardrail_from_db", lambda guardrail: synced.append(guardrail)
    )
    monkeypatch.setattr(read_through_module, "_initialized_guardrail", lambda guardrail_name: MagicMock())

    assert await _resync_guardrails("enc-guardrail") is True
    assert synced[0]["litellm_params"]["api_key"] == "vendor-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", ["agent-id", "Agent name"])
async def test_agent_read_through_hydrates_identity_binding(
    lookup, clean_agent_registry, fresh_agent_read_through, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.common_utils.registry_read_through import get_agent_with_read_through

    binding = {
        "agent_id": "agent-id",
        "provider": "microsoft_entra",
        "tenant_id": "tenant",
        "client_id": "client",
        "issuer": "https://login.microsoftonline.com/tenant/v2.0",
        "revision": "revision",
    }

    async def load_row(*, where, include):
        if where == {"agent_id": "Agent name"}:
            return None
        row = FakeAgentRow("agent-id", "Agent name").model_dump()
        return SimpleNamespace(model_dump=lambda: {**row, "identity": binding if include.get("identity") else None})

    prisma = MagicMock()
    prisma.db.litellm_agentstable.find_unique = AsyncMock(side_effect=load_row)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    agent = await get_agent_with_read_through(lookup)
    assert agent is not None
    assert agent.identity is not None
    assert agent.identity.model_dump(include=set(binding)) == binding
    assert clean_agent_registry.get_agent_by_id(agent_id="agent-id").identity == agent.identity


def test_model_is_loaded_matches_router_model_names_and_deployment_ids(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server
    from litellm import Router
    from litellm.proxy.common_utils.registry_read_through import _model_is_loaded

    router: Final = Router(
        model_list=[
            {
                "model_name": "loaded-model",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
                "model_info": {"id": "loaded-deployment-id"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)

    assert _model_is_loaded("loaded-model") is True
    assert _model_is_loaded("loaded-deployment-id") is True
    assert _model_is_loaded("model-created-on-a-sibling") is False

    monkeypatch.setattr(proxy_server, "llm_router", None)
    assert _model_is_loaded("loaded-model") is False


@pytest.mark.asyncio
async def test_model_read_through_answers_a_loaded_model_without_reading_the_db(monkeypatch: pytest.MonkeyPatch):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm import Router
    from litellm.proxy.common_utils.registry_read_through import model_registry_read_through

    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(side_effect=AssertionError("db read"))
    router: Final = Router(
        model_list=[
            {
                "model_name": "wired-loaded-model",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    assert await model_registry_read_through.attempt("wired-loaded-model") is True
    prisma_client.db.litellm_proxymodeltable.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_guardrail_read_through_answers_a_loaded_guardrail_without_reading_the_db(
    monkeypatch: pytest.MonkeyPatch,
):
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import guardrail_registry_read_through
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER
    from litellm.types.guardrails import Guardrail

    guardrail_id: Final = "wired-loaded-guardrail-id"
    guardrail_name: Final = "wired-loaded-guardrail"
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_guardrailstable.find_first = AsyncMock(side_effect=AssertionError("db read"))
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    IN_MEMORY_GUARDRAIL_HANDLER.sync_guardrail_from_db(
        guardrail=Guardrail(**dict(FakeGuardrailRow(guardrail_id, guardrail_name)))
    )
    try:
        assert await guardrail_registry_read_through.attempt(guardrail_name) is True
        prisma_client.db.litellm_guardrailstable.find_first.assert_not_awaited()
    finally:
        IN_MEMORY_GUARDRAIL_HANDLER.delete_in_memory_guardrail(guardrail_id)


@pytest.mark.asyncio
async def test_agent_read_through_answers_a_loaded_agent_without_reading_the_db(
    clean_agent_registry: "AgentRegistry", monkeypatch: pytest.MonkeyPatch
):
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import agent_registry_read_through
    from litellm.types.agents import AgentResponse

    monkeypatch.setattr(proxy_server, "store_model_in_db", False)
    clean_agent_registry.register_agent(
        agent_config=AgentResponse.model_validate(
            FakeAgentRow("wired-loaded-agent-id", "wired-loaded-agent").model_dump()
        )
    )

    assert await agent_registry_read_through.attempt("wired-loaded-agent-id") is True
    assert await agent_registry_read_through.attempt("wired-loaded-agent") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("router_holds_the_name_after_reconcile", [True, False])
async def test_resync_model_deployments_reconciles_every_db_row_when_the_missed_model_is_an_auto_router(
    monkeypatch: pytest.MonkeyPatch, router_holds_the_name_after_reconcile: bool
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
    from litellm.proxy.common_utils.registry_read_through import _resync_model_deployments

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-auto-router-read-through")
    router_name: Final = "auto-router-created-on-a-sibling-replica"
    row: Final = MagicMock()
    row.model_name = router_name
    row.litellm_params = {
        "model": encrypt_value_helper("auto_router/complexity-router"),
        "auto_router_config": {"model_tiers": {"SIMPLE": "simple-tier", "COMPLEX": "complex-tier"}},
    }
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[row])
    router: Final = MagicMock()
    registry: Final = router.registry.return_value
    registry.names.return_value = []
    registry.has_deployment.return_value = False
    reconciled: Final = MagicMock()

    async def reconcile_every_row(prisma_client: object, proxy_logging_obj: object) -> None:
        reconciled(prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj)
        registry.names.return_value = (
            [router_name, "simple-tier", "complex-tier"] if router_holds_the_name_after_reconcile else []
        )

    def reject_single_row_path(db_models: object) -> None:
        raise AssertionError(f"an auto router row took the single-row path: {db_models}")

    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", None)
    monkeypatch.setattr(proxy_server.proxy_config, "add_deployment", reconcile_every_row)
    monkeypatch.setattr(proxy_server.proxy_config, "_add_deployment", reject_single_row_path)

    assert await _resync_model_deployments(router_name) is router_holds_the_name_after_reconcile
    reconciled.assert_called_once_with(prisma_client=prisma_client, proxy_logging_obj=proxy_server.proxy_logging_obj)


@pytest.mark.asyncio
@pytest.mark.usefixtures("stored_params_salt")
async def test_resync_model_deployments_by_ids_reads_the_writer_not_a_lagging_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from litellm.proxy.common_utils.registry_read_through import resync_model_deployments_by_ids
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer: Final = MagicMock(name="writer_prisma")
    writer.litellm_proxymodeltable = _ProxyModelTable((_backup_row("backup-id"),))
    replica_table: Final = _ProxyModelTable(())
    replica: Final = MagicMock(name="lagging_replica_prisma")
    replica.litellm_proxymodeltable = replica_table
    router: Final = _router_serving("primary-id")
    _serve_db_models(
        monkeypatch,
        RoutingPrismaWrapper(
            writer=PrismaWrapper(original_prisma=writer, iam_token_db_auth=False),
            reader=PrismaWrapper(original_prisma=replica, iam_token_db_auth=False),
        ),
        router,
    )

    await resync_model_deployments_by_ids(("backup-id",))

    assert router.registry().has_deployment("backup-id")
    assert replica_table.reads == 0


@pytest.mark.asyncio
async def test_unknown_model_name_read_through_reads_the_replica_without_waiting_for_a_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import _resync_model_deployments
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer: Final = MagicMock(name="writer_prisma")
    writer.litellm_proxymodeltable = SimpleNamespace(find_many=AsyncMock(return_value=[]), create=AsyncMock())
    replica: Final = MagicMock(name="replica_prisma")
    replica.litellm_proxymodeltable = SimpleNamespace(find_many=AsyncMock(return_value=[]), create=AsyncMock())
    prisma_client: Final = MagicMock()
    prisma_client.db = RoutingPrismaWrapper(
        writer=PrismaWrapper(original_prisma=writer, iam_token_db_auth=False),
        reader=PrismaWrapper(original_prisma=replica, iam_token_db_auth=False),
    )
    reload_in_progress: Final = asyncio.Lock()
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(proxy_server, "MODEL_RECONCILE_LOCK", reload_in_progress)

    async with reload_in_progress:
        found: Final = await asyncio.wait_for(_resync_model_deployments("no-such-model"), timeout=2)

    assert found is False
    writer.litellm_proxymodeltable.find_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.usefixtures("stored_params_salt")
async def test_resync_model_deployments_by_ids_installs_the_row_a_concurrent_reload_left_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import resync_model_deployments_by_ids

    table: Final = _ProxyModelTable((_backup_row("m-1", model="openai/gpt-before-update"),))
    router: Final = _router_serving("primary-id")
    _serve_db_models(monkeypatch, _Db(table), router)
    reload_in_progress: Final = asyncio.Lock()
    monkeypatch.setattr(proxy_server, "MODEL_RECONCILE_LOCK", reload_in_progress)

    async with reload_in_progress:
        targeted_load: Final = asyncio.create_task(resync_model_deployments_by_ids(("m-1",)))
        for _ in range(5):
            await asyncio.sleep(0)
        assert not router.registry().has_deployment("m-1")
        table.rows = (_backup_row("m-1", model="openai/gpt-after-update"),)
    await targeted_load

    installed: Final = router.registry().deployment("m-1")
    assert installed is not None
    assert installed.litellm_params.model == "openai/gpt-after-update"


@pytest.mark.asyncio
@pytest.mark.usefixtures("stored_params_salt")
@pytest.mark.parametrize(
    ("stored_credential_name", "credentials_on_this_worker", "api_key_this_worker_uses"),
    [
        ("shared-openai", (), "sk-from-db"),
        ("shared-openai", ("another-openai",), "sk-from-db"),
        ("shared-openai", ("shared-openai",), "sk-on-this-worker"),
        (None, (), None),
    ],
)
async def test_resync_model_deployments_by_ids_reloads_credentials_only_for_one_this_worker_lacks(
    monkeypatch: pytest.MonkeyPatch,
    stored_credential_name: str | None,
    credentials_on_this_worker: tuple[str, ...],
    api_key_this_worker_uses: str | None,
) -> None:
    import litellm
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
    from litellm.proxy.common_utils.registry_read_through import resync_model_deployments_by_ids
    from litellm.types.utils import CredentialItem

    model_params: Final = {"model": "openai/gpt-backup"}
    row: Final = _stored_model_row(
        "backup-id",
        model_params if stored_credential_name is None else model_params | {"litellm_credential_name": stored_credential_name},
    )
    stored_credential: Final = CredentialItem(
        credential_name="shared-openai",
        credential_values={"api_key": encrypt_value_helper("sk-from-db")},
        credential_info={},
    )
    router: Final = _router_serving("primary-id")
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(credential_name=name, credential_values={"api_key": "sk-on-this-worker"}, credential_info={})
            for name in credentials_on_this_worker
        ],
    )
    _serve_db_models(monkeypatch, _Db(_ProxyModelTable((row,)), _CredentialsTable((stored_credential,))), router)

    await resync_model_deployments_by_ids(("backup-id",))

    credential: Final = CredentialAccessor.find_credential("shared-openai")
    assert (credential.credential_values.get("api_key") if credential else None) == api_key_this_worker_uses
    assert router.registry().has_deployment("backup-id")


@pytest.mark.asyncio
@pytest.mark.usefixtures("stored_params_salt")
async def test_resync_model_deployments_by_ids_installs_the_ordinary_rows_queued_with_an_auto_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.common_utils.registry_read_through import resync_model_deployments_by_ids

    table: Final = _ProxyModelTable(
        (
            _stored_model_row("auto-router-id", {"model": "auto_router/complexity-router"}),
            _backup_row("backup-id"),
        )
    )
    db: Final = _Db(table)
    router: Final = _router_serving("primary-id")
    _serve_db_models(monkeypatch, db, router)

    await resync_model_deployments_by_ids(("auto-router-id", "backup-id"))

    assert _served_ids(router.registry().deployments()) == {"primary-id", "backup-id"}
    assert _served_ids(proxy_server.llm_model_list) == {"primary-id", "backup-id"}
    assert table.reads == 1
    assert db.other_tables_read == []
    assert not proxy_server.MODEL_RECONCILE_LOCK.locked()
