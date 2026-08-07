"""Unit tests for the ADEPT router."""

import asyncio
import hashlib
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# AdeptTemplateRouter tests (mock the Prisma-backed store — no live DB needed)
# ---------------------------------------------------------------------------


def _make_template_router(mock_storage, conversations_threshold=10, trainer_url=None):
    from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
        AdeptTemplateRouter,
    )

    mock_router_instance = MagicMock()
    mock_router_instance.get_model_ids.return_value = ["router-id-1"]

    with patch(
        "litellm.router_strategy.adept_router.store.implementation.prisma.AdeptPrismaRepo",
        return_value=mock_storage,
    ):
        router = AdeptTemplateRouter(
            model_name="adept_router_test",
            litellm_router_instance=mock_router_instance,
            pg_url="postgresql://user:pass@localhost:5432/db",
            tag_prefix="var",
            conversations_threshold=conversations_threshold,
            trainer_url=trainer_url,
        )
    router.template_store = mock_storage
    return router


def test_adept_template_router_route_miss():
    mock_storage = AsyncMock()
    mock_storage.match_by_hash.return_value = None

    router = _make_template_router(mock_storage)
    result = asyncio.run(router.route("What is 2 + 2?"))
    assert result is None


def test_adept_template_router_route_hit():
    from litellm.router_strategy.adept_router.store.store_template import StoredTemplate

    mock_storage = AsyncMock()
    mock_storage.match_by_hash.return_value = "tmpl-abc"
    mock_storage.get_template.return_value = StoredTemplate(
        id="tmpl-abc",
        template="Get order {ID} for {EMAIL}",
        template_hash="hash-abc",
        router_id="router-id-1",
        target_model="gpt-4o",
        additional_information=None,
        created_at=None,
    )

    router = _make_template_router(mock_storage)
    result = asyncio.run(router.route("Get order ORD-123 for user@example.com"))
    assert result is not None
    assert result["target_model"] == "gpt-4o"


def test_threshold_modulo_triggers_at_multiples():
    """Trainer should be called at 5, 10, 15... but not at 7."""
    mock_storage = AsyncMock()
    mock_storage.match_by_hash.return_value = "tmpl-1"
    mock_storage.store_conversation.return_value = True
    mock_storage.store_template.return_value = "tmpl-1"

    router = _make_template_router(mock_storage, conversations_threshold=5, trainer_url="http://trainer.test")

    with patch.object(router, "_trigger_trainer", new_callable=AsyncMock) as mock_trigger:
        # count=5 -> triggers
        mock_storage.count_conversation_by_template_id.return_value = 5
        asyncio.run(router.store_conversation("prompt", "response"))
        mock_trigger.assert_awaited_once_with("tmpl-1")

        mock_trigger.reset_mock()

        # count=7 -> does not trigger
        mock_storage.count_conversation_by_template_id.return_value = 7
        asyncio.run(router.store_conversation("prompt", "response"))
        mock_trigger.assert_not_awaited()

        # count=10 -> triggers again
        mock_storage.count_conversation_by_template_id.return_value = 10
        asyncio.run(router.store_conversation("prompt", "response"))
        mock_trigger.assert_awaited_once_with("tmpl-1")


def test_trainer_url_used_in_trigger():
    """_trigger_trainer fires POST to trainer_url, skips if not set."""
    mock_storage = AsyncMock()
    router_with = _make_template_router(mock_storage, trainer_url="http://my-trainer.internal")
    router_without = _make_template_router(mock_storage, trainer_url=None)

    with patch(
        "litellm.router_strategy.adept_router.template.implementation.adept_template_router.httpx.post"
    ) as mock_post:
        asyncio.run(router_with._trigger_trainer("tmpl-xyz"))
        mock_post.assert_called_once()
        assert "tmpl-xyz" in mock_post.call_args[1]["url"]

        mock_post.reset_mock()
        asyncio.run(router_without._trigger_trainer("tmpl-xyz"))
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# System prompt isolation tests
# ---------------------------------------------------------------------------


def test_different_system_prompts_produce_different_hashes():
    """Two tools with the same XML structure but different system prompts must not collide."""
    from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
        AdeptTemplateRouter,
    )

    hash_a = AdeptTemplateRouter._hash_template("<doc></doc>", system_prompt="You are an invoice extractor.")
    hash_b = AdeptTemplateRouter._hash_template("<doc></doc>", system_prompt="You are a contract reviewer.")
    assert hash_a != hash_b


def test_same_tool_always_produces_same_hash():
    """Identical system prompt + same tag structure must always hash to the same value."""
    from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
        AdeptTemplateRouter,
    )

    system = "You are a ticket classifier."
    hash_1 = AdeptTemplateRouter._hash_template("<ticket></ticket>", system_prompt=system)
    hash_2 = AdeptTemplateRouter._hash_template("<ticket></ticket>", system_prompt=system)
    assert hash_1 == hash_2


def test_no_system_prompt_falls_back_to_user_message_hash():
    """Without a system prompt the hash is identical to hashing the masked template alone."""
    from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
        AdeptTemplateRouter,
    )

    masked = "<doc></doc>"
    expected = hashlib.sha256(masked.encode()).hexdigest()
    assert AdeptTemplateRouter._hash_template(masked, system_prompt=None) == expected
    assert AdeptTemplateRouter._hash_template(masked) == expected


# ---------------------------------------------------------------------------
# Router.py integration: detection and registration
# ---------------------------------------------------------------------------


def _make_minimal_litellm_params(**kwargs):
    from litellm.types.router import LiteLLM_Params

    return LiteLLM_Params(**kwargs)


def test_is_adept_router_deployment():
    from litellm.router import Router

    router = Router(model_list=[])
    lp = _make_minimal_litellm_params(model="adept/my_adept")
    assert router._is_adept_router_deployment(lp) is True


def test_adept_router_excluded_from_auto_router():
    from litellm.router import Router

    router = Router(model_list=[])
    lp = _make_minimal_litellm_params(model="adept/my_adept")
    assert router._is_auto_router_deployment(lp) is False


def test_adept_router_prefix_is_not_semantic_auto_router():
    from litellm.router import Router

    router = Router(model_list=[])
    lp = _make_minimal_litellm_params(model="auto_router/my_semantic_router")
    assert router._is_adept_router_deployment(lp) is False
    assert router._is_auto_router_deployment(lp) is True


def test_adept_routers_dict_exists_on_router():
    from litellm.router import Router

    router = Router(model_list=[])
    assert hasattr(router, "adept_routers")
    assert isinstance(router.adept_routers, dict)
    assert hasattr(router, "init_adept_router_deployment")
    assert callable(router.init_adept_router_deployment)


def test_init_adept_router_deployment_requires_pg_host():
    """init_adept_router_deployment raises ValueError when pg_host is missing."""
    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    deployment = Deployment(
        model_name="my_adept",
        litellm_params=LiteLLM_Params(
            model="adept/my_adept",
            adept_router_default_model="gpt-4o",
            # adept_router_pg_host intentionally omitted
        ),
        model_info=ModelInfo(),
    )

    try:
        router.init_adept_router_deployment(deployment)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "adept_router_pg_host" in str(e)


def test_init_adept_router_deployment_registers_router():
    """init_adept_router_deployment wires up an AdeptRouter with correct params."""
    from unittest.mock import patch as _patch

    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    deployment = Deployment(
        model_name="my_adept",
        litellm_params=LiteLLM_Params(
            model="adept/my_adept",
            adept_router_default_model="gpt-4o",
            adept_router_pg_host="db.internal.com",
            adept_router_pg_port=5432,
            adept_router_pg_database="adept_db",
            adept_router_pg_user="user",
            adept_router_pg_password="pass",
            adept_router_conversations_threshold=20,
            adept_router_trainer_url="http://trainer.internal",
        ),
        model_info=ModelInfo(),
    )

    mock_adept = MagicMock()
    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        return_value=mock_adept,
    ) as MockAdeptRouter:
        router.init_adept_router_deployment(deployment)

    assert "my_adept" in router.adept_routers
    call_kwargs = MockAdeptRouter.call_args[1]
    assert "postgresql://user:pass@db.internal.com:5432/adept_db" in call_kwargs["pg_url"]
    assert call_kwargs["conversations_threshold"] == 20
    assert call_kwargs["trainer_url"] == "http://trainer.internal"


# ---------------------------------------------------------------------------
# Callback registration, routing decision, URL encoding, caching
# ---------------------------------------------------------------------------


def test_callback_registered_after_init():
    """After init_adept_router_deployment, AdeptRouter must appear in the async success callbacks."""
    from unittest.mock import patch as _patch

    import litellm
    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    deployment = Deployment(
        model_name="cb_test_adept",
        litellm_params=LiteLLM_Params(
            model="adept/cb_test_adept",
            adept_router_default_model="gpt-4o",
            adept_router_pg_host="db.internal.com",
            adept_router_pg_database="adept_db",
            adept_router_pg_user="user",
            adept_router_pg_password="pass",
        ),
        model_info=ModelInfo(),
    )

    mock_adept = MagicMock()
    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        return_value=mock_adept,
    ):
        router.init_adept_router_deployment(deployment)

    assert mock_adept in litellm.callbacks


def _make_success_event_adept(model_name="adept/test", default_model="gpt-4o"):
    """An AdeptRouter with mocked template_router and seeding disabled, for callback tests."""
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = model_name
    adept.default_model = default_model
    adept.litellm_router_instance = MagicMock()
    adept.template_router = AsyncMock()
    adept._seeded = True
    return adept


def _model_response(content="output", prompt_tokens=10, completion_tokens=20, total_tokens=30):
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    response = ModelResponse(choices=[Choices(message=Message(content=content))])
    response.usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
    return response


def test_routing_decision_stored_in_conversation():
    """routed_to_slm=True is persisted in conversation additional_information."""
    import datetime

    adept = _make_success_event_adept()
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end = datetime.datetime(2024, 1, 1, 0, 0, 1)

    kwargs = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "my-slm",
        "response_cost": 0.001,
        "litellm_params": {"metadata": {"model_group": "adept/test", "adept_routed_to_slm": True}},
    }

    asyncio.run(adept.async_log_success_event(kwargs, _model_response(), start, end))

    call_args = adept.template_router.store_conversation.call_args
    assert call_args is not None
    # routed_to_slm is the last positional arg
    assert call_args[0][-1] is True


def test_routing_decision_fallback_stored():
    """routed_to_slm=False is persisted when fallback was used."""
    import datetime

    adept = _make_success_event_adept()
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end = datetime.datetime(2024, 1, 1, 0, 0, 1)

    kwargs = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4o",
        "response_cost": 0.005,
        "litellm_params": {"metadata": {"model_group": "adept/test", "adept_routed_to_slm": False}},
    }

    asyncio.run(adept.async_log_success_event(kwargs, _model_response(), start, end))

    call_args = adept.template_router.store_conversation.call_args
    assert call_args is not None
    assert call_args[0][-1] is False


def test_success_event_skips_foreign_and_untagged_requests():
    """The success callback is global, so it fires for every proxy request. It must store rows
    only for requests routed through THIS adept model: a request whose model_group is absent or
    belongs to another deployment is skipped, so non-ADEPT traffic and other ADEPT deployments
    never pollute or duplicate this store's conversations."""
    import datetime

    adept = _make_success_event_adept()
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end = datetime.datetime(2024, 1, 1, 0, 0, 1)
    base_kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"}

    asyncio.run(
        adept.async_log_success_event(
            {**base_kwargs, "litellm_params": {"metadata": {"model_group": "other-model"}}},
            _model_response(),
            start,
            end,
        )
    )
    asyncio.run(
        adept.async_log_success_event(
            {**base_kwargs, "litellm_params": {"metadata": {}}}, _model_response(), start, end
        )
    )

    adept.template_router.store_conversation.assert_not_called()


def test_pre_routing_hook_stashes_routed_to_slm_in_metadata():
    """async_pre_routing_hook records the SLM decision in the request metadata dict.

    A bare top-level request_kwargs key never reaches the logging callback, so the
    decision must live in metadata (the channel model_group already travels through).
    """
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = "adept/test"
    adept.default_model = "big-llm"
    adept.template_router = AsyncMock()
    adept._seeded = True

    messages = [{"role": "user", "content": "hello"}]

    adept.template_router.route.return_value = {"template_id": "t1", "target_model": "slm-x"}
    matched_kwargs = {"metadata": {}}
    matched_resp = asyncio.run(
        adept.async_pre_routing_hook(model="adept/test", request_kwargs=matched_kwargs, messages=messages)
    )
    assert matched_resp.model == "slm-x"
    assert matched_kwargs["metadata"]["adept_routed_to_slm"] is True

    adept.template_router.route.return_value = None
    miss_kwargs = {"metadata": {}}
    miss_resp = asyncio.run(
        adept.async_pre_routing_hook(model="adept/test", request_kwargs=miss_kwargs, messages=messages)
    )
    assert miss_resp.model == "big-llm"
    assert miss_kwargs["metadata"]["adept_routed_to_slm"] is False


def test_routed_to_slm_survives_pre_hook_to_success_event():
    """Regression: the SLM decision set in the pre-routing hook reaches
    async_log_success_event through the shared request metadata dict and is persisted.

    Models how litellm threads request metadata into litellm_params.metadata. With the old
    top-level kwargs key this handoff dropped the flag and routed_to_slm was never stored.
    """
    import datetime

    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = "adept/test"
    adept.default_model = "big-llm"
    adept.template_router = AsyncMock()
    adept._seeded = True
    adept.template_router.route.return_value = {"template_id": "t1", "target_model": "slm-x"}

    messages = [{"role": "user", "content": "hello"}]
    metadata = {"model_group": "adept/test"}
    asyncio.run(
        adept.async_pre_routing_hook(model="adept/test", request_kwargs={"metadata": metadata}, messages=messages)
    )
    assert metadata["adept_routed_to_slm"] is True

    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end = datetime.datetime(2024, 1, 1, 0, 0, 1)

    success_kwargs = {
        "messages": messages,
        "model": "slm-x",
        "litellm_params": {"metadata": metadata},
    }
    asyncio.run(adept.async_log_success_event(success_kwargs, _model_response(content="out"), start, end))

    call_args = adept.template_router.store_conversation.call_args
    assert call_args is not None
    assert call_args[0][-1] is True


def test_pg_url_special_chars_encoded():
    """Passwords with @, :, / must be percent-encoded in the PG URL."""
    from unittest.mock import patch as _patch

    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    deployment = Deployment(
        model_name="url_enc_test",
        litellm_params=LiteLLM_Params(
            model="adept/url_enc_test",
            adept_router_default_model="gpt-4o",
            adept_router_pg_host="db.host",
            adept_router_pg_database="mydb",
            adept_router_pg_user="adept_user",
            adept_router_pg_password="p@ss:w/rd",
        ),
        model_info=ModelInfo(),
    )

    captured_url = {}

    def capture_adept(model_name, default_model, litellm_router_instance, pg_url, **kwargs):
        captured_url["pg_url"] = pg_url
        return MagicMock()

    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        side_effect=capture_adept,
    ):
        router.init_adept_router_deployment(deployment)

    pg_url = captured_url["pg_url"]
    assert "p%40ss%3Aw%2Frd" in pg_url, f"Expected encoded password in URL, got: {pg_url}"
    assert "p@ss:w/rd" not in pg_url


def test_seed_config_missing_description_logs_warning():
    """_ensure_seeded warns and skips entries without a description."""
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = "adept/test"
    adept.default_model = "gpt-4o"
    adept.litellm_router_instance = MagicMock()
    adept.template_router = AsyncMock()
    adept._seed_config = [{"target_model": "my-slm"}]  # missing description
    adept._seeded = False
    adept._seed_lock = asyncio.Lock()

    with patch("litellm.router_strategy.adept_router.adept_router.verbose_router_logger") as mock_log:
        asyncio.run(adept._ensure_seeded())
        warning_calls = [str(c) for c in mock_log.warning.call_args_list]
        assert any("description" in w for w in warning_calls)

    adept.template_router.seed_template.assert_not_called()


def test_router_id_cached_after_first_call():
    """get_router_id() should call get_model_ids only once regardless of how many times it's called."""
    mock_storage = AsyncMock()
    mock_storage.match_by_hash.return_value = None

    router = _make_template_router(mock_storage)
    router._router_id_cache = None  # ensure cache is clear

    router.get_router_id()
    router.get_router_id()
    router.get_router_id()

    assert router.litellm_router_instance.get_model_ids.call_count == 1


# ---------------------------------------------------------------------------
# AdeptPrismaRepo store tests: row mapping/guards with a mocked Prisma client
# (no DB), plus a real-database integration test that runs only when a prisma
# engine and an ADEPT_TEST_DB_URL are configured.
# ---------------------------------------------------------------------------


def test_prisma_repo_row_mapping_and_guards():
    """The store maps raw rows to StoredTemplate, serializes JSON payloads, issues the right SQL,
    and rejects a conversation with no template_id, all without a live database."""
    import litellm.router_strategy.adept_router.store.implementation.prisma as prisma_mod
    from litellm.router_strategy.adept_router.store.implementation.prisma import (
        AdeptPrismaRepo,
        _CountRow,
        _IdRow,
        _TemplateRow,
    )

    repo = AdeptPrismaRepo("postgresql://u:p@localhost:5432/mockdb")
    client = MagicMock()
    client.query_raw = AsyncMock()
    client.execute_raw = AsyncMock()
    prisma_mod._CLIENTS[repo._db_url] = client  # inject a fake connected client into the per-URL registry

    client.query_raw.return_value = [_IdRow(id="tmpl-1")]
    assert asyncio.run(repo.match_by_hash("h", "r")) == "tmpl-1"
    client.query_raw.return_value = []
    assert asyncio.run(repo.match_by_hash("h", "r")) is None

    client.query_raw.return_value = [
        _TemplateRow(id="t", template="skel", router_id="r", target_model="m", additional_information={"a": 1})
    ]
    stored = asyncio.run(repo.get_template("t"))
    assert stored is not None
    assert stored.id == "t" and stored.target_model == "m"
    assert stored.additional_information == {"a": 1}
    client.query_raw.return_value = []
    assert asyncio.run(repo.get_template("missing")) is None

    assert asyncio.run(repo.store_conversation("p", "resp", "t", {"routed_to_slm": True})) is True
    assert "INSERT INTO conversations" in client.execute_raw.call_args[0][0]
    # guard: no template_id -> False, and no SQL issued for it
    client.execute_raw.reset_mock()
    assert asyncio.run(repo.store_conversation("p", "resp", None)) is False
    client.execute_raw.assert_not_called()

    client.query_raw.return_value = [_CountRow(c=3)]
    assert asyncio.run(repo.count_conversation_by_template_id("t")) == 3


def test_prisma_repo_rejects_empty_db_url():
    """A misconfigured (empty) connection URL fails fast with a clear error."""
    from litellm.router_strategy.adept_router.store.implementation.prisma import AdeptPrismaRepo

    with pytest.raises(ValueError, match="PostgreSQL connection URL"):
        AdeptPrismaRepo("")


def test_prisma_store_real_db_roundtrip():
    """Real end-to-end against a live PostgreSQL via the actual Prisma client: covers table
    creation, ON CONFLICT concurrency safety, JSON round-trip, and the counter. Skipped unless a
    prisma engine and an ADEPT_TEST_DB_URL are configured (so it runs locally / in the E2E env,
    not in the dependency-light unit CI where no database or engine is present)."""
    from litellm._uuid import uuid

    if not os.environ.get("PRISMA_QUERY_ENGINE_BINARY"):
        pytest.skip("prisma query engine not configured (set PRISMA_QUERY_ENGINE_BINARY)")
    db_url = os.environ.get("ADEPT_TEST_DB_URL")
    if not db_url:
        pytest.skip("no ADEPT_TEST_DB_URL configured")

    import litellm.router_strategy.adept_router.store.implementation.prisma as prisma_mod
    from litellm.router_strategy.adept_router.store.implementation.prisma import AdeptPrismaRepo

    repo = AdeptPrismaRepo(db_url)
    router_id = "test-router-" + uuid.uuid4().hex[:8]
    template_hash = uuid.uuid4().hex

    async def run() -> None:
        surviving = await repo.store_template(
            template_id=uuid.uuid4().hex,
            template="skeleton",
            template_hash=template_hash,
            target_model="slm-a",
            router_id=router_id,
            additional_information={"system_prompt": "sys"},
        )
        assert surviving is not None
        # A concurrent duplicate (same router_id + hash) no-ops and resolves to the same id.
        again = await repo.store_template(
            template_id=uuid.uuid4().hex,
            template="skeleton",
            template_hash=template_hash,
            target_model="",
            router_id=router_id,
        )
        assert again == surviving
        assert await repo.match_by_hash(template_hash, router_id) == surviving

        stored = await repo.get_template(surviving)
        assert stored is not None
        assert stored.target_model == "slm-a"
        assert stored.additional_information == {"system_prompt": "sys"}

        assert await repo.count_conversation_by_template_id(surviving) == 0
        assert await repo.store_conversation("p", "resp", surviving, {"routed_to_slm": True, "model": "slm-a"}) is True
        assert await repo.count_conversation_by_template_id(surviving) == 1

        client = await prisma_mod._get_client(db_url)
        await client.execute_raw("DELETE FROM conversations WHERE template_id = $1", surviving)
        await client.execute_raw("DELETE FROM templates WHERE router_id = $1", router_id)
        await client.disconnect()
        prisma_mod._CLIENTS.pop(db_url, None)

    asyncio.run(run())


def test_prisma_repo_reuses_one_client_per_url():
    """A router rebuild drops the old repo and builds a new one for the same database URL; the
    store must reuse the existing client instead of connecting a second one and orphaning the
    first (the connection-leak guard)."""
    import litellm.router_strategy.adept_router.store.implementation.prisma as prisma_mod
    from litellm.router_strategy.adept_router.store.implementation.prisma import AdeptPrismaRepo

    url = "postgresql://u:p@localhost:5432/leaktest"
    prisma_mod._CLIENTS.pop(url, None)
    connected = []

    class _FakeClient:
        async def connect(self):
            connected.append(self)

        async def execute_raw(self, *args, **kwargs):
            return 0

    with patch.object(prisma_mod, "Prisma", side_effect=lambda datasource: _FakeClient()):

        async def run():
            AdeptPrismaRepo(url)  # first router
            c1 = await prisma_mod._get_client(url)
            AdeptPrismaRepo(url)  # simulate a rebuild: a fresh repo for the same URL
            c2 = await prisma_mod._get_client(url)
            return c1, c2

        c1, c2 = asyncio.run(run())

    assert c1 is c2  # reused, not reconnected
    assert len(connected) == 1  # connected exactly once across both repos -> no leak
    prisma_mod._CLIENTS.pop(url, None)


def test_trigger_trainer_uses_httpx():
    """_trigger_trainer must use httpx.post, not requests.post."""
    mock_storage = AsyncMock()
    router = _make_template_router(mock_storage, trainer_url="http://trainer.test")

    with patch(
        "litellm.router_strategy.adept_router.template.implementation.adept_template_router.httpx.post"
    ) as mock_httpx:
        asyncio.run(router._trigger_trainer("tmpl-httpx-test"))
        mock_httpx.assert_called_once()
        call_kwargs = mock_httpx.call_args[1]
        assert "tmpl-httpx-test" in call_kwargs["url"]
        assert call_kwargs["timeout"] == 10


# ---------------------------------------------------------------------------
# Rebuild-on-change tests: editing an ADEPT deployment in the DB should
# refresh the in-memory router without requiring a proxy restart.
# ---------------------------------------------------------------------------


def _make_adept_deployment(
    model_name: str = "fin_agent",
    trainer_url: str | None = None,
    threshold: int | None = None,
    tag_prefix: str | None = None,
):
    """Helper: build a Deployment for the rebuild-on-change tests."""
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    return Deployment(
        model_name=model_name,
        litellm_params=LiteLLM_Params(
            model=f"adept/{model_name}",
            adept_router_default_model="gpt-4o",
            adept_router_pg_host="db.internal.com",
            adept_router_pg_database="adept_db",
            adept_router_pg_user="user",
            adept_router_pg_password="pass",
            adept_router_trainer_url=trainer_url,
            adept_router_conversations_threshold=threshold,
            adept_router_tag_prefix=tag_prefix,
        ),
        model_info=ModelInfo(),
    )


def test_init_adept_router_idempotent_when_params_unchanged():
    """
    Calling init twice with identical params must not rebuild the AdeptRouter —
    the second call is a no-op so the DB-sync loop doesn't churn callbacks.
    """
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from litellm.router import Router

    router = Router(model_list=[])
    deployment = _make_adept_deployment(trainer_url="http://trainer.internal", threshold=10)

    existing_mock = MagicMock()
    existing_mock.default_model = "gpt-4o"
    existing_mock.template_router = MagicMock(
        trainer_url="http://trainer.internal",
        conversations_threshold=10,
        tag_prefix="",
    )

    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        return_value=existing_mock,
    ) as MockAdeptRouter:
        router.init_adept_router_deployment(deployment)
        first_instance = router.adept_routers["fin_agent"]
        router.init_adept_router_deployment(deployment)

    assert MockAdeptRouter.call_count == 1
    assert router.adept_routers["fin_agent"] is first_instance


def test_init_adept_router_rebuilds_when_trainer_url_changes():
    """
    Editing trainer_url in the DB row must rebuild the in-memory AdeptRouter on
    the next sync tick — otherwise edits silently never take effect (the bug
    that hid 30 conversations' worth of trainer notifications).
    """
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from litellm.router import Router

    router = Router(model_list=[])

    initial_mock = MagicMock()
    initial_mock.default_model = "gpt-4o"
    initial_mock.template_router = MagicMock(trainer_url=None, conversations_threshold=10, tag_prefix="")
    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        return_value=initial_mock,
    ):
        router.init_adept_router_deployment(_make_adept_deployment(trainer_url=None))

    assert router.adept_routers["fin_agent"] is initial_mock

    rebuilt_mock = MagicMock()
    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        return_value=rebuilt_mock,
    ) as MockAdeptRouter:
        router.init_adept_router_deployment(_make_adept_deployment(trainer_url="http://trainer.internal"))

    MockAdeptRouter.assert_called_once()
    assert MockAdeptRouter.call_args[1]["trainer_url"] == "http://trainer.internal"
    assert router.adept_routers["fin_agent"] is rebuilt_mock
    assert router.adept_routers["fin_agent"] is not initial_mock
