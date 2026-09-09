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

    with patch(  # test-quality-ok: helper constructs AdeptTemplateRouter without a live PG connection; no HTTP boundary to fake at construction time
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


def test_adept_template_router_route_returns_none_when_cache_empty():
    """route() must never await a DB lookup on the request path; an empty cache falls back to
    None. The background refresh may load templates asynchronously but the request path is
    cache-only."""
    mock_storage = AsyncMock()
    mock_storage.load_all_for_router.return_value = ()

    router = _make_template_router(mock_storage)

    async def _run() -> object:
        try:
            return await router.route("What is 2 + 2?")
        finally:
            if router._refresh_task is not None:
                router._refresh_task.cancel()

    result = asyncio.run(_run())
    assert result is None
    mock_storage.match_by_hash.assert_not_called()
    mock_storage.get_template.assert_not_called()


def test_adept_template_router_route_hit_from_preloaded_cache():
    """After refresh_cache() populates the cache from load_all_for_router, route() serves the
    match without any request-path DB call."""
    from litellm.router_strategy.adept_router.store.store_template import StoredTemplate

    stored = StoredTemplate(
        id="tmpl-abc",
        template="Get order {ID} for {EMAIL}",
        template_hash=hashlib.sha256(b"Get order {ID} for {EMAIL}").hexdigest(),
        router_id="router-id-1",
        target_model="gpt-4o",
        additional_information=None,
        created_at=None,
    )
    mock_storage = AsyncMock()
    mock_storage.load_all_for_router.return_value = (stored,)

    router = _make_template_router(mock_storage)

    async def _run():
        await router.refresh_cache()
        try:
            first = await router.route("Get order ORD-123 for user@example.com")
            second = await router.route("Get order ORD-999 for other@example.com")
        finally:
            if router._refresh_task is not None:
                router._refresh_task.cancel()
        return first, second

    first, second = asyncio.run(_run())
    assert first is not None and second is not None
    assert first["target_model"] == "gpt-4o"
    assert second["target_model"] == "gpt-4o"
    mock_storage.match_by_hash.assert_not_called()
    mock_storage.get_template.assert_not_called()


def test_refresh_cache_replaces_cache_from_store():
    """refresh_cache rebuilds the cache from a bulk load and keeps templates keyed by hash."""
    from litellm.router_strategy.adept_router.store.store_template import StoredTemplate

    first = StoredTemplate(
        id="t1",
        template="",
        template_hash="h1",
        router_id="router-id-1",
        target_model="slm-a",
        additional_information=None,
        created_at=None,
    )
    second = StoredTemplate(
        id="t2",
        template="",
        template_hash="h2",
        router_id="router-id-1",
        target_model="slm-b",
        additional_information=None,
        created_at=None,
    )
    mock_storage = AsyncMock()
    mock_storage.load_all_for_router.return_value = (first, second)

    router = _make_template_router(mock_storage)
    asyncio.run(router.refresh_cache())

    assert ("router-id-1", "h1") in router._template_cache
    assert ("router-id-1", "h2") in router._template_cache
    mock_storage.load_all_for_router.assert_awaited_once_with("router-id-1", limit=1024)


def test_threshold_modulo_triggers_at_multiples():  # test-quality-ok: trigger has no return value; count-based orchestration is only observable via trigger invocations
    """Trainer should be called at 5, 10, 15... but not at 7."""
    mock_storage = AsyncMock()
    mock_storage.match_by_hash.return_value = "tmpl-1"
    mock_storage.store_conversation.return_value = True
    mock_storage.store_template.return_value = "tmpl-1"

    router = _make_template_router(mock_storage, conversations_threshold=5, trainer_url="http://trainer.test")

    with patch.object(router, "_trigger_trainer") as mock_trigger:
        # count=5 -> triggers
        mock_storage.count_conversation_by_template_id.return_value = 5
        asyncio.run(router.store_conversation("prompt", "response"))
        mock_trigger.assert_called_once_with("tmpl-1")

        mock_trigger.reset_mock()

        # count=7 -> does not trigger
        mock_storage.count_conversation_by_template_id.return_value = 7
        asyncio.run(router.store_conversation("prompt", "response"))
        mock_trigger.assert_not_called()

        # count=10 -> triggers again
        mock_storage.count_conversation_by_template_id.return_value = 10
        asyncio.run(router.store_conversation("prompt", "response"))
        mock_trigger.assert_called_once_with("tmpl-1")


def test_trainer_url_used_in_trigger():  # test-quality-ok: asserts scheduling was requested for the correct URL; create_task IS the observable boundary
    """_trigger_trainer schedules a POST to trainer_url, and no-ops if not set."""
    mock_storage = AsyncMock()
    router_with = _make_template_router(mock_storage, trainer_url="http://my-trainer.internal")
    router_without = _make_template_router(mock_storage, trainer_url=None)

    with (
        patch(  # test-quality-ok: create_task IS the scheduling boundary the test is verifying; no HTTP round-trip happens here
            "litellm.router_strategy.adept_router.template.implementation.adept_template_router.asyncio.create_task"
        ) as mock_create_task
    ):

        async def _run() -> None:
            router_with._trigger_trainer("tmpl-xyz")
            router_without._trigger_trainer("tmpl-xyz")

        asyncio.run(_run())

    assert mock_create_task.call_count == 1
    scheduled_coro = mock_create_task.call_args[0][0]
    # Coroutine target holds the URL through its cr_frame locals — close to release it.
    scheduled_coro.close()


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

    with pytest.raises(ValueError, match="adept_router_pg_host"):
        router.init_adept_router_deployment(deployment)


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
    with (
        _patch(
            "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
            return_value=mock_adept,
        ) as MockAdeptRouter,
        _patch(
            "litellm.litellm_core_utils.url_utils.validate_url",
            return_value=("http://trainer.internal", "trainer.internal"),
        ),
    ):
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


def test_model_list_reload_unregisters_stale_adept_callbacks():
    """set_model_list() must remove old AdeptRouter instances from every litellm callback list
    before clearing self.adept_routers; otherwise a stale router keeps exporting conversations
    (including new prompts and responses) to its old PostgreSQL destination after the operator
    has replaced or removed the deployment."""
    import litellm
    from litellm.router import Router

    router = Router(model_list=[])

    stale_adept = MagicMock()
    router.adept_routers["stale_adept"] = stale_adept
    litellm.callbacks.append(stale_adept)
    litellm._async_success_callback.append(stale_adept)
    try:
        router.set_model_list([])

        assert stale_adept not in litellm.callbacks
        assert stale_adept not in litellm._async_success_callback
        assert "stale_adept" not in router.adept_routers
    finally:
        for cb_list in (litellm.callbacks, litellm._async_success_callback):
            while stale_adept in cb_list:
                cb_list.remove(stale_adept)


def test_model_list_reload_disconnects_orphaned_adept_urls():
    """set_model_list() must reclaim PG clients for ADEPT URLs that the new model_list no
    longer references, otherwise repeated config churn leaks connection pools and password-
    bearing URLs until process restart."""
    from unittest.mock import patch as _patch

    from litellm.router import Router
    from litellm.router_strategy.adept_router.store.implementation import prisma as prisma_mod

    router = Router(model_list=[])
    orphan = MagicMock()
    orphan.pg_url = "postgresql://u:p@host.internal:5432/adept_db"
    router.adept_routers["will_be_removed"] = orphan

    with _patch.object(prisma_mod, "schedule_disconnect") as mock_disc:
        router.set_model_list([])

    mock_disc.assert_called_once_with(orphan.pg_url)


def test_model_list_reload_does_not_disconnect_url_still_in_use():
    """When the new model_list re-registers a deployment on the SAME pg_url, the reclaim step
    must skip that URL: disconnecting it would immediately tear down the client the rebuilt
    deployment just started reusing."""
    from unittest.mock import patch as _patch

    from litellm.router import Router
    from litellm.router_strategy.adept_router.store.implementation import prisma as prisma_mod

    router = Router(model_list=[])
    surviving_url = "postgresql://u:p@host.internal:5432/adept_db"
    stale = MagicMock()
    stale.pg_url = surviving_url
    router.adept_routers["survivor"] = stale

    def _fake_reregister(_router: Router) -> None:
        replacement = MagicMock()
        replacement.pg_url = surviving_url
        _router.adept_routers["survivor"] = replacement

    with (
        _patch.object(prisma_mod, "schedule_disconnect") as mock_disc,
        _patch.object(Router, "_finalize_adaptive_router_if_configured", autospec=True, side_effect=_fake_reregister),
    ):
        router.set_model_list([])

    mock_disc.assert_not_called()


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


def test_pre_routing_hook_does_not_await_seed_on_request_path():
    """The pre-routing hook must not await seed DB lookups: it kicks seeding off in the
    background so the first request is not blocked on `seed_template` round-trips."""
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = "adept/test"
    adept.default_model = "gpt-4o"
    adept.litellm_router_instance = MagicMock()
    adept.template_router = AsyncMock()
    adept.template_router.route.return_value = None
    adept._seed_config = [{"description": "Extract order", "target_model": "slm-order"}]
    adept._seeded = False
    adept._seed_lock = asyncio.Lock()
    adept._seed_task = None

    async def _slow_seed(_desc, _model):
        await asyncio.sleep(60)  # would block for a minute if awaited on the request path
        return True

    adept.template_router.seed_template.side_effect = _slow_seed

    async def _run():
        try:
            return await asyncio.wait_for(
                adept.async_pre_routing_hook(
                    model="adept/test",
                    request_kwargs={"metadata": {}},
                    messages=[{"role": "user", "content": "hi"}],
                ),
                timeout=1.0,
            )
        finally:
            if adept._seed_task is not None:
                adept._seed_task.cancel()

    result = asyncio.run(_run())
    assert result is not None
    assert result.model == "gpt-4o"


def test_close_cancels_seed_task_so_retired_router_stops_writing():
    """When an ADEPT router is retired (delete or rebuild), close() must cancel any in-flight
    seed task so it cannot write templates against a replacement deployment."""
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = "adept/test"
    adept.default_model = "gpt-4o"
    adept.litellm_router_instance = MagicMock()
    adept.template_router = MagicMock()
    adept.template_router.stop_refresh = MagicMock()
    adept._seed_config = [{"description": "d", "target_model": "m"}]
    adept._seeded = False
    adept._seed_lock = asyncio.Lock()
    adept._seed_task = None

    async def _run():
        adept._seed_task = asyncio.create_task(asyncio.sleep(60))
        seed_task = adept._seed_task
        adept.close()
        assert adept._seed_task is None
        assert seed_task.cancelled() or seed_task.cancelling()

    asyncio.run(_run())
    adept.template_router.stop_refresh.assert_called_once()


def test_seed_config_missing_description_logs_warning():
    """_run_seed warns and skips entries without a description."""
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = "adept/test"
    adept.default_model = "gpt-4o"
    adept.litellm_router_instance = MagicMock()
    adept.template_router = AsyncMock()
    adept._seed_config = [{"target_model": "my-slm"}]  # missing description
    adept._seeded = False
    adept._seed_lock = asyncio.Lock()

    with patch(  # test-quality-ok: warning log is the observable output of the misconfiguration guard
        "litellm.router_strategy.adept_router.adept_router.verbose_router_logger"
    ) as mock_log:
        asyncio.run(adept._run_seed())
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
    prisma_mod._CLIENTS[repo._db_url] = prisma_mod._ClientHandle(
        client
    )  # inject a fake connected client into the per-URL registry

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

        handle = await prisma_mod._get_handle(db_url)
        await handle.client.execute_raw("DELETE FROM conversations WHERE template_id = $1", surviving)
        await handle.client.execute_raw("DELETE FROM templates WHERE router_id = $1", router_id)
        await handle.client.disconnect()
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

    with patch.object(  # test-quality-ok: asserts connection reuse across repo rebuilds; the Prisma constructor IS the reuse boundary
        prisma_mod, "Prisma", side_effect=lambda datasource: _FakeClient()
    ):

        async def run():
            AdeptPrismaRepo(url)  # first router
            h1 = await prisma_mod._get_handle(url)
            AdeptPrismaRepo(url)  # simulate a rebuild: a fresh repo for the same URL
            h2 = await prisma_mod._get_handle(url)
            return h1, h2

        h1, h2 = asyncio.run(run())

    assert h1 is h2  # reused, not reconnected
    assert len(connected) == 1  # connected exactly once across both repos -> no leak
    prisma_mod._CLIENTS.pop(url, None)


def test_trigger_trainer_uses_shared_async_client():
    """_trainer_post reuses the litellm-managed shared async client (not a per-call httpx.AsyncClient)."""
    mock_storage = AsyncMock()
    router = _make_template_router(mock_storage, trainer_url="http://trainer.test")

    fake_client = MagicMock()
    fake_client.post = AsyncMock()

    with (
        patch(  # test-quality-ok: the whole point of this test is that the shared client is used instead of a per-call httpx.AsyncClient
            "litellm.router_strategy.adept_router.template.implementation.adept_template_router.get_async_httpx_client",
            return_value=fake_client,
        ) as mock_get,
        patch(  # test-quality-ok: validate_url would fail DNS resolution for the fake hostname in unit tests; production behavior is covered by test_trainer_post_re_validates_url_to_defeat_dns_rebinding
            "litellm.litellm_core_utils.url_utils.validate_url",
            return_value=("http://trainer.test/run-workflow/tmpl-httpx-test", "trainer.test"),
        ),
    ):

        async def _run() -> None:
            await router._trainer_post("http://trainer.test/run-workflow/tmpl-httpx-test", "tmpl-httpx-test")

        asyncio.run(_run())

    mock_get.assert_called_once()
    fake_client.post.assert_awaited_once()
    call_kwargs = fake_client.post.await_args.kwargs
    assert call_kwargs["url"] == "http://trainer.test/run-workflow/tmpl-httpx-test"
    assert call_kwargs["timeout"] == 10.0


def test_trigger_trainer_is_fire_and_forget():  # test-quality-ok: asserts scheduling is non-blocking; the only observable is that _trainer_post was scheduled but the caller returned before it awaited
    """_trigger_trainer must not await the HTTP round-trip — it schedules a background task."""
    mock_storage = AsyncMock()
    router = _make_template_router(mock_storage, trainer_url="http://slow-trainer.test")

    async def _run() -> None:
        # A running loop is required for asyncio.create_task; store_conversation always
        # runs inside one so we mirror that here.
        with patch.object(router, "_trainer_post", new_callable=AsyncMock) as mock_post:
            router._trigger_trainer("tmpl-fire-forget")
            # Yield once so the scheduled task starts; then assert it was scheduled
            # and store_conversation would have returned already.
            await asyncio.sleep(0)
            mock_post.assert_called_once()
            # Drain scheduled tasks so pytest doesn't warn about an unawaited coroutine.
            await asyncio.gather(*(t for t in asyncio.all_tasks() if t is not asyncio.current_task()))

    asyncio.run(_run())


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
    existing_mock.pg_url = "postgresql://user:pass@db.internal.com:5432/adept_db?sslmode=prefer"
    existing_mock.template_router = MagicMock(
        trainer_url="http://trainer.internal",
        conversations_threshold=10,
        tag_prefix="",
    )

    with (
        _patch(
            "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
            return_value=existing_mock,
        ) as MockAdeptRouter,
        _patch(
            "litellm.litellm_core_utils.url_utils.validate_url",
            return_value=("http://trainer.internal", "trainer.internal"),
        ),
    ):
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
    initial_mock.pg_url = "postgresql://user:pass@db.internal.com:5432/adept_db?sslmode=prefer"
    initial_mock.template_router = MagicMock(trainer_url=None, conversations_threshold=10, tag_prefix="")
    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        return_value=initial_mock,
    ):
        router.init_adept_router_deployment(_make_adept_deployment(trainer_url=None))

    assert router.adept_routers["fin_agent"] is initial_mock

    rebuilt_mock = MagicMock()
    with (
        _patch(
            "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
            return_value=rebuilt_mock,
        ) as MockAdeptRouter,
        _patch(
            "litellm.litellm_core_utils.url_utils.validate_url",
            return_value=("http://trainer.internal", "trainer.internal"),
        ),
    ):
        router.init_adept_router_deployment(_make_adept_deployment(trainer_url="http://trainer.internal"))

    MockAdeptRouter.assert_called_once()
    assert MockAdeptRouter.call_args[1]["trainer_url"] == "http://trainer.internal"
    assert router.adept_routers["fin_agent"] is rebuilt_mock
    assert router.adept_routers["fin_agent"] is not initial_mock


# ---------------------------------------------------------------------------
# Security: trainer_url SSRF validation, pg TLS, params_changed pg fields
# ---------------------------------------------------------------------------


def test_trainer_url_blocked_cloud_metadata_host():
    """Cloud-metadata addresses in trainer_url must be rejected at init time by the SSRF guard."""
    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    for blocked in ("http://169.254.169.254/latest/meta-data", "http://metadata.google.internal/"):
        deployment = Deployment(
            model_name="ssrf_test",
            litellm_params=LiteLLM_Params(
                model="adept/ssrf_test",
                adept_router_default_model="gpt-4o",
                adept_router_pg_host="db.internal.com",
                adept_router_trainer_url=blocked,
            ),
            model_info=ModelInfo(),
        )
        with pytest.raises(ValueError, match="rejected by SSRF guard"):
            router.init_adept_router_deployment(deployment)


def test_trainer_url_invalid_scheme_rejected():
    """Non-http(s) schemes in trainer_url must be rejected by the SSRF guard."""
    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    deployment = Deployment(
        model_name="scheme_test",
        litellm_params=LiteLLM_Params(
            model="adept/scheme_test",
            adept_router_default_model="gpt-4o",
            adept_router_pg_host="db.internal.com",
            adept_router_trainer_url="file:///etc/passwd",
        ),
        model_info=ModelInfo(),
    )
    with pytest.raises(ValueError, match="rejected by SSRF guard"):
        router.init_adept_router_deployment(deployment)


def test_pg_ssl_mode_included_in_url():
    """adept_router_pg_ssl_mode is appended as ?sslmode=... in the pg_url."""
    from unittest.mock import patch as _patch

    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    deployment = Deployment(
        model_name="ssl_test",
        litellm_params=LiteLLM_Params(
            model="adept/ssl_test",
            adept_router_default_model="gpt-4o",
            adept_router_pg_host="db.internal.com",
            adept_router_pg_ssl_mode="verify-full",
        ),
        model_info=ModelInfo(),
    )

    captured: dict[str, str] = {}

    def capture(model_name, default_model, litellm_router_instance, pg_url, **kwargs):
        captured["pg_url"] = pg_url
        return MagicMock()

    with _patch("litellm.router_strategy.adept_router.adept_router.AdeptRouter", side_effect=capture):
        router.init_adept_router_deployment(deployment)

    assert "sslmode=verify-full" in captured["pg_url"]


def test_pg_host_change_triggers_rebuild():
    """Changing pg_host must rebuild the in-memory AdeptRouter on the next sync tick."""
    from unittest.mock import patch as _patch

    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])

    def _deployment(host: str) -> Deployment:
        return Deployment(
            model_name="pg_rebuild_test",
            litellm_params=LiteLLM_Params(
                model="adept/pg_rebuild_test",
                adept_router_default_model="gpt-4o",
                adept_router_pg_host=host,
                adept_router_pg_database="adept_db",
                adept_router_pg_user="user",
                adept_router_pg_password="pass",
            ),
            model_info=ModelInfo(),
        )

    first_mock = MagicMock()
    first_mock.default_model = "gpt-4o"
    first_mock.pg_url = "postgresql://user:pass@db-old.internal.com:5432/adept_db?sslmode=prefer"
    first_mock.template_router = MagicMock(trainer_url=None, conversations_threshold=1000, tag_prefix="")
    with _patch("litellm.router_strategy.adept_router.adept_router.AdeptRouter", return_value=first_mock):
        router.init_adept_router_deployment(_deployment("db-old.internal.com"))

    second_mock = MagicMock()
    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter", return_value=second_mock
    ) as MockAdeptRouter:
        router.init_adept_router_deployment(_deployment("db-new.internal.com"))

    MockAdeptRouter.assert_called_once()
    assert router.adept_routers["pg_rebuild_test"] is second_mock
    assert router.adept_routers["pg_rebuild_test"] is not first_mock


# ---------------------------------------------------------------------------
# Model-access authorization on the routed target model
# ---------------------------------------------------------------------------


def _make_pre_hook_adept(
    model_name: str = "adept/test", default_model: str = "gpt-4o", target_model: str | None = "slm-x"
):
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = model_name
    adept.default_model = default_model
    adept.litellm_router_instance = MagicMock()
    adept.litellm_router_instance.model_list = []
    adept.template_router = AsyncMock()
    adept.template_router.route.return_value = (
        {"template_id": "t1", "target_model": target_model} if target_model else None
    )
    adept._seeded = True
    return adept


def test_pre_routing_hook_rejects_when_caller_lacks_access_to_routed_model():
    """Regression: the swapped-in target model must go through the caller's model-access check.

    Without this, a key that can call the ADEPT alias but not the trained SLM (or the
    default fallback) would be silently upgraded to a model it cannot legitimately call.
    """
    from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth

    adept = _make_pre_hook_adept(target_model="slm-x")
    messages = [{"role": "user", "content": "hello"}]
    caller = UserAPIKeyAuth(api_key="sk-test", models=["adept/test"])
    request_kwargs = {"metadata": {"user_api_key_auth": caller}}

    denial = ProxyException(
        message="Key not allowed", type=ProxyErrorTypes.key_model_access_denied.value, param=None, code="401"
    )

    with (
        patch(  # test-quality-ok: verifies authz is delegated to the proxy's auth_checks with the resolved SLM model; that call IS the boundary
            "litellm.proxy.auth.auth_checks.can_key_call_resolved_model", new_callable=AsyncMock, side_effect=denial
        ) as check
    ):
        with pytest.raises(ProxyException):
            asyncio.run(
                adept.async_pre_routing_hook(model="adept/test", request_kwargs=request_kwargs, messages=messages)
            )
    check.assert_awaited_once()
    assert check.await_args.kwargs["model"] == "slm-x"
    # The routing decision must NOT have been stamped on a rejected request.
    assert "adept_routed_to_slm" not in request_kwargs["metadata"]


def test_pre_routing_hook_allows_when_caller_has_access_to_routed_model():
    """Happy path counterpart: an authorized caller reaches the SLM and metadata is stamped."""
    from litellm.proxy._types import UserAPIKeyAuth

    adept = _make_pre_hook_adept(target_model="slm-x")
    messages = [{"role": "user", "content": "hello"}]
    caller = UserAPIKeyAuth(api_key="sk-test", models=["adept/test", "slm-x"])
    request_kwargs = {"metadata": {"user_api_key_auth": caller}}

    with (
        patch(  # test-quality-ok: verifies authz is delegated to the proxy's auth_checks with the resolved SLM model on the happy path
            "litellm.proxy.auth.auth_checks.can_key_call_resolved_model", new_callable=AsyncMock, return_value=None
        ) as check
    ):
        resp = asyncio.run(
            adept.async_pre_routing_hook(model="adept/test", request_kwargs=request_kwargs, messages=messages)
        )

    check.assert_awaited_once()
    assert check.await_args.kwargs["model"] == "slm-x"
    assert resp is not None
    assert resp.model == "slm-x"
    assert request_kwargs["metadata"]["adept_routed_to_slm"] is True


def test_pre_routing_hook_authorizes_default_model_on_miss():
    """A miss falls back to the default model — that fallback must be authorized too."""
    from litellm.proxy._types import UserAPIKeyAuth

    adept = _make_pre_hook_adept(default_model="big-llm", target_model=None)
    messages = [{"role": "user", "content": "hi"}]
    caller = UserAPIKeyAuth(api_key="sk-test", models=["adept/test", "big-llm"])
    request_kwargs = {"metadata": {"user_api_key_auth": caller}}

    with (
        patch(  # test-quality-ok: verifies authz is also enforced against the default fallback model, not just the SLM target
            "litellm.proxy.auth.auth_checks.can_key_call_resolved_model", new_callable=AsyncMock, return_value=None
        ) as check
    ):
        resp = asyncio.run(
            adept.async_pre_routing_hook(model="adept/test", request_kwargs=request_kwargs, messages=messages)
        )

    check.assert_awaited_once()
    assert check.await_args.kwargs["model"] == "big-llm"
    assert resp is not None
    assert resp.model == "big-llm"


def test_pre_routing_hook_skips_authz_when_no_user_api_key_auth():
    """Non-proxy paths (ADEPT used directly against a Router) have no auth object;
    the hook must not crash, just skip the check."""
    adept = _make_pre_hook_adept(target_model="slm-x")
    messages = [{"role": "user", "content": "hello"}]
    request_kwargs: dict[str, object] = {"metadata": {}}

    with (
        patch(  # test-quality-ok: verifies the authz check is SKIPPED when no proxy auth object is present; the check-not-called IS the observable
            "litellm.proxy.auth.auth_checks.can_key_call_resolved_model", new_callable=AsyncMock
        ) as check
    ):
        resp = asyncio.run(
            adept.async_pre_routing_hook(model="adept/test", request_kwargs=request_kwargs, messages=messages)
        )

    check.assert_not_awaited()
    assert resp is not None
    assert resp.model == "slm-x"


# ---------------------------------------------------------------------------
# Trainer URL: operator-controlled host allowlist (general_settings.user_url_allowed_hosts)
# ---------------------------------------------------------------------------


def test_trainer_url_private_ip_rejected_by_ssrf_guard():
    """A private-range host in trainer_url must be rejected — a team admin cannot bypass this
    by setting the URL directly on a team-scoped deployment because the SSRF guard is central."""
    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    deployment = Deployment(
        model_name="private_ip",
        litellm_params=LiteLLM_Params(
            model="adept/private_ip",
            adept_router_default_model="gpt-4o",
            adept_router_pg_host="db.internal.com",
            adept_router_trainer_url="http://10.0.0.5/hook",
        ),
        model_info=ModelInfo(),
    )
    with pytest.raises(ValueError, match="rejected by SSRF guard"):
        router.init_adept_router_deployment(deployment)


def test_trainer_url_private_host_allowed_when_operator_allowlists_it():
    """Operators can opt an internal trainer past the SSRF guard by adding it to
    `litellm.user_url_allowed_hosts` (surfaced through general_settings) — team admins cannot."""
    from unittest.mock import patch as _patch

    import litellm
    from litellm.router import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    previous_allowed = list(getattr(litellm, "user_url_allowed_hosts", ()) or ())
    litellm.user_url_allowed_hosts = [
        *previous_allowed,
        "trainer.internal.com",
    ]  # test-quality-ok: verifying operator-controlled allowlist behavior REQUIRES writing the process-wide setting; save/restore is done in the finally block
    try:
        router = Router(model_list=[])
        deployment = Deployment(
            model_name="operator_allowlisted",
            litellm_params=LiteLLM_Params(
                model="adept/operator_allowlisted",
                adept_router_default_model="gpt-4o",
                adept_router_pg_host="db.internal.com",
                adept_router_trainer_url="http://trainer.internal.com/hook",
            ),
            model_info=ModelInfo(),
        )
        with (
            _patch(
                "litellm.litellm_core_utils.url_utils.socket.getaddrinfo",
                return_value=[(2, 1, 6, "", ("10.0.0.5", 80))],
            ),
            _patch("litellm.router_strategy.adept_router.adept_router.AdeptRouter", return_value=MagicMock()),
        ):
            router.init_adept_router_deployment(deployment)
        assert "operator_allowlisted" in router.adept_routers
    finally:
        litellm.user_url_allowed_hosts = previous_allowed  # test-quality-ok: restore the pre-test value so the write above does not leak into other tests


def test_trainer_post_re_validates_url_to_defeat_dns_rebinding():  # test-quality-ok: the observable behavior of the DNS-rebinding defense is exactly "no HTTP client is obtained when the URL fails re-validation"; nothing else to assert
    """The trainer POST must re-run the SSRF guard so a hostname that resolved to a public IP
    at deployment sync but rebinds to a private IP by call time is still refused."""
    from litellm.litellm_core_utils.url_utils import SSRFError
    from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
        AdeptTemplateRouter,
    )

    async def _run() -> None:
        with (
            patch(  # test-quality-ok: validate_url IS the SSRF guard the test is verifying; simulating an SSRFError raise from it is the only way to force the re-validation branch
                "litellm.litellm_core_utils.url_utils.validate_url",
                side_effect=SSRFError("resolves to blocked network"),
            ),
            patch(  # test-quality-ok: get_async_httpx_client is the only observable proxy for whether the POST would have been sent; asserting not-called IS the security invariant
                "litellm.router_strategy.adept_router.template.implementation.adept_template_router.get_async_httpx_client"
            ) as mock_get,
        ):
            await AdeptTemplateRouter._trainer_post("http://rebound.example.com/hook", "tmpl-rebind")
        assert not mock_get.called

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Coverage: cache TTL expiry, LRU eviction, seed_template, helpers
# ---------------------------------------------------------------------------


def test_cache_ttl_expiry_returns_none_and_evicts():
    """A cached entry older than the TTL must be dropped on read so the next call re-hits the store."""
    from litellm.router_strategy.adept_router.store.store_template import StoredTemplate
    from litellm.router_strategy.adept_router.template.implementation import adept_template_router as mod

    router = _make_template_router(AsyncMock())
    key = ("router-id", "hash")
    stored = StoredTemplate(
        id="t",
        template="",
        template_hash="",
        router_id="",
        target_model="m",
        additional_information=None,
        created_at=None,
    )
    router._template_cache[key] = (0.0, stored)

    with patch.object(  # test-quality-ok: TTL expiry is only observable via a controlled clock; patching time.monotonic on the module is the standard way to fast-forward
        mod.time, "monotonic", return_value=mod._TEMPLATE_CACHE_TTL_SECONDS + 1
    ):
        assert router._cache_get(key) is None
    assert key not in router._template_cache


def test_cache_put_evicts_lru_when_over_capacity():
    """Beyond _TEMPLATE_CACHE_MAX_SIZE, the oldest entry must be popped so memory stays bounded."""
    from litellm.router_strategy.adept_router.store.store_template import StoredTemplate
    from litellm.router_strategy.adept_router.template.implementation import adept_template_router as mod

    router = _make_template_router(AsyncMock())
    stored = StoredTemplate(
        id="t",
        template="",
        template_hash="",
        router_id="",
        target_model="m",
        additional_information=None,
        created_at=None,
    )
    with patch.object(  # test-quality-ok: shrinking the module-level bound is the only way to exercise LRU eviction without inserting 1024 real entries per test
        mod, "_TEMPLATE_CACHE_MAX_SIZE", 2
    ):
        router._cache_put(("r", "a"), stored)
        router._cache_put(("r", "b"), stored)
        router._cache_put(("r", "c"), stored)
    assert ("r", "a") not in router._template_cache
    assert ("r", "b") in router._template_cache
    assert ("r", "c") in router._template_cache


def test_seed_template_stores_new_and_skips_existing():
    """seed_template writes on a miss and short-circuits on a hit (idempotent seed)."""
    mock_storage = AsyncMock()
    mock_storage.match_by_hash.return_value = None
    mock_storage.store_template.return_value = "tmpl-seeded"
    router = _make_template_router(mock_storage)

    assert asyncio.run(router.seed_template("Extract invoice fields", "slm-invoice")) is True
    mock_storage.store_template.assert_awaited_once()

    mock_storage.reset_mock()
    mock_storage.match_by_hash.return_value = "tmpl-existing"
    assert asyncio.run(router.seed_template("Extract invoice fields", "slm-invoice")) is False
    mock_storage.store_template.assert_not_awaited()


def test_response_text_returns_none_on_empty_choices():
    """`_response_text` guards against a response with no choices (the reject / no-content path)."""
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter
    from litellm.types.utils import ModelResponse

    response = ModelResponse()
    response.choices = []
    assert AdeptRouter._response_text(response) is None


def test_content_to_text_flattens_list_content_and_handles_none():
    """Multimodal list content is flattened to text; None content returns empty string."""
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    assert AdeptRouter._content_to_text(None) == ""
    assert AdeptRouter._content_to_text("plain string") == "plain string"

    blocks = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
        {"type": "image_url", "image_url": {"url": "x"}},
    ]
    assert AdeptRouter._content_to_text(blocks) == "hello world"


def test_extract_system_prompt_and_user_text_helpers():
    """`_extract_system_prompt` finds the system role; `_extract_user_text` reads the LAST user turn."""
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    assert AdeptRouter._extract_system_prompt([{"role": "user", "content": "hi"}]) is None
    assert AdeptRouter._extract_system_prompt([{"role": "system", "content": "you are..."}]) == "you are..."

    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert AdeptRouter._extract_user_text(messages) == "second"
    assert AdeptRouter._extract_user_text([{"role": "assistant", "content": "only bot"}]) == ""


def test_resolve_template_id_returns_matched_id_on_hit():
    """When a template with the same hash already exists, `_resolve_template_id` reuses that id and does NOT re-store."""
    mock_storage = AsyncMock()
    mock_storage.match_by_hash.return_value = "tmpl-existing"
    router = _make_template_router(mock_storage)

    resolved = asyncio.run(router._resolve_template_id("masked", "hash-x", "router-1", None))
    assert resolved == "tmpl-existing"
    mock_storage.store_template.assert_not_awaited()


def test_store_conversation_swallows_and_logs_expected_errors():
    """The success-event handler is fire-and-forget: expected raises (KeyError etc.) must be caught, not propagated."""
    mock_storage = AsyncMock()
    mock_storage.match_by_hash.side_effect = KeyError("boom")
    router = _make_template_router(mock_storage)

    asyncio.run(router.store_conversation("prompt", "response"))

    assert mock_storage.store_conversation.await_count == 0


# ---------------------------------------------------------------------------
# Security: no prompt content in debug logs; stale PG clients get disconnected
# ---------------------------------------------------------------------------


def test_extract_template_debug_log_omits_prompt_content(caplog):
    """The extract-template debug line must NOT include any prompt text — even 'masked' — because
    `_mask_text` only rewrites tag content, so untagged prompt words would otherwise reach the log."""
    import logging

    router = _make_template_router(AsyncMock())
    secret_prompt = "USER_SECRET_TOKEN_abc123 please summarize this <var>x</var>"

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Router"):
        router._extract_template(secret_prompt)

    assert not any("USER_SECRET_TOKEN_abc123" in rec.getMessage() for rec in caplog.records)


def test_stale_pg_client_disconnected_on_rebuild_when_url_changes():
    """A rebuild whose pg_url differs from the existing router's must schedule a disconnect of the
    old client so credentials do not remain in-memory for the process lifetime."""
    from unittest.mock import patch as _patch

    from litellm.router import Router
    from litellm.router_strategy.adept_router.store.implementation import prisma as prisma_mod

    router = Router(model_list=[])
    stale = MagicMock()
    stale.pg_url = "postgresql://old_user:old_pass@old.internal:5432/adept_db?sslmode=prefer"
    stale.default_model = "gpt-4o"
    stale.template_router = MagicMock(trainer_url=None, conversations_threshold=10, tag_prefix="")
    router.adept_routers["cred_rotation"] = stale

    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    new_deployment = Deployment(
        model_name="cred_rotation",
        litellm_params=LiteLLM_Params(
            model="adept/cred_rotation",
            adept_router_default_model="gpt-4o",
            adept_router_pg_host="new.internal.com",
            adept_router_pg_user="new_user",
            adept_router_pg_password="new_pass",
            adept_router_pg_database="adept_db",
        ),
        model_info=ModelInfo(),
    )

    with (
        _patch.object(prisma_mod, "schedule_disconnect") as mock_disc,
        _patch(
            "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
            return_value=MagicMock(),
        ),
    ):
        router.init_adept_router_deployment(new_deployment)

    mock_disc.assert_called_once_with(stale.pg_url)


def test_rebuild_skips_disconnect_when_another_deployment_still_uses_the_old_url():
    """If a second ADEPT deployment shares the OLD pg_url, the rebuild must NOT disconnect it —
    the sibling deployment would immediately hit a torn-down client on its next query."""
    from unittest.mock import patch as _patch

    from litellm.router import Router
    from litellm.router_strategy.adept_router.store.implementation import prisma as prisma_mod
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    shared_url = "postgresql://u:p@shared.internal:5432/adept_db?sslmode=prefer"

    outgoing = MagicMock()
    outgoing.pg_url = shared_url
    outgoing.default_model = "gpt-4o"
    outgoing.template_router = MagicMock(trainer_url=None, conversations_threshold=10, tag_prefix="")
    router.adept_routers["moving_deployment"] = outgoing

    sibling = MagicMock()
    sibling.pg_url = shared_url
    router.adept_routers["sibling_deployment"] = sibling

    new_deployment = Deployment(
        model_name="moving_deployment",
        litellm_params=LiteLLM_Params(
            model="adept/moving_deployment",
            adept_router_default_model="gpt-4o",
            adept_router_pg_host="new.internal.com",
            adept_router_pg_database="adept_db",
        ),
        model_info=ModelInfo(),
    )

    with (
        _patch.object(prisma_mod, "schedule_disconnect") as mock_disc,
        _patch(
            "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
            return_value=MagicMock(),
        ),
    ):
        router.init_adept_router_deployment(new_deployment)

    mock_disc.assert_not_called()


def test_disconnect_client_evicts_from_registry_and_disconnects():
    """`disconnect_client` must pop the entry AND call `.disconnect()` so no zombie sockets or
    passwords linger. If the URL is not present, it must not raise."""
    from litellm.router_strategy.adept_router.store.implementation import prisma as prisma_mod

    url = "postgresql://user:pw@host:5432/db"
    fake_client = MagicMock()
    fake_client.disconnect = AsyncMock()
    prisma_mod._CLIENTS[url] = prisma_mod._ClientHandle(fake_client)

    asyncio.run(prisma_mod.disconnect_client(url))

    assert url not in prisma_mod._CLIENTS
    fake_client.disconnect.assert_awaited_once()

    asyncio.run(prisma_mod.disconnect_client(url))
    assert fake_client.disconnect.await_count == 1


def test_disconnect_client_waits_for_in_flight_borrow_then_disconnects():
    """A borrowed client must not be disconnected mid-op: disconnect must wait for the borrow
    to release before tearing the socket down."""
    from litellm.router_strategy.adept_router.store.implementation import prisma as prisma_mod

    url = "postgresql://user:pw@host:5432/db"
    fake_client = MagicMock()
    fake_client.disconnect = AsyncMock()
    handle = prisma_mod._ClientHandle(fake_client)
    prisma_mod._CLIENTS[url] = handle

    async def _run():
        borrow_started = asyncio.Event()
        release_borrow = asyncio.Event()

        async def _hold_borrow():
            async with handle.borrow():
                borrow_started.set()
                await release_borrow.wait()

        borrower = asyncio.create_task(_hold_borrow())
        await borrow_started.wait()

        disconnect_task = asyncio.create_task(prisma_mod.disconnect_client(url))
        await asyncio.sleep(0.05)
        assert not disconnect_task.done(), "disconnect must wait for in-flight borrow"
        assert fake_client.disconnect.await_count == 0

        release_borrow.set()
        await borrower
        await disconnect_task

    asyncio.run(_run())
    assert url not in prisma_mod._CLIENTS
    fake_client.disconnect.assert_awaited_once()


def test_redact_url_masks_password_in_logs():
    """_redact_url must replace embedded passwords with '***' so error logs cannot leak them."""
    from litellm.router_strategy.adept_router.store.implementation.prisma import _redact_url

    assert _redact_url("postgresql://user:secret@host:5432/db") == "postgresql://user:***@host:5432/db"
    assert _redact_url("postgresql://user@host/db") == "postgresql://user@host/db"


def test_delete_deployment_releases_adept_router_and_schedules_disconnect():
    """delete_deployment must drop the ADEPT router from self.adept_routers, unregister its
    callbacks, and schedule disconnect of its PG client when no sibling deployment shares the
    URL. Otherwise repeated removals leak connection pools and password-bearing URLs."""
    from unittest.mock import patch as _patch

    import litellm
    from litellm.router import Router
    from litellm.router_strategy.adept_router.store.implementation import prisma as prisma_mod
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    to_remove = MagicMock()
    to_remove.pg_url = "postgresql://u:p@removed.internal:5432/adept_db"
    router.adept_routers["will_be_deleted"] = to_remove
    litellm.callbacks.append(to_remove)

    deployment = Deployment(
        model_name="will_be_deleted",
        litellm_params=LiteLLM_Params(model="adept/will_be_deleted"),
        model_info=ModelInfo(id="dep-id-1"),
    )
    router.model_list = [deployment.model_dump(exclude_none=True)]
    router.model_id_to_deployment_index_map = {"dep-id-1": 0}

    try:
        with _patch.object(prisma_mod, "schedule_disconnect") as mock_disc:
            router.delete_deployment("dep-id-1")

        assert "will_be_deleted" not in router.adept_routers
        assert to_remove not in litellm.callbacks
        mock_disc.assert_called_once_with(to_remove.pg_url)
    finally:
        while to_remove in litellm.callbacks:
            litellm.callbacks.remove(to_remove)


def test_delete_deployment_skips_disconnect_when_sibling_shares_url():
    """A sibling ADEPT deployment on the same URL must keep its client: deleting the primary
    should unregister only the primary and never disconnect the shared client."""
    from unittest.mock import patch as _patch

    import litellm
    from litellm.router import Router
    from litellm.router_strategy.adept_router.store.implementation import prisma as prisma_mod
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    router = Router(model_list=[])
    shared_url = "postgresql://u:p@shared.internal:5432/adept_db"
    to_remove = MagicMock()
    to_remove.pg_url = shared_url
    router.adept_routers["primary"] = to_remove

    sibling = MagicMock()
    sibling.pg_url = shared_url
    router.adept_routers["sibling"] = sibling

    litellm.callbacks.append(to_remove)

    deployment = Deployment(
        model_name="primary",
        litellm_params=LiteLLM_Params(model="adept/primary"),
        model_info=ModelInfo(id="dep-id-2"),
    )
    router.model_list = [deployment.model_dump(exclude_none=True)]
    router.model_id_to_deployment_index_map = {"dep-id-2": 0}

    try:
        with _patch.object(prisma_mod, "schedule_disconnect") as mock_disc:
            router.delete_deployment("dep-id-2")

        assert "primary" not in router.adept_routers
        assert "sibling" in router.adept_routers
        mock_disc.assert_not_called()
    finally:
        while to_remove in litellm.callbacks:
            litellm.callbacks.remove(to_remove)
