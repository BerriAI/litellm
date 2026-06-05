"""Unit tests for the ADEPT router."""

import hashlib
from typing import Optional
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# AdeptTemplateRouter tests (mock PostgresTemplateRepo — no live DB needed)
# ---------------------------------------------------------------------------


def _make_template_router(mock_storage, conversations_threshold=10, trainer_url=None):
    from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
        AdeptTemplateRouter,
    )

    mock_router_instance = MagicMock()
    mock_router_instance.get_model_ids.return_value = ["router-id-1"]

    with patch(
        "litellm.router_strategy.adept_router.store.implementation.postgresql.PostgresTemplateRepo",
        return_value=mock_storage,
    ):
        router = AdeptTemplateRouter(
            model_name="adept_router_test",
            litellm_router_instance=mock_router_instance,
            pg_url="postgresql+psycopg2://user:pass@localhost:5432/db",
            tag_prefix="var",
            conversations_threshold=conversations_threshold,
            trainer_url=trainer_url,
        )
    router.template_store = mock_storage
    return router


def test_adept_template_router_route_miss():
    mock_storage = MagicMock()
    mock_storage.match_by_hash.return_value = None

    router = _make_template_router(mock_storage)
    result = router.route("What is 2 + 2?")
    assert result is None


def test_adept_template_router_route_hit():
    mock_storage = MagicMock()
    mock_storage.match_by_hash.return_value = "tmpl-abc"
    mock_storage.get_template.return_value = {
        "id": "tmpl-abc",
        "template": "Get order {ID} for {EMAIL}",
        "target_model": "gpt-4o",
        "additional_information": None,
    }

    router = _make_template_router(mock_storage)
    result = router.route("Get order ORD-123 for user@example.com")
    assert result is not None
    assert result["target_model"] == "gpt-4o"


def test_threshold_modulo_triggers_at_multiples():
    """Trainer should be called at 5, 10, 15... but not at 7."""
    mock_storage = MagicMock()
    mock_storage.match_by_hash.return_value = "tmpl-1"
    mock_storage.get_template.return_value = None  # not used in store_conversation
    mock_storage.store_conversation.return_value = True
    mock_storage.store_template.return_value = True

    router = _make_template_router(
        mock_storage, conversations_threshold=5, trainer_url="http://trainer.test"
    )

    with patch.object(router, "_trigger_trainer") as mock_trigger:
        # count=5 → triggers
        mock_storage.count_conversation_by_template_id.return_value = 5
        router.store_conversation("prompt", "response")
        mock_trigger.assert_called_once_with("tmpl-1")

        mock_trigger.reset_mock()

        # count=7 → does not trigger
        mock_storage.count_conversation_by_template_id.return_value = 7
        router.store_conversation("prompt", "response")
        mock_trigger.assert_not_called()

        # count=10 → triggers again
        mock_storage.count_conversation_by_template_id.return_value = 10
        router.store_conversation("prompt", "response")
        mock_trigger.assert_called_once_with("tmpl-1")


def test_trainer_url_used_in_trigger():
    """_trigger_trainer fires POST to trainer_url, skips if not set."""
    mock_storage = MagicMock()
    router_with = _make_template_router(
        mock_storage, trainer_url="http://my-trainer.internal"
    )
    router_without = _make_template_router(mock_storage, trainer_url=None)

    with patch(
        "litellm.router_strategy.adept_router.template.implementation.adept_template_router.httpx.post"
    ) as mock_post:
        router_with._trigger_trainer("tmpl-xyz")
        mock_post.assert_called_once()
        assert "tmpl-xyz" in mock_post.call_args[1]["url"]

        mock_post.reset_mock()
        router_without._trigger_trainer("tmpl-xyz")
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# System prompt isolation tests
# ---------------------------------------------------------------------------


def test_different_system_prompts_produce_different_hashes():
    """Two tools with the same XML structure but different system prompts must not collide."""
    from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
        AdeptTemplateRouter,
    )

    hash_a = AdeptTemplateRouter._hash_template(
        "<doc></doc>", system_prompt="You are an invoice extractor."
    )
    hash_b = AdeptTemplateRouter._hash_template(
        "<doc></doc>", system_prompt="You are a contract reviewer."
    )
    assert hash_a != hash_b


def test_same_tool_always_produces_same_hash():
    """Identical system prompt + same tag structure must always hash to the same value."""
    from litellm.router_strategy.adept_router.template.implementation.adept_template_router import (
        AdeptTemplateRouter,
    )

    system = "You are a ticket classifier."
    hash_1 = AdeptTemplateRouter._hash_template(
        "<ticket></ticket>", system_prompt=system
    )
    hash_2 = AdeptTemplateRouter._hash_template(
        "<ticket></ticket>", system_prompt=system
    )
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
    assert (
        "postgresql+psycopg2://user:pass@db.internal.com:5432/adept_db"
        in call_kwargs["pg_url"]
    )
    assert call_kwargs["conversations_threshold"] == 20
    assert call_kwargs["trainer_url"] == "http://trainer.internal"


# ---------------------------------------------------------------------------
# New tests: callback registration, routing decision, URL encoding, caching
# ---------------------------------------------------------------------------


def test_callback_registered_after_init():
    """After init_adept_router_deployment, AdeptRouter must appear in the async success callbacks."""
    import litellm
    from unittest.mock import patch as _patch
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

    # add_litellm_callback routes to litellm.callbacks (the unified list)
    assert mock_adept in litellm.callbacks


def test_routing_decision_stored_in_conversation():
    """routed_to_slm=True is persisted in conversation additional_information."""
    import asyncio
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    mock_router = MagicMock()
    mock_router.get_model_ids.return_value = ["router-id-1"]
    mock_template_router = MagicMock()

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = "adept/test"
    adept.default_model = "gpt-4o"
    adept.litellm_router_instance = mock_router
    adept.template_router = mock_template_router

    start = MagicMock()
    end = MagicMock()
    end.__sub__ = MagicMock(return_value=MagicMock(total_seconds=lambda: 0.1))

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="output"))]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    kwargs = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "my-slm",
        "response_cost": 0.001,
        "adept_routed_to_slm": True,
    }

    asyncio.get_event_loop().run_until_complete(
        adept.async_log_success_event(kwargs, response, start, end)
    )

    call_args = mock_template_router.store_conversation.call_args
    assert call_args is not None
    # routed_to_slm is the last positional arg
    assert call_args[0][-1] is True


def test_routing_decision_fallback_stored():
    """routed_to_slm=False is persisted when fallback was used."""
    import asyncio
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    mock_router = MagicMock()
    mock_router.get_model_ids.return_value = ["router-id-1"]
    mock_template_router = MagicMock()

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = "adept/test"
    adept.default_model = "gpt-4o"
    adept.litellm_router_instance = mock_router
    adept.template_router = mock_template_router

    start = MagicMock()
    end = MagicMock()
    end.__sub__ = MagicMock(return_value=MagicMock(total_seconds=lambda: 0.1))

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="output"))]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    kwargs = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4o",
        "response_cost": 0.005,
        "adept_routed_to_slm": False,
    }

    asyncio.get_event_loop().run_until_complete(
        adept.async_log_success_event(kwargs, response, start, end)
    )

    call_args = mock_template_router.store_conversation.call_args
    assert call_args is not None
    assert call_args[0][-1] is False


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

    def capture_adept(
        model_name, default_model, litellm_router_instance, pg_url, **kwargs
    ):
        captured_url["pg_url"] = pg_url
        return MagicMock()

    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        side_effect=capture_adept,
    ):
        router.init_adept_router_deployment(deployment)

    pg_url = captured_url["pg_url"]
    assert (
        "p%40ss%3Aw%2Frd" in pg_url
    ), f"Expected encoded password in URL, got: {pg_url}"
    assert "p@ss:w/rd" not in pg_url


def test_seed_config_missing_description_logs_warning():
    """_seed_templates warns and skips entries without a description."""
    from litellm.router_strategy.adept_router.adept_router import AdeptRouter

    mock_router = MagicMock()
    mock_router.get_model_ids.return_value = ["router-id-1"]
    mock_template_router = MagicMock()

    adept = AdeptRouter.__new__(AdeptRouter)
    adept.model_name = "adept/test"
    adept.default_model = "gpt-4o"
    adept.litellm_router_instance = mock_router
    adept.template_router = mock_template_router

    with patch(
        "litellm.router_strategy.adept_router.adept_router.verbose_router_logger"
    ) as mock_log:
        adept._seed_templates([{"target_model": "my-slm"}])  # missing description
        warning_calls = [str(c) for c in mock_log.warning.call_args_list]
        assert any("description" in w for w in warning_calls)

    mock_template_router.template_store.store_template.assert_not_called()


def test_router_id_cached_after_first_call():
    """get_router_id() should call get_model_ids only once regardless of how many times it's called."""
    mock_storage = MagicMock()
    mock_storage.match_by_hash.return_value = None

    router = _make_template_router(mock_storage)
    router._router_id_cache = None  # ensure cache is clear

    router.get_router_id()
    router.get_router_id()
    router.get_router_id()

    assert router.litellm_router_instance.get_model_ids.call_count == 1


def test_trigger_trainer_uses_httpx():
    """_trigger_trainer must use httpx.post, not requests.post."""
    mock_storage = MagicMock()
    router = _make_template_router(mock_storage, trainer_url="http://trainer.test")

    with patch(
        "litellm.router_strategy.adept_router.template.implementation.adept_template_router.httpx.post"
    ) as mock_httpx:
        router._trigger_trainer("tmpl-httpx-test")
        mock_httpx.assert_called_once()
        call_kwargs = mock_httpx.call_args[1]
        assert "tmpl-httpx-test" in call_kwargs["url"]
        assert call_kwargs["timeout"] == 10


def test_store_template_race_condition_safe():
    """
    A concurrent duplicate insert (same router_id + template_hash) must not raise.
    store_template uses ON CONFLICT DO NOTHING — the second insert is a no-op.
    """
    from sqlalchemy.exc import IntegrityError
    from litellm.router_strategy.adept_router.store.implementation.postgresql import (
        PostgresTemplateRepo,
    )

    repo = PostgresTemplateRepo.__new__(PostgresTemplateRepo)

    call_count = {"n": 0}

    def mock_session_context():
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            mock_sess = MagicMock()
            # Second execute raises IntegrityError — but ON CONFLICT DO NOTHING should prevent this
            # at the DB level. Here we verify the code handles it gracefully anyway.
            if call_count["n"] > 0:
                mock_sess.execute.side_effect = IntegrityError(
                    "stmt", "params", Exception("unique")
                )
            call_count["n"] += 1
            yield mock_sess

        return ctx()

    repo.Session = mock_session_context
    repo.engine = MagicMock()

    # First call should succeed
    result = repo.store_template(
        template_id="tid-1",
        template="<doc></doc>",
        template_hash="abc123",
        target_model="",
        router_id="router-1",
    )
    # Second call simulates the race — should return None gracefully, not raise
    result2 = repo.store_template(
        template_id="tid-2",
        template="<doc></doc>",
        template_hash="abc123",
        target_model="",
        router_id="router-1",
    )
    assert result is not None  # first succeeded
    assert result2 is None  # gracefully handled the conflict


# ---------------------------------------------------------------------------
# Rebuild-on-change tests: editing an ADEPT deployment in the DB should
# refresh the in-memory router without requiring a proxy restart.
# ---------------------------------------------------------------------------


def _make_adept_deployment(
    model_name: str = "fin_agent",
    trainer_url: Optional[str] = None,
    threshold: Optional[int] = None,
    tag_prefix: Optional[str] = None,
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
    from unittest.mock import MagicMock, patch as _patch
    from litellm.router import Router

    router = Router(model_list=[])
    deployment = _make_adept_deployment(
        trainer_url="http://trainer.internal", threshold=10
    )

    # Mock must expose the same attribute values the deployment carries, otherwise
    # the params-diff check sees MagicMock != real-value and forces a rebuild.
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

    # Constructor invoked once; in-memory router unchanged on second call.
    assert MockAdeptRouter.call_count == 1
    assert router.adept_routers["fin_agent"] is first_instance


def test_init_adept_router_rebuilds_when_trainer_url_changes():
    """
    Editing trainer_url in the DB row must rebuild the in-memory AdeptRouter on
    the next sync tick — otherwise edits silently never take effect (the bug
    that hid 30 conversations' worth of trainer notifications).
    """
    from unittest.mock import MagicMock, patch as _patch
    from litellm.router import Router

    router = Router(model_list=[])

    # First init: trainer_url unset
    initial_mock = MagicMock()
    initial_mock.default_model = "gpt-4o"
    initial_mock.template_router = MagicMock(
        trainer_url=None, conversations_threshold=10, tag_prefix=""
    )
    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        return_value=initial_mock,
    ):
        router.init_adept_router_deployment(_make_adept_deployment(trainer_url=None))

    assert router.adept_routers["fin_agent"] is initial_mock

    # Second init: trainer_url now set — must rebuild.
    rebuilt_mock = MagicMock()
    with _patch(
        "litellm.router_strategy.adept_router.adept_router.AdeptRouter",
        return_value=rebuilt_mock,
    ) as MockAdeptRouter:
        router.init_adept_router_deployment(
            _make_adept_deployment(trainer_url="http://trainer.internal")
        )

    MockAdeptRouter.assert_called_once()
    assert MockAdeptRouter.call_args[1]["trainer_url"] == "http://trainer.internal"
    assert router.adept_routers["fin_agent"] is rebuilt_mock
    assert router.adept_routers["fin_agent"] is not initial_mock
