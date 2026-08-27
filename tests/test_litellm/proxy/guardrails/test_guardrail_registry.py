from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_registry import (
    get_guardrail_initializer_from_hooks,
    GuardrailRegistry,
    InMemoryGuardrailHandler,
)
from litellm.types.guardrails import GuardrailEventHooks, Guardrail, LitellmParams


def test_get_guardrail_initializer_from_hooks():
    initializers = get_guardrail_initializer_from_hooks()
    assert "aim" in initializers


def test_guardrail_class_registry():
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    assert "aim" in guardrail_class_registry
    assert "aporia" in guardrail_class_registry


def test_noma_registry_resolution():
    from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaGuardrail
    from litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2 import NomaV2Guardrail
    from litellm.proxy.guardrails.guardrail_registry import (
        guardrail_class_registry,
        guardrail_initializer_registry,
    )

    assert guardrail_class_registry["noma"] is NomaGuardrail
    assert guardrail_class_registry["noma_v2"] is NomaV2Guardrail
    assert "noma" in guardrail_initializer_registry
    assert "noma_v2" in guardrail_initializer_registry


@pytest.mark.parametrize(
    "configured, expected",
    [(None, True), (False, False), (True, True)],
)
def test_initialize_guardrail_run_in_parallel_preserves_constructor_default(configured, expected):
    """
    A guardrail whose constructor sets run_in_parallel=True must keep that default when
    the config omits the key; only an explicit config value may override it. The
    previous code wrote bool(None)==False on every instance, silently disabling the
    opt-in for such guardrails.
    """
    from litellm.proxy.guardrails import guardrail_registry as registry_module

    def _initializer(litellm_params, guardrail):
        return CustomGuardrail(
            guardrail_name=guardrail["guardrail_name"],
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
            run_in_parallel=True,
        )

    registry_module.guardrail_initializer_registry["parallel_default_test"] = _initializer
    try:
        params = {"guardrail": "parallel_default_test", "mode": "pre_call"}
        if configured is not None:
            params["run_in_parallel"] = configured

        handler = InMemoryGuardrailHandler()
        result = handler.initialize_guardrail(
            guardrail={"guardrail_name": "cf-parallel-default", "litellm_params": params},
        )

        stored = handler.guardrail_id_to_custom_guardrail[result["guardrail_id"]]
        assert stored.run_in_parallel is expected
    finally:
        registry_module.guardrail_initializer_registry.pop("parallel_default_test", None)


def _register_noop_initializer(guardrail_type: str):
    from litellm.proxy.guardrails import guardrail_registry as registry_module

    def _initializer(litellm_params, guardrail):
        return CustomGuardrail(
            guardrail_name=guardrail["guardrail_name"],
            event_hook=GuardrailEventHooks.pre_call,
            default_on=False,
        )

    registry_module.guardrail_initializer_registry[guardrail_type] = _initializer
    return registry_module


def _config_guardrail(name: str, guardrail_type: str, guardrail_id=None) -> dict:
    guardrail = {
        "guardrail_name": name,
        "litellm_params": {"guardrail": guardrail_type, "mode": "pre_call"},
    }
    if guardrail_id is not None:
        guardrail["guardrail_id"] = guardrail_id
    return guardrail


def test_config_guardrail_id_is_stable_across_boots():
    """
    Config guardrails used to get a fresh uuid4 per process, so ids from a
    previous boot (or another replica) 404'd on /guardrails/{id}/info even
    though the guardrail was alive.
    """
    registry_module = _register_noop_initializer("stable_id_test")
    try:
        first_boot = InMemoryGuardrailHandler().initialize_guardrail(
            guardrail=_config_guardrail("tooling", "stable_id_test")
        )
        second_boot = InMemoryGuardrailHandler().initialize_guardrail(
            guardrail=_config_guardrail("tooling", "stable_id_test")
        )

        assert first_boot["guardrail_id"] == second_boot["guardrail_id"]
    finally:
        registry_module.guardrail_initializer_registry.pop("stable_id_test", None)


def test_explicit_config_guardrail_id_wins_over_derived_id():
    registry_module = _register_noop_initializer("explicit_id_test")
    try:
        result = InMemoryGuardrailHandler().initialize_guardrail(
            guardrail=_config_guardrail("tooling", "explicit_id_test", guardrail_id="my-explicit-id")
        )

        assert result["guardrail_id"] == "my-explicit-id"
    finally:
        registry_module.guardrail_initializer_registry.pop("explicit_id_test", None)


def test_duplicate_config_guardrail_names_get_distinct_stable_ids():
    """
    Duplicate guardrail_name entries are legitimate (load balancing across
    deployments); each occurrence must keep its own id, stable across boots.
    """
    registry_module = _register_noop_initializer("dup_name_test")
    try:
        handler = InMemoryGuardrailHandler()
        first = handler.initialize_guardrail(guardrail=_config_guardrail("dup", "dup_name_test"))
        second = handler.initialize_guardrail(guardrail=_config_guardrail("dup", "dup_name_test"))

        rebooted_handler = InMemoryGuardrailHandler()
        rebooted_first = rebooted_handler.initialize_guardrail(guardrail=_config_guardrail("dup", "dup_name_test"))
        rebooted_second = rebooted_handler.initialize_guardrail(guardrail=_config_guardrail("dup", "dup_name_test"))

        assert first["guardrail_id"] != second["guardrail_id"]
        assert first["guardrail_id"] == rebooted_first["guardrail_id"]
        assert second["guardrail_id"] == rebooted_second["guardrail_id"]
        assert len(handler.IN_MEMORY_GUARDRAILS) == 2
    finally:
        registry_module.guardrail_initializer_registry.pop("dup_name_test", None)


def test_update_in_memory_guardrail():
    handler = InMemoryGuardrailHandler()
    handler.guardrail_id_to_custom_guardrail["123"] = CustomGuardrail(
        guardrail_name="test-guardrail",
        default_on=False,
        event_hook=GuardrailEventHooks.pre_call,
    )

    handler.update_in_memory_guardrail(
        "123",
        Guardrail(
            guardrail_name="test-guardrail",
            litellm_params=LitellmParams(guardrail="test-guardrail", mode="pre_call", default_on=True),
        ),
    )

    assert (
        handler.guardrail_id_to_custom_guardrail["123"].should_run_guardrail(
            data={}, event_type=GuardrailEventHooks.pre_call
        )
        is True
    )
    assert handler.guardrail_id_to_custom_guardrail["123"].event_hook is GuardrailEventHooks.pre_call


def _make_guardrail(guardrail_id: str, name: str = "g") -> Guardrail:
    return Guardrail(
        guardrail_id=guardrail_id,
        guardrail_name=name,
        litellm_params=LitellmParams(guardrail=name, mode="pre_call", default_on=False),
    )


def test_reconcile_db_guardrails_drops_stale_db_entries_only():
    """
    The reconcile pass must drop in-memory entries marked source='db' that are
    missing from the DB result, and never touch source='config' entries.
    Models the multi-pod case where another pod deleted a DB-backed guardrail.
    """
    handler = InMemoryGuardrailHandler()

    # Two DB-backed entries on this pod (synced from earlier polling cycles)
    handler.IN_MEMORY_GUARDRAILS["db-keep"] = _make_guardrail("db-keep")
    handler.IN_MEMORY_GUARDRAILS["db-stale"] = _make_guardrail("db-stale")
    handler._sources["db-keep"] = "db"
    handler._sources["db-stale"] = "db"

    # One config-loaded entry that must survive reconciliation
    handler.IN_MEMORY_GUARDRAILS["cfg"] = _make_guardrail("cfg")
    handler._sources["cfg"] = "config"

    # The DB now only contains db-keep — db-stale was deleted on another pod.
    removed = handler.reconcile_db_guardrails(db_guardrail_ids={"db-keep"})

    assert removed == ["db-stale"]
    assert "db-stale" not in handler.IN_MEMORY_GUARDRAILS
    assert "db-stale" not in handler._sources
    assert "db-keep" in handler.IN_MEMORY_GUARDRAILS
    assert "cfg" in handler.IN_MEMORY_GUARDRAILS
    assert handler._sources["cfg"] == "config"


def test_reconcile_does_not_drop_config_entries_missing_from_db():
    """A config-only guardrail (no DB row) must never be reconciled away."""
    handler = InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS["cfg-only"] = _make_guardrail("cfg-only")
    handler._sources["cfg-only"] = "config"

    removed = handler.reconcile_db_guardrails(db_guardrail_ids=set())

    assert removed == []
    assert "cfg-only" in handler.IN_MEMORY_GUARDRAILS


def test_get_source_returns_marker_set_at_insert():
    handler = InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS["a"] = _make_guardrail("a")
    handler._sources["a"] = "db"
    handler.IN_MEMORY_GUARDRAILS["b"] = _make_guardrail("b")
    handler._sources["b"] = "config"

    assert handler.get_source("a") == "db"
    assert handler.get_source("b") == "config"
    assert handler.get_source("missing") is None


def test_delete_in_memory_guardrail_clears_source_marker():
    handler = InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS["a"] = _make_guardrail("a")
    handler._sources["a"] = "db"

    handler.delete_in_memory_guardrail("a")

    assert "a" not in handler.IN_MEMORY_GUARDRAILS
    assert "a" not in handler._sources
    assert handler.get_source("a") is None


def test_list_config_guardrails_excludes_db_sourced():
    """LIT-2529: read surfaces union DB rows with config guardrails; db-sourced
    in-memory entries would double-count (or resurrect stale ones), so exclude them."""
    handler = InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS["cfg"] = _make_guardrail("cfg", name="config-one")
    handler._sources["cfg"] = "config"
    handler.IN_MEMORY_GUARDRAILS["db"] = _make_guardrail("db", name="db-one")
    handler._sources["db"] = "db"

    config_guardrails = handler.list_config_guardrails()

    assert [g["guardrail_id"] for g in config_guardrails] == ["cfg"]


def test_get_config_guardrail_by_id_returns_config_only():
    """LIT-2529: the detail/logs fallback must return config-owned guardrails and
    treat a db-sourced (stale) or missing id as a miss."""
    handler = InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS["cfg"] = _make_guardrail("cfg", name="config-one")
    handler._sources["cfg"] = "config"
    handler.IN_MEMORY_GUARDRAILS["db"] = _make_guardrail("db", name="db-one")
    handler._sources["db"] = "db"

    assert handler.get_config_guardrail_by_id("cfg")["guardrail_name"] == "config-one"
    assert handler.get_config_guardrail_by_id("db") is None
    assert handler.get_config_guardrail_by_id("missing") is None


def test_initialize_guardrail_early_return_updates_source_marker():
    """
    When initialize_guardrail is called for a guardrail that already exists
    in memory, the early-return path must still honor the caller's source.
    Otherwise a racing polling tick that placed a DB entry in memory first
    would leave a later config-init call wrongly marked as 'db' (or vice
    versa), and the entry would be reconciled with the wrong classification.
    """
    handler = InMemoryGuardrailHandler()
    # Simulate a polling tick already placing the entry as DB-backed.
    handler.IN_MEMORY_GUARDRAILS["collide"] = _make_guardrail("collide", name="bedrock")
    handler._sources["collide"] = "db"

    # Config init re-visits the same id (e.g., hot-reload, or UUID collision).
    g = Guardrail(
        guardrail_id="collide",
        guardrail_name="bedrock",
        litellm_params=LitellmParams(guardrail="bedrock", mode="pre_call", default_on=False),
    )
    handler.initialize_guardrail(guardrail=g, source="config")

    assert handler.get_source("collide") == "config"

    # And the symmetric direction: db sync should override an entry left
    # marked as 'config' from a stale init path.
    handler.initialize_guardrail(guardrail=g, source="db")
    assert handler.get_source("collide") == "db"


def test_sync_guardrail_from_db_marks_source_db_when_unchanged():
    """
    sync_guardrail_from_db must enforce source='db' even when params are
    unchanged, so a config entry whose UUID happens to collide with a later
    DB row gets re-tagged correctly.
    """
    handler = InMemoryGuardrailHandler()
    g = _make_guardrail("collide")
    handler.IN_MEMORY_GUARDRAILS["collide"] = g
    handler._sources["collide"] = "config"

    handler.sync_guardrail_from_db(g)

    assert handler.get_source("collide") == "db"


def _db_litellm_params() -> dict:
    """
    Shape produced by GuardrailRegistry.get_all_guardrails_from_db: litellm_params
    is a raw dict (not a LitellmParams), holding only the keys originally stored,
    a non-schema extra key, and plain-string enum values.
    """
    return {
        "guardrail": "litellm_content_filter",
        "mode": "pre_call",
        "default_on": True,
        "version": 2,
        "blocked_words": [{"keyword": "secret", "action": "BLOCK"}],
    }


def test_unchanged_db_params_do_not_register_as_changed():
    """
    A DB poll returns litellm_params as a raw dict while the in-memory copy is a
    LitellmParams whose model_dump() fills every field default and coerces enums.
    The two shapes must compare equal when the config is identical; otherwise
    every poll cycle re-initializes the guardrail indefinitely.
    """
    handler = InMemoryGuardrailHandler()
    raw = _db_litellm_params()
    gid = "11111111-1111-1111-1111-111111111111"
    handler.IN_MEMORY_GUARDRAILS[gid] = Guardrail(
        guardrail_id=gid,
        guardrail_name="cf",
        litellm_params=LitellmParams(**raw),
    )

    new = Guardrail(guardrail_id=gid, guardrail_name="cf", litellm_params=dict(raw))
    assert handler._has_guardrail_params_changed(gid, new) is False


def test_changed_db_params_register_as_changed():
    """Normalizing both sides must still surface a genuine config change."""
    handler = InMemoryGuardrailHandler()
    raw = _db_litellm_params()
    gid = "22222222-2222-2222-2222-222222222222"
    handler.IN_MEMORY_GUARDRAILS[gid] = Guardrail(
        guardrail_id=gid,
        guardrail_name="cf",
        litellm_params=LitellmParams(**raw),
    )

    changed = {**raw, "blocked_words": [{"keyword": "different", "action": "BLOCK"}]}
    new = Guardrail(guardrail_id=gid, guardrail_name="cf", litellm_params=changed)
    assert handler._has_guardrail_params_changed(gid, new) is True


def test_unnormalizable_db_params_register_as_changed_without_raising():
    """
    A DB row whose litellm_params fail LitellmParams validation must not crash the
    poll loop. The comparison falls back to treating the guardrail as changed so it
    re-initializes (and surfaces the bad row in logs) rather than propagating the
    validation error up through the polling cycle.
    """
    handler = InMemoryGuardrailHandler()
    raw = _db_litellm_params()
    gid = "55555555-5555-5555-5555-555555555555"
    handler.IN_MEMORY_GUARDRAILS[gid] = Guardrail(
        guardrail_id=gid,
        guardrail_name="cf",
        litellm_params=LitellmParams(**raw),
    )

    malformed = {**raw, "default_on": "not-a-bool-xyz"}
    new = Guardrail(guardrail_id=gid, guardrail_name="cf", litellm_params=malformed)
    assert handler._has_guardrail_params_changed(gid, new) is True


def _all_callback_lists():
    import litellm

    return [
        litellm.callbacks,
        litellm.success_callback,
        litellm.failure_callback,
        litellm._async_success_callback,
        litellm._async_failure_callback,
    ]


def test_delete_in_memory_guardrail_removes_callback_from_all_lists():
    """
    Request handling promotes guardrail callbacks from litellm.callbacks into the
    success/failure/async lists. delete_in_memory_guardrail must purge the callback
    from every list, otherwise a re-initialized guardrail leaves its old instance
    stranded in those lists and instances accumulate.
    """
    handler = InMemoryGuardrailHandler()
    callback = CustomGuardrail(
        guardrail_name="cf-delete",
        default_on=True,
        event_hook=GuardrailEventHooks.pre_call,
    )
    gid = "33333333-3333-3333-3333-333333333333"
    handler.IN_MEMORY_GUARDRAILS[gid] = _make_guardrail(gid, "cf-delete")
    handler._sources[gid] = "db"
    handler.guardrail_id_to_custom_guardrail[gid] = callback

    lists = _all_callback_lists()
    snapshots = [list(cb_list) for cb_list in lists]
    try:
        for cb_list in lists:
            cb_list.append(callback)

        handler.delete_in_memory_guardrail(gid)

        for cb_list in lists:
            assert callback not in cb_list
    finally:
        for cb_list, snapshot in zip(lists, snapshots):
            cb_list[:] = snapshot


def test_repeated_db_sync_does_not_accumulate_runner_instances():
    """
    End-to-end regression for the OOM: across repeated DB polls (with the config
    genuinely changing each cycle to force re-initialization), exactly one live
    guardrail instance must exist across all callback lists. On the unfixed code
    the stale instance lingers in the success/failure lists and the distinct count
    climbs above one.
    """
    import litellm

    handler = InMemoryGuardrailHandler()
    gid = "44444444-4444-4444-4444-444444444444"
    name = "cf-accum"

    def db_guardrail(word: str) -> Guardrail:
        params = {
            **_db_litellm_params(),
            "blocked_words": [{"keyword": word, "action": "BLOCK"}],
        }
        return Guardrail(guardrail_id=gid, guardrail_name=name, litellm_params=params)

    def promote_into_request_lists() -> None:
        manager = litellm.logging_callback_manager
        for callback in list(litellm.callbacks):
            manager.add_litellm_success_callback(callback)
            manager.add_litellm_failure_callback(callback)
            manager.add_litellm_async_success_callback(callback)
            manager.add_litellm_async_failure_callback(callback)

    def distinct_runner_instances() -> int:
        seen = set()
        for callback in litellm.logging_callback_manager._get_all_callbacks():
            if isinstance(callback, CustomGuardrail) and getattr(callback, "guardrail_name", None) == name:
                seen.add(id(callback))
        return len(seen)

    lists = _all_callback_lists()
    snapshots = [list(cb_list) for cb_list in lists]
    try:
        for cycle in range(5):
            handler.sync_guardrail_from_db(db_guardrail(f"word-{cycle}"))
            promote_into_request_lists()

        assert distinct_runner_instances() == 1
    finally:
        for cb_list, snapshot in zip(lists, snapshots):
            cb_list[:] = snapshot


def _judge_guardrail(guardrail_id: str) -> Guardrail:
    return Guardrail(
        guardrail_id=guardrail_id,
        guardrail_name="quality-judge",
        litellm_params={
            "guardrail": "llm_as_a_judge",
            "mode": "post_call",
            "judge_model": "my-judge-alias",
            "overall_threshold": 80,
            "on_failure": "log",
            "criteria": [{"name": "helpfulness", "weight": 100, "description": "helpful?"}],
        },
    )


def test_db_synced_judge_guardrail_uses_lazy_router_provider():
    """A judge guardrail created/synced through a DB path must resolve the active
    Router lazily at call time (issue: UI-created guardrails failed open because the
    Router was captured at construction; a guardrail created before the Router
    existed captured None and never recovered). Asserting the default provider is
    wired guarantees the instance reads the live global rather than a stale value."""
    from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import (
        LLMAsAJudgeGuardrail,
        _default_router_provider,
    )

    handler = InMemoryGuardrailHandler()

    lists = _all_callback_lists()
    snapshots = [list(cb_list) for cb_list in lists]
    try:
        handler.sync_guardrail_from_db(_judge_guardrail("judge-db"))

        instance = handler.guardrail_id_to_custom_guardrail["judge-db"]
        assert isinstance(instance, LLMAsAJudgeGuardrail)
        assert instance._router_provider is _default_router_provider
    finally:
        for cb_list, snapshot in zip(lists, snapshots):
            cb_list[:] = snapshot


def test_reinitialized_judge_guardrail_uses_lazy_router_provider():
    from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import (
        LLMAsAJudgeGuardrail,
        _default_router_provider,
    )

    handler = InMemoryGuardrailHandler()

    lists = _all_callback_lists()
    snapshots = [list(cb_list) for cb_list in lists]
    try:
        handler.reinitialize_guardrail(_judge_guardrail("judge-reinit"), source="db")

        instance = handler.guardrail_id_to_custom_guardrail["judge-reinit"]
        assert isinstance(instance, LLMAsAJudgeGuardrail)
        assert instance._router_provider is _default_router_provider
    finally:
        for cb_list, snapshot in zip(lists, snapshots):
            cb_list[:] = snapshot


class TestScanOnlyToolResultsInitRefusal:
    """A guardrail whose role filtering never scans tool results must be rejected at
    initialization when configured with scan_only_tool_results, instead of booting a
    proxy that silently scans nothing on every request."""

    def _initialize(self, name: str, params: dict):
        lists = _all_callback_lists()
        snapshots = [list(cb_list) for cb_list in lists]
        try:
            return InMemoryGuardrailHandler().initialize_guardrail(
                guardrail={"guardrail_name": name, "litellm_params": params},
            )
        finally:
            for cb_list, snapshot in zip(lists, snapshots):
                cb_list[:] = snapshot

    def test_panw_prisma_airs_with_scan_only_tool_results_is_rejected(self):
        with pytest.raises(ValueError, match="never scans tool results"):
            self._initialize(
                "panw-scan-only-combo",
                {
                    "guardrail": "panw_prisma_airs",
                    "mode": "pre_call",
                    "api_key": "test-key",
                    "profile_name": "test-profile",
                    "scan_only_tool_results": True,
                },
            )

    def test_bedrock_latest_role_with_scan_only_tool_results_is_rejected(self):
        with pytest.raises(ValueError, match="never scans tool results"):
            self._initialize(
                "bedrock-latest-role-scan-only-combo",
                {
                    "guardrail": "bedrock",
                    "mode": "pre_call",
                    "guardrailIdentifier": "gr-1",
                    "guardrailVersion": "1",
                    "experimental_use_latest_role_message_only": True,
                    "scan_only_tool_results": True,
                },
            )

    def test_bedrock_without_latest_role_accepts_scan_only_tool_results(self):
        result = self._initialize(
            "bedrock-scan-only-ok",
            {
                "guardrail": "bedrock",
                "mode": "pre_call",
                "guardrailIdentifier": "gr-1",
                "guardrailVersion": "1",
                "scan_only_tool_results": True,
            },
        )
        assert result is not None

    def test_prompt_security_default_tool_filtering_rejects_scan_only_tool_results(self, monkeypatch):
        monkeypatch.delenv("PROMPT_SECURITY_CHECK_TOOL_RESULTS", raising=False)
        with pytest.raises(ValueError, match="never scans tool results"):
            self._initialize(
                "prompt-security-scan-only-combo",
                {
                    "guardrail": "prompt_security",
                    "mode": "pre_call",
                    "api_key": "test-key",
                    "api_base": "https://ps.example.com",
                    "scan_only_tool_results": True,
                },
            )

    def test_prompt_security_check_tool_results_accepts_scan_only_tool_results(self, monkeypatch):
        monkeypatch.setenv("PROMPT_SECURITY_CHECK_TOOL_RESULTS", "true")
        result = self._initialize(
            "prompt-security-scan-only-ok",
            {
                "guardrail": "prompt_security",
                "mode": "pre_call",
                "api_key": "test-key",
                "api_base": "https://ps.example.com",
                "scan_only_tool_results": True,
            },
        )
        assert result is not None

    def test_skip_tool_message_with_scan_only_tool_results_is_rejected(self):
        with pytest.raises(ValueError, match="skip_tool_message_in_guardrail are enabled together"):
            self._initialize(
                "bedrock-skip-tool-scan-only-combo",
                {
                    "guardrail": "bedrock",
                    "mode": "pre_call",
                    "guardrailIdentifier": "gr-1",
                    "guardrailVersion": "1",
                    "skip_tool_message_in_guardrail": True,
                    "scan_only_tool_results": True,
                },
            )


@pytest.mark.asyncio
async def test_update_guardrail_in_db_raises_when_row_missing():
    prisma_client = MagicMock()
    prisma_client.db.litellm_guardrailstable.update = AsyncMock(return_value=None)

    with pytest.raises(
        Exception,
        match=r"^Error updating guardrail in DB: Guardrail not found, passed guardrail_id=missing-guardrail$",
    ):
        await GuardrailRegistry().update_guardrail_in_db(
            guardrail_id="missing-guardrail",
            guardrail=Guardrail(
                guardrail_name="missing-guardrail",
                litellm_params=LitellmParams(guardrail="bedrock", mode="pre_call"),
            ),
            prisma_client=prisma_client,
        )


def test_reinitialize_guardrail_restores_previous_on_failure():
    """A reinitialization whose new params make the guardrail constructor raise must
    restore the previous instance instead of leaving the guardrail silently removed:
    an enforcing guardrail must never fail open because an update was bad."""
    from litellm.proxy.guardrails import guardrail_registry as registry_module

    def _initializer(litellm_params, guardrail):
        if litellm_params.api_key == "boom":
            raise ValueError("invalid updated params")
        return CustomGuardrail(
            guardrail_name=guardrail["guardrail_name"],
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
        )

    registry_module.guardrail_initializer_registry["restore_test"] = _initializer
    try:
        handler = InMemoryGuardrailHandler()
        created = handler.initialize_guardrail(
            guardrail={
                "guardrail_name": "restore-me",
                "litellm_params": {"guardrail": "restore_test", "mode": "pre_call", "api_key": "ok"},
            },
        )
        guardrail_id = created["guardrail_id"]
        original_instance = handler.guardrail_id_to_custom_guardrail[guardrail_id]

        with pytest.raises(ValueError, match="invalid updated params"):
            handler.reinitialize_guardrail(
                guardrail={
                    "guardrail_id": guardrail_id,
                    "guardrail_name": "restore-me",
                    "litellm_params": {"guardrail": "restore_test", "mode": "pre_call", "api_key": "boom"},
                },
            )

        assert guardrail_id in handler.IN_MEMORY_GUARDRAILS
        restored = handler.guardrail_id_to_custom_guardrail[guardrail_id]
        assert restored is not None and restored is not original_instance
        assert restored.guardrail_name == "restore-me"
    finally:
        registry_module.guardrail_initializer_registry.pop("restore_test", None)
