from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_registry import (
    _normalize_model_group_allowlist,
    get_guardrail_initializer_from_hooks,
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


class TestNormalizeModelGroupAllowlist:
    def test_none_returns_empty_frozenset(self):
        assert _normalize_model_group_allowlist(None) == frozenset()

    def test_empty_list_returns_empty_frozenset(self):
        assert _normalize_model_group_allowlist([]) == frozenset()

    def test_normalizes_to_lowercase_and_strips_whitespace(self):
        result = _normalize_model_group_allowlist(["  AI-Gateway-Low  ", "AI-Gateway-High"])
        assert result == frozenset({"ai-gateway-low", "ai-gateway-high"})

    def test_deduplicates_entries(self):
        result = _normalize_model_group_allowlist(["group-a", "GROUP-A", "  group-a  "])
        assert result == frozenset({"group-a"})

    def test_drops_blank_entries(self):
        result = _normalize_model_group_allowlist(["valid-group", "  ", ""])
        assert result == frozenset({"valid-group"})

    def test_returns_frozenset(self):
        result = _normalize_model_group_allowlist(["group-a"])
        assert isinstance(result, frozenset)


def test_initialize_guardrail_sets_apply_guardrail_to_model_groups_on_callback():
    handler = InMemoryGuardrailHandler()

    callback = CustomGuardrail(
        guardrail_name="test-guardrail",
        default_on=True,
        event_hook=GuardrailEventHooks.pre_call,
    )
    litellm_params = LitellmParams(
        guardrail="test-guardrail",
        mode="pre_call",
        default_on=True,
        apply_guardrail_to_model_groups=["Group-A", "  GROUP-B  "],
    )
    setattr(callback, "skip_system_message_in_guardrail", None)
    setattr(callback, "skip_tool_message_in_guardrail", None)
    setattr(
        callback,
        "apply_guardrail_to_model_groups",
        _normalize_model_group_allowlist(
            getattr(litellm_params, "apply_guardrail_to_model_groups", None)
        ),
    )

    assert getattr(callback, "apply_guardrail_to_model_groups") == frozenset({"group-a", "group-b"})

