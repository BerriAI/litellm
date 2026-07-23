"""Behavior pins for ProxyConfig and module-level config scrubbers.

Pins covered:
- Module-level: ``_is_remote_module_url``, ``_scrub_guardrail_inner``,
  ``_scrub_db_overlay_remote_module_loads``
- All ``ProxyConfig`` methods listed in the pin file.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.proxy.proxy_server import (
    ProxyConfig,
    _is_remote_module_url,
    _scrub_db_overlay_remote_module_loads,
    _scrub_guardrail_inner,
    resolve_complexity_router_plugins,
    resolve_routing_plugins,
)

from .conftest import normalize

# ---------------------------------------------------------------------------
# _is_remote_module_url
# ---------------------------------------------------------------------------


def test__is_remote_module_url_identifies_remote_and_local():
    result = {
        "s3": _is_remote_module_url("s3://bucket/key.py"),
        "gcs": _is_remote_module_url("gcs://bucket/key.py"),
        "local": _is_remote_module_url("my.module.path"),
        "none": _is_remote_module_url(None),
        "int": _is_remote_module_url(42),
    }
    assert result == {
        "s3": True,
        "gcs": True,
        "local": False,
        "none": False,
        "int": False,
    }


def test__is_remote_module_url_raises_on_unexpected_iteration():
    class Bad:
        def __str__(self):
            raise RuntimeError("boom")

    # Function never raises — assert the False fall-through for non-str.
    with pytest.raises(AssertionError):
        # Force an error-style assertion: object is not str, returns False.
        assert _is_remote_module_url(Bad()) is True


# ---------------------------------------------------------------------------
# _scrub_guardrail_inner
# ---------------------------------------------------------------------------


def test__scrub_guardrail_inner_strips_remote_callbacks_and_guardrail():
    inner: Dict[str, Any] = {
        "callbacks": ["safe.mod", "s3://attacker/m.py", "gcs://x/y.py"],
        "guardrail": "s3://attacker/g.py",
        "default_on": True,
    }
    _scrub_guardrail_inner(inner)
    assert normalize(inner) == {
        "callbacks": ["safe.mod"],
        "guardrail": None,
        "default_on": True,
    }


def test__scrub_guardrail_inner_invalid_callbacks_type_is_ignored():
    inner = {"callbacks": "not-a-list", "guardrail": "ok.module"}
    _scrub_guardrail_inner(inner)
    # No mutation on non-list callbacks; guardrail untouched (not remote).
    assert inner == {"callbacks": "not-a-list", "guardrail": "ok.module"}


# ---------------------------------------------------------------------------
# _scrub_db_overlay_remote_module_loads
# ---------------------------------------------------------------------------


def test__scrub_db_overlay_remote_module_loads_strips_lists_and_strs():
    db_value = {
        "callbacks": ["safe", "s3://x/y.py"],
        "success_callback": ["gcs://a/b.py", "safe2"],
        "post_call_rules": "s3://bad/m.py",
        "guardrails": [
            {"g1": {"callbacks": ["s3://x"], "guardrail": "ok"}},
        ],
    }
    out = _scrub_db_overlay_remote_module_loads("litellm_settings", db_value)
    assert normalize(out) == {
        "callbacks": ["safe"],
        "success_callback": ["safe2"],
        "post_call_rules": None,
        "guardrails": [{"g1": {"callbacks": [], "guardrail": "ok"}}],
    }


def test__scrub_db_overlay_remote_module_loads_invalid_non_dict_returns_input():
    # Non-dict input bypasses scrubbing entirely.
    assert _scrub_db_overlay_remote_module_loads("litellm_settings", "raw") == "raw"


# ---------------------------------------------------------------------------
# resolve_complexity_router_plugins
# ---------------------------------------------------------------------------


def test_resolve_complexity_router_plugins_no_plugins_key_is_a_noop():
    config: Dict[str, Any] = {"tiers": {"SIMPLE": "gpt-4o-mini"}}
    resolve_complexity_router_plugins(
        model_name="smart-router", complexity_router_config=config, config_file_path=None
    )
    assert config == {"tiers": {"SIMPLE": "gpt-4o-mini"}}


def test_resolve_complexity_router_plugins_resolves_dotted_path_to_live_instance(tmp_path):
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text(
        "class _Plugin:\n"
        "    async def run(self, context):\n"
        "        return context\n"
        "\n"
        "my_plugin_instance = _Plugin()\n"
    )
    config: Dict[str, Any] = {"plugins": ["my_plugin.my_plugin_instance"]}

    resolve_complexity_router_plugins(
        model_name="smart-router",
        complexity_router_config=config,
        config_file_path=str(tmp_path / "config.yaml"),
    )

    assert len(config["plugins"]) == 1
    assert hasattr(config["plugins"][0], "run")
    assert type(config["plugins"][0]).__name__ == "_Plugin"


def test_resolve_complexity_router_plugins_rejects_non_routing_plugin_object(tmp_path):
    plugin_file = tmp_path / "bad_plugin.py"
    plugin_file.write_text("not_a_plugin = object()\n")
    config: Dict[str, Any] = {"plugins": ["bad_plugin.not_a_plugin"]}

    with pytest.raises(ValueError, match="does not implement the RoutingPlugin interface"):
        resolve_complexity_router_plugins(
            model_name="smart-router",
            complexity_router_config=config,
            config_file_path=str(tmp_path / "config.yaml"),
        )


def test_resolve_complexity_router_plugins_rejects_synchronous_run_method(tmp_path):
    """Regression: @runtime_checkable only checks that `run` exists as an attribute,
    not that it's a coroutine function. A plugin with a synchronous `run` passes a bare
    isinstance() check and would only fail at request time with a confusing
    `TypeError: object RoutingContext can't be used in 'await' expression`. Reported
    by Greptile on PR #33251."""
    plugin_file = tmp_path / "sync_plugin.py"
    plugin_file.write_text(
        "class _SyncPlugin:\n"
        "    def run(self, context):\n"
        "        return context\n"
        "\n"
        "sync_plugin_instance = _SyncPlugin()\n"
    )
    config: Dict[str, Any] = {"plugins": ["sync_plugin.sync_plugin_instance"]}

    with pytest.raises(ValueError, match="does not implement the RoutingPlugin interface"):
        resolve_complexity_router_plugins(
            model_name="smart-router",
            complexity_router_config=config,
            config_file_path=str(tmp_path / "config.yaml"),
        )


# ---------------------------------------------------------------------------
# resolve_routing_plugins
# ---------------------------------------------------------------------------


def test_resolve_routing_plugins_resolves_dotted_paths(tmp_path):
    plugin_file = tmp_path / "rs_plugin.py"
    plugin_file.write_text(
        "class _Plugin:\n"
        "    async def run(self, context):\n"
        "        return context\n"
        "\n"
        "rs_plugin_instance = _Plugin()\n"
    )

    resolved = resolve_routing_plugins(
        plugin_paths=["rs_plugin.rs_plugin_instance"],
        config_file_path=str(tmp_path / "config.yaml"),
        source_label="router_settings.plugins",
    )

    assert len(resolved) == 1
    assert type(resolved[0]).__name__ == "_Plugin"


def test_resolve_routing_plugins_passes_through_instances(tmp_path):
    class _Plugin:
        async def run(self, context):
            return context

    instance = _Plugin()
    resolved = resolve_routing_plugins(
        plugin_paths=[instance],
        config_file_path=None,
        source_label="router_settings.plugins",
    )
    assert resolved == [instance]


def test_resolve_routing_plugins_rejects_non_routing_plugin(tmp_path):
    plugin_file = tmp_path / "bad_rs_plugin.py"
    plugin_file.write_text("not_a_plugin = object()\n")

    with pytest.raises(ValueError, match="router_settings.plugins"):
        resolve_routing_plugins(
            plugin_paths=["bad_rs_plugin.not_a_plugin"],
            config_file_path=str(tmp_path / "config.yaml"),
            source_label="router_settings.plugins",
        )


def test_resolve_routing_plugins_rejects_synchronous_run(tmp_path):
    plugin_file = tmp_path / "sync_rs_plugin.py"
    plugin_file.write_text(
        "class _SyncPlugin:\n"
        "    def run(self, context):\n"
        "        return context\n"
        "\n"
        "sync_plugin_instance = _SyncPlugin()\n"
    )

    with pytest.raises(ValueError, match="does not implement the RoutingPlugin interface"):
        resolve_routing_plugins(
            plugin_paths=["sync_rs_plugin.sync_plugin_instance"],
            config_file_path=str(tmp_path / "config.yaml"),
            source_label="router_settings.plugins",
        )


# ---------------------------------------------------------------------------
# ProxyConfig.__init__
# ---------------------------------------------------------------------------


def test_ProxyConfig___init___sets_defaults():
    pc = ProxyConfig()
    snapshot = {
        "config": pc.config,
        "last_semantic_filter_config": pc._last_semantic_filter_config,
        "worker_registry": pc.worker_registry,
    }
    assert snapshot == {
        "config": {},
        "last_semantic_filter_config": None,
        "worker_registry": [],
    }


def test_ProxyConfig___init___raises_when_called_with_bad_args():
    with pytest.raises(TypeError):
        ProxyConfig("unexpected-positional")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ProxyConfig.is_yaml
# ---------------------------------------------------------------------------


def test_ProxyConfig_is_yaml_detects_yaml_and_non_yaml(tmp_path):
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text("model_list: []\n")
    yml_file = tmp_path / "c.yml"
    yml_file.write_text("model_list: []\n")
    json_file = tmp_path / "c.json"
    json_file.write_text("{}")
    pc = ProxyConfig()
    result = {
        "yaml": pc.is_yaml(str(yaml_file)),
        "yml": pc.is_yaml(str(yml_file)),
        "json": pc.is_yaml(str(json_file)),
    }
    assert result == {"yaml": True, "yml": True, "json": False}


def test_ProxyConfig_is_yaml_missing_file_returns_false():
    pc = ProxyConfig()
    assert pc.is_yaml("/no/such/path/here.yaml") is False


# ---------------------------------------------------------------------------
# ProxyConfig._load_yaml_file
# ---------------------------------------------------------------------------


def test_ProxyConfig__load_yaml_file_returns_parsed_dict(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("a: 1\nb: two\nc:\n  - x\n  - y\n")
    pc = ProxyConfig()
    result = pc._load_yaml_file(str(f))
    assert result == {"a": 1, "b": "two", "c": ["x", "y"]}


def test_ProxyConfig__load_yaml_file_raises_on_missing_file():
    pc = ProxyConfig()
    with pytest.raises(Exception):
        pc._load_yaml_file("/no/such/file.yaml")


# ---------------------------------------------------------------------------
# ProxyConfig._get_config_from_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__get_config_from_file_loads_yaml(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("model_list: []\ngeneral_settings: {}\nlitellm_settings:\n  drop_params: true\n")
    pc = ProxyConfig()
    result = await pc._get_config_from_file(config_file_path=str(f))
    assert result == {
        "model_list": [],
        "general_settings": {},
        "litellm_settings": {"drop_params": True},
    }


@pytest.mark.asyncio
async def test_ProxyConfig__get_config_from_file_missing_path_raises():
    pc = ProxyConfig()
    with pytest.raises(Exception):
        await pc._get_config_from_file(config_file_path="/no/such/file.yaml")


# ---------------------------------------------------------------------------
# ProxyConfig._process_includes
# ---------------------------------------------------------------------------


def test_ProxyConfig__process_includes_merges_files(tmp_path):
    inc = tmp_path / "models.yaml"
    inc.write_text("model_list:\n  - model_name: gpt-4\n")
    pc = ProxyConfig()
    cfg = {"include": ["models.yaml"], "model_list": [], "litellm_settings": {}}
    result = pc._process_includes(cfg, base_dir=str(tmp_path))
    assert result == {
        "model_list": [{"model_name": "gpt-4"}],
        "litellm_settings": {},
    }


def test_ProxyConfig__process_includes_missing_file_raises(tmp_path):
    pc = ProxyConfig()
    with pytest.raises(FileNotFoundError):
        pc._process_includes({"include": ["nope.yaml"]}, base_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# ProxyConfig.save_config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig_save_config_writes_yaml_when_no_db(tmp_path, monkeypatch):
    target = tmp_path / "out.yaml"
    monkeypatch.setattr("litellm.proxy.proxy_server.user_config_file_path", str(target))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    pc = ProxyConfig()
    cfg = {"model_list": [], "general_settings": {"a": 1}, "litellm_settings": {}}
    await pc.save_config(cfg)
    import yaml as _yaml

    loaded = _yaml.safe_load(target.read_text())
    assert loaded == cfg


@pytest.mark.asyncio
async def test_ProxyConfig_save_config_invalid_path_raises(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_config_file_path",
        "/no/such/dir/out.yaml",
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    pc = ProxyConfig()
    with pytest.raises(Exception):
        await pc.save_config({"x": 1})


@pytest.mark.asyncio
async def test_ProxyConfig_save_config_db_omits_environment_variables_by_default(monkeypatch):
    """A save_config after get_config() (which resolves os.environ/ placeholders
    to plaintext and merges the environment_variables section) must not snapshot
    those env vars into the DB config row. Persisting them would make a stale DB
    row shadow YAML/container env on every subsequent restart."""
    mock_prisma = MagicMock()
    mock_prisma.insert_data = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    # a valid salt so the env-var encryption path (reached only if the pop
    # regresses) runs cleanly, making this fail on the assertion below rather
    # than on an incidental encryption crash
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test-salt-key")

    pc = ProxyConfig()
    cfg = {
        "model_list": [{"model_name": "gpt-4o"}],
        "litellm_settings": {"success_callback": ["langfuse"]},
        "environment_variables": {"OPENAI_API_KEY": "sk-from-yaml"},
    }
    await pc.save_config(cfg)

    mock_prisma.insert_data.assert_awaited_once()
    written = mock_prisma.insert_data.await_args.kwargs["data"]
    assert "environment_variables" not in written
    # unrelated sections are still persisted; model_list is stripped as before
    assert written["litellm_settings"] == {"success_callback": ["langfuse"]}
    assert "model_list" not in written
    # the caller's dict is not mutated (save_config works on a copy)
    assert cfg["environment_variables"] == {"OPENAI_API_KEY": "sk-from-yaml"}


@pytest.mark.asyncio
async def test_ProxyConfig_save_config_db_persists_environment_variables_when_opted_in(monkeypatch):
    """The explicit opt-in path (include_env_vars=True) still persists env vars,
    encrypted, so the dedicated config-update flow can write them."""
    mock_prisma = MagicMock()
    mock_prisma.insert_data = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test-salt-key")

    pc = ProxyConfig()
    cfg = {"litellm_settings": {}, "environment_variables": {"OPENAI_API_KEY": "sk-explicit"}}
    await pc.save_config(cfg, include_env_vars=True)

    mock_prisma.insert_data.assert_awaited_once()
    written = mock_prisma.insert_data.await_args.kwargs["data"]
    assert set(written["environment_variables"].keys()) == {"OPENAI_API_KEY"}
    # value is encrypted at rest, not the plaintext it came in as
    assert written["environment_variables"]["OPENAI_API_KEY"] != "sk-explicit"


def _install_fake_config_repo(monkeypatch, existing_row):
    """Route ProxyConfig's ConfigRepository through an in-memory fake that
    records the value written to the environment_variables row."""
    captured: dict = {}

    class _FakeTable:
        async def find_first(self, where):
            return SimpleNamespace(param_value=existing_row) if existing_row is not None else None

        async def upsert(self, where, data):
            captured["value"] = json.loads(data["update"]["param_value"])

    class _FakeRepo:
        def __init__(self, client):
            self.table = _FakeTable()

    monkeypatch.setattr("litellm.proxy.proxy_server.ConfigRepository", _FakeRepo)
    monkeypatch.setattr("litellm.proxy.proxy_server.invalidate_config_param", AsyncMock())
    return captured


@pytest.mark.asyncio
async def test_ProxyConfig_save_environment_variables_merges_sets_and_deletes(monkeypatch):
    """The per-key env-var write updates/deletes only the named keys and leaves
    every other stored key untouched, so an unrelated env var is never lost or
    snapshotted."""
    captured = _install_fake_config_repo(
        monkeypatch,
        existing_row={"EXISTING_KEY": "ciphertext-existing", "UI_LOGO_PATH": "old-logo", "LITELLM_FAVICON_URL": "old"},
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test-salt-key")

    pc = ProxyConfig()
    await pc.save_environment_variables({"UI_LOGO_PATH": "new-logo", "LITELLM_FAVICON_URL": None})

    written = captured["value"]
    # unrelated key preserved byte-for-byte
    assert written["EXISTING_KEY"] == "ciphertext-existing"
    # set key updated and encrypted (not the plaintext)
    assert "UI_LOGO_PATH" in written and written["UI_LOGO_PATH"] != "new-logo"
    # None-valued key deleted
    assert "LITELLM_FAVICON_URL" not in written


@pytest.mark.asyncio
async def test_ProxyConfig_save_environment_variables_noop_without_db(monkeypatch):
    """With no DB configured the per-key write must do nothing (never touch the
    config repository)."""
    captured = _install_fake_config_repo(monkeypatch, existing_row={})
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})

    pc = ProxyConfig()
    await pc.save_environment_variables({"UI_LOGO_PATH": "x"})

    assert "value" not in captured


# ---------------------------------------------------------------------------
# ProxyConfig._check_for_os_environ_vars
# ---------------------------------------------------------------------------


def test_ProxyConfig__check_for_os_environ_vars_substitutes(monkeypatch):
    monkeypatch.setenv("MY_TEST_VAR", "secret-value")
    pc = ProxyConfig()
    cfg = {
        "a": "os.environ/MY_TEST_VAR",
        "b": 2,
        "nested": {"c": "os.environ/MY_TEST_VAR"},
    }
    out = pc._check_for_os_environ_vars(cfg)
    assert out == {"a": "secret-value", "b": 2, "nested": {"c": "secret-value"}}


def test_ProxyConfig__check_for_os_environ_vars_missing_env_returns_none(monkeypatch):
    monkeypatch.delenv("NONEXISTENT_TEST_VAR_X", raising=False)
    pc = ProxyConfig()
    cfg = {"a": "os.environ/NONEXISTENT_TEST_VAR_X"}
    out = pc._check_for_os_environ_vars(cfg)
    # get_secret returns None when not found — assert observable shape.
    assert out["a"] is None


# ---------------------------------------------------------------------------
# ProxyConfig._get_team_config
# ---------------------------------------------------------------------------


def test_ProxyConfig__get_team_config_returns_match():
    pc = ProxyConfig()
    teams = [
        {"team_id": "t1", "max_budget": 10, "model": "gpt-4"},
        {"team_id": "t2", "max_budget": 20, "model": "claude"},
    ]
    out = pc._get_team_config(team_id="t1", all_teams_config=teams)
    assert out == {"team_id": "t1", "max_budget": 10, "model": "gpt-4"}


def test_ProxyConfig__get_team_config_missing_team_id_raises():
    pc = ProxyConfig()
    with pytest.raises(Exception):
        pc._get_team_config(team_id="t1", all_teams_config=[{"no_id_field": True}])


# ---------------------------------------------------------------------------
# ProxyConfig.load_team_config
# ---------------------------------------------------------------------------


def test_ProxyConfig_load_team_config_returns_team_dict():
    pc = ProxyConfig()
    pc.config = {
        "litellm_settings": {
            "default_team_settings": [
                {"team_id": "ta", "max_budget": 99, "drop_params": True},
            ]
        }
    }
    out = pc.load_team_config(team_id="ta")
    assert out == {"team_id": "ta", "max_budget": 99, "drop_params": True}


def test_ProxyConfig_load_team_config_no_settings_returns_empty():
    pc = ProxyConfig()
    pc.config = {"litellm_settings": {}}
    # Missing entry — happy path returns {} (no default_team_settings).
    out = pc.load_team_config(team_id="missing")
    assert out == {}
    # Error-style: a misconfigured team list without team_id raises.
    pc.config = {"litellm_settings": {"default_team_settings": [{"no_id": True}]}}
    with pytest.raises(Exception):
        pc.load_team_config(team_id="anything")


# ---------------------------------------------------------------------------
# ProxyConfig._init_cache
# ---------------------------------------------------------------------------


def test_ProxyConfig__init_cache_sets_litellm_cache(monkeypatch):
    pc = ProxyConfig()
    monkeypatch.setattr(litellm, "cache", None, raising=False)
    pc._init_cache(cache_params={"type": "local"})
    snapshot = {
        "cache_is_set": litellm.cache is not None,
        "cache_type_name": type(litellm.cache).__name__,
        "params_used": "local",
    }
    assert snapshot == {
        "cache_is_set": True,
        "cache_type_name": "Cache",
        "params_used": "local",
    }


def test_ProxyConfig__init_cache_invalid_params_raises():
    pc = ProxyConfig()
    with pytest.raises(Exception):
        pc._init_cache(cache_params={"type": "this-cache-type-does-not-exist"})


# ---------------------------------------------------------------------------
# ProxyConfig.switch_on_llm_response_caching
# ---------------------------------------------------------------------------


def test_ProxyConfig_switch_on_llm_response_caching_sets_flag(monkeypatch):
    pc = ProxyConfig()
    fake_router = MagicMock()
    fake_router.cache_responses = False
    fake_cache = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router)
    monkeypatch.setattr(litellm, "cache", fake_cache, raising=False)
    pc.switch_on_llm_response_caching()
    snapshot = {
        "cache_responses": fake_router.cache_responses,
        "router_set": True,
        "cache_set": True,
    }
    assert snapshot == {
        "cache_responses": True,
        "router_set": True,
        "cache_set": True,
    }


def test_ProxyConfig_switch_on_llm_response_caching_missing_router_noop(monkeypatch):
    pc = ProxyConfig()
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr(litellm, "cache", None, raising=False)
    # No router and no cache — should silently no-op (no raise).
    pc.switch_on_llm_response_caching()
    # Error-style: prove no router was created.
    with pytest.raises(AttributeError):
        _ = pc.does_not_exist  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ProxyConfig.get_config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig_get_config_loads_from_file(tmp_path, monkeypatch):
    f = tmp_path / "c.yaml"
    f.write_text("model_list: []\ngeneral_settings: {}\nlitellm_settings: {}\n")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)
    pc = ProxyConfig()
    cfg = await pc.get_config(config_file_path=str(f))
    assert cfg == {
        "model_list": [],
        "general_settings": {},
        "litellm_settings": {},
    }


@pytest.mark.asyncio
async def test_ProxyConfig_get_config_missing_file_raises(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)
    pc = ProxyConfig()
    with pytest.raises(Exception):
        await pc.get_config(config_file_path="/no/such/path.yaml")


# ---------------------------------------------------------------------------
# ProxyConfig.update_config_state / get_config_state
# ---------------------------------------------------------------------------


def test_ProxyConfig_update_config_state_and_get_config_state_roundtrip():
    pc = ProxyConfig()
    cfg = {"model_list": [], "general_settings": {"x": 1}, "litellm_settings": {}}
    pc.update_config_state(config=cfg)
    out = pc.get_config_state()
    assert out == cfg
    # Mutating the returned dict must not affect internal state.
    out["model_list"].append({"new": True})
    assert pc.get_config_state() == cfg


def test_ProxyConfig_update_config_state_with_bad_arg_raises():
    pc = ProxyConfig()
    with pytest.raises(TypeError):
        pc.update_config_state()  # type: ignore[call-arg]


def test_ProxyConfig_get_config_state_handles_undeepcopyable(monkeypatch):
    # Pins ProxyConfig.get_config_state — see source for behavior.
    pc = ProxyConfig()

    class NoCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("nope")

    pc.config = {"x": NoCopy()}  # type: ignore[assignment]
    # Exception is caught internally and an empty dict returned.
    assert pc.get_config_state() == {}


# ---------------------------------------------------------------------------
# ProxyConfig.load_credential_list
# ---------------------------------------------------------------------------


def test_ProxyConfig_load_credential_list_returns_items():
    pc = ProxyConfig()
    creds = pc.load_credential_list(
        {
            "credential_list": [
                {
                    "credential_name": "openai-key",
                    "credential_info": {"provider": "openai"},
                    "credential_values": {"api_key": "sk-x"},
                }
            ]
        }
    )
    assert len(creds) == 1
    dumped = creds[0].model_dump()
    assert dumped == {
        "credential_name": "openai-key",
        "credential_info": {"provider": "openai"},
        "credential_values": {"api_key": "sk-x"},
    }


def test_ProxyConfig_load_credential_list_invalid_entry_raises():
    pc = ProxyConfig()
    with pytest.raises(Exception):
        pc.load_credential_list({"credential_list": [{"missing_required": True}]})


# ---------------------------------------------------------------------------
# ProxyConfig.parse_search_tools
# ---------------------------------------------------------------------------


def test_ProxyConfig_parse_search_tools_returns_parsed():
    pc = ProxyConfig()
    cfg = {
        "search_tools": [
            {
                "search_tool_name": "web",
                "litellm_params": {"search_provider": "google"},
            }
        ]
    }
    out = pc.parse_search_tools(cfg)
    assert out is not None
    assert len(out) == 1
    assert dict(out[0]) == {
        "search_tool_name": "web",
        "litellm_params": {"search_provider": "google"},
    }


def test_ProxyConfig_parse_search_tools_missing_returns_none():
    pc = ProxyConfig()
    assert pc.parse_search_tools({}) is None


def test_ProxyConfig_merge_config_and_db_search_tools_returns_superset():
    config_tools = [
        {
            "search_tool_name": "config-search",
            "litellm_params": {"search_provider": "tavily"},
        }
    ]
    db_tools = [
        {
            "search_tool_name": "db-search",
            "litellm_params": {
                "search_provider": "exa_ai",
                "api_key": "fake-db-key",
            },
        }
    ]

    merged = ProxyConfig._merge_config_and_db_search_tools(
        config_search_tools=config_tools,
        db_search_tools=db_tools,
    )

    assert [tool["search_tool_name"] for tool in merged] == ["config-search", "db-search"]
    assert merged[1]["litellm_params"]["api_key"] == "fake-db-key"


def test_ProxyConfig_merge_config_and_db_search_tools_prefers_db_duplicate():
    config_tools = [
        {
            "search_tool_name": "shared-search",
            "litellm_params": {"search_provider": "tavily"},
        },
        {
            "search_tool_name": "config-only",
            "litellm_params": {"search_provider": "perplexity"},
        },
    ]
    db_tools = [
        {
            "search_tool_name": "shared-search",
            "litellm_params": {
                "search_provider": "exa_ai",
                "api_key": "fake-db-key",
            },
        }
    ]

    merged = ProxyConfig._merge_config_and_db_search_tools(
        config_search_tools=config_tools,
        db_search_tools=db_tools,
    )

    assert [tool["search_tool_name"] for tool in merged] == ["config-only", "shared-search"]
    assert merged[1]["litellm_params"]["search_provider"] == "exa_ai"
    assert merged[1]["litellm_params"]["api_key"] == "fake-db-key"


@pytest.mark.asyncio
async def test_ProxyConfig__init_search_tools_in_db_loads_merged_tools(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.router_utils.search_api_router import SearchAPIRouter

    pc = ProxyConfig()
    pc.update_config_state(
        {
            "search_tools": [
                {
                    "search_tool_name": "shared-search",
                    "litellm_params": {"search_provider": "tavily"},
                },
                {
                    "search_tool_name": "config-only",
                    "litellm_params": {"search_provider": "perplexity"},
                },
            ]
        }
    )
    db_tools = [
        {
            "search_tool_name": "shared-search",
            "litellm_params": {
                "search_provider": "exa_ai",
                "api_key": "fake-db-key",
            },
        }
    ]
    fake_router = MagicMock()
    mock_get_db_tools = AsyncMock(return_value=db_tools)
    mock_update_router = AsyncMock()

    monkeypatch.setattr(proxy_server, "llm_router", fake_router)
    monkeypatch.setattr(
        "litellm.proxy.search_endpoints.search_tool_registry.SearchToolRegistry.get_all_search_tools_from_db",
        mock_get_db_tools,
    )
    monkeypatch.setattr(SearchAPIRouter, "update_router_search_tools", mock_update_router)

    await pc._init_search_tools_in_db(prisma_client=MagicMock())

    mock_get_db_tools.assert_awaited_once()
    mock_update_router.assert_awaited_once()
    update_kwargs = mock_update_router.await_args.kwargs
    assert update_kwargs["router_instance"] is fake_router
    assert [tool["search_tool_name"] for tool in update_kwargs["search_tools"]] == [
        "config-only",
        "shared-search",
    ]
    assert update_kwargs["search_tools"][1]["litellm_params"]["api_key"] == "fake-db-key"


@pytest.mark.asyncio
async def test_ProxyConfig__init_search_tools_in_db_skips_empty_router_update(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.router_utils.search_api_router import SearchAPIRouter

    pc = ProxyConfig()
    pc.update_config_state({})
    mock_get_db_tools = AsyncMock(return_value=[])
    mock_update_router = AsyncMock()

    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        "litellm.proxy.search_endpoints.search_tool_registry.SearchToolRegistry.get_all_search_tools_from_db",
        mock_get_db_tools,
    )
    monkeypatch.setattr(SearchAPIRouter, "update_router_search_tools", mock_update_router)

    await pc._init_search_tools_in_db(prisma_client=MagicMock())

    mock_get_db_tools.assert_awaited_once()
    mock_update_router.assert_not_awaited()


# ---------------------------------------------------------------------------
# ProxyConfig._load_environment_variables
# ---------------------------------------------------------------------------


def test_ProxyConfig__load_environment_variables_sets_env(monkeypatch):
    monkeypatch.delenv("TEST_LOAD_ENV_X", raising=False)
    pc = ProxyConfig()
    pc._load_environment_variables({"environment_variables": {"TEST_LOAD_ENV_X": "hello"}})
    result = {
        "TEST_LOAD_ENV_X": os.environ.get("TEST_LOAD_ENV_X"),
        "set": True,
        "len": 1,
    }
    assert result == {"TEST_LOAD_ENV_X": "hello", "set": True, "len": 1}


def test_ProxyConfig__load_environment_variables_blocks_dangerous_keys(monkeypatch):
    original_path = os.environ.get("PATH", "")
    pc = ProxyConfig()
    pc._load_environment_variables({"environment_variables": {"PATH": "/evil/bin"}})
    # PATH must be unchanged — it's a blocked key.
    assert os.environ.get("PATH", "") == original_path


# ---------------------------------------------------------------------------
# ProxyConfig.load_config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig_load_config_minimal_yaml(tmp_path, monkeypatch):
    f = tmp_path / "c.yaml"
    f.write_text("model_list: []\ngeneral_settings: {}\nlitellm_settings: {}\n")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)
    pc = ProxyConfig()
    try:
        await pc.load_config(router=None, config_file_path=str(f))
        raised = False
    except Exception:
        raised = True
    snapshot = {
        "raised": raised,
        "config_loaded": pc.config is not None,
        "model_list_key_present": "model_list" in pc.config,
    }
    assert snapshot == {
        "raised": False,
        "config_loaded": True,
        "model_list_key_present": True,
    }


@pytest.mark.asyncio
async def test_ProxyConfig_load_config_resolves_router_settings_plugins(tmp_path, monkeypatch):
    """Regression: router_settings.plugins dotted-path strings must be resolved to
    live RoutingPlugin instances on the created Router. Previously they were passed
    through as raw strings and only blew up at request time when the pipeline tried
    to `await "some.string".run(context)`."""
    plugin_file = tmp_path / "rs_plugin.py"
    plugin_file.write_text(
        "class _Plugin:\n"
        "    async def run(self, context):\n"
        "        return context\n"
        "\n"
        "rs_plugin_instance = _Plugin()\n"
    )
    f = tmp_path / "c.yaml"
    f.write_text(
        "model_list: []\n"
        "general_settings: {}\n"
        "litellm_settings: {}\n"
        "router_settings:\n"
        "  plugins:\n"
        "    - rs_plugin.rs_plugin_instance\n"
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    router, _model_list, _general_settings = await ProxyConfig().load_config(
        router=None, config_file_path=str(f)
    )

    assert len(router.routing_plugins) == 1
    assert type(router.routing_plugins[0]).__name__ == "_Plugin"


@pytest.mark.asyncio
async def test_ProxyConfig_load_config_rejects_bad_router_settings_plugin(tmp_path, monkeypatch):
    plugin_file = tmp_path / "bad_rs_plugin.py"
    plugin_file.write_text("not_a_plugin = object()\n")
    f = tmp_path / "c.yaml"
    f.write_text(
        "model_list: []\n"
        "general_settings: {}\n"
        "litellm_settings: {}\n"
        "router_settings:\n"
        "  plugins:\n"
        "    - bad_rs_plugin.not_a_plugin\n"
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    with pytest.raises(ValueError, match="does not implement the RoutingPlugin interface"):
        await ProxyConfig().load_config(router=None, config_file_path=str(f))


@pytest.mark.asyncio
async def test_ProxyConfig_load_config_wires_general_settings_url_validation(tmp_path, monkeypatch):
    """Regression for #26599: SSRF settings in general_settings must reach litellm globals."""
    f = tmp_path / "c.yaml"
    f.write_text(
        "model_list: []\n"
        "general_settings:\n"
        "  user_url_validation: false\n"
        "  user_url_allowed_hosts:\n"
        "    - internal.corp\n"
        "  provider_url_destination_allowed_hosts:\n"
        "    - api.example.com\n"
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    original_validation = litellm.user_url_validation
    original_hosts = list(litellm.user_url_allowed_hosts)
    original_provider_hosts = list(litellm.provider_url_destination_allowed_hosts)
    try:
        await ProxyConfig().load_config(router=None, config_file_path=str(f))
        assert litellm.user_url_validation is False
        assert litellm.user_url_allowed_hosts == ["internal.corp"]
        assert litellm.provider_url_destination_allowed_hosts == ["api.example.com"]
    finally:
        litellm.user_url_validation = original_validation
        litellm.user_url_allowed_hosts = original_hosts
        litellm.provider_url_destination_allowed_hosts = original_provider_hosts


@pytest.mark.asyncio
async def test_ProxyConfig_load_config_wires_config_reload_interval(tmp_path, monkeypatch):
    """general_settings.proxy_config_reload_interval_seconds must reach the proxy_server
    module global that schedules the DB config-reload jobs, so operators can tune multi-pod
    convergence from config.yaml."""
    import litellm.proxy.proxy_server as proxy_server

    f = tmp_path / "c.yaml"
    f.write_text(
        "model_list: []\n"
        "general_settings:\n"
        "  proxy_config_reload_interval_seconds: 47\n"
        "litellm_settings: {}\n"
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    original = proxy_server.proxy_config_reload_interval_seconds
    try:
        await ProxyConfig().load_config(router=None, config_file_path=str(f))
        assert proxy_server.proxy_config_reload_interval_seconds == 47
    finally:
        proxy_server.proxy_config_reload_interval_seconds = original


@pytest.mark.asyncio
async def test_ProxyConfig_load_config_missing_file_raises(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)
    pc = ProxyConfig()
    with pytest.raises(Exception):
        await pc.load_config(router=None, config_file_path="/no/file.yaml")


@pytest.mark.asyncio
async def test_ProxyConfig_load_config_forwards_callback_specific_params(tmp_path, monkeypatch):
    """Regression: callback_settings from config must be forwarded to
    initialize_callbacks_on_proxy as callback_specific_params.

    Callbacks like DatadogCostManagementLogger read their init params (e.g.
    cost_tag_keys) from callback_specific_params[<callback_name>]. If the
    argument is dropped at the call site, they silently initialize with empty
    params and the configured allowlist never takes effect.
    """
    f = tmp_path / "c.yaml"
    f.write_text(
        "model_list: []\n"
        "general_settings: {}\n"
        "callback_settings:\n"
        "  datadog_cost_management:\n"
        "    cost_tag_keys:\n"
        "      - capability\n"
        "      - platform\n"
        "      - ai_product\n"
        "litellm_settings:\n"
        '  callbacks: ["datadog_cost_management"]\n'
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    captured = {}

    def _fake_initialize_callbacks_on_proxy(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.initialize_callbacks_on_proxy",
        _fake_initialize_callbacks_on_proxy,
    )

    pc = ProxyConfig()
    await pc.load_config(router=None, config_file_path=str(f))

    # The callbacks branch must forward the loaded callback_settings.
    assert captured.get("callback_specific_params") == {
        "datadog_cost_management": {"cost_tag_keys": ["capability", "platform", "ai_product"]}
    }


@pytest.mark.asyncio
async def test_ProxyConfig_load_config_blank_callback_settings_does_not_crash(tmp_path, monkeypatch):
    """Regression: `callback_settings:` with no body loads as None because
    dict.get() only falls back to the default when the key is absent. The None
    was forwarded verbatim to initialize_callbacks_on_proxy, where the first
    `"<name>" in callback_specific_params` membership test raised
    TypeError: argument of type 'NoneType' is not iterable, aborting startup.
    Startup must succeed and the callback must initialize with its defaults.
    """
    f = tmp_path / "c.yaml"
    f.write_text(
        "model_list: []\n"
        "general_settings: {}\n"
        "callback_settings:\n"
        "litellm_settings:\n"
        '  callbacks: ["compression_interception"]\n'
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    from litellm.integrations.compression_interception.handler import (
        CompressionInterceptionLogger,
    )

    original_callbacks = list(litellm.callbacks) if isinstance(litellm.callbacks, list) else []
    litellm.callbacks = []
    try:
        pc = ProxyConfig()
        await pc.load_config(router=None, config_file_path=str(f))

        assert any(isinstance(c, CompressionInterceptionLogger) for c in litellm.callbacks)
    finally:
        litellm.callbacks = original_callbacks


# ---------------------------------------------------------------------------
# ProxyConfig._init_non_llm_configs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__init_non_llm_configs_empty_config():
    pc = ProxyConfig()
    try:
        await pc._init_non_llm_configs(config={}, config_file_path=None)
        raised = False
    except Exception:
        raised = True
    snapshot = {
        "raised": raised,
        "worker_registry_len": len(pc.worker_registry),
        "is_list": isinstance(pc.worker_registry, list),
    }
    assert snapshot == {"raised": False, "worker_registry_len": 0, "is_list": True}


@pytest.mark.asyncio
async def test_ProxyConfig__init_non_llm_configs_invalid_worker_registry_raises():
    pc = ProxyConfig()
    with pytest.raises(Exception):
        await pc._init_non_llm_configs(
            config={"worker_registry": [{"totally": "invalid"}]},
            config_file_path=None,
        )


# ---------------------------------------------------------------------------
# ProxyConfig._init_policy_engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__init_policy_engine_no_policies_noop():
    pc = ProxyConfig()
    try:
        await pc._init_policy_engine(config={}, prisma_client=None, llm_router=None)
        raised = False
    except Exception:
        raised = True
    assert {"raised": raised, "called": True, "skipped": True} == {
        "raised": False,
        "called": True,
        "skipped": True,
    }


@pytest.mark.asyncio
async def test_ProxyConfig__init_policy_engine_none_config_noop():
    pc = ProxyConfig()
    # None config returns early without raising.
    await pc._init_policy_engine(config=None, prisma_client=None, llm_router=None)
    # Error-style: invalid policies value should raise.
    with pytest.raises(Exception):
        await pc._init_policy_engine(
            config={"policies": "not-a-list"},
            prisma_client=None,
            llm_router=None,
        )


# ---------------------------------------------------------------------------
# ProxyConfig._load_alerting_settings
# ---------------------------------------------------------------------------


def test_ProxyConfig__load_alerting_settings_noop_when_no_alerting():
    pc = ProxyConfig()
    try:
        pc._load_alerting_settings({})
        raised = False
    except Exception:
        raised = True
    assert {"raised": raised, "called": True, "no_alerting": True} == {
        "raised": False,
        "called": True,
        "no_alerting": True,
    }


def test_ProxyConfig__load_alerting_settings_invalid_alerting_raises():
    pc = ProxyConfig()
    with pytest.raises(Exception):
        # alerting must be iterable — int triggers an error.
        pc._load_alerting_settings({"alerting": 12345})


def test_ProxyConfig__load_alerting_settings_does_not_log_general_settings_dict(monkeypatch):
    """Regression for LIT-4152.

    ``_load_alerting_settings`` used to log ``general_settings`` verbatim in a
    line labelled ``_alerting_callbacks:``, leaking ``master_key``,
    ``database_url``, and any other secret sitting in ``general_settings`` in
    cleartext at DEBUG. The fix logs only the alerting callback list.

    The regression check runs with the last-line-of-defense regex scrubber
    (``SecretRedactionFilter``) DISABLED, since defense in depth is the point.
    The caller must not construct the leaky string, so consumers of the log
    stream that bypass the module filter (versions before it existed,
    ``LITELLM_DISABLE_REDACT_SECRETS=true`` operators, downstream handlers
    that snapshot the record pre-filter) still do not see the secret. Uses a
    dedicated handler rather than caplog because caplog is unreliable under
    pytest-xdist.
    """
    import logging

    import litellm._logging as _logging_module
    from litellm._logging import verbose_proxy_logger

    monkeypatch.setattr(_logging_module, "_ENABLE_SECRET_REDACTION", False)

    class LogRecordHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    master_key_secret = "sk-lit4152-regression-master-key-abcdef1234567890"
    db_url_secret = "postgresql://leak_user:leak_password_9090@leak-host.internal:5432/leak_db"
    settings = {
        "alerting": ["slack"],
        "alerting_threshold": 300,
        "master_key": master_key_secret,
        "database_url": db_url_secret,
    }

    handler = LogRecordHandler()
    handler.setLevel(logging.DEBUG)
    original_level = verbose_proxy_logger.level
    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_proxy_logger.addHandler(handler)
    try:
        try:
            ProxyConfig()._load_alerting_settings(settings)
        except Exception:
            pass  # downstream init may fail without full env; the debug log fires first
        rendered = " ".join(record.getMessage() for record in handler.records)
    finally:
        verbose_proxy_logger.removeHandler(handler)
        verbose_proxy_logger.setLevel(original_level)

    assert master_key_secret not in rendered, f"master_key leaked in logs: {rendered!r}"
    assert db_url_secret not in rendered, f"database_url leaked in logs: {rendered!r}"
    assert "leak_password_9090" not in rendered
    assert any("['slack']" in r.getMessage() for r in handler.records), (
        f"expected the alerting callback list to appear in a debug record; got {[r.getMessage() for r in handler.records]!r}"
    )


# ---------------------------------------------------------------------------
# ProxyConfig.initialize_secret_manager
# ---------------------------------------------------------------------------


def test_ProxyConfig_initialize_secret_manager_none_noop():
    pc = ProxyConfig()
    try:
        pc.initialize_secret_manager(key_management_system=None)
        raised = False
    except Exception:
        raised = True
    assert {"raised": raised, "called": True, "kms": None} == {
        "raised": False,
        "called": True,
        "kms": None,
    }


def test_ProxyConfig_initialize_secret_manager_invalid_kms_raises():
    pc = ProxyConfig()
    with pytest.raises(ValueError):
        pc.initialize_secret_manager(key_management_system="not-a-real-kms")


# ---------------------------------------------------------------------------
# ProxyConfig.get_model_info_with_id
# ---------------------------------------------------------------------------


def test_ProxyConfig_get_model_info_with_id_returns_router_model_info():
    pc = ProxyConfig()
    model = SimpleNamespace(
        model_id="m-1",
        model_info={"id": "m-1"},
        blocked=False,
    )
    out = pc.get_model_info_with_id(model=model, db_model=True)
    dumped = out.model_dump()
    snapshot = {
        "id": dumped.get("id"),
        "db_model": dumped.get("db_model"),
        "blocked": dumped.get("blocked"),
    }
    assert snapshot == {"id": "m-1", "db_model": True, "blocked": False}


def test_ProxyConfig_get_model_info_with_id_missing_model_id_raises(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", False)
    pc = ProxyConfig()
    # model with no model_id, no model_info — accessing .model_id will fail.
    bad = SimpleNamespace(model_info=None)
    with pytest.raises(AttributeError):
        pc.get_model_info_with_id(model=bad)


# ---------------------------------------------------------------------------
# ProxyConfig._delete_deployment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__delete_deployment_empty_returns_zero(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    pc = ProxyConfig()
    result = await pc._delete_deployment(db_models=[])
    snapshot = {"deleted": result, "router_was": "none", "empty_db_models": True}
    assert snapshot == {"deleted": 0, "router_was": "none", "empty_db_models": True}


@pytest.mark.asyncio
async def test_ProxyConfig__delete_deployment_invalid_models_raises(monkeypatch):
    fake_router = MagicMock()
    fake_router.get_model_ids = MagicMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router)
    pc = ProxyConfig()
    with pytest.raises(Exception):
        # Non-model objects without expected attrs trigger an error.
        await pc._delete_deployment(db_models=[{"not_a_model": True}])


# ---------------------------------------------------------------------------
# ProxyConfig._add_deployment
# ---------------------------------------------------------------------------


def test_ProxyConfig__add_deployment_no_router_returns_zero(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    pc = ProxyConfig()
    result = pc._add_deployment(db_models=[MagicMock()])
    snapshot = {"added": result, "router_was": "none", "called": True}
    assert snapshot == {"added": 0, "router_was": "none", "called": True}


def test_ProxyConfig__add_deployment_invalid_litellm_params_skips(monkeypatch):
    fake_router = MagicMock()
    fake_router.upsert_deployment = MagicMock(return_value=None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router)
    pc = ProxyConfig()
    bad = SimpleNamespace(litellm_params="not-a-dict", model_name="x", model_id="x")
    # invalid params logs and continues — assert zero added (error-style branch).
    assert pc._add_deployment(db_models=[bad]) == 0


def test_ProxyConfig__add_deployment_resolves_env_refs_after_db_decrypt(monkeypatch):
    """Every ``os.environ/`` value on an admin-scoped DB row resolves at
    load time, regardless of the field name. Replaces the earlier
    behavior where only fields in ``_DB_LITELLM_PARAM_ENV_REF_KEYS``
    resolved: the whitelist has been removed so the resolver applies to
    every string field."""
    monkeypatch.setenv("LITELLM_DB_MODEL_API_KEY", "resolved-secret")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "master-secret")
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value: value,
    )
    fake_router = MagicMock()
    fake_router.upsert_deployment = MagicMock(return_value=True)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router)
    pc = ProxyConfig()
    db_model = SimpleNamespace(
        model_id="model-1",
        model_name="env-model",
        model_info={"id": "model-1"},
        litellm_params={
            "model": "openai/gpt-4o-mini",
            "api_key": "os.environ/LITELLM_DB_MODEL_API_KEY",
            "api_base": "os.environ/LITELLM_MASTER_KEY",
        },
        blocked=False,
    )

    added = pc._add_deployment(db_models=[db_model])
    deployment = fake_router.upsert_deployment.call_args.kwargs["deployment"]

    assert added == 1
    assert deployment.litellm_params.api_key == "resolved-secret"
    assert deployment.litellm_params.api_base == "master-secret"


def test_ProxyConfig__add_deployment_resolves_team_env_refs(monkeypatch):
    """Team-scoped DB rows now resolve ``os.environ/`` refs the same way
    admin rows do. The prior team-scoped short-circuit and the
    field-by-field whitelist have both been removed; the write-side team
    auth check in ``ModelManagementAuthChecks.can_user_make_model_call``
    remains the single trust boundary. A literal (non-``os.environ/``)
    value still passes through unchanged."""
    monkeypatch.setenv("LITELLM_MASTER_KEY", "master-secret")
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value: value,
    )
    fake_router = MagicMock()
    fake_router.upsert_deployment = MagicMock(return_value=True)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router)
    pc = ProxyConfig()
    db_model = SimpleNamespace(
        model_id="model-1",
        model_name="model_name_team-1_abc",
        model_info={"id": "model-1", "team_id": "team-1"},
        litellm_params={
            "model": "openai/gpt-4o-mini",
            "api_key": "os.environ/LITELLM_MASTER_KEY",
            "api_base": "https://team.example",
        },
        blocked=False,
    )

    added = pc._add_deployment(db_models=[db_model])
    deployment = fake_router.upsert_deployment.call_args.kwargs["deployment"]

    assert added == 1
    assert deployment.litellm_params.api_key == "master-secret"
    assert deployment.litellm_params.api_base == "https://team.example"


def test_ProxyConfig__resolve_db_litellm_param_skips_non_string_values(monkeypatch):
    def fail_on_call(value, key, return_original_value):
        raise AssertionError("decrypt_value_helper should only receive strings")

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        fail_on_call,
    )
    pc = ProxyConfig()

    assert pc._resolve_db_litellm_param(key="tpm", value=100) == 100


def test_ProxyConfig__add_deployment_resolves_env_refs_for_aws_bedrock_auth_params(
    monkeypatch,
):
    """Regression: DB-stored Bedrock/SageMaker auth params like
    ``aws_role_name: os.environ/BEDROCK_ASSUME_ROLE_ARN`` must resolve at
    DB-load time. PR #30867 removed request-time expansion in
    ``BaseAWSLLM.get_credentials``; without DB-load resolution the literal
    string reaches STS and fails with ``ValidationError: ... is invalid``."""
    aws_env = {
        "aws_session_token": ("BEDROCK_SESSION_TOKEN", "resolved-session-token"),
        "aws_region_name": ("BEDROCK_REGION", "us-east-1"),
        "aws_session_name": ("BEDROCK_SESSION_NAME", "resolved-session"),
        "aws_profile_name": ("BEDROCK_PROFILE", "resolved-profile"),
        "aws_role_name": (
            "BEDROCK_ASSUME_ROLE_ARN",
            "arn:aws:iam::123456789012:role/resolved",
        ),
        "aws_web_identity_token": ("BEDROCK_WEB_IDENTITY_TOKEN", "resolved-token"),
        "aws_sts_endpoint": (
            "BEDROCK_STS_ENDPOINT",
            "https://sts.us-east-1.amazonaws.com",
        ),
        "aws_external_id": ("BEDROCK_EXTERNAL_ID", "resolved-external-id"),
        "aws_bedrock_runtime_endpoint": (
            "BEDROCK_RUNTIME_ENDPOINT",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
        ),
        "aws_bedrock_project_id": ("BEDROCK_PROJECT_ID", "resolved-project-id"),
        "aws_batch_role_arn": (
            "BEDROCK_BATCH_ROLE_ARN",
            "arn:aws:iam::123456789012:role/batch",
        ),
        "aws_workspace_id": ("BEDROCK_WORKSPACE_ID", "resolved-workspace-id"),
    }
    for _, (env_name, env_value) in aws_env.items():
        monkeypatch.setenv(env_name, env_value)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value: value,
    )
    fake_router = MagicMock()
    fake_router.upsert_deployment = MagicMock(return_value=True)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router)
    pc = ProxyConfig()
    litellm_params: Dict[str, Any] = {"model": "bedrock/anthropic.claude-v2"}
    for key, (env_name, _) in aws_env.items():
        litellm_params[key] = f"os.environ/{env_name}"
    db_model = SimpleNamespace(
        model_id="model-1",
        model_name="bedrock-model",
        model_info={"id": "model-1"},
        litellm_params=litellm_params,
        blocked=False,
    )

    added = pc._add_deployment(db_models=[db_model])
    deployment = fake_router.upsert_deployment.call_args.kwargs["deployment"]

    assert added == 1
    for key, (_, expected) in aws_env.items():
        assert getattr(deployment.litellm_params, key) == expected, key


def test_ProxyConfig__add_deployment_resolves_env_refs_on_arbitrary_field(monkeypatch):
    """A made-up field name that was never on the removed whitelist still
    resolves ``os.environ/`` refs. Pins the "no whitelist" invariant:
    the resolver applies to every string field, not a curated list."""
    monkeypatch.setenv("SOME_CUSTOM_ENV", "resolved-custom-value")
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value: value,
    )
    fake_router = MagicMock()
    fake_router.upsert_deployment = MagicMock(return_value=True)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router)
    pc = ProxyConfig()
    db_model = SimpleNamespace(
        model_id="model-1",
        model_name="custom-field-model",
        model_info={"id": "model-1"},
        litellm_params={
            "model": "openai/gpt-4o-mini",
            "some_future_field": "os.environ/SOME_CUSTOM_ENV",
        },
        blocked=False,
    )

    added = pc._add_deployment(db_models=[db_model])
    deployment = fake_router.upsert_deployment.call_args.kwargs["deployment"]

    assert added == 1
    assert deployment.litellm_params.some_future_field == "resolved-custom-value"


# ---------------------------------------------------------------------------
# ProxyConfig.decrypt_model_list_from_db
# ---------------------------------------------------------------------------


def test_ProxyConfig_decrypt_model_list_from_db_returns_decrypted(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value: value,
    )
    pc = ProxyConfig()
    m = SimpleNamespace(
        model_id="m-1",
        model_name="gpt-4",
        model_info={"id": "m-1"},
        litellm_params={"api_key": "sk-x", "model": "gpt-4"},
        blocked=False,
    )
    out = pc.decrypt_model_list_from_db(new_models=[m])
    assert len(out) == 1
    snapshot = {
        "model_name": out[0]["model_name"],
        "params_model": out[0]["litellm_params"]["model"],
        "id_present": "id" in out[0].get("model_info", {}),
    }
    assert snapshot == {
        "model_name": "gpt-4",
        "params_model": "gpt-4",
        "id_present": True,
    }


def test_ProxyConfig_decrypt_model_list_from_db_resolves_env_refs_after_db_decrypt(
    monkeypatch,
):
    """Path B (feeding /v2/model/info fallback and /model/info fallback)
    resolves every ``os.environ/`` field on admin-scoped rows, mirroring
    path A. Both paths now share the same universal-resolution shape."""
    monkeypatch.setenv("LITELLM_DB_MODEL_API_KEY", "resolved-secret")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "master-secret")
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value: (
            "os.environ/LITELLM_DB_MODEL_API_KEY"
            if key == "api_key"
            else "os.environ/LITELLM_MASTER_KEY"
            if key == "api_base"
            else value
        ),
    )
    pc = ProxyConfig()
    m = SimpleNamespace(
        model_id="model-1",
        model_name="env-model",
        model_info={"id": "model-1"},
        litellm_params={
            "api_key": "encrypted-env-ref",
            "api_base": "encrypted-api-base-env-ref",
            "model": "openai/gpt-4o-mini",
        },
        blocked=False,
    )

    out = pc.decrypt_model_list_from_db(new_models=[m])

    assert out[0]["litellm_params"]["api_key"] == "resolved-secret"
    assert out[0]["litellm_params"]["api_base"] == "master-secret"


def test_ProxyConfig_decrypt_model_list_from_db_resolves_team_env_refs_after_db_decrypt(
    monkeypatch,
):
    """Team-scoped rows on path B resolve ``os.environ/`` refs just like
    admin rows do. Pairs with
    ``test_ProxyConfig__add_deployment_resolves_team_env_refs`` on path
    A — both paths now agree on the trust model."""
    monkeypatch.setenv("LITELLM_MASTER_KEY", "master-secret")
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value: "os.environ/LITELLM_MASTER_KEY" if key == "api_key" else value,
    )
    pc = ProxyConfig()
    m = SimpleNamespace(
        model_id="model-1",
        model_name="model_name_team-1_abc",
        model_info={"id": "model-1", "team_id": "team-1"},
        litellm_params={
            "api_key": "encrypted-env-ref",
            "api_base": "https://team.example",
            "model": "openai/gpt-4o-mini",
        },
        blocked=False,
    )

    out = pc.decrypt_model_list_from_db(new_models=[m])

    assert out[0]["litellm_params"]["api_key"] == "master-secret"
    assert out[0]["litellm_params"]["api_base"] == "https://team.example"


def test_ProxyConfig_decrypt_model_list_from_db_invalid_params_skips():
    pc = ProxyConfig()
    bad = SimpleNamespace(model_id="m-1", model_name="x", model_info={}, litellm_params="not-a-dict")
    out = pc.decrypt_model_list_from_db(new_models=[bad])
    # Invalid entries skipped — empty list returned.
    assert out == []


# ---------------------------------------------------------------------------
# ProxyConfig._update_llm_router
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__update_llm_router_no_models_smoke(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-master")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    pc = ProxyConfig()

    async def fake_get_config(*args, **kwargs):
        return {}

    monkeypatch.setattr(pc, "get_config", fake_get_config)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_config",
        pc,
    )
    try:
        await pc._update_llm_router(new_models=[], proxy_logging_obj=MagicMock())
        raised = False
    except Exception:
        raised = True
    snapshot = {"raised": raised, "called": True, "models": "empty"}
    assert snapshot == {"raised": False, "called": True, "models": "empty"}


@pytest.mark.asyncio
async def test_ProxyConfig__update_llm_router_bad_proxy_logging_raises(monkeypatch):
    pc = ProxyConfig()

    async def fake_get_config():
        # alerting present + non-list general_settings to trigger the alerting branch.
        return {"general_settings": {"alerting": ["slack"]}}

    fake_router = MagicMock()
    fake_router.update_settings = MagicMock()
    monkeypatch.setattr(pc, "get_config", fake_get_config)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-x")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {"alerting": ["email"]})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", pc)
    # Passing None for proxy_logging_obj triggers AttributeError in _add_general_settings_from_db_config
    # when it calls proxy_logging_obj.update_values.
    with pytest.raises(AttributeError):
        await pc._update_llm_router(new_models=[], proxy_logging_obj=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProxyConfig._add_callback_from_db_to_in_memory_litellm_callbacks
# ---------------------------------------------------------------------------


def test_ProxyConfig__add_callback_from_db_to_in_memory_litellm_callbacks_adds(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "callbacks", [], raising=False)
    pc = ProxyConfig()
    pc._add_callback_from_db_to_in_memory_litellm_callbacks(
        callback="my_custom_cb",
        event_types=["success", "failure"],
        existing_callbacks=[],
    )
    snapshot = {
        "in_callbacks": "my_custom_cb" in litellm.callbacks,
        "count": len(litellm.callbacks),
        "method_called": True,
    }
    assert snapshot == {"in_callbacks": True, "count": 1, "method_called": True}


def test_ProxyConfig__add_callback_from_db_to_in_memory_litellm_callbacks_invalid_event_raises(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "callbacks", [], raising=False)
    pc = ProxyConfig()
    # For a "known" callback, event_types is iterated — non-iterable raises TypeError.
    with pytest.raises(TypeError):
        pc._add_callback_from_db_to_in_memory_litellm_callbacks(
            callback="lago",  # in _known_custom_logger_compatible_callbacks
            event_types=12345,  # type: ignore[arg-type]
            existing_callbacks=[],
        )


# ---------------------------------------------------------------------------
# ProxyConfig._add_callbacks_from_db_config
# ---------------------------------------------------------------------------


def test_ProxyConfig__add_callbacks_from_db_config_processes_lists(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [], raising=False)
    monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "failure_callback", [], raising=False)
    pc = ProxyConfig()
    cfg = {
        "litellm_settings": {
            "callbacks": ["cb_a"],
            "success_callback": ["s_a"],
            "failure_callback": ["f_a"],
        }
    }
    pc._add_callbacks_from_db_config(cfg)
    snapshot = {
        "cb_added": "cb_a" in litellm.callbacks,
        "success_added": "s_a" in litellm.success_callback,
        "failure_added": "f_a" in litellm.failure_callback,
    }
    assert snapshot == {
        "cb_added": True,
        "success_added": True,
        "failure_added": True,
    }


def test_ProxyConfig__add_callbacks_from_db_config_bad_config_raises():
    pc = ProxyConfig()
    with pytest.raises(AttributeError):
        # Non-dict input — .get will fail.
        pc._add_callbacks_from_db_config(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProxyConfig._encrypt_env_variables
# ---------------------------------------------------------------------------


def test_ProxyConfig__encrypt_env_variables_returns_dict(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.encrypt_value_helper",
        lambda value, new_encryption_key=None: f"ENC[{value}]",
    )
    pc = ProxyConfig()
    out = pc._encrypt_env_variables({"A": "1", "B": "2", "C": "3"})
    assert out == {"A": "ENC[1]", "B": "ENC[2]", "C": "ENC[3]"}


def test_ProxyConfig__encrypt_env_variables_invalid_raises():
    pc = ProxyConfig()
    with pytest.raises(AttributeError):
        # Non-dict input — .items() fails.
        pc._encrypt_env_variables(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProxyConfig._decrypt_and_set_db_env_variables
# ---------------------------------------------------------------------------


def test_ProxyConfig__decrypt_and_set_db_env_variables_sets_env(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value=False: value + "-dec",
    )
    monkeypatch.delenv("KEY_X", raising=False)
    monkeypatch.delenv("KEY_Y", raising=False)
    pc = ProxyConfig()
    out = pc._decrypt_and_set_db_env_variables({"KEY_X": "x", "KEY_Y": "y"})
    snapshot = {
        "KEY_X_env": os.environ.get("KEY_X"),
        "KEY_Y_env": os.environ.get("KEY_Y"),
        "returned_keys": sorted(out.keys()),
    }
    assert snapshot == {
        "KEY_X_env": "x-dec",
        "KEY_Y_env": "y-dec",
        "returned_keys": ["KEY_X", "KEY_Y"],
    }


def test_ProxyConfig__decrypt_and_set_db_env_variables_invalid_dict_raises():
    pc = ProxyConfig()
    with pytest.raises(AttributeError):
        pc._decrypt_and_set_db_env_variables("not-a-dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProxyConfig._decrypt_db_variables
# ---------------------------------------------------------------------------


def test_ProxyConfig__decrypt_db_variables_returns_decrypted(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value: f"D({value})",
    )
    pc = ProxyConfig()
    out = pc._decrypt_db_variables({"a": "1", "b": "2", "c": "3"})
    assert out == {"a": "D(1)", "b": "D(2)", "c": "D(3)"}


def test_ProxyConfig__decrypt_db_variables_invalid_raises():
    pc = ProxyConfig()
    with pytest.raises(AttributeError):
        pc._decrypt_db_variables(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProxyConfig._encrypt_env_variables_for_db
# ---------------------------------------------------------------------------


def test_ProxyConfig__encrypt_env_variables_for_db_idempotent(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.decrypt_value_helper",
        lambda value, key, return_original_value: value,
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.encrypt_value_helper",
        lambda value, new_encryption_key=None: f"ENC[{value}]",
    )
    pc = ProxyConfig()
    out = pc._encrypt_env_variables_for_db({"A": "1", "B": "2", "C": "3"})
    assert out == {"A": "ENC[1]", "B": "ENC[2]", "C": "ENC[3]"}


def test_ProxyConfig__encrypt_env_variables_for_db_invalid_raises():
    pc = ProxyConfig()
    with pytest.raises(AttributeError):
        pc._encrypt_env_variables_for_db(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProxyConfig._parse_router_settings_value
# ---------------------------------------------------------------------------


def test_ProxyConfig__parse_router_settings_value_handles_inputs():
    result = {
        "dict": ProxyConfig._parse_router_settings_value({"a": 1}),
        "yaml_string": ProxyConfig._parse_router_settings_value("a: 1\nb: 2"),
        "none": ProxyConfig._parse_router_settings_value(None),
    }
    assert result == {
        "dict": {"a": 1},
        "yaml_string": {"a": 1, "b": 2},
        "none": None,
    }


def test_ProxyConfig__parse_router_settings_value_invalid_returns_none():
    # Non-dict, non-parseable scalar -> None.
    assert ProxyConfig._parse_router_settings_value(12345) is None
    # Empty dict -> None (not truthy).
    assert ProxyConfig._parse_router_settings_value({}) is None


# ---------------------------------------------------------------------------
# ProxyConfig._get_hierarchical_router_settings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__get_hierarchical_router_settings_key_wins():
    pc = ProxyConfig()
    fake_key = SimpleNamespace(
        router_settings={"timeout": 30, "retries": 2, "model": "gpt-4"},
        team_id=None,
    )
    out = await pc._get_hierarchical_router_settings(
        user_api_key_dict=fake_key,
        prisma_client=None,
        proxy_logging_obj=None,
    )
    assert out == {"timeout": 30, "retries": 2, "model": "gpt-4"}


@pytest.mark.asyncio
async def test_ProxyConfig__get_hierarchical_router_settings_missing_returns_none():
    pc = ProxyConfig()
    fake_key = SimpleNamespace(router_settings=None, team_id=None)
    out = await pc._get_hierarchical_router_settings(
        user_api_key_dict=fake_key,
        prisma_client=None,
        proxy_logging_obj=None,
    )
    assert out is None


# ---------------------------------------------------------------------------
# ProxyConfig._add_router_settings_from_db_config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__add_router_settings_from_db_config_updates_router():
    pc = ProxyConfig()
    fake_router = MagicMock()
    fake_router.update_settings = MagicMock()
    fake_prisma = MagicMock()
    fake_prisma.db.litellm_config.find_first = AsyncMock(
        return_value=SimpleNamespace(param_value={"timeout": 30, "retries": 2, "fallbacks": []})
    )
    config_data = {"router_settings": {"timeout": 10}}
    await pc._add_router_settings_from_db_config(
        config_data=config_data,
        llm_router=fake_router,
        prisma_client=fake_prisma,
    )
    snapshot = {
        "called": fake_router.update_settings.called,
        "call_count": fake_router.update_settings.call_count,
        "kwargs_keys": sorted(list(fake_router.update_settings.call_args.kwargs.keys())),
    }
    assert snapshot == {
        "called": True,
        "call_count": 1,
        "kwargs_keys": ["fallbacks", "retries", "timeout"],
    }


@pytest.mark.asyncio
async def test_ProxyConfig__add_router_settings_from_db_config_none_router_noop():
    pc = ProxyConfig()
    # No router and no prisma — should silently return.
    await pc._add_router_settings_from_db_config(config_data={}, llm_router=None, prisma_client=None)
    # Error-style: bad call signature raises.
    with pytest.raises(TypeError):
        await pc._add_router_settings_from_db_config()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ProxyConfig._add_general_settings_from_db_config
# ---------------------------------------------------------------------------


def test_ProxyConfig__add_general_settings_from_db_config_merges_alerting():
    pc = ProxyConfig()
    proxy_logging = MagicMock()
    general = {"alerting": ["slack"]}
    config_data = {"general_settings": {"alerting": ["email", "slack"]}}
    pc._add_general_settings_from_db_config(
        config_data=config_data,
        general_settings=general,
        proxy_logging_obj=proxy_logging,
    )
    snapshot = {
        "alerting": sorted(general["alerting"]),
        "logging_called": proxy_logging.update_values.called,
        "merged_count": len(general["alerting"]),
    }
    assert snapshot == {
        "alerting": ["email", "slack"],
        "logging_called": True,
        "merged_count": 2,
    }


def test_ProxyConfig__add_general_settings_from_db_config_bad_config_raises():
    pc = ProxyConfig()
    with pytest.raises(AttributeError):
        pc._add_general_settings_from_db_config(
            config_data=None,  # type: ignore[arg-type]
            general_settings={},
            proxy_logging_obj=MagicMock(),
        )


# ---------------------------------------------------------------------------
# ProxyConfig._reschedule_spend_log_cleanup_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__reschedule_spend_log_cleanup_job_no_scheduler(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.scheduler", None)
    pc = ProxyConfig()
    try:
        await pc._reschedule_spend_log_cleanup_job()
        raised = False
    except Exception:
        raised = True
    snapshot = {"raised": raised, "called": True, "scheduler_was": "none"}
    assert snapshot == {"raised": False, "called": True, "scheduler_was": "none"}


@pytest.mark.asyncio
async def test_ProxyConfig__reschedule_spend_log_cleanup_job_invalid_cron(monkeypatch):
    fake_scheduler = MagicMock()
    fake_scheduler.remove_job = MagicMock()
    fake_scheduler.add_job = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.scheduler", fake_scheduler)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {
            "maximum_spend_logs_retention_period": "1d",
            "maximum_spend_logs_cleanup_cron": "INVALID CRON STRING",
        },
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    pc = ProxyConfig()
    # Invalid cron is caught and logged — does not raise outward.
    await pc._reschedule_spend_log_cleanup_job()
    # But add_job should not have been called for the invalid cron path.
    assert fake_scheduler.add_job.call_count == 0


# ---------------------------------------------------------------------------
# ProxyConfig._update_general_settings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__update_general_settings_updates_max_parallel(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {},
    )
    pc = ProxyConfig()
    await pc._update_general_settings(
        {
            "max_parallel_requests": 7,
            "global_max_parallel_requests": 99,
            "ui_access_mode": "admin_only",
        }
    )
    from litellm.proxy import proxy_server as ps

    snapshot = {
        "max_parallel_requests": ps.general_settings.get("max_parallel_requests"),
        "global_max_parallel_requests": ps.general_settings.get("global_max_parallel_requests"),
        "ui_access_mode": ps.general_settings.get("ui_access_mode"),
    }
    assert snapshot == {
        "max_parallel_requests": 7,
        "global_max_parallel_requests": 99,
        "ui_access_mode": "admin_only",
    }


@pytest.mark.asyncio
async def test_ProxyConfig__update_general_settings_none_input_noop():
    pc = ProxyConfig()
    # None input returns early.
    result = await pc._update_general_settings(db_general_settings=None)
    assert result is None
    # Error-style: dict() will fail on non-mapping non-None input.
    with pytest.raises(Exception):
        await pc._update_general_settings(db_general_settings=12345)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_ProxyConfig__update_general_settings_env_true_overrides_db_false(
    monkeypatch,
):
    """STORE_MODEL_IN_DB=True env var must win over a stale DB `false`.

    Regression test for the case where a periodic config refresh pulled
    ``store_model_in_db: false`` from the DB's general_settings and silently
    disabled the feature even though the operator set STORE_MODEL_IN_DB=True.
    """
    monkeypatch.setenv("STORE_MODEL_IN_DB", "True")
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

    pc = ProxyConfig()
    await pc._update_general_settings({"store_model_in_db": False})

    from litellm.proxy import proxy_server as ps

    assert ps.store_model_in_db is True
    assert ps.general_settings.get("store_model_in_db") is True


@pytest.mark.asyncio
async def test_ProxyConfig__update_general_settings_db_false_applied_without_env(
    monkeypatch,
):
    """Without the env override, a DB `false` still takes effect as before."""
    monkeypatch.delenv("STORE_MODEL_IN_DB", raising=False)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

    pc = ProxyConfig()
    await pc._update_general_settings({"store_model_in_db": False})

    from litellm.proxy import proxy_server as ps

    assert ps.store_model_in_db is False
    assert ps.general_settings.get("store_model_in_db") is False


# ---------------------------------------------------------------------------
# ProxyConfig._update_config_fields
# ---------------------------------------------------------------------------


def test_ProxyConfig__update_config_fields_merges_dict():
    pc = ProxyConfig()
    current = {"general_settings": {"a": 1, "b": 2}}
    out = pc._update_config_fields(
        current_config=current,
        param_name="general_settings",
        db_param_value={"b": 3, "c": 4, "d": 5},
    )
    assert out == {"general_settings": {"a": 1, "b": 3, "c": 4, "d": 5}}


def test_ProxyConfig__update_config_fields_invalid_param_raises():
    pc = ProxyConfig()
    with pytest.raises(Exception):
        # Missing required arg.
        pc._update_config_fields(current_config={}, param_name="general_settings")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ProxyConfig._update_config_from_db
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ProxyConfig__update_config_from_db_does_not_log_general_settings_secrets(
    monkeypatch,
):
    """Regression for LIT-4152 on the store_model_in_db path.

    ``_update_config_from_db`` logged each DB ``param_value`` verbatim at DEBUG;
    for ``general_settings`` that value is the whole dict, leaking ``master_key``
    and ``database_url`` the same way the startup config load did. The value now
    routes through the recursive redactor. Asserted with the module regex
    scrubber (``_ENABLE_SECRET_REDACTION``) disabled so the caller itself must
    not build the leaky string. The merge into the returned config must still
    carry the raw values, proving only the log record is redacted.
    """
    import logging

    import litellm._logging as _logging_module
    from litellm._logging import verbose_proxy_logger

    monkeypatch.setattr(_logging_module, "_ENABLE_SECRET_REDACTION", False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def _fake_decrypt_value_helper(value, key, **_kwargs):
        return value

    monkeypatch.setattr("litellm.proxy.proxy_server.decrypt_value_helper", _fake_decrypt_value_helper)

    master_key_secret = "sk-lit4152-db-path-master-key-abcdef1234567890"
    db_url_secret = "postgresql://leak_user:leak_password_9090@leak-host.internal:5432/leak_db"
    env_db_url_secret = "postgresql://env_leak_user:env_leak_password_9090@env-leak-host.internal:5432/env_leak_db"
    nested_webhook_secret = "https://hooks.slack.com/services/T0/B0/db-path-webhook-secret"

    responses = {
        "general_settings": SimpleNamespace(
            param_name="general_settings",
            param_value={
                "master_key": master_key_secret,
                "database_url": db_url_secret,
                "alert_to_webhook_url": {"budget_alerts": nested_webhook_secret},
            },
        ),
        "router_settings": None,
        "litellm_settings": None,
        "environment_variables": SimpleNamespace(
            param_name="environment_variables",
            param_value={"DATABASE_URL": env_db_url_secret},
        ),
    }

    async def _fake_get_config_param(prisma_client, key):
        return responses[key]

    monkeypatch.setattr("litellm.proxy.proxy_server.get_config_param", _fake_get_config_param)

    class LogRecordHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = LogRecordHandler()
    handler.setLevel(logging.DEBUG)
    original_level = verbose_proxy_logger.level
    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_proxy_logger.addHandler(handler)
    try:
        merged = await ProxyConfig()._update_config_from_db(
            prisma_client=MagicMock(),
            config={"general_settings": {}},
            store_model_in_db=True,
        )
        rendered = " ".join(record.getMessage() for record in handler.records)
    finally:
        verbose_proxy_logger.removeHandler(handler)
        verbose_proxy_logger.setLevel(original_level)

    for secret in (
        master_key_secret,
        db_url_secret,
        env_db_url_secret,
        nested_webhook_secret,
        "leak_password_9090",
        "env_leak_password_9090",
    ):
        assert secret not in rendered, f"leak: {secret} in {rendered!r}"
    assert merged["general_settings"]["master_key"] == master_key_secret
    assert merged["general_settings"]["database_url"] == db_url_secret
    assert merged["environment_variables"]["DATABASE_URL"] == env_db_url_secret


@pytest.mark.asyncio
async def test_ProxyConfig_load_config_redacts_secret_litellm_setting_keeps_plain(tmp_path, monkeypatch):
    """Regression for LIT-4152 on the ``litellm_settings`` apply loop.

    ``load_config`` logged ``setting litellm.<key>=<value>`` verbatim at DEBUG,
    so a secret-bearing setting such as ``api_key`` leaked in cleartext. The
    value now routes through ``_redact_general_setting_value``. Crucially the
    redaction must be surgical: a secret-named key is masked, but a plain
    operational setting like ``num_retries`` must still log its real value, so
    the debug line keeps its signal. Asserted with the module regex scrubber
    (``_ENABLE_SECRET_REDACTION``) disabled.
    """
    import logging

    import litellm._logging as _logging_module
    from litellm._logging import verbose_proxy_logger

    monkeypatch.setattr(_logging_module, "_ENABLE_SECRET_REDACTION", False)

    api_key_secret = "sk-lit4152-litellm-settings-secret-abcdef1234567890"
    f = tmp_path / "c.yaml"
    f.write_text(
        f"model_list: []\ngeneral_settings: {{}}\nlitellm_settings:\n  api_key: {api_key_secret}\n  num_retries: 7\n"
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    class LogRecordHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = LogRecordHandler()
    handler.setLevel(logging.DEBUG)
    original_level = verbose_proxy_logger.level
    original_api_key = getattr(litellm, "api_key", None)
    original_num_retries = getattr(litellm, "num_retries", None)
    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_proxy_logger.addHandler(handler)
    try:
        await ProxyConfig().load_config(router=None, config_file_path=str(f))
        rendered = " ".join(record.getMessage() for record in handler.records)
    finally:
        verbose_proxy_logger.removeHandler(handler)
        verbose_proxy_logger.setLevel(original_level)
        litellm.api_key = original_api_key
        litellm.num_retries = original_num_retries

    assert api_key_secret not in rendered, f"api_key leaked in logs: {rendered!r}"
    assert "num_retries=7" in rendered, (
        f"non-secret num_retries value was over-redacted; expected it visible in {rendered!r}"
    )
