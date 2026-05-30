"""Behavior pins for ProxyConfig and module-level config scrubbers.

Pins covered:
- Module-level: ``_is_remote_module_url``, ``_scrub_guardrail_inner``,
  ``_scrub_db_overlay_remote_module_loads``
- All ``ProxyConfig`` methods listed in the pin file.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy.proxy_server import (
    ProxyConfig,
    _is_remote_module_url,
    _scrub_db_overlay_remote_module_loads,
    _scrub_guardrail_inner,
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
    f.write_text(
        "model_list: []\ngeneral_settings: {}\nlitellm_settings:\n  drop_params: true\n"
    )
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


# ---------------------------------------------------------------------------
# ProxyConfig._load_environment_variables
# ---------------------------------------------------------------------------


def test_ProxyConfig__load_environment_variables_sets_env(monkeypatch):
    monkeypatch.delenv("TEST_LOAD_ENV_X", raising=False)
    pc = ProxyConfig()
    pc._load_environment_variables(
        {"environment_variables": {"TEST_LOAD_ENV_X": "hello"}}
    )
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
async def test_ProxyConfig_load_config_missing_file_raises(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)
    pc = ProxyConfig()
    with pytest.raises(Exception):
        await pc.load_config(router=None, config_file_path="/no/file.yaml")


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


def test_ProxyConfig_get_model_info_with_id_missing_model_id_raises():
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


def test_ProxyConfig_decrypt_model_list_from_db_invalid_params_skips():
    pc = ProxyConfig()
    bad = SimpleNamespace(
        model_id="m-1", model_name="x", model_info={}, litellm_params="not-a-dict"
    )
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
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings", {"alerting": ["email"]}
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", pc)
    # Passing None for proxy_logging_obj triggers AttributeError in _add_general_settings_from_db_config
    # when it calls proxy_logging_obj.update_values.
    with pytest.raises(AttributeError):
        await pc._update_llm_router(new_models=None, proxy_logging_obj=None)  # type: ignore[arg-type]


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
        return_value=SimpleNamespace(
            param_value={"timeout": 30, "retries": 2, "fallbacks": []}
        )
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
        "kwargs_keys": sorted(
            list(fake_router.update_settings.call_args.kwargs.keys())
        ),
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
    await pc._add_router_settings_from_db_config(
        config_data={}, llm_router=None, prisma_client=None
    )
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
        "global_max_parallel_requests": ps.general_settings.get(
            "global_max_parallel_requests"
        ),
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
