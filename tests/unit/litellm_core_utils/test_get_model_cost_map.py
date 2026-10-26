"""Catalog fetch purity, publication, and import-only retry compatibility."""

import copy
import json
import logging
import os
import random
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from unittest.mock import Mock

import pytest

import litellm
from litellm._catalog import CatalogUnavailable, Reloaded, _bootstrap, _fetch, _install, reload
from litellm._catalog._fetch import Bundled, File, Remote, fetch
from litellm._catalog._install import Provenance, install
from litellm._catalog._load import LoadedCatalog, load, served_rows
from litellm.litellm_core_utils import get_model_cost_map as get_model_cost_map_module
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_capability_generalizations,
    match_routing_generalization,
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.get_model_cost_map import (
    FALLBACK_GENERALIZATIONS_KEY,
    GetModelCostMap,
    _count_model_entries,
    get_model_cost_map,
    get_model_cost_map_provenance,
    git_blob_id,
)
from litellm.rust_bridge._native import FetchedRegistry, ModelRegistry, RegistryUnavailable


def _load_root_cost_map() -> dict:
    path = os.path.join(os.path.dirname(__file__), "../../../model_prices_and_context_window.json")
    with open(path) as f:
        return json.load(f)


def _bundled_blob_id() -> str:
    path = os.path.join(os.path.dirname(__file__), "../../../model_prices_and_context_window.json")
    with open(path, "rb") as f:
        return git_blob_id(f.read())


def test_git_blob_id_is_what_git_hash_object_prints():
    assert git_blob_id(b'{"gpt-5.4-mini": {"mode": "chat"}}\n') == "18b9a8381e13a3b38a2128f184f631f95829e987"


def test_fetch_policy_matches_the_fixture_the_rust_catalog_reads():
    fixture_path: Final = Path(__file__).parents[3] / "litellm-rust/crates/model-catalog/tests/git_blob_id.json"
    fixture: Final = json.loads(fixture_path.read_text())
    blob: Final = fixture["git_blob_id"]
    retry: Final = fixture["retry"]
    assert git_blob_id(blob["example_body_utf8"].encode()) == blob["example_revision"]
    assert (
        retry["max_attempts"],
        retry["max_wait_seconds"],
        retry["minimum_model_count"],
        retry["minimum_reference_ratio"],
    ) == (
        get_model_cost_map_module.MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
        get_model_cost_map_module.MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS,
        get_model_cost_map_module.MODEL_COST_MAP_MIN_MODEL_COUNT,
        get_model_cost_map_module.MODEL_COST_MAP_MAX_SHRINK_RATIO,
    )


def _make_models(n: int) -> dict:
    return {f"model-{i}": {"litellm_provider": "openai", "mode": "chat"} for i in range(n)}


def test_count_model_entries_excludes_reserved_keys():
    m = _make_models(3)
    m["sample_spec"] = {"foo": "bar"}
    m[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}
    assert _count_model_entries(m) == 3


def test_validation_rejects_truly_shrunk_file_even_with_meta_keys():
    """A file with only a handful of real models must be rejected as corrupt,
    and the extra meta keys must not inflate the count past the minimum."""
    shrunk = _make_models(5)
    shrunk["sample_spec"] = {"foo": "bar"}
    shrunk[FALLBACK_GENERALIZATIONS_KEY] = {"rules": [{"name": "x"}]}

    assert (
        GetModelCostMap.validate_model_cost_map(
            fetched_map=shrunk,
            backup_model_count=2000,
            min_model_count=50,
        )
        is False
    )


def test_validation_accepts_healthy_file_with_meta_keys():
    healthy = _make_models(2000)
    healthy["sample_spec"] = {"foo": "bar"}
    healthy[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}

    assert (
        GetModelCostMap.validate_model_cost_map(
            fetched_map=healthy,
            backup_model_count=2000,
            min_model_count=50,
        )
        is True
    )


def test_validation_rejects_significant_shrink_vs_backup():
    # 600 real models vs a 2000-model backup is below the 50% shrink threshold.
    shrunk = _make_models(600)
    shrunk[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}
    assert (
        GetModelCostMap.validate_model_cost_map(
            fetched_map=shrunk,
            backup_model_count=2000,
            min_model_count=50,
            max_shrink_ratio=0.5,
        )
        is False
    )


def test_install_pops_key_and_installs_rules():
    previous = list(get_fallback_generalization_rules())
    try:
        raw = _make_models(2)
        raw[FALLBACK_GENERALIZATIONS_KEY] = {
            "rules": [
                {
                    "name": "rule",
                    "pattern": r"^widget-",
                    "model_info": {"litellm_provider": "openai"},
                }
            ]
        }
        catalog = load(json.dumps(raw).encode())
        assert isinstance(catalog, LoadedCatalog)
        install(catalog, provenance=Provenance("local"))
        finalized = litellm.model_cost

        # The reserved key is removed from the returned model map ...
        assert FALLBACK_GENERALIZATIONS_KEY not in finalized
        # ... and its rules are installed into the generalizations module.
        assert match_routing_generalization("widget-9") == "openai"
    finally:
        set_fallback_generalizations(previous)


def test_install_with_no_block_clears_rules():
    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations([{"name": "stale", "pattern": r"^x", "model_info": {"a": 1}}])
        catalog = load(json.dumps(_make_models(2)).encode())
        assert isinstance(catalog, LoadedCatalog)
        install(catalog, provenance=Provenance("local"))
        assert match_capability_generalizations("x-1") is None
    finally:
        set_fallback_generalizations(previous)


def test_shipped_backup_carries_the_claude_routing_rules():
    """The bundled backup must ship the Claude routing rules so a fresh install
    (or an offline fallback) routes unknown Claude models without code changes.
    Bedrock-syntax ids must hit the bedrock rule before the bare-id Anthropic rule."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup.get(FALLBACK_GENERALIZATIONS_KEY, {}).get("rules", [])
    names = [r.get("name") for r in rules]
    assert names.index("bedrock-claude-ids") < names.index("anthropic-claude-ids")

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        assert match_routing_generalization("claude-opus-4-9") == "anthropic"
        assert match_routing_generalization("global.anthropic.claude-opus-4-9") == "bedrock"
    finally:
        set_fallback_generalizations(previous)


def test_shipped_routing_rules_never_match_through_an_unrecognized_namespace():
    """Routing rules decide ``litellm_provider`` for otherwise-unknown ids, and the
    proxy's wildcard access check (``can_key_call_model`` with a ``bedrock/*`` key)
    trusts that inference: it rebuilds ``{provider}/{model}`` and matches it against
    the key's patterns. A routing pattern that matches as a substring lets
    ``bedrockz/anthropic.claude-...`` resolve to bedrock and slip through a
    ``bedrock/*`` key, so every shipped routing rule must anchor to the start of
    the name and never match an id carrying an unrecognized namespace prefix."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]

    routing_rules = [r for r in rules if "litellm_provider" in r["model_info"]]
    assert routing_rules
    assert all(r["pattern"].startswith("^") for r in routing_rules)

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        for bedrock_id in [
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-v2:1",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "global.anthropic.claude-fable-5-20260120-v1:0",
        ]:
            assert match_routing_generalization(bedrock_id) == "bedrock", bedrock_id
        for namespaced in [
            "bedrockz/anthropic.claude-3-5-sonnet-20240620",
            "bedrockz/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrockz/claude-3-5-sonnet-20240620",
        ]:
            assert match_routing_generalization(namespaced) is None, namespaced
    finally:
        set_fallback_generalizations(previous)


def test_shipped_backup_marks_claude_4_6_plus_adaptive_not_4_0():
    """Adaptive thinking is data, not code. The bundled backup must carry
    supports_adaptive_thinking on genuine Claude >= 4.6 entries (every provider
    route) and on the version-gated anthropic-claude-adaptive-thinking rule for
    unmapped future Claudes, while leaving the dated Claude 4.0 names
    ("...-4-20250514") unflagged so a date can never be mistaken for a 4.6+ minor
    version. The version-neutral claude-family-baseline capability rule must not flag
    it, so an unmapped sub-4.6 name resolves but stays non-adaptive. The adaptive rule
    carries only its delta; capability unioning stacks it onto the baseline, so the
    baseline block is never duplicated across rules and no rule needs ``extends``."""
    backup = GetModelCostMap.load_local_model_cost_map()

    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]
    baseline_rule = next(r for r in rules if r.get("name") == "claude-family-baseline")
    adaptive_rule = next(r for r in rules if r.get("name") == "claude-adaptive-thinking")
    assert "supports_adaptive_thinking" not in baseline_rule["model_info"]
    assert "litellm_provider" not in baseline_rule["model_info"]
    assert adaptive_rule["model_info"] == {"supports_adaptive_thinking": True}
    assert all("extends" not in r for r in rules)

    for adaptive in [
        "anthropic.claude-opus-4-8",
        "vertex_ai/claude-opus-4-6@default",
        "us.anthropic.claude-sonnet-4-6",
        "azure_ai/claude-opus-4-7",
    ]:
        assert backup[adaptive]["supports_adaptive_thinking"] is True, adaptive

    for non_adaptive in [
        "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-opus-4-5",
    ]:
        assert "supports_adaptive_thinking" not in backup[non_adaptive], non_adaptive


class TestCheckIsValidDict:
    """Unit tests for _check_is_valid_dict."""

    def test_should_reject_non_dict(self):
        """Non-dict should fail."""
        assert GetModelCostMap._check_is_valid_dict("not a dict") is False

    def test_should_reject_empty_dict(self):
        """Empty dict should fail."""
        assert GetModelCostMap._check_is_valid_dict({}) is False

    def test_should_reject_list(self):
        """List should fail."""
        assert GetModelCostMap._check_is_valid_dict([1, 2, 3]) is False

    def test_should_reject_none(self):
        """None should fail."""
        assert GetModelCostMap._check_is_valid_dict(None) is False

    def test_should_accept_non_empty_dict(self):
        """Non-empty dict should pass."""
        assert GetModelCostMap._check_is_valid_dict({"model": {}}) is True


class TestCheckModelCountNotReduced:
    """Unit tests for _check_model_count_not_reduced."""

    def test_should_reject_too_few_models(self):
        """Fetched map with fewer models than min_model_count should fail."""
        small_map = {f"model-{i}": {} for i in range(5)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=small_map, backup_model_count=0, min_model_count=10
            )
            is False
        )

    def test_should_reject_significant_shrinkage(self):
        """Fetched map that shrunk >50% vs backup should fail."""
        fetched = {f"model-{i}": {} for i in range(40)}  # 40% of 100
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is False
        )

    def test_should_accept_when_above_threshold(self):
        """Fetched map at 60% of backup (above 50% threshold) should pass."""
        fetched = {f"model-{i}": {} for i in range(60)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is True
        )

    def test_should_accept_growth(self):
        """Fetched map larger than backup should pass."""
        fetched = {f"model-{i}": {} for i in range(120)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is True
        )

    def test_should_accept_with_empty_backup(self):
        """When backup is empty, only min_model_count matters."""
        fetched = {f"model-{i}": {} for i in range(15)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=0, min_model_count=10
            )
            is True
        )


class TestValidateModelCostMap:
    """Unit tests for validate_model_cost_map (combines both checks)."""

    def test_should_reject_non_dict(self):
        """Non-dict should fail at check 1."""
        assert GetModelCostMap.validate_model_cost_map(fetched_map="not a dict", backup_model_count=0) is False

    def test_should_reject_empty_map(self):
        """Empty dict should fail at check 1."""
        assert GetModelCostMap.validate_model_cost_map(fetched_map={}, backup_model_count=0) is False

    def test_should_reject_significant_shrinkage(self):
        """Should fail at check 2 (shrinkage)."""
        fetched = {f"model-{i}": {} for i in range(40)}
        assert (
            GetModelCostMap.validate_model_cost_map(fetched_map=fetched, backup_model_count=100, min_model_count=10)
            is False
        )

    def test_should_accept_valid_map(self):
        """Should pass both checks."""
        fetched = {f"model-{i}": {} for i in range(120)}
        assert (
            GetModelCostMap.validate_model_cost_map(fetched_map=fetched, backup_model_count=100, min_model_count=10)
            is True
        )

    def test_should_accept_equal_size_map(self):
        """Equal size should pass both checks."""
        fetched = {f"model-{i}": {} for i in range(100)}
        assert (
            GetModelCostMap.validate_model_cost_map(fetched_map=fetched, backup_model_count=100, min_model_count=10)
            is True
        )


def test_sample_spec_is_raw_metadata_not_a_served_model():
    assert "sample_spec" in GetModelCostMap.load_local_model_cost_map()
    assert "sample_spec" not in litellm.model_cost
    assert type(litellm.model_cost) is dict


def test_local_map_is_embedded_not_read_from_the_backup(monkeypatch: pytest.MonkeyPatch):
    from litellm._catalog._load import LoadedCatalog, load, served_rows

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    backup_path: Final = Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json"
    original: Final = backup_path.read_bytes()
    try:
        backup_path.write_bytes(b'{"replacement": {"mode": "chat"}}')
        loaded: Final = load(None)
        assert isinstance(loaded, LoadedCatalog)
        result: Final = get_model_cost_map(url="")
        assert result == served_rows(loaded)
        assert GetModelCostMap.load_local_model_cost_map() == _load_root_cost_map()
        assert get_model_cost_map_module.get_model_cost_map_source_info()["source_revision"] == _bundled_blob_id()
    finally:
        backup_path.write_bytes(original)


def _read_every_key(model_cost: dict) -> dict[str, tuple[object, object, object]]:
    get_model_cost_map_module.adopt_model_cost_map(model_cost)
    litellm.get_model_info.cache_clear()

    def outcome(read: Callable[[], object]) -> object:
        try:
            return read()
        except Exception as error:  # noqa: BLE001  # compare existing public exception contracts
            return type(error).__name__

    return {
        key: (
            outcome(lambda key=key: litellm.get_model_info(key)),
            outcome(lambda key=key: litellm.cost_per_token(model=key, prompt_tokens=1000, completion_tokens=100)),
            model_cost[key],
        )
        for key in model_cost
    }


def test_every_key_reads_the_same_as_the_json_parsed_map(monkeypatch: pytest.MonkeyPatch):
    from litellm._catalog._load import LoadedCatalog, load, served_rows

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    raw: Final = _load_root_cost_map()
    set_fallback_generalizations(raw[FALLBACK_GENERALIZATIONS_KEY]["rules"])
    legacy: Final = get_model_cost_map_module._expand_model_aliases(
        {key: row for key, row in raw.items() if key not in {"sample_spec", FALLBACK_GENERALIZATIONS_KEY}}
    )
    from_json: Final = _read_every_key(legacy)
    loaded: Final = load(None)
    assert isinstance(loaded, LoadedCatalog)
    rows: Final = served_rows(loaded)
    assert rows == loaded.registry.served() == legacy
    native: Final = _read_every_key(rows)
    assert native == from_json


def _catalog(rows: dict) -> LoadedCatalog:
    result: Final = load(json.dumps(rows).encode())
    assert isinstance(result, LoadedCatalog)
    return result


def _catalog_file(tmp_path: Path, rows: dict) -> Path:
    path: Final = tmp_path / "catalog.json"
    path.write_text(json.dumps(rows))
    return path


def test_explicit_side_read_is_pure_even_with_local_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw: Final = _load_root_cost_map()
    raw["side-read-only"] = {"mode": "chat", "litellm_provider": "openai", "metadata": {"version": 1}}
    raw[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}
    path: Final = _catalog_file(tmp_path, raw)
    live: Final = litellm.model_cost
    before: Final = copy.deepcopy(live)
    provenance: Final = get_model_cost_map_module.get_model_cost_map_source_info()
    rules: Final = get_fallback_generalization_rules()
    snapshot: Final = GetModelCostMap.loaded_model_cost_map()
    installed: Final = copy.deepcopy(_install.STATE.rows)
    threads: Final = frozenset(threading.enumerate())
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    result: Final = get_model_cost_map(path.as_uri(), max_attempts=20)

    assert result["side-read-only"]["metadata"] == {"version": 1}
    result["side-read-only"]["metadata"]["version"] = 2
    assert get_model_cost_map(path.as_uri())["side-read-only"]["metadata"] == {"version": 1}
    assert litellm.model_cost is live
    assert live == before
    assert get_model_cost_map_module.get_model_cost_map_source_info() == provenance
    assert get_fallback_generalization_rules() == rules
    assert GetModelCostMap.loaded_model_cost_map() is snapshot
    assert _install.STATE.rows == installed
    assert frozenset(threading.enumerate()) == threads
    assert get_model_cost_map("") == served_rows(LoadedCatalog(ModelRegistry.bundled()))


@pytest.mark.parametrize("url", ["https://catalog.invalid/prices", "s3://bucket/prices.json"])
def test_side_read_does_exactly_one_native_attempt_without_retry_thread(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure: Final = Mock(spec=RegistryUnavailable, reason="transient", retryable=True, retry_after=9)
    native: Final = Mock(return_value=failure)
    monkeypatch.setattr(_fetch, "fetch_once", native)
    threads: Final = frozenset(threading.enumerate())
    before: Final = get_model_cost_map_module.get_model_cost_map_source_info()

    assert get_model_cost_map(url, max_attempts=15) == get_model_cost_map("")
    native.assert_called_once_with(url, if_none_match=None, timeout_seconds=5)
    assert frozenset(threading.enumerate()) == threads
    assert get_model_cost_map_module.get_model_cost_map_source_info() == before


@pytest.mark.parametrize(
    "url", ["https://catalog.invalid/prices", "http://catalog.invalid/prices", "s3://bucket/prices"]
)
def test_explicit_multi_attempt_fetch_delegates_to_native_retry(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path: Final = _catalog_file(tmp_path, _load_root_cost_map())
    outcome: Final = _fetch.fetch_once(path.as_uri(), if_none_match=None, timeout_seconds=1)
    assert isinstance(outcome, FetchedRegistry)
    native: Final = Mock(return_value=outcome)
    monkeypatch.setattr(_fetch, "fetch_with_retry", native)
    result: Final = fetch(Remote(url), max_attempts=4, timeout=2.5)
    native.assert_called_once_with(url, if_none_match=None, timeout_seconds=2.5, max_attempts=4)
    assert isinstance(result, LoadedCatalog)
    assert result.registry.source_revision == git_blob_id(path.read_bytes())
    assert result.registry.source_validator == outcome.validator


def test_native_file_and_bundled_fetch(tmp_path: Path) -> None:
    path: Final = _catalog_file(tmp_path, {**_load_root_cost_map(), "file-model": {"mode": "chat"}})
    result: Final = fetch(File(str(path)))
    assert isinstance(result, LoadedCatalog)
    assert served_rows(result)["file-model"]["mode"] == "chat"
    assert result.registry.source_revision == git_blob_id(path.read_bytes())
    assert isinstance(fetch(Bundled()), LoadedCatalog)
    assert isinstance(fetch(File(str(tmp_path / "missing"))), CatalogUnavailable)


@pytest.mark.parametrize("register", [True, False])
def test_reload_preserves_python_metadata_and_nested_edits(register: bool) -> None:
    install(_catalog({"model": {"mode": "chat", "input_cost_per_token": 1}}), provenance=Provenance("local"))
    original: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    metadata: Final = {"when": original, "coordinates": (1, 2)}
    if register:
        litellm.register_model({"model": {"python_metadata": metadata}})
    else:
        litellm.model_cost["model"]["python_metadata"] = metadata
    updated: Final = _catalog({"model": {"mode": "chat", "input_cost_per_token": 2}})
    install(updated, provenance=Provenance("local"))
    assert litellm.model_cost["model"]["input_cost_per_token"] == 2
    assert litellm.model_cost["model"]["python_metadata"] == metadata
    later: Final = datetime(2026, 2, 1, tzinfo=timezone.utc)
    litellm.model_cost["model"]["python_metadata"]["when"] = later
    install(updated, provenance=Provenance("local"))
    assert litellm.model_cost["model"]["python_metadata"] == {"when": later, "coordinates": (1, 2)}
    del litellm.model_cost["model"]["python_metadata"]
    install(updated, provenance=Provenance("local"))
    assert "python_metadata" not in litellm.model_cost["model"]


def test_register_url_uses_explicit_source_under_local_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw: Final = _load_root_cost_map()
    raw["url-registration"] = {"mode": "chat", "litellm_provider": "openai", "input_cost_per_token": 0.123}
    path: Final = _catalog_file(tmp_path, raw)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    live: Final = litellm.model_cost
    assert litellm.register_model(path.as_uri()) == path.as_uri()
    assert litellm.model_cost is live
    assert live["url-registration"]["input_cost_per_token"] == 0.123


def test_failed_url_registration_is_a_single_logged_noop(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    before: Final = copy.deepcopy(litellm.model_cost)
    registrations: Final = copy.deepcopy(litellm.utils._runtime_registered_model_cost)
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        url: Final = (tmp_path / "missing.json").as_uri()
        assert litellm.register_model(url) == url
    assert litellm.model_cost == before
    assert litellm.utils._runtime_registered_model_cost == registrations
    assert sum("register_model fetch failed" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("custom_dict", [False, True])
def test_install_keeps_live_identity_direct_nested_edits_and_tombstones(
    custom_dict: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CostMap(dict):
        def __getitem__(self, key):
            self.last_read = key
            return super().__getitem__(key)

        def clear(self) -> None:
            pytest.fail("installer must not clear the live dict")

        def update(self, *args, **kwargs) -> None:
            pytest.fail("installer must use native row publication, not Python update")

    old: Final = {
        "retained": {"mode": "chat", "metadata": {"version": 1}, "input_cost_per_token": 1.0},
        "deleted": {"mode": "chat"},
        "removed": {"mode": "chat"},
    }
    monkeypatch.setattr(litellm, "model_cost", CostMap() if custom_dict else {})
    monkeypatch.setattr(_install.STATE, "rows", {})
    install(_catalog(old), provenance=Provenance("local"))
    live: Final = litellm.model_cost
    live["retained"]["metadata"]["version"] = 7
    live["direct"] = {"mode": "chat", "input_cost_per_token": 9.0}
    del live["deleted"]

    new: Final = {
        "retained": {"mode": "chat", "metadata": {"version": 2}, "input_cost_per_token": 3.0},
        "deleted": {"mode": "chat", "input_cost_per_token": 2.0},
        "added": {"mode": "chat"},
    }
    assert install(_catalog(new), provenance=Provenance("local")) == len(new)
    assert litellm.model_cost is live
    assert live == {
        "retained": {"mode": "chat", "metadata": {"version": 7}, "input_cost_per_token": 3.0},
        "direct": {"mode": "chat", "input_cost_per_token": 9.0},
        "added": {"mode": "chat"},
    }
    if custom_dict:
        assert live["added"]["mode"] == "chat"
        assert live.last_read == "added"


def test_install_sparse_registration_order_survives_multiple_reloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(_install.STATE, "rows", {})
    old: Final = {
        "gpt-4o": {
            "mode": "chat",
            "litellm_provider": "openai",
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
        }
    }
    install(_catalog(old), provenance=Provenance("local"))
    litellm.register_model({"gpt-4o": {"input_cost_per_token": 3.0}})
    litellm.model_cost["gpt-4o"]["input_cost_per_token"] = 4.0
    new: Final = {"gpt-4o": {**old["gpt-4o"], "input_cost_per_token": 5.0, "output_cost_per_token": 6.0}}
    install(_catalog(new), provenance=Provenance("local"))
    assert litellm.model_cost["gpt-4o"]["input_cost_per_token"] == 4.0
    assert litellm.model_cost["gpt-4o"]["output_cost_per_token"] == 6.0
    litellm.register_model({"gpt-4o": {"input_cost_per_token": 7.0}})
    install(_catalog(new), provenance=Provenance("local"))
    assert litellm.model_cost["gpt-4o"]["input_cost_per_token"] == 7.0
    assert litellm.model_cost["gpt-4o"]["output_cost_per_token"] == 6.0


def test_paused_live_iterator_keeps_python_key_change_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(_install.STATE, "rows", {})
    install(_catalog({"old": {"mode": "chat"}}), provenance=Provenance("local"))
    ready: Final = threading.Event()
    resume: Final = threading.Event()
    errors: Final = []

    def read() -> None:
        try:
            for _ in litellm.model_cost:
                ready.set()
                resume.wait()
        except RuntimeError as error:
            errors.append(str(error))

    reader: Final = threading.Thread(target=read)
    reader.start()
    try:
        assert ready.wait(10)
        install(_catalog({"new": {"mode": "chat"}, "extra": {"mode": "chat"}}), provenance=Provenance("local"))
    finally:
        resume.set()
        reader.join(10)
    assert not reader.is_alive()
    assert errors == ["dictionary changed size during iteration"]


def test_snapshot_readers_never_observe_empty_or_partial_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(_install.STATE, "rows", {})
    old: Final = {f"old-{i}": {"mode": "chat"} for i in range(300)}
    new: Final = {f"new-{i}": {"mode": "chat"} for i in range(300)}
    install(_catalog(old), provenance=Provenance("local"))
    ready: Final = threading.Event()
    saw_new: Final = threading.Event()
    stop: Final = threading.Event()
    failures: Final = []
    observations: Final = set()

    def read() -> None:
        while not stop.is_set():
            snapshot = tuple(litellm.model_cost.items())
            observations.add(tuple(key for key, _ in snapshot))
            if dict(snapshot) == new:
                saw_new.set()
            if dict(snapshot) != old and dict(snapshot) != new:
                failures.append(snapshot)
            ready.set()

    reader: Final = threading.Thread(target=read)
    reader.start()
    try:
        assert ready.wait(10)
        for rows in (new, old, new, old):
            install(_catalog(rows), provenance=Provenance("local"))
            if rows is new:
                assert saw_new.wait(10)
    finally:
        stop.set()
        reader.join(10)
    assert not reader.is_alive()
    assert observations
    assert not failures


@pytest.mark.asyncio
async def test_reload_filters_rejections_before_integrity_and_keeps_previous_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    raw: Final = _load_root_cost_map()
    raw["bad-row"] = {"mode": "chat", "input_cost_per_token": -1}
    path: Final = _catalog_file(tmp_path, raw)
    live: Final = litellm.model_cost
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        result: Final = await reload(path.as_uri())
    assert isinstance(result, Reloaded)
    assert litellm.model_cost is live
    assert "bad-row" not in live
    assert sum("bad-row.input_cost_per_token" in record.message for record in caplog.records) == 1
    before: Final = copy.deepcopy(live)
    provenance: Final = get_model_cost_map_module.get_model_cost_map_source_info()
    path.write_text(json.dumps({f"bad-{i}": {"mode": "telepathy"} for i in range(len(raw))}))
    with caplog.at_level(logging.CRITICAL, logger="LiteLLM"):
        failed: Final = await reload(path.as_uri())
    assert isinstance(failed, CatalogUnavailable)
    assert litellm.model_cost == before
    assert get_model_cost_map_module.get_model_cost_map_source_info() == provenance


@pytest.mark.asyncio
async def test_reload_publishes_provenance_snapshot_and_invalidates_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    raw: Final = _load_root_cost_map()
    raw["gpt-4o"]["input_cost_per_token"] = 0.543
    path: Final = _catalog_file(tmp_path, raw)
    live: Final = litellm.model_cost
    litellm.get_model_info("gpt-4o")
    assert isinstance(await reload(path.as_uri()), Reloaded)
    assert litellm.model_cost is live
    assert litellm.get_model_info("gpt-4o")["input_cost_per_token"] == 0.543
    assert GetModelCostMap.loaded_model_cost_map()["gpt-4o"]["input_cost_per_token"] == 0.543
    assert get_model_cost_map_provenance() == {"source_revision": git_blob_id(path.read_bytes()), "etag": None}
    assert get_model_cost_map_module.get_model_cost_map_source_info()["source"] == "remote"


@pytest.mark.parametrize("terminal", [False, True])
def test_import_retry_preserves_first_retry_after_budget_and_terminal_stop(
    terminal: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path: Final = _catalog_file(tmp_path, _load_root_cost_map())
    success: Final = _fetch.fetch_once(path.as_uri(), if_none_match=None, timeout_seconds=1)
    first: Final = Mock(spec=RegistryUnavailable, reason="busy", retryable=True, retry_after=7)
    second: Final = Mock(
        spec=RegistryUnavailable, reason="missing" if terminal else "busy", retryable=not terminal, retry_after=None
    )
    native: Final = Mock(side_effect=[first, second, success])
    waits: Final = []
    done: Final = threading.Event()
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(_bootstrap, "fetch_once", native)
    original_retry: Final = _bootstrap._retry

    def finish(*args) -> None:
        try:
            original_retry(*args)
        finally:
            done.set()

    def sleep(wait: float) -> None:
        waits.append(wait)
        litellm.model_cost["post-import-write"] = {"mode": "chat"}

    monkeypatch.setattr(_bootstrap, "_retry", finish)
    _bootstrap.initialize("https://catalog.invalid/prices", max_attempts=3, sleep=sleep, rng=random.Random(0))
    assert done.wait(10)
    assert waits[0] == 7
    assert native.call_count == (2 if terminal else 3)
    if not terminal:
        assert len(waits) == 2
        assert 4 <= waits[1] <= 5
    assert litellm.model_cost["post-import-write"]["mode"] == "chat"


def test_import_retry_exhaustion_keeps_full_attempt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = Mock(spec=RegistryUnavailable, reason="busy", retryable=True, retry_after=100)
    native: Final = Mock(return_value=failure)
    waits: Final = []
    monkeypatch.setattr(_bootstrap, "fetch_once", native)
    before: Final = copy.deepcopy(litellm.model_cost)
    _bootstrap._retry(
        "s3://bucket/prices", 5, 3, waits.append, random.Random(0), _bootstrap._FetchAttemptRetryable("busy", 100)
    )
    assert native.call_count == 2
    assert waits == [30, 30]
    assert litellm.model_cost == before
