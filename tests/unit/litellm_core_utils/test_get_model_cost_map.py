"""
Tests for model-cost-map loading through the Rust catalog: fetching, retrying, validating, installing into the bound
dict, provenance and the extraction of the ``fallback_generalizations`` block out of the raw map
"""

import asyncio
import contextlib
import functools
import hashlib
import importlib
import json
import logging
import os
import sys
import tempfile
import threading
import types
from collections.abc import Iterator
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm._catalog import CATALOG, Catalog, Installed, Parsed, Unavailable
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
    get_model_cost_map,
    install_initial_model_cost_map,
    reload_model_cost_map,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.utils import _invalidate_model_cost_lowercase_map
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome

_ROOT: Final = Path(__file__).resolve().parents[3]


def git_blob_id(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


@functools.lru_cache(maxsize=1)
def _real_map_bytes() -> bytes:
    return (_ROOT / "model_prices_and_context_window.json").read_bytes()


def _load_root_cost_map() -> dict:
    return json.loads(_real_map_bytes())


def _make_models(n: int) -> dict:
    return {f"model-{i}": {"litellm_provider": "openai", "mode": "chat"} for i in range(n)}


@functools.lru_cache(maxsize=1)
def _bundled_document() -> dict:
    return CATALOG.bundled().document()


def _with(edit) -> bytes:
    document = json.loads(json.dumps(_bundled_document()))
    edit(document)
    return json.dumps(document).encode()


class _Server:
    """A local HTTP server answering each GET with the next queued reply, the last one repeating"""

    def __init__(self, replies: list[tuple[int, dict[str, str], bytes]]) -> None:
        self.replies = replies
        self.requests: list[dict[str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.requests.append(dict(self.headers))
                status, headers, body = outer.replies[min(len(outer.requests), len(outer.replies)) - 1]
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/model_prices.json"


@contextlib.contextmanager
def _serving(replies: list[tuple[int, dict[str, str], bytes]]) -> Iterator[_Server]:
    server = _Server(replies)
    thread = threading.Thread(target=server.httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.httpd.shutdown()
        server.httpd.server_close()


def _bound_catalog() -> tuple[types.ModuleType, Catalog]:
    module = types.ModuleType("fake_litellm")
    module.model_cost = {}
    catalog = Catalog.bound_to(module, "model_cost")
    catalog.install(catalog.bundled())
    return module, catalog


def _file(tmp_path: Path, body: bytes) -> str:
    path = tmp_path / f"map-{git_blob_id(body)[:12]}.json"
    path.write_bytes(body)
    return path.as_uri()


@pytest.fixture(autouse=True)
def _unset_local_cost_map_env(monkeypatch):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)


def _retry_threads():
    return [thread for thread in threading.enumerate() if thread.name == "litellm-model-cost-map-retry"]


# ---------------------------------------------------------------------------
# revision, validation and fallback rules
# ---------------------------------------------------------------------------


def test_the_revision_is_what_git_hash_object_prints(tmp_path):
    body = _real_map_bytes()
    fetched = CATALOG.load(_file(tmp_path, body), timeout=5, attempts=1)
    assert isinstance(fetched, Parsed)
    assert fetched.revision == git_blob_id(body)


def test_the_bundled_revision_is_the_root_json_blob_id():
    assert CATALOG.bundled().revision == git_blob_id(_real_map_bytes())


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b'"not a dict"', id="not_an_object"),
        pytest.param(b"{}", id="empty"),
        pytest.param(b"[1, 2, 3]", id="list"),
        pytest.param(b"null", id="null"),
        pytest.param(json.dumps(_make_models(5)).encode(), id="below_minimum"),
        pytest.param(
            json.dumps({**_make_models(60), "sample_spec": {}, FALLBACK_GENERALIZATIONS_KEY: {"rules": []}}).encode(),
            id="shrunk_with_meta_keys",
        ),
    ],
)
def test_a_map_that_fails_mains_whole_map_checks_is_unavailable(tmp_path, body):
    assert isinstance(CATALOG.load(_file(tmp_path, body), timeout=5, attempts=1), Unavailable)


def test_a_healthy_map_with_meta_keys_is_fetched(tmp_path):
    body = _with(lambda document: document.update({"sample_spec": {"foo": "bar"}}))
    assert isinstance(CATALOG.load(_file(tmp_path, body), timeout=5, attempts=1), Parsed)


def test_a_loaded_map_installs_its_rules_and_never_serves_the_block(tmp_path):
    previous = list(get_fallback_generalization_rules())
    rules = {"rules": [{"name": "rule", "pattern": r"^widget-", "model_info": {"litellm_provider": "openai"}}]}
    try:
        fetched = CATALOG.load(
            _file(tmp_path, _with(lambda document: document.update({FALLBACK_GENERALIZATIONS_KEY: rules}))),
            timeout=5,
            attempts=1,
        )
        assert isinstance(fetched, Parsed)

        get_model_cost_map_module._note_loaded(fetched)

        assert FALLBACK_GENERALIZATIONS_KEY not in fetched.served()
        assert match_routing_generalization("widget-9") == "openai"
    finally:
        set_fallback_generalizations(previous)


def test_a_loaded_map_with_no_block_clears_rules(tmp_path):
    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations([{"name": "stale", "pattern": r"^x", "model_info": {"a": 1}}])
        fetched = CATALOG.load(
            _file(tmp_path, _with(lambda document: document.pop(FALLBACK_GENERALIZATIONS_KEY))),
            timeout=5,
            attempts=1,
        )
        assert isinstance(fetched, Parsed)
        get_model_cost_map_module._note_loaded(fetched)
        assert match_capability_generalizations("x-1") is None
    finally:
        set_fallback_generalizations(previous)


def test_shipped_backup_carries_the_claude_routing_rules():
    """The bundled backup must ship the Claude routing rules so a fresh install
    (or an offline fallback) routes unknown Claude models without code changes.
    Bedrock-syntax ids must hit the bedrock rule before the bare-id Anthropic rule."""
    backup = CATALOG.bundled().document()
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
    backup = CATALOG.bundled().document()
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
    backup = CATALOG.bundled().document()

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


# ---------------------------------------------------------------------------
# reload_model_cost_map: retry/backoff behavior for runtime reloads
# ---------------------------------------------------------------------------


def test_reload_retries_429_honoring_retry_after():
    _, catalog = _bound_catalog()
    with _serving([(429, {"Retry-After": "0"}, b""), (200, {}, _real_map_bytes())]) as server:
        result = reload_model_cost_map(server.url, catalog)
    assert isinstance(result, Installed)
    assert len(server.requests) == 2


def test_reload_gives_up_after_max_attempts():
    _, catalog = _bound_catalog()
    with _serving([(503, {"Retry-After": "0"}, b"")]) as server:
        result = reload_model_cost_map(server.url, catalog)
    assert isinstance(result, Unavailable)
    assert result.reason == f"gave up after 3 attempts: HTTP 503 from {server.url}"
    assert len(server.requests) == 3


@pytest.mark.parametrize("status", [404, 501])
def test_reload_non_retryable_status_fails_immediately(status):
    _, catalog = _bound_catalog()
    with _serving([(status, {}, b"")]) as server:
        result = reload_model_cost_map(server.url, catalog)
    assert isinstance(result, Unavailable)
    assert result.reason == f"HTTP {status} from {server.url}"
    assert len(server.requests) == 1


def test_reload_invalid_json_fails_immediately():
    _, catalog = _bound_catalog()
    with _serving([(200, {}, b"not json")]) as server:
        result = reload_model_cost_map(server.url, catalog)
    assert isinstance(result, Unavailable)
    assert "invalid JSON" in result.reason
    assert len(server.requests) == 1


def test_reload_shrunk_map_fails_integrity_and_keeps_the_map():
    module, catalog = _bound_catalog()
    before = dict(module.model_cost)
    with _serving([(200, {}, json.dumps(_make_models(60)).encode())]) as server:
        result = reload_model_cost_map(server.url, catalog)
    assert isinstance(result, Unavailable)
    assert "below 0.5 of reference count" in result.reason
    assert module.model_cost == before


@pytest.mark.parametrize("flag", ["LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_OFFLINE_MODE"])
def test_a_reload_under_the_local_env_override_installs_the_bundled_map_without_a_request(monkeypatch, flag):
    _, catalog = _bound_catalog()
    monkeypatch.setenv(flag, "True")
    edited = _with(lambda document: document["gpt-5.4-mini"].update({"input_cost_per_token": 0.5}))

    with _serving([(200, {}, edited)]) as server:
        result = reload_model_cost_map(server.url, catalog)

    assert isinstance(result, Installed)
    assert server.requests == []
    provenance = catalog.provenance()
    assert (provenance.source, provenance.url, provenance.is_env_forced) == ("local", None, True)
    assert provenance.source_revision == git_blob_id(_real_map_bytes())


def test_reload_records_the_blob_id_of_the_bytes_served_and_the_fetch_etag():
    _, catalog = _bound_catalog()
    body = _real_map_bytes()
    with _serving([(200, {"ETag": 'W/"abc123"'}, body)]) as server:
        reload_model_cost_map(server.url, catalog)
    provenance = catalog.provenance()
    assert provenance is not None
    assert (provenance.source_revision, provenance.etag, provenance.source, provenance.url) == (
        git_blob_id(body),
        'W/"abc123"',
        "remote",
        server.url,
    )


def test_loaded_catalog_snapshot_follows_the_fetched_map_and_ignores_later_registrations(monkeypatch):
    _, catalog = _bound_catalog()
    edited = _with(lambda document: document["gpt-5.4-mini"].update({"max_input_tokens": 777}))
    with _serving([(200, {}, edited)]) as server:
        reload_model_cost_map(server.url, catalog)

    monkeypatch.setattr(litellm, "model_cost", dict(GetModelCostMap.loaded_model_cost_map()))
    litellm.register_model({"gpt-5.4-mini": {"max_input_tokens": 2048}}, persist_across_reloads=False)
    assert litellm.model_cost["gpt-5.4-mini"]["max_input_tokens"] == 2048
    assert GetModelCostMap.loaded_model_cost_map()["gpt-5.4-mini"]["max_input_tokens"] == 777


def test_loaded_catalog_fields_follow_each_fetched_map():
    _, catalog = _bound_catalog()
    added = _with(lambda document: document["gpt-5.4-mini"].update({"supports_litellm_qa_capability": True}))
    with _serving([(200, {}, added)]) as server:
        reload_model_cost_map(server.url, catalog)
    fields_after_adding: Final = GetModelCostMap.loaded_model_cost_map_fields()
    with _serving([(200, {}, _real_map_bytes())]) as server:
        reload_model_cost_map(server.url, catalog)

    assert "supports_litellm_qa_capability" in fields_after_adding
    assert "supports_litellm_qa_capability" not in GetModelCostMap.loaded_model_cost_map_fields()
    assert "max_input_tokens" in GetModelCostMap.loaded_model_cost_map_fields()


def test_reload_revision_follows_the_bytes_not_the_url():
    _, catalog = _bound_catalog()
    edited = _with(lambda document: document["gpt-5.4-mini"].update({"input_cost_per_token": 0.5}))
    with _serving([(200, {}, _real_map_bytes()), (200, {}, edited)]) as server:
        first = reload_model_cost_map(server.url, catalog)
        first_revision = catalog.provenance().source_revision
        second = reload_model_cost_map(server.url, catalog)
    assert isinstance(first, Installed) and isinstance(second, Installed)
    assert first_revision != catalog.provenance().source_revision == git_blob_id(edited)


def test_a_bundled_reload_reports_the_bundled_blob_id_without_an_etag(monkeypatch):
    _, catalog = _bound_catalog()
    with _serving([(200, {"ETag": 'W/"remote"'}, _real_map_bytes())]) as server:
        reload_model_cost_map(server.url, catalog)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    reload_model_cost_map(server.url, catalog)

    provenance = catalog.provenance()
    assert (provenance.source_revision, provenance.etag, provenance.source, provenance.is_env_forced) == (
        git_blob_id(_real_map_bytes()),
        None,
        "local",
        True,
    )


def test_reload_stamps_loaded_at_when_the_load_starts():
    _, catalog = _bound_catalog()
    before = datetime.now(timezone.utc)
    with _serving([(200, {}, _real_map_bytes())]) as server:
        reload_model_cost_map(server.url, catalog)
    loaded_at = catalog.provenance().loaded_at
    assert before <= loaded_at <= datetime.now(timezone.utc)


def test_a_failed_reload_keeps_the_provenance_of_the_map_in_use():
    _, catalog = _bound_catalog()
    kept = catalog.provenance()
    with _serving([(404, {}, b"")]) as server:
        reload_model_cost_map(server.url, catalog)
    assert catalog.provenance().source_revision == kept.source_revision
    assert catalog.provenance().loaded_at == kept.loaded_at


def test_reload_reads_a_file_url(tmp_path):
    module, catalog = _bound_catalog()
    path = tmp_path / "map.json"
    path.write_bytes(_with(lambda document: document.update({"only-in-the-file": {"litellm_provider": "openai"}})))

    result = reload_model_cost_map(path.as_uri(), catalog)

    assert isinstance(result, Installed)
    assert "only-in-the-file" in module.model_cost
    assert catalog.provenance().source == "remote"


# ---------------------------------------------------------------------------
# install_initial_model_cost_map: the import-time load
# ---------------------------------------------------------------------------


class _SyncSleepRecorder:
    """Injected as the retry thread's first wait so the boot path is asserted without delay"""

    def __init__(self, block=False):
        self.waits = []
        self.block = block
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, seconds: float) -> None:
        if self.block:
            self.started.set()
            self.release.wait(timeout=10)
        self.waits.append(seconds)


def test_boot_load_success_does_not_start_background_retry():
    module, catalog = _bound_catalog()
    sleeper = _SyncSleepRecorder()
    with _serving([(200, {}, _real_map_bytes())]) as server:
        install_initial_model_cost_map(server.url, catalog, sleeper)
    assert len(server.requests) == 1
    assert sleeper.waits == []
    assert _retry_threads() == []
    assert module.model_cost.keys() >= _load_root_cost_map().keys() - {"sample_spec", FALLBACK_GENERALIZATIONS_KEY}
    assert catalog.provenance().source == "remote"


def test_boot_load_transient_failure_installs_local_then_background_retry_installs_remote(monkeypatch):
    from litellm import utils as litellm_utils

    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    for name, provider_models in tuple(vars(litellm).items()):
        if name.endswith("_models") and isinstance(provider_models, set):
            monkeypatch.setattr(litellm, name, set(provider_models))
    monkeypatch.setattr(litellm, "models_by_provider", dict(litellm.models_by_provider))
    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )
    catalog = Catalog.bound_to(litellm, "model_cost")
    held = litellm.model_cost
    remote = _with(
        lambda document: (
            document.update({"claude-remote-only-test": {"litellm_provider": "anthropic", "mode": "chat"}}),
            document["gpt-4o"].update({"input_cost_per_token": 0.125}),
        )
    )
    sleeper = _SyncSleepRecorder(block=True)
    litellm.register_model({"my-runtime-model": {"litellm_provider": "custom", "max_input_tokens": 4321}})

    with _serving([(503, {"Retry-After": "0"}, b""), (200, {}, remote)]) as server:
        install_initial_model_cost_map(server.url, catalog, sleeper)
        assert sleeper.started.wait(timeout=10)
        threads = _retry_threads()
        try:
            assert len(server.requests) == 1
            assert len(threads) == 1
            assert "claude-remote-only-test" not in litellm.model_cost
            provenance = catalog.provenance()
            assert provenance.source == "local"
            assert provenance.fallback_reason.startswith("Remote fetch failed:")
            sleeper.release.set()
            for thread in threads:
                thread.join(timeout=10)
            assert all(not thread.is_alive() for thread in threads)
            assert sleeper.waits == [0.0]
            assert len(server.requests) == 2
            assert "claude-remote-only-test" in litellm.model_cost
            assert litellm.model_cost is held
            assert held["gpt-4o"]["input_cost_per_token"] == 0.125
            assert "claude-remote-only-test" in litellm.anthropic_models
            assert "my-runtime-model" in litellm.model_cost
            assert catalog.provenance().source == "remote"
            assert catalog.provenance().fallback_reason is None
        finally:
            sleeper.release.set()
            for thread in _retry_threads():
                thread.join(timeout=10)


def test_boot_load_does_not_retry_non_retryable_failure():
    _, catalog = _bound_catalog()
    sleeper = _SyncSleepRecorder()
    with _serving([(404, {}, b"")]) as server:
        install_initial_model_cost_map(server.url, catalog, sleeper)
    assert len(server.requests) == 1
    assert sleeper.waits == []
    assert _retry_threads() == []
    provenance = catalog.provenance()
    assert provenance.source == "local"
    assert provenance.fallback_reason == f"Remote fetch failed: HTTP 404 from {server.url}"
    assert provenance.url == server.url


def test_boot_load_falls_back_when_the_url_does_not_parse():
    module, catalog = _bound_catalog()
    install_initial_model_cost_map("http://[bad/model_prices.json", catalog, _SyncSleepRecorder())
    assert _retry_threads() == []
    assert module.model_cost.keys() >= _load_root_cost_map().keys() - {"sample_spec", FALLBACK_GENERALIZATIONS_KEY}
    assert "is not a URL" in (catalog.provenance().fallback_reason or "")


@pytest.mark.parametrize("flag", ["LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_OFFLINE_MODE"])
def test_boot_load_respects_local_env_override(monkeypatch, flag):
    monkeypatch.setenv(flag, "True")
    module, catalog = _bound_catalog()
    with _serving([(200, {}, _real_map_bytes())]) as server:
        install_initial_model_cost_map(server.url, catalog, _SyncSleepRecorder())
    assert server.requests == []
    assert len(module.model_cost) > 100
    assert catalog.provenance().is_env_forced is True


def test_boot_load_records_the_blob_id_of_the_bytes_served_and_the_fetch_etag():
    _, catalog = _bound_catalog()
    body = _real_map_bytes()
    with _serving([(200, {"ETag": 'W/"boot"'}, body)]) as server:
        install_initial_model_cost_map(server.url, catalog, _SyncSleepRecorder())
    provenance = catalog.provenance()
    assert (provenance.source, provenance.etag, provenance.source_revision) == ("remote", 'W/"boot"', git_blob_id(body))


def test_boot_load_fallback_to_the_bundled_map_reports_its_blob_id_and_drops_the_remote_etag():
    _, catalog = _bound_catalog()
    with _serving([(200, {"ETag": 'W/"boot"'}, _real_map_bytes())]) as server:
        install_initial_model_cost_map(server.url, catalog, _SyncSleepRecorder())
    with _serving([(404, {}, b"")]) as server:
        install_initial_model_cost_map(server.url, catalog, _SyncSleepRecorder())
    provenance = catalog.provenance()
    assert (provenance.source, provenance.etag, provenance.source_revision) == (
        "local",
        None,
        git_blob_id(_real_map_bytes()),
    )


def test_boot_load_that_fails_the_integrity_check_reports_the_bundled_map_not_the_rejected_fetch():
    _, catalog = _bound_catalog()
    shrunk_body = b'{"gpt-5.4-mini": {"mode": "chat", "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}}'
    with _serving([(200, {"ETag": 'W/"shrunk"'}, shrunk_body)]) as server:
        install_initial_model_cost_map(server.url, catalog, _SyncSleepRecorder())
    provenance = catalog.provenance()
    assert provenance.source == "local"
    assert "below minimum" in provenance.fallback_reason
    assert provenance.etag is None
    assert provenance.source_revision == git_blob_id(_real_map_bytes())


@pytest.mark.parametrize(
    ("argv0", "request_count"),
    [
        ("/some/venv/bin/lite", 0),
        ("/some/venv/bin/lite.exe", 0),
        ("/some/venv/bin/python", 1),
    ],
)
def test_boot_load_skips_remote_fetch_for_cli_processes(
    monkeypatch: pytest.MonkeyPatch, argv0: str, request_count: int
) -> None:
    monkeypatch.setattr(sys, "argv", [argv0, "--version"])
    _, catalog = _bound_catalog()
    with _serving([(200, {}, _real_map_bytes())]) as server:
        install_initial_model_cost_map(server.url, catalog, _SyncSleepRecorder())
    assert len(server.requests) == request_count
    assert catalog.provenance().source == ("local" if request_count == 0 else "remote")
    assert catalog.provenance().is_env_forced is False


def test_get_model_cost_map_returns_the_remote_rows_without_installing_them():
    module, catalog = _bound_catalog()
    remote = _with(lambda document: document.update({"only-remote": {"litellm_provider": "openai"}}))
    with _serving([(200, {}, remote)]) as server:
        cost_map = get_model_cost_map(server.url)
    assert "only-remote" in cost_map
    assert "only-remote" not in litellm.model_cost


def test_get_model_cost_map_warns_that_an_injected_client_is_ignored():
    with _serving([(200, {}, _real_map_bytes())]) as server, pytest.warns(DeprecationWarning):
        get_model_cost_map(server.url, client=object())


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}


def _backup_keys():
    return CATALOG.bundled().served().keys()


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param((200, {}, b"not json"), id="invalid_json"),
        pytest.param((200, {}, b"{}"), id="empty"),
        pytest.param((200, {}, b"[]"), id="not_an_object"),
        pytest.param((200, {}, json.dumps(_make_models(11)).encode()), id="shrinks_dramatically"),
        pytest.param((404, {}, b""), id="not_found"),
    ],
)
def test_get_model_cost_map_falls_back_to_the_bundled_map(reply):
    with _serving([reply]) as server:
        result = get_model_cost_map(server.url, max_attempts=1)
    assert len(server.requests) == 1
    assert result.keys() == _backup_keys()


def test_get_model_cost_map_uses_the_local_map_when_the_env_var_is_set(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    with _serving([(200, {}, b"{}")]) as server:
        result = get_model_cost_map(server.url)
    assert server.requests == []
    assert result.keys() == _backup_keys()


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestBackupModelCostMapExists:
    """Validates the local backup file is always present and valid."""

    def test_should_have_backup_file(self):
        """The bundled model cost map must be embedded and loadable."""
        backup = CATALOG.bundled().served()
        assert isinstance(backup, dict)
        assert len(backup) > 0, "Backup model cost map is empty"

    def test_should_have_minimum_models_in_backup(self):
        """The backup must contain a reasonable number of models."""
        backup = CATALOG.bundled().served()
        assert len(backup) > 100, f"Backup has only {len(backup)} models, expected > 100"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestBadHostedModelCostMap:
    """
    Simulates the hosted model cost map being bad (invalid JSON / corrupted).

    When the hosted map is bad, get_model_cost_map() falls back to the local
    backup. These tests verify that after fallback:
    - get_model_info() still works for models in the backup
    - litellm.completion() still works
    """

    def test_should_model_info_pass_after_bad_hosted_map(self):
        """
        If the hosted map is bad, get_model_cost_map falls back to the local
        backup. get_model_info should still work for models in the backup.
        """
        with _serving([(200, {}, b"not json")]) as server:
            fallback_map = get_model_cost_map(server.url)

        original = litellm.model_cost
        litellm.model_cost = fallback_map
        try:
            # gpt-4o is in every backup — should work fine
            info = litellm.get_model_info("gpt-4o")
            assert info is not None
            assert info["input_cost_per_token"] > 0
        finally:
            litellm.model_cost = original

    def test_should_completion_pass_after_bad_hosted_map(self):
        """
        If the hosted map is bad, litellm.completion() should still work.

        Uses litellm's built-in mock_response param so the real completion
        path is exercised (routing, cost calculator, logging) without
        needing API credentials.
        """
        # Simulate bad hosted map → fallback to backup
        with _serving([(200, {}, b"not json")]) as server:
            fallback_map = get_model_cost_map(server.url)

        original = litellm.model_cost
        litellm.model_cost = fallback_map
        try:
            # mock_response goes through the real completion path —
            # routing, cost calculator, logging — but skips the HTTP call
            response = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "say hi"}],
                mock_response="hello from mock",
            )
            assert response is not None
            assert response.choices[0].message.content == "hello from mock"
        finally:
            litellm.model_cost = original


@pytest.fixture()
def _vcr_outcome_gate_local_testing(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS_local_testing:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS_local_testing.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()


_SCALAR_DEFAULTS_local_testing = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}


@pytest.fixture(scope="module")
def setup_and_teardown_local_testing():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield


@pytest.mark.usefixtures(
    "_vcr_outcome_gate_local_testing",
    "isolate_litellm_state",
    "setup_and_teardown_local_testing",
)
def test_get_model_cost_map():
    try:
        print(litellm.get_model_cost_map(url="fake-url"))
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")


# ---------------------------------------------------------------------------
# Install through the Rust catalog
# ---------------------------------------------------------------------------


def _remote(edit) -> Parsed:
    path = Path(tempfile.mkdtemp()) / "map.json"
    path.write_bytes(_with(edit))
    fetched = CATALOG.load(path.as_uri(), timeout=5, attempts=1)
    assert isinstance(fetched, Parsed)
    return fetched


def test_sample_spec_is_not_served_but_stays_in_the_bundled_document():
    assert "sample_spec" not in litellm.model_cost
    assert "sample_spec" in CATALOG.bundled().document()


def test_a_fetched_row_that_fails_validation_is_replaced_by_bundled_or_left_out(caplog):
    module, catalog = _bound_catalog()
    bundled_gpt_4o = dict(module.model_cost["gpt-4o"])

    def edit(document):
        document["gpt-4o"]["input_cost_per_token"] = -1
        document["acme/new"] = {"litellm_provider": "acme", "input_cost_per_token": "free"}

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        installed = get_model_cost_map_module._install(_remote(edit), catalog)

    assert module.model_cost["gpt-4o"] == bundled_gpt_4o
    assert "acme/new" not in module.model_cost
    assert {error.key for error in installed.rejected} == {"gpt-4o", "acme/new"}
    warnings = [record for record in caplog.records if "failed validation" in record.getMessage()]
    assert len(warnings) == 1
    assert "gpt-4o" in warnings[0].getMessage() and "acme/new" in warnings[0].getMessage()


def test_a_field_the_fetched_row_lacks_is_not_served():
    module, catalog = _bound_catalog()

    catalog.install(_remote(lambda document: document["gpt-4o"].pop("max_output_tokens")))

    assert "max_output_tokens" not in module.model_cost["gpt-4o"]


@pytest.mark.parametrize(
    "drop",
    [lambda document: document.pop("gpt-4o"), lambda document: document.update({"gpt-4o": None})],
    ids=["left_out", "null"],
)
def test_a_key_the_fetched_map_does_not_serve_is_removed(drop) -> None:
    module, catalog = _bound_catalog()

    catalog.install(_remote(drop))

    assert "gpt-4o" not in module.model_cost


def test_a_null_field_in_the_fetched_map_removes_it():
    module, catalog = _bound_catalog()

    catalog.install(_remote(lambda document: document["gpt-4o"].update({"max_output_tokens": None})))

    assert "max_output_tokens" not in module.model_cost["gpt-4o"]


def _direct_writes(model_cost: dict) -> None:
    model_cost["gpt-4o"]["input_cost_per_token"] = 1.0
    model_cost["my-own-model"] = {"litellm_provider": "openai", "input_cost_per_token": 3.0}
    model_cost["gpt-4o-mini"] = {"litellm_provider": "openai", "input_cost_per_token": 4.0}
    del model_cost["gpt-4.1"]


def _changed_upstream(document: dict) -> None:
    document["gpt-4o"]["input_cost_per_token"] = 9.0
    document["gpt-4o-mini"]["input_cost_per_token"] = 9.0
    document["gpt-4o-mini"]["max_input_tokens"] = 1


def test_direct_writes_survive_a_reload():
    module, catalog = _bound_catalog()
    held = module.model_cost
    _direct_writes(module.model_cost)

    catalog.install(_remote(_changed_upstream))

    assert module.model_cost is held
    assert module.model_cost["gpt-4o"]["input_cost_per_token"] == 1.0
    assert module.model_cost["my-own-model"]["input_cost_per_token"] == 3.0
    assert module.model_cost["gpt-4o-mini"]["input_cost_per_token"] == 4.0
    assert "max_input_tokens" not in module.model_cost["gpt-4o-mini"]
    assert "gpt-4.1" not in module.model_cost


def test_rebinding_the_dict_then_reloading_drops_direct_writes():
    module, catalog = _bound_catalog()
    _direct_writes(module.model_cost)
    module.model_cost = dict(module.model_cost)

    catalog.install(_remote(_changed_upstream))

    assert module.model_cost["gpt-4o"]["input_cost_per_token"] == 9.0
    assert "my-own-model" not in module.model_cost
    assert module.model_cost["gpt-4o-mini"]["max_input_tokens"] == 1
    assert "gpt-4.1" in module.model_cost


def test_install_keeps_a_row_object_whose_fields_did_not_change():
    module, catalog = _bound_catalog()
    row = module.model_cost["gpt-4o"]

    catalog.install(_remote(lambda document: document["gpt-4o-mini"].update({"input_cost_per_token": 9.0})))

    assert module.model_cost["gpt-4o"] is row


def test_install_refuses_a_bound_attribute_that_is_not_a_dict():
    module, catalog = _bound_catalog()
    module.model_cost = types.MappingProxyType({})

    with pytest.raises(TypeError):
        catalog.install(catalog.bundled())


def test_the_local_map_is_the_embedded_one_not_the_backup_file(monkeypatch, tmp_path):
    (tmp_path / "model_prices_and_context_window_backup.json").write_text(
        json.dumps({"only-in-the-backup": {"litellm_provider": "openai"}})
    )
    monkeypatch.setattr(get_model_cost_map_module, "files", lambda _package: tmp_path, raising=False)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    cost_map = get_model_cost_map(url="http://127.0.0.1:9/never-fetched.json")

    assert "only-in-the-backup" not in cost_map
    assert cost_map.keys() == CATALOG.bundled().served().keys()
