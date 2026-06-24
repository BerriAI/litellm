"""Tests for the Asqav local-first audit-log callback.

All tests are self-contained: they use only stdlib + the integration module
itself.  No LLM API calls, no network, no external services.

Chain property tests verify:
- Appending N records produces a valid chain (every hash links to its predecessor).
- Mutating one byte in any record causes verify_chain to detect the break.
- A chain survives a process restart (state loaded from the tail of the file).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import importlib
import importlib.util
import types

# ---------------------------------------------------------------------------
# Bootstrap: load litellm.integrations.asqav.asqav without triggering
# litellm/__init__.py (which needs tokenizers, openai, etc.).  We stub the
# minimal litellm sub-modules the integration actually imports at the top of
# its file, then load the module via importlib.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, _REPO_ROOT)


def _stub_litellm_deps() -> None:
    """Install minimal stubs for litellm sub-modules imported by asqav.py."""
    if "litellm" in sys.modules:
        return  # already loaded (e.g. in the full test suite)

    # litellm package stub
    pkg = types.ModuleType("litellm")
    sys.modules["litellm"] = pkg

    # litellm._logging stub
    class _VL:
        def debug(self, *a: object, **k: object) -> None:
            pass

        def warning(self, *a: object, **k: object) -> None:
            pass

    log_mod = types.ModuleType("litellm._logging")
    log_mod.verbose_logger = _VL()  # type: ignore[attr-defined]
    sys.modules["litellm._logging"] = log_mod

    # litellm.integrations package + custom_logger stub
    integrations_pkg = types.ModuleType("litellm.integrations")
    sys.modules["litellm.integrations"] = integrations_pkg
    pkg.integrations = integrations_pkg  # type: ignore[attr-defined]

    class _CustomLogger:
        def __init__(self, **kw: object) -> None:
            pass

    cl_mod = types.ModuleType("litellm.integrations.custom_logger")
    cl_mod.CustomLogger = _CustomLogger  # type: ignore[attr-defined]
    sys.modules["litellm.integrations.custom_logger"] = cl_mod

    # litellm.types stubs (imported in type annotations only)
    types_pkg = types.ModuleType("litellm.types")
    sys.modules["litellm.types"] = types_pkg
    utils_mod = types.ModuleType("litellm.types.utils")
    sys.modules["litellm.types.utils"] = utils_mod

    # litellm.llms stubs (kept minimal; asqav.py no longer imports from
    # litellm.llms.custom_httpx.http_handler, but other litellm internals may
    # still need the package hierarchy present during import resolution).
    llms_pkg = types.ModuleType("litellm.llms")
    sys.modules["litellm.llms"] = llms_pkg
    custom_httpx_pkg = types.ModuleType("litellm.llms.custom_httpx")
    sys.modules["litellm.llms.custom_httpx"] = custom_httpx_pkg
    http_handler_mod = types.ModuleType("litellm.llms.custom_httpx.http_handler")
    sys.modules["litellm.llms.custom_httpx.http_handler"] = http_handler_mod

    # litellm.types.llms stub
    types_llms_pkg = types.ModuleType("litellm.types.llms")
    sys.modules["litellm.types.llms"] = types_llms_pkg
    custom_http_mod = types.ModuleType("litellm.types.llms.custom_http")
    sys.modules["litellm.types.llms.custom_http"] = custom_http_mod


_stub_litellm_deps()

# Now load the integration module directly.
_asqav_path = os.path.join(_REPO_ROOT, "litellm", "integrations", "asqav", "asqav.py")
_spec = importlib.util.spec_from_file_location(
    "litellm.integrations.asqav.asqav", _asqav_path
)
assert _spec and _spec.loader
_asqav_module = importlib.util.module_from_spec(_spec)
sys.modules["litellm.integrations.asqav.asqav"] = _asqav_module
_spec.loader.exec_module(_asqav_module)  # type: ignore[union-attr]

AsqavLogger = _asqav_module.AsqavLogger
_GENESIS_HASH = _asqav_module._GENESIS_HASH
_canonical_bytes = _asqav_module._canonical_bytes
_content_digest = _asqav_module._content_digest
_sha256_hex = _asqav_module._sha256_hex
_cloud_sign_payload = _asqav_module._cloud_sign_payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kwargs(model: str = "gpt-4o", content: str = "hello") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "litellm_call_id": f"test-{content[:8]}",
    }


def _make_response(content: str = "world") -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    resp._hidden_params = {}
    return resp


def _make_times() -> tuple:
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    return start, end


def _logger_at(path: str) -> AsqavLogger:
    return AsqavLogger(log_path=path, redact_content=True)


def _append_n(logger: AsqavLogger, n: int) -> None:
    start, end = _make_times()
    for i in range(n):
        logger.log_success_event(
            kwargs=_make_kwargs(content=f"msg-{i}"),
            response_obj=_make_response(content=f"resp-{i}"),
            start_time=start,
            end_time=end,
        )


def _read_records(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


def test_sha256_hex_is_64_chars() -> None:
    h = _sha256_hex(b"hello")
    assert len(h) == 64
    assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_canonical_bytes_is_deterministic() -> None:
    d = {"b": 2, "a": 1, "c": [3, 4]}
    b1 = _canonical_bytes(d)
    b2 = _canonical_bytes({"c": [3, 4], "a": 1, "b": 2})
    assert b1 == b2


def test_content_digest_returns_none_for_none() -> None:
    assert _content_digest(None) is None


def test_content_digest_is_stable() -> None:
    d1 = _content_digest("hello world")
    d2 = _content_digest("hello world")
    assert d1 == d2
    assert d1 is not None and len(d1) == 64


# ---------------------------------------------------------------------------
# Chain property tests
# ---------------------------------------------------------------------------


def test_single_record_genesis_chain(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    start, end = _make_times()

    logger.log_success_event(
        kwargs=_make_kwargs(),
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    assert len(records) == 1
    r = records[0]
    assert r["seq"] == 0
    assert r["prev_hash"] == _GENESIS_HASH
    assert r["status"] == "success"
    assert "record_hash" in r
    assert len(r["record_hash"]) == 64


def test_chain_links_correctly_for_n_records(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    n = 10
    _append_n(logger, n)

    records = _read_records(path)
    assert len(records) == n

    # First record's prev_hash is the genesis sentinel.
    assert records[0]["prev_hash"] == _GENESIS_HASH

    # Each subsequent record's prev_hash equals the hash of the prior record.
    for i in range(1, n):
        assert (
            records[i]["prev_hash"] == records[i - 1]["record_hash"]
        ), f"Chain broken between records {i-1} and {i}"


def test_verify_chain_passes_on_valid_log(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    _append_n(logger, 5)

    ok, msg = logger.verify_chain(path)
    assert ok is True, f"Expected valid chain but got: {msg}"
    assert msg == "ok"


def test_verify_chain_detects_record_hash_tampering(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    _append_n(logger, 5)

    records = _read_records(path)
    # Corrupt the model field of record 2 (middle of chain).
    records[2]["model"] = "tampered-model"

    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")

    ok, msg = logger.verify_chain(path)
    assert ok is False
    assert "hash mismatch" in msg


def test_verify_chain_detects_prev_hash_tampering(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    _append_n(logger, 5)

    records = _read_records(path)
    # Recompute record_hash after tampering prev_hash to bypass the first check.
    records[3]["prev_hash"] = "a" * 64
    hashable = {k: v for k, v in records[3].items() if k != "record_hash"}
    records[3]["record_hash"] = _sha256_hex(_canonical_bytes(hashable))

    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")

    ok, msg = logger.verify_chain(path)
    assert ok is False
    assert "chain break" in msg


def test_verify_chain_detects_deleted_record(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    _append_n(logger, 5)

    records = _read_records(path)
    # Remove record 2 - record 3's prev_hash will no longer match record 1's hash.
    del records[2]

    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")

    ok, msg = logger.verify_chain(path)
    assert ok is False


def test_chain_resumes_after_process_restart(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")

    # First "process": write 3 records.
    logger1 = _logger_at(path)
    _append_n(logger1, 3)
    hash_after_first = logger1._prev_hash

    # Second "process": new logger instance reads the tail and continues.
    logger2 = _logger_at(path)
    assert (
        logger2._prev_hash == hash_after_first
    ), "Second logger did not resume chain from tail of existing log"
    _append_n(logger2, 3)

    # Full 6-record chain should verify clean.
    ok, msg = logger2.verify_chain(path)
    assert ok is True, f"Chain broken across restart: {msg}"


def test_seq_counter_restored_after_restart(tmp_path) -> None:
    """P1 regression: _call_count (and thus seq) must resume from the last
    persisted record's seq field after a process restart.

    Before the fix, _load_chain_tail restored _prev_hash but left _call_count
    at 0, so the second "process" would emit seq=0 again instead of continuing
    from where the first process stopped.
    """
    path = str(tmp_path / "audit.jsonl")

    # First "process": write 5 records (seq 0..4).
    logger1 = _logger_at(path)
    _append_n(logger1, 5)
    records_after_first = _read_records(path)
    assert (
        records_after_first[-1]["seq"] == 4
    ), "sanity: last seq from first process is 4"

    # Second "process": new instance reads the tail.
    logger2 = _logger_at(path)
    assert (
        logger2._call_count == 5
    ), f"_call_count not restored: expected 5, got {logger2._call_count}"

    # Writing one more record must produce seq=5, not seq=0.
    _append_n(logger2, 1)
    records = _read_records(path)
    assert len(records) == 6
    assert (
        records[5]["seq"] == 5
    ), f"seq reset after restart: expected 5, got {records[5]['seq']}"

    # The full chain must also pass integrity verification.
    ok, msg = logger2.verify_chain(path)
    assert ok is True, f"Chain broken after restart: {msg}"


def test_failure_event_is_logged(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    start, end = _make_times()

    logger.log_failure_event(
        kwargs=_make_kwargs(),
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    assert len(records) == 1
    assert records[0]["status"] == "failure"


def test_logger_does_not_raise_on_malformed_response(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    start, end = _make_times()

    bad_response = object()  # not a ModelResponse at all

    # Should not raise; logger is fail-soft.
    logger.log_success_event(
        kwargs=_make_kwargs(),
        response_obj=bad_response,
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    assert len(records) == 1
    assert records[0]["status"] == "success"


def test_content_digest_stored_not_plaintext_by_default(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    start, end = _make_times()

    logger.log_success_event(
        kwargs=_make_kwargs(content="my secret prompt"),
        response_obj=_make_response(content="my secret response"),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    raw = json.dumps(records[0])
    assert "my secret prompt" not in raw
    assert "my secret response" not in raw
    assert "messages_digest" in records[0]
    assert "response_content_digest" in records[0]


def test_seq_increments_across_calls(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    _append_n(logger, 4)

    records = _read_records(path)
    seqs = [r["seq"] for r in records]
    assert seqs == [0, 1, 2, 3]


def test_latency_ms_is_computed(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    start, end = _make_times()  # 1-second gap

    logger.log_success_event(
        kwargs=_make_kwargs(),
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    assert records[0]["latency_ms"] == 1000


def test_no_background_threads_spawned(tmp_path) -> None:
    """Local-only logger must not spawn background threads."""
    path = str(tmp_path / "audit.jsonl")
    logger = AsqavLogger(log_path=path, redact_content=True)
    start, end = _make_times()

    before = threading.active_count()
    logger.log_success_event(
        kwargs=_make_kwargs(),
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )
    after = threading.active_count()
    assert after <= before


# ---------------------------------------------------------------------------
# Concurrency, restart with large records, repr
# ---------------------------------------------------------------------------


def test_concurrent_callbacks_keep_chain_ordered(tmp_path) -> None:
    """Records from concurrent threads land on disk in seq order.

    Regression test for the out-of-order-write race: seq/prev_hash assignment
    and the file write must happen under the same lock, otherwise two threads
    can write their records in reversed order and break verify_chain.
    """
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    start, end = _make_times()
    n_threads = 8
    per_thread = 25
    barrier = threading.Barrier(n_threads)

    def _worker(tid: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            logger.log_success_event(
                kwargs=_make_kwargs(content=f"t{tid}-m{i}"),
                response_obj=_make_response(content=f"t{tid}-r{i}"),
                start_time=start,
                end_time=end,
            )

    threads = [
        threading.Thread(target=_worker, args=(tid,)) for tid in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = _read_records(path)
    assert len(records) == n_threads * per_thread
    assert [r["seq"] for r in records] == list(range(n_threads * per_thread))
    ok, msg = logger.verify_chain(path)
    assert ok is True, f"Chain broken under concurrent writes: {msg}"


def test_restart_resumes_chain_when_last_record_exceeds_4kb(tmp_path) -> None:
    """A record larger than the old 4 KB tail buffer survives a restart.

    Regression test for the silent chain reset: the tail read must widen until
    it contains the whole last line instead of truncating it.
    """
    path = str(tmp_path / "audit.jsonl")
    logger1 = _logger_at(path)
    start, end = _make_times()

    big_kwargs = _make_kwargs(content="big")
    big_kwargs["metadata"] = {"blob": "x" * 10_000}
    logger1.log_success_event(
        kwargs=big_kwargs,
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )
    assert os.path.getsize(path) > 4096
    hash_after_first = logger1._prev_hash

    logger2 = _logger_at(path)
    assert (
        logger2._prev_hash == hash_after_first
    ), "Restart did not resume the chain from a record larger than 4 KB"
    _append_n(logger2, 2)

    ok, msg = logger2.verify_chain(path)
    assert ok is True, f"Chain broken across restart with large record: {msg}"


def test_repr_shows_log_path_and_redact(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = AsqavLogger(log_path=path, redact_content=True)

    r = repr(logger)
    assert "AsqavLogger" in r
    assert path in r


def test_async_hooks_write_records(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    start, end = _make_times()

    async def _run() -> None:
        await logger.async_log_success_event(
            kwargs=_make_kwargs(content="async-ok"),
            response_obj=_make_response(),
            start_time=start,
            end_time=end,
        )
        await logger.async_log_failure_event(
            kwargs=_make_kwargs(content="async-fail"),
            response_obj=_make_response(),
            start_time=start,
            end_time=end,
        )

    asyncio.run(_run())

    records = _read_records(path)
    assert [r["status"] for r in records] == ["success", "failure"]
    ok, msg = logger.verify_chain(path)
    assert ok is True, msg


def test_redact_content_false_stores_plaintext(tmp_path) -> None:
    path = str(tmp_path / "audit.jsonl")
    logger = AsqavLogger(log_path=path, redact_content=False)
    start, end = _make_times()

    logger.log_success_event(
        kwargs=_make_kwargs(content="visible prompt"),
        response_obj=_make_response(content="visible response"),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    assert records[0]["messages"][0]["content"] == "visible prompt"
    assert records[0]["response_content"] == "visible response"


# ---------------------------------------------------------------------------
# New regression tests (must FAIL before the corresponding fix is applied)
# ---------------------------------------------------------------------------


def test_audit_log_file_created_with_0600_perms(tmp_path) -> None:
    """Veria Medium: audit log must be created with mode 0600.

    With a standard 022 umask, plain open(..., 'a') produces 0644, which lets
    other local users read the log.  The fix creates via os.open with 0o600 and
    chmods an existing file to 0600 before appending.
    """
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    start, end = _make_times()

    logger.log_success_event(
        kwargs=_make_kwargs(),
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )

    assert os.path.exists(path), "audit log was not created"
    mode_octal = oct(os.stat(path).st_mode)[-3:]
    assert mode_octal == "600", f"audit log has mode {mode_octal}, expected 600"


def test_proxy_identity_metadata_attributed_in_record(tmp_path) -> None:
    """Veria Medium: proxy identity fields from litellm_params.metadata must
    appear in the logged record's metadata.

    user_api_key_user_id, team_id, org_id, and key_alias live under
    kwargs['litellm_params']['metadata'], not at kwargs root.  Records written
    without reading that sub-dict have no proxy attribution.
    """
    path = str(tmp_path / "audit.jsonl")
    logger = _logger_at(path)
    start, end = _make_times()

    kwargs = _make_kwargs()
    kwargs["litellm_params"] = {
        "metadata": {
            "user_api_key_user_id": "user-abc",
            "user_api_key_team_id": "team-xyz",
            "user_api_key_org_id": "org-123",
            "user_api_key_alias": "my-key",
            # sensitive values that must NOT be persisted
            "user_api_key": "sk-secret-12345",
            "Authorization": "Bearer sk-secret-12345",
        }
    }

    logger.log_success_event(
        kwargs=kwargs,
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    meta = records[0]["metadata"]
    assert (
        meta.get("user_api_key_user_id") == "user-abc"
    ), "user_api_key_user_id not attributed in record"
    assert (
        meta.get("user_api_key_team_id") == "team-xyz"
    ), "user_api_key_team_id not attributed in record"
    # Sensitive fields must be filtered out
    assert "user_api_key" not in meta, "raw api key leaked into record metadata"
    assert "Authorization" not in meta, "auth header leaked into record metadata"


def test_multiworker_flock_guard_documented_or_implemented(tmp_path) -> None:
    """Veria Medium: the multi-worker limitation must be documented in the
    class docstring (or, if fcntl is used, the lock must serialize cross-process
    writes).  This test checks for the docstring acknowledgement.
    """
    import inspect

    doc = inspect.getdoc(AsqavLogger) or ""
    assert (
        "single" in doc.lower() or "flock" in doc.lower() or "worker" in doc.lower()
    ), "AsqavLogger docstring must document the single-writer / multi-worker limitation"


# ---------------------------------------------------------------------------
# Optional cloud signing (opt-in; default off)
#
# These tests stub litellm's _get_httpx_client so no real network is touched.
# The local hash chain stays the source of truth; the cloud receipt is bound
# into the record when the opt-in env vars are set.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

_HTTP_HANDLER_MOD = "litellm.llms.custom_httpx.http_handler"


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeHttpxClient:
    """Records the last POST and returns a canned response or raises."""

    def __init__(self, response=None, raise_exc=None) -> None:
        self._response = response
        self._raise_exc = raise_exc
        self.calls: list = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


@pytest.fixture
def _patch_httpx(monkeypatch):
    """Install a fake _get_httpx_client on the stubbed http_handler module."""
    mod = sys.modules[_HTTP_HANDLER_MOD]

    def _install(client: _FakeHttpxClient) -> _FakeHttpxClient:
        monkeypatch.setattr(
            mod, "_get_httpx_client", lambda *a, **k: client, raising=False
        )
        return client

    return _install


@pytest.fixture
def _cloud_env(monkeypatch):
    monkeypatch.setenv("ASQAV_API_KEY", "sk_test_fake")
    monkeypatch.setenv("ASQAV_AGENT_ID", "agent-123")
    monkeypatch.delenv("ASQAV_API_BASE", raising=False)


def _cloud_logger(path: str) -> AsqavLogger:
    # log_path passed positionally so redact stays the env-driven default (True).
    return AsqavLogger(log_path=path)


def test_cloud_sign_payload_carries_digests_not_content() -> None:
    record = {
        "messages_digest": "a" * 64,
        "response_content_digest": "b" * 64,
        "call_id": "call-1",
        "model": "gpt-4o",
        "status": "success",
        "seq": 0,
    }
    body = _cloud_sign_payload(record, "c" * 64)
    raw = json.dumps(body)
    assert body["action_type"] == "litellm:completion"
    # The SIGNED hash is the pre-receipt content hash, not the prompt digest.
    # The cloud signs body["hash"] verbatim, so this is what the receipt binds.
    assert body["hash"] == "sha256:" + "c" * 64
    assert body["hash_algo"] == "sha256"
    assert body["metadata"]["response_content_digest"] == "b" * 64
    # messages_digest stays in metadata for reference only.
    assert body["metadata"]["messages_digest"] == "a" * 64
    assert body["metadata"]["record_hash"] == "c" * 64
    # No raw prompt/response text ever leaves; only digests.
    assert "hello" not in raw
    assert "world" not in raw


def test_cloud_sign_records_signature_when_key_set(
    tmp_path, _cloud_env, _patch_httpx
) -> None:
    path = str(tmp_path / "audit.jsonl")
    resp = _FakeResponse(
        201,
        {
            "signature_id": "sig-abc",
            "verification_url": "https://api.asqav.com/v/sig-abc",
            "action_id": "act-xyz",
            "mode": "hash",
        },
    )
    client = _patch_httpx(_FakeHttpxClient(response=resp))

    logger = _cloud_logger(path)
    start, end = _make_times()
    logger.log_success_event(
        kwargs=_make_kwargs(content="secret prompt"),
        response_obj=_make_response(content="secret response"),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    assert len(records) == 1
    cloud = records[0]["asqav_cloud"]
    assert cloud["signature_id"] == "sig-abc"
    assert cloud["verification_url"] == "https://api.asqav.com/v/sig-abc"

    # Exactly one POST, to the agent sign endpoint, with the X-API-Key header.
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["url"] == "https://api.asqav.com/api/v1/agents/agent-123/sign"
    assert call["headers"]["X-API-Key"] == "sk_test_fake"

    # Digests-only on the wire; no raw content.
    sent = json.dumps(call["json"])
    assert "secret prompt" not in sent
    assert "secret response" not in sent
    assert call["json"]["hash"].startswith("sha256:")

    # The bound receipt does not break the local hash chain.
    ok, msg = logger.verify_chain(path)
    assert ok is True, msg


def _posted_signed_hash(call: dict) -> str:
    """The hex the cloud actually signs: body["hash"] minus the sha256: prefix.

    The cloud signs body["hash"] verbatim in hash mode and drops metadata keys
    outside its whitelist (record_hash is not whitelisted), so body["hash"] is
    the only field that proves what the receipt binds.
    """
    signed = call["json"]["hash"]
    assert signed.startswith("sha256:"), signed
    return signed[len("sha256:") :]


def test_cloud_sign_signs_real_pre_receipt_hash(
    tmp_path, _cloud_env, _patch_httpx
) -> None:
    """The SIGNED hash (body["hash"]) is a real 64-hex content hash equal to
    sha256(canonical(record content)).

    The cloud signs body["hash"], not metadata, so the binding lives there. This
    asserts the signed field equals the re-derived pre-receipt content hash.
    """
    path = str(tmp_path / "audit.jsonl")
    resp = _FakeResponse(201, {"signature_id": "sig-1", "mode": "hash"})
    client = _patch_httpx(_FakeHttpxClient(response=resp))

    logger = _cloud_logger(path)
    start, end = _make_times()
    logger.log_success_event(
        kwargs=_make_kwargs(content="bind me"),
        response_obj=_make_response(content="bound"),
        start_time=start,
        end_time=end,
    )

    assert len(client.calls) == 1
    signed = _posted_signed_hash(client.calls[0])
    assert isinstance(signed, str) and len(signed) == 64
    int(signed, 16)  # must be valid hex

    # Re-derive the pre-receipt hash from the stored record: canonicalize the
    # content minus record_hash and asqav_cloud. It must equal the signed hash.
    record = _read_records(path)[0]
    content = {
        k: v for k, v in record.items() if k not in ("record_hash", "asqav_cloud")
    }
    assert signed == _sha256_hex(_canonical_bytes(content))


@pytest.mark.parametrize(
    "tampered_field, tampered_value",
    [
        ("model", "attacker-model"),
        ("response_content_digest", "f" * 64),
        ("status", "failure"),
        ("seq", 999),
        ("prev_hash", "9" * 64),
    ],
)
def test_cloud_signature_binds_full_record_not_just_prompt(
    tmp_path, _cloud_env, _patch_httpx, tampered_field, tampered_value
) -> None:
    """The SIGNED hash binds the whole canonical record, so tampering any
    content field (including non-prompt fields like model, status, response
    digest, seq, prev_hash) changes the hash the cloud signed.

    Against the prior code body["hash"] was sha256:{messages_digest} (the prompt
    only), so tampering model/status/response/seq/prev_hash did NOT change the
    signed hash and the receipt validated a forged record. This is the end-to-end
    binding regression: the signed field must move with every content field.
    """
    path = str(tmp_path / "audit.jsonl")
    resp = _FakeResponse(201, {"signature_id": "sig-1", "mode": "hash"})
    client = _patch_httpx(_FakeHttpxClient(response=resp))

    logger = _cloud_logger(path)
    start, end = _make_times()
    logger.log_success_event(
        kwargs=_make_kwargs(model="gpt-4o", content="same"),
        response_obj=_make_response(content="resp"),
        start_time=start,
        end_time=end,
    )

    signed_hash = _posted_signed_hash(client.calls[0])
    record = _read_records(path)[0]
    content = {
        k: v for k, v in record.items() if k not in ("record_hash", "asqav_cloud")
    }
    # The signed hash matches the untouched content the cloud committed to.
    assert signed_hash == _sha256_hex(_canonical_bytes(content))
    # Tampering the field flips the signed hash, so the receipt cannot validate
    # a record with this field altered. Fails against the prompt-only binding.
    assert tampered_field in content
    tampered = {**content, tampered_field: tampered_value}
    assert _sha256_hex(_canonical_bytes(tampered)) != signed_hash


def test_cloud_sign_fail_soft_on_raise(tmp_path, _cloud_env, _patch_httpx) -> None:
    path = str(tmp_path / "audit.jsonl")
    _patch_httpx(_FakeHttpxClient(raise_exc=RuntimeError("connection refused")))

    logger = _cloud_logger(path)
    start, end = _make_times()
    # Must not raise; the local line is still written.
    logger.log_success_event(
        kwargs=_make_kwargs(),
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    assert len(records) == 1
    assert records[0]["status"] == "success"
    assert "asqav_cloud" not in records[0]
    ok, msg = logger.verify_chain(path)
    assert ok is True, msg


def test_cloud_sign_fail_soft_on_non_2xx(tmp_path, _cloud_env, _patch_httpx) -> None:
    path = str(tmp_path / "audit.jsonl")
    resp = _FakeResponse(401, {"detail": "invalid api key"})
    _patch_httpx(_FakeHttpxClient(response=resp))

    logger = _cloud_logger(path)
    start, end = _make_times()
    logger.log_success_event(
        kwargs=_make_kwargs(),
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    assert len(records) == 1
    assert "asqav_cloud" not in records[0]
    ok, msg = logger.verify_chain(path)
    assert ok is True, msg


def test_default_off_is_byte_identical_without_key(
    tmp_path, monkeypatch, _patch_httpx
) -> None:
    """With no key, the record carries no cloud field and never calls httpx."""
    monkeypatch.delenv("ASQAV_API_KEY", raising=False)
    monkeypatch.delenv("ASQAV_AGENT_ID", raising=False)
    client = _patch_httpx(_FakeHttpxClient(raise_exc=AssertionError("must not call")))

    path = str(tmp_path / "audit.jsonl")
    logger = _cloud_logger(path)
    start, end = _make_times()
    logger.log_success_event(
        kwargs=_make_kwargs(),
        response_obj=_make_response(),
        start_time=start,
        end_time=end,
    )

    records = _read_records(path)
    assert len(records) == 1
    assert "asqav_cloud" not in records[0]
    # Same field set a pure-local logger would write.
    assert set(records[0]) == {
        "seq",
        "ts",
        "prev_hash",
        "call_id",
        "model",
        "status",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "finish_reason",
        "provider_request_id",
        "messages_digest",
        "response_content_digest",
        "metadata",
        "record_hash",
    }
    assert len(client.calls) == 0
