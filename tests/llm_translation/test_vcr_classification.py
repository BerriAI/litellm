"""Unit tests for the VCR classification + observability layer.

Covers:
- per-item respx detection (module scan, marker, fixture)
- skip-reason tagging in ``apply_vcr_auto_marker_to_items``
- verdict classification (HIT / MISS:RECORDED / MISS:OVERFLOW / MISS:NOT_PERSISTED /
  PARTIAL / NOOP / UNMARKED:LIVE_CALL / UNMARKED:NO_TRAFFIC)
- AWS SigV4 fingerprint stability
- session-end summary rendering
- live-call host classification
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Optional

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_conftest_common import (  # noqa: E402
    SKIP_REASON_FILE_OPT_OUT,
    SKIP_REASON_INCOMPATIBLE,
    SKIP_REASON_PRE_MARKED,
    SKIP_REASON_RESPX,
    SKIP_REASON_RESPX_MODULE,
    VCR_SKIP_REASON_USER_ATTR,
    VERDICT_HIT,
    VERDICT_MISS_NOT_PERSISTED,
    VERDICT_MISS_OVERFLOW,
    VERDICT_MISS_RECORDED,
    VERDICT_NOOP_NO_TRAFFIC,
    VERDICT_PARTIAL,
    VERDICT_UNMARKED_LIVE_CALL,
    VERDICT_UNMARKED_NO_TRAFFIC,
    _RESPX_MODULE_CACHE,
    _classify_marked_test,
    _compute_key_fingerprint,
    _is_live_call_host,
    _reset_session_stats,
    _stable_key_value,
    aggregate_report_outcome,
    apply_vcr_auto_marker_to_items,
    emit_vcr_classification_summary,
    install_live_call_probe,
    record_vcr_outcome,
    session_stats_snapshot,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def vcr_enabled(monkeypatch):
    monkeypatch.setenv("CASSETTE_REDIS_URL", "redis://stub")
    monkeypatch.delenv("LITELLM_VCR_DISABLE", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)


@pytest.fixture(autouse=True)
def _reset_module_caches():
    _reset_session_stats()
    _RESPX_MODULE_CACHE.clear()
    yield
    _reset_session_stats()
    _RESPX_MODULE_CACHE.clear()


# ---------------------------------------------------------------------------
# AWS SigV4 fingerprint stability — the Bedrock cassette overflow root cause
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Live-call host classification
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# apply_vcr_auto_marker_to_items: skip-reason tagging
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Session-end summary
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# xdist controller aggregation
#
# _session_stats lives in module-global memory. Under xdist that memory is
# per-worker, so the controller's pytest_terminal_summary would render an
# empty summary without these aggregation hooks. The tests below simulate
# the controller receiving teardown reports produced by workers.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Live-call probe
# ---------------------------------------------------------------------------


def test_live_call_probe_records_known_llm_hosts(vcr_enabled, monkeypatch):
    """The probe should record outbound TCP connections to known LLM
    provider hosts (and ignore localhost / RFC1918 / unknown hosts)."""
    finalizers = []

    class _Node:
        pass

    request = SimpleNamespace(node=_Node(), addfinalizer=lambda fn: finalizers.append(fn))
    probe = install_live_call_probe(request, None)
    assert probe is not None

    import socket

    # Manually invoke the patched function — we don't actually open a
    # connection because that would hit the network. The probe records
    # at the *call site* before delegating, and the original
    # ``socket.create_connection`` will then fail; we swallow that.
    try:
        socket.create_connection(("api.openai.com", 443), timeout=0.001)
    except Exception:
        pass
    try:
        socket.create_connection(("127.0.0.1", 6379), timeout=0.001)
    except Exception:
        pass

    # Restore via finalizers before asserting so the rest of the test
    # session is unaffected.
    for fn in finalizers:
        fn()

    hosts = getattr(request.node, "vcr_live_call_hosts", [])
    assert "api.openai.com" in hosts
    assert "127.0.0.1" not in hosts
