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
    apply_vcr_auto_marker_to_items,
    emit_vcr_classification_summary,
    install_live_call_probe,
    record_vcr_outcome,
    session_stats_snapshot,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubItem:
    """Pytest item double sufficient for the auto-marker logic."""

    def __init__(
        self,
        nodeid: str,
        path: str,
        *,
        markers: Optional[list[str]] = None,
        fixturenames: Optional[list[str]] = None,
        module=None,
    ) -> None:
        self.nodeid = nodeid
        self.path = path
        self._markers = list(markers or [])
        self.fixturenames = list(fixturenames or [])
        self.module = module
        self.user_properties: list = []

    def get_closest_marker(self, name: str):
        return name if name in self._markers else None

    def add_marker(self, marker):
        # ``pytest.mark.vcr`` is a MarkDecorator; rely on its ``name``.
        name = getattr(marker, "name", str(marker))
        self._markers.append(name)


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


def test_should_extract_only_aws_access_key_from_sigv4_authorization():
    """Two Bedrock requests with the same access key but different
    timestamps and signatures must produce the same fingerprint, otherwise
    every CI run pushes a new episode into the cassette."""
    auth_today = (
        "AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE12345/20260512/us-east-1/"
        "bedrock/aws4_request, SignedHeaders=host;x-amz-date, "
        "Signature=AAAAAAAA"
    )
    auth_tomorrow = (
        "AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE12345/20260513/us-east-1/"
        "bedrock/aws4_request, SignedHeaders=host;x-amz-date, "
        "Signature=BBBBBBBB"
    )
    today = _stable_key_value("Authorization", auth_today)
    tomorrow = _stable_key_value("Authorization", auth_tomorrow)
    assert today == tomorrow == "aws-sigv4:AKIAEXAMPLE12345"


def test_should_keep_bearer_authorization_unchanged():
    """OpenAI ``Bearer <key>`` headers are stable as-is — keep them."""
    out = _stable_key_value("Authorization", "Bearer sk-1234")
    assert out == "Bearer sk-1234"


def test_should_produce_stable_fingerprint_across_sigv4_signatures():
    """``_compute_key_fingerprint`` should not change when only the SigV4
    signature/timestamp rotates."""
    req_a = SimpleNamespace(
        headers={
            "authorization": (
                "AWS4-HMAC-SHA256 Credential=AKIA1/20260101/us-east-1/"
                "bedrock/aws4_request, SignedHeaders=host, Signature=AAA"
            )
        }
    )
    req_b = SimpleNamespace(
        headers={
            "authorization": (
                "AWS4-HMAC-SHA256 Credential=AKIA1/20260512/us-east-1/"
                "bedrock/aws4_request, SignedHeaders=host;x-amz-date, "
                "Signature=ZZZ"
            )
        }
    )
    assert _compute_key_fingerprint(req_a) == _compute_key_fingerprint(req_b)


def test_should_distinguish_different_aws_access_keys():
    """Two different access keys must produce different fingerprints so
    cassettes recorded under one identity never serve another."""
    req_a = SimpleNamespace(
        headers={
            "authorization": "AWS4-HMAC-SHA256 Credential=AKIAONE/x/y/z/aws4_request, Signature=A"
        }
    )
    req_b = SimpleNamespace(
        headers={
            "authorization": "AWS4-HMAC-SHA256 Credential=AKIATWO/x/y/z/aws4_request, Signature=A"
        }
    )
    assert _compute_key_fingerprint(req_a) != _compute_key_fingerprint(req_b)


# ---------------------------------------------------------------------------
# Live-call host classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host,expected",
    [
        ("api.openai.com", True),
        ("api.anthropic.com", True),
        ("bedrock-runtime.us-east-1.amazonaws.com", True),
        ("bedrock-runtime-fips.us-east-1.amazonaws.com", True),
        ("api.us-east-1.bedrock-runtime.amazonaws.com", False),
        ("foo.bar.openai.com", True),
        ("127.0.0.1", False),
        ("localhost", False),
        ("10.0.0.1", False),
        ("172.16.0.1", False),
        ("redis.example.com", False),
        ("", False),
    ],
)
def test_should_classify_live_call_hosts(host, expected):
    assert _is_live_call_host(host) is expected


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------


def _cassette(played: int, dirty: bool, total: int):
    class _Sized:
        def __init__(self, n):
            self.n = n
            self.play_count = played
            self.dirty = dirty

        def __len__(self):
            return self.n

    return _Sized(total)


def test_should_classify_pure_replay_as_hit():
    assert (
        _classify_marked_test(_cassette(played=3, dirty=False, total=3)) == VERDICT_HIT
    )


def test_should_classify_no_traffic_as_noop():
    assert (
        _classify_marked_test(_cassette(played=0, dirty=False, total=0))
        == VERDICT_NOOP_NO_TRAFFIC
    )


def test_should_classify_pure_record_as_miss_recorded():
    assert (
        _classify_marked_test(_cassette(played=0, dirty=True, total=1))
        == VERDICT_MISS_RECORDED
    )


def test_should_classify_mixed_replay_and_record_as_partial():
    assert (
        _classify_marked_test(_cassette(played=2, dirty=True, total=4))
        == VERDICT_PARTIAL
    )


def test_should_classify_overflow_only_when_dirty_episodes_were_recorded():
    """Cassettes that exceed ``MAX_EPISODES_PER_CASSETTE`` (50) are
    refused for save — but only when ``dirty=True`` (new episodes were
    actually recorded that the persister would refuse). Replaying an
    already-large cassette with no new traffic is healthy: the persister
    never tries to save, so the cache state is stable and the next run
    will replay too."""
    assert (
        _classify_marked_test(_cassette(played=0, dirty=True, total=51))
        == VERDICT_MISS_OVERFLOW
    )
    assert (
        _classify_marked_test(_cassette(played=10, dirty=True, total=52))
        == VERDICT_MISS_OVERFLOW
    )


def test_should_classify_large_cassette_with_no_new_episodes_as_hit():
    """``total > 50`` + ``dirty=False`` means everything was replayed
    from cache; no save attempt happens, so this is a healthy HIT, not
    OVERFLOW."""
    assert (
        _classify_marked_test(_cassette(played=51, dirty=False, total=51))
        == VERDICT_HIT
    )
    assert (
        _classify_marked_test(_cassette(played=60, dirty=False, total=60))
        == VERDICT_HIT
    )


# ---------------------------------------------------------------------------
# apply_vcr_auto_marker_to_items: skip-reason tagging
# ---------------------------------------------------------------------------


def _make_module_with_source(tmp_path, src: str, name: str):
    p = tmp_path / f"{name}.py"
    p.write_text(src)
    mod = SimpleNamespace(__file__=str(p))
    return mod, str(p)


def test_should_apply_vcr_marker_to_clean_test(vcr_enabled, tmp_path):
    mod, p = _make_module_with_source(tmp_path, "def test_x(): pass\n", "clean")
    item = _StubItem("clean.py::test_x", p, module=mod)
    apply_vcr_auto_marker_to_items([item])
    assert item.get_closest_marker("vcr") == "vcr"


def test_should_skip_per_item_when_respx_marker_present(vcr_enabled, tmp_path):
    mod, p = _make_module_with_source(tmp_path, "def test_x(): pass\n", "respx_marker")
    item = _StubItem("respx_marker.py::test_x", p, markers=["respx"], module=mod)
    apply_vcr_auto_marker_to_items([item])
    assert item.get_closest_marker("vcr") is None
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_RESPX


def test_should_skip_per_item_when_respx_mock_fixture_present(vcr_enabled, tmp_path):
    mod, p = _make_module_with_source(tmp_path, "def test_x(): pass\n", "respx_fixture")
    item = _StubItem(
        "respx_fixture.py::test_x", p, fixturenames=["respx_mock"], module=mod
    )
    apply_vcr_auto_marker_to_items([item])
    assert item.get_closest_marker("vcr") is None
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_RESPX


def test_should_tag_pre_marked_items_so_summary_can_show_them(vcr_enabled, tmp_path):
    mod, p = _make_module_with_source(tmp_path, "def test_x(): pass\n", "premarked")
    item = _StubItem("premarked.py::test_x", p, markers=["vcr"], module=mod)
    apply_vcr_auto_marker_to_items([item])
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_PRE_MARKED


def test_should_tag_skip_files_with_respx_module_when_module_actually_uses_respx(
    vcr_enabled, tmp_path
):
    """A file in ``skip_files`` whose module *does* call respx should be
    labeled as a real conflict (respx_conflict_module), not a dead opt-out."""
    mod, p = _make_module_with_source(
        tmp_path,
        "import respx\n@pytest.mark.respx\ndef test_x(): pass\n",
        "real_respx",
    )
    item = _StubItem("real_respx.py::test_x", p, module=mod)
    apply_vcr_auto_marker_to_items([item], skip_files={"real_respx.py"})
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_RESPX_MODULE


def test_should_tag_skip_files_with_file_opt_out_when_module_does_not_use_respx(
    vcr_enabled, tmp_path
):
    """A file in ``skip_files`` whose module never wires up respx is a
    dead skip-list entry — surface it so we can prune."""
    mod, p = _make_module_with_source(
        tmp_path,
        "from respx import MockRouter  # dead import\ndef test_x(): pass\n",
        "dead_skip",
    )
    item = _StubItem("dead_skip.py::test_x", p, module=mod)
    apply_vcr_auto_marker_to_items([item], skip_files={"dead_skip.py"})
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_FILE_OPT_OUT


def test_should_not_flag_respx_mentioned_in_comment_or_docstring(vcr_enabled, tmp_path):
    """Substring scans of source text false-positive on
    ``# Previously used respx.mock`` and similar — defeats the dead
    skip-list pruning goal. AST-based detection ignores comments and
    string literals."""
    src = (
        '"""Module docstring mentions respx.mock and @pytest.mark.respx and respx_mock."""\n'
        "# Previously tried respx.mock but switched to vcrpy\n"
        "# Old code did `with respx.mock(): ...`\n"
        "x = '@respx.mock'  # string literal, not a real decorator\n"
        "def test_x():\n"
        "    pass\n"
    )
    mod, p = _make_module_with_source(tmp_path, src, "comment_respx")
    item = _StubItem("comment_respx.py::test_x", p, module=mod)
    apply_vcr_auto_marker_to_items([item], skip_files={"comment_respx.py"})
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_FILE_OPT_OUT


def test_should_flag_real_respx_mark_decorator_via_ast(vcr_enabled, tmp_path):
    src = "import pytest\n" "@pytest.mark.respx\n" "def test_x(respx_mock): pass\n"
    mod, p = _make_module_with_source(tmp_path, src, "real_respx_mark")
    item = _StubItem("real_respx_mark.py::test_x", p, module=mod)
    apply_vcr_auto_marker_to_items([item], skip_files={"real_respx_mark.py"})
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_RESPX_MODULE


def test_should_flag_real_respx_with_block_via_ast(vcr_enabled, tmp_path):
    src = "import respx\n" "def test_x():\n" "    with respx.mock():\n" "        pass\n"
    mod, p = _make_module_with_source(tmp_path, src, "real_respx_with")
    item = _StubItem("real_respx_with.py::test_x", p, module=mod)
    apply_vcr_auto_marker_to_items([item], skip_files={"real_respx_with.py"})
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_RESPX_MODULE


def test_should_flag_respx_mock_call_at_module_scope_via_ast(vcr_enabled, tmp_path):
    src = "import respx\nmock = respx.mock()\ndef test_x(): pass\n"
    mod, p = _make_module_with_source(tmp_path, src, "real_respx_call")
    item = _StubItem("real_respx_call.py::test_x", p, module=mod)
    apply_vcr_auto_marker_to_items([item], skip_files={"real_respx_call.py"})
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_RESPX_MODULE


def test_should_tag_nodeid_suffix_skips_as_incompatible(vcr_enabled, tmp_path):
    mod, p = _make_module_with_source(tmp_path, "def test_x(): pass\n", "incompat")
    item = _StubItem("incompat.py::test_prompt_caching", p, module=mod)
    apply_vcr_auto_marker_to_items(
        [item], skip_nodeid_suffixes=("::test_prompt_caching",)
    )
    assert getattr(item, VCR_SKIP_REASON_USER_ATTR) == SKIP_REASON_INCOMPATIBLE


# ---------------------------------------------------------------------------
# Session-end summary
# ---------------------------------------------------------------------------


class _FakeReporter:
    def __init__(self):
        self.lines: list[str] = []

    def write_sep(self, sep, title="", **kwargs):
        self.lines.append(f"=== {title}" if title else "===")

    def write_line(self, line):
        self.lines.append(line)

    @property
    def output(self):
        return "\n".join(self.lines)


def test_should_render_overflow_section_when_any_test_overflowed(vcr_enabled):
    """The OVERFLOW section is the cost-leak signal: if it's empty, no
    cassettes are silently being refused; if it's not empty, those tests
    re-bill on every run."""
    request = SimpleNamespace(
        node=SimpleNamespace(
            nodeid="t::overflow",
            user_properties=[],
            rep_call=SimpleNamespace(passed=True),
        )
    )
    cassette = _cassette(played=0, dirty=True, total=51)
    cassette._path = None  # avoid mark_test_outcome side-effects
    record_vcr_outcome(request, cassette)

    reporter = _FakeReporter()
    emit_vcr_classification_summary(reporter)
    assert "VCR CACHE CLASSIFICATION SUMMARY" in reporter.output
    assert "VCR MISS:OVERFLOW" in reporter.output
    assert "CASSETTE OVERFLOW" in reporter.output
    assert "t::overflow" in reporter.output


def test_should_render_unmarked_live_call_section_with_hosts(vcr_enabled):
    request_node = SimpleNamespace(
        nodeid="t::leak",
        user_properties=[],
        rep_call=SimpleNamespace(passed=True),
    )
    setattr(request_node, VCR_SKIP_REASON_USER_ATTR, SKIP_REASON_RESPX)
    setattr(request_node, "vcr_live_call_hosts", ["api.openai.com"])
    request = SimpleNamespace(node=request_node)

    record_vcr_outcome(request, None)

    snap = session_stats_snapshot()
    assert snap["unmarked_live_call_tests"] == [("t::leak", ["api.openai.com"])]
    assert snap["verdict_counts"][VERDICT_UNMARKED_LIVE_CALL] == 1

    reporter = _FakeReporter()
    emit_vcr_classification_summary(reporter)
    assert "UNMARKED TESTS WITH LIVE API CALLS" in reporter.output
    assert "api.openai.com" in reporter.output
    assert "t::leak" in reporter.output


def test_should_record_unmarked_no_traffic_when_test_skipped_vcr_but_did_not_call_out(
    vcr_enabled,
):
    request_node = SimpleNamespace(
        nodeid="t::clean_skip",
        user_properties=[],
        rep_call=SimpleNamespace(passed=True),
    )
    setattr(request_node, VCR_SKIP_REASON_USER_ATTR, SKIP_REASON_INCOMPATIBLE)
    request = SimpleNamespace(node=request_node)

    record_vcr_outcome(request, None)

    snap = session_stats_snapshot()
    assert snap["verdict_counts"][VERDICT_UNMARKED_NO_TRAFFIC] == 1
    assert snap["skip_reason_counts"][SKIP_REASON_INCOMPATIBLE] == 1


def test_should_demote_miss_recorded_to_not_persisted_when_test_failed(vcr_enabled):
    """If a test failed, ``save_cassette`` skips persisting — that means
    the next CI run will hit live again. The verdict must reflect that."""
    request = SimpleNamespace(
        node=SimpleNamespace(
            nodeid="t::failed",
            user_properties=[],
            rep_call=SimpleNamespace(passed=False),
        )
    )
    cassette = _cassette(played=0, dirty=True, total=1)
    cassette._path = None
    record_vcr_outcome(request, cassette)

    snap = session_stats_snapshot()
    assert snap["verdict_counts"].get(VERDICT_MISS_NOT_PERSISTED) == 1


def test_should_emit_no_summary_when_no_tests_observed(vcr_enabled):
    reporter = _FakeReporter()
    emit_vcr_classification_summary(reporter)
    assert reporter.output == ""


# ---------------------------------------------------------------------------
# Live-call probe
# ---------------------------------------------------------------------------


def test_should_skip_live_probe_when_vcr_active(vcr_enabled):
    """When the test *is* VCR-marked (cassette truthy), we don't install
    the probe — vcrpy intercepts above the socket layer, so any
    'connection' would be vcrpy's own bookkeeping and not real spend."""
    request = SimpleNamespace(node=SimpleNamespace(), addfinalizer=lambda fn: None)
    fake_cassette = SimpleNamespace(play_count=0, dirty=False)
    probe = install_live_call_probe(request, fake_cassette)
    assert probe is None


def test_live_call_probe_records_known_llm_hosts(vcr_enabled, monkeypatch):
    """The probe should record outbound TCP connections to known LLM
    provider hosts (and ignore localhost / RFC1918 / unknown hosts)."""
    finalizers = []

    class _Node:
        pass

    request = SimpleNamespace(
        node=_Node(), addfinalizer=lambda fn: finalizers.append(fn)
    )
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
