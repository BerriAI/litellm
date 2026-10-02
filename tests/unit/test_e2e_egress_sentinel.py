"""Tests for .github/scripts/e2e_egress_sentinel.py.

The replay lane's zero-egress proof is only as good as this sentinel: it pins the
provider hosts to a local sink and counts every connection that reaches them, so
a single escaped provider call turns the run red. The contract locked in here is
that the counter counts (each accepted connection is exactly one recorded hit),
that ``assert-empty`` is the pass/fail gate around that count, and that the hosts
file it edits is always handed back exactly as it was found.
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "e2e_egress_sentinel.py"
_spec: Final = importlib.util.spec_from_file_location("e2e_egress_sentinel", _MODULE_PATH)
sentinel: Final = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sentinel  # @dataclass(slots=True) rebuilds via sys.modules
_spec.loader.exec_module(sentinel)


def _hit_lines(hits_file: Path) -> list[str]:
    if not hits_file.exists():
        return []
    return [line for line in hits_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_pin_block_lists_every_host_against_the_sink():
    block = sentinel._pin_block("127.0.0.1", ("api.openai.com", "api.anthropic.com"))
    assert "127.0.0.1\tapi.openai.com" in block
    assert "127.0.0.1\tapi.anthropic.com" in block
    assert sentinel._BLOCK_BEGIN in block and sentinel._BLOCK_END in block


def test_install_and_restore_round_trips_an_existing_hosts_file(tmp_path):
    hosts = tmp_path / "hosts"
    original = "127.0.0.1\tlocalhost\n255.255.255.255\tbroadcasthost\n"
    hosts.write_text(original, encoding="utf-8")

    saved = sentinel._install_pins(hosts, "127.0.0.1", ("api.openai.com",))
    assert saved == original.encode()
    assert "api.openai.com" in hosts.read_text(encoding="utf-8")

    sentinel._restore_pins(hosts, saved)
    assert hosts.read_text(encoding="utf-8") == original


def test_install_on_a_missing_hosts_file_creates_then_restore_empties(tmp_path):
    hosts = tmp_path / "hosts"
    saved = sentinel._install_pins(hosts, "127.0.0.1", ("api.anthropic.com",))
    assert saved == b""
    assert "api.anthropic.com" in hosts.read_text(encoding="utf-8")

    sentinel._restore_pins(hosts, saved)
    assert hosts.read_text(encoding="utf-8") == ""


def test_assert_empty_passes_when_no_calls(tmp_path):
    absent = tmp_path / "absent.jsonl"
    assert sentinel.assert_empty(absent) == 0

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n  \n", encoding="utf-8")
    assert sentinel.assert_empty(empty) == 0


def test_assert_empty_fails_when_calls_recorded(tmp_path):
    hits = tmp_path / "hits.jsonl"
    hits.write_text('{"port": 443, "peer": ["127.0.0.1", 5]}\n', encoding="utf-8")
    assert sentinel.assert_empty(hits) == 1


def test_accept_loop_records_exactly_one_hit_per_connection(tmp_path):
    hits_file = tmp_path / "hits.jsonl"
    hits_file.write_text("", encoding="utf-8")
    listener = sentinel._bind("127.0.0.1", 0)
    port = listener.getsockname()[1]
    stop = threading.Event()
    hits = sentinel._HitLog(path=hits_file, _lock=threading.Lock())
    worker = threading.Thread(target=sentinel._serve_socket, args=(listener, port, hits, stop), daemon=True)
    worker.start()

    try:
        for _ in range(3):
            conn = socket.create_connection(("127.0.0.1", port), timeout=2)
            conn.close()
        deadline = time.time() + 3
        while time.time() < deadline and len(_hit_lines(hits_file)) < 3:
            time.sleep(0.02)
    finally:
        stop.set()
        listener.close()
        worker.join(timeout=3)

    assert len(_hit_lines(hits_file)) == 3


def test_serve_end_to_end_pins_counts_and_restores(tmp_path):
    hosts = tmp_path / "hosts"
    hosts.write_text("127.0.0.1\tlocalhost\n", encoding="utf-8")
    hits = tmp_path / "hits.jsonl"
    ready = tmp_path / "ready"
    pidf = tmp_path / "pid"
    port = _free_port()

    proc = subprocess.Popen(
        [
            sys.executable,
            str(_MODULE_PATH),
            "serve",
            "--host",
            "api.openai.com",
            "--host",
            "api.anthropic.com",
            "--sink-address",
            "127.0.0.1",
            "--port",
            str(port),
            "--hits-file",
            str(hits),
            "--ready-file",
            str(ready),
            "--pid-file",
            str(pidf),
            "--hosts-file",
            str(hosts),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for(lambda: ready.exists(), timeout=10)
        pinned = hosts.read_text(encoding="utf-8")
        assert "api.openai.com" in pinned and "api.anthropic.com" in pinned

        for _ in range(2):
            conn = socket.create_connection(("127.0.0.1", port), timeout=2)
            conn.close()
        _wait_for(lambda: len(_hit_lines(hits)) >= 2, timeout=5)
        assert sentinel.assert_empty(hits) == 1
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    assert hosts.read_text(encoding="utf-8") == "127.0.0.1\tlocalhost\n"
    assert not ready.exists()


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _wait_for(predicate, *, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition never became true within the timeout")
