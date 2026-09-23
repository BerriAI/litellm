#!/usr/bin/env python3
"""Merge smoke harness: bounded checks run inside a loopback-only Linux network namespace."""

# ruff: noqa: T201  # CLI harness: stdout/stderr lines are the reported result

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, NoReturn, TextIO, cast

import pytest

EXPECTED_CASES: Final = (
    "CHAT-JSON",
    "CHAT-TEXT-STREAM",
    "CHAT-TOOL-STREAM",
    "MODEL-ALLOW",
    "MODEL-DENY",
    "COST-EXPLICIT",
    "COST-ZERO",
    "LOG-CONTENT-ON",
    "LOG-CONTENT-OFF",
    "CALLBACK-SUCCESS",
    "CALLBACK-FAILURE",
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    ok: bool
    detail: str = ""


@dataclass(slots=True)
class _Args:
    command: str = ""
    no_child: bool = False
    expect: str = ""
    litellm_bin: str | None = None
    lite_bin: str | None = None
    diagnostics_dir: str = ""
    ready_deadline: float = 120.0
    shutdown_deadline: float = 20.0
    poll_interval: float = 0.5
    manifest: str = ""
    rootdir: str | None = None


def fail(reason: str) -> NoReturn:
    print(f"merge-smoke: FAIL {reason}", file=sys.stderr)
    sys.exit(1)


def ok(step: str) -> None:
    print(f"merge-smoke: OK {step}")


def tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError as exc:
        return f"<cannot read {path}: {exc}>"


def cmd_verify_isolation(args: _Args) -> int:
    if os.geteuid() == 0:
        fail("verify-isolation must run unprivileged (geteuid()==0)")
    try:
        socket.create_connection(("192.0.2.1", 9), timeout=3)
    except OSError as exc:
        print(f"external connect blocked as expected: errno={exc.errno} {exc}")
    else:
        fail("external TCP connect to 192.0.2.1:9 succeeded; namespace is not isolated")
    listener: Final = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port: Final = cast(int, listener.getsockname()[1])
    client: Final = socket.create_connection(("127.0.0.1", port), timeout=5)
    accepted: Final = listener.accept()
    accepted[0].close()
    client.close()
    listener.close()
    print(f"loopback connect ok on 127.0.0.1:{port}")
    if not args.no_child:
        proc: Final = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "verify-isolation", "--no-child"],
            timeout=30,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            fail(f"child process did not inherit isolation: {proc.stderr.strip()}")
        print("child process inherits isolation")
    ok("verify-isolation")
    return 0


def cmd_interpreter(args: _Args) -> int:
    print(sys.version)
    print(sys.executable)
    actual: Final = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual != args.expect:
        fail(f"interpreter is {actual}, expected {args.expect}")
    ok(f"interpreter {actual}")
    return 0


def _run_cli(argv: Sequence[str], label: str) -> CheckResult:
    try:
        proc: Final = subprocess.run(list(argv), timeout=120, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return CheckResult(ok=False, detail=f"{label} timed out after 120s")
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return CheckResult(ok=False, detail=f"{label} exited {proc.returncode}")
    return CheckResult(ok=True)


def cmd_cli(args: _Args) -> int:
    venv_bin: Final = Path(sys.executable).parent
    litellm_bin: Final = Path(args.litellm_bin) if args.litellm_bin else venv_bin / "litellm"
    lite_bin: Final = Path(args.lite_bin) if args.lite_bin else venv_bin / "lite"
    commands: Final = (
        ("import litellm", [sys.executable, "-c", "import litellm"]),
        ("litellm --version", [str(litellm_bin), "--version"]),
        ("lite version", [str(lite_bin), "version"]),
    )
    for label, argv in commands:
        result = _run_cli(argv, label)
        if not result.ok:
            fail(result.detail)
        ok(label)
    return 0


def _free_port() -> int:
    sock: Final = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port: Final = cast(int, sock.getsockname()[1])
    sock.close()
    return port


_CONFIG_TEMPLATE: Final = """model_list:
  - model_name: smoke-model
    litellm_params:
      model: openai/smoke-model
      api_base: http://127.0.0.1:9/v1
      api_key: synthetic-key
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
"""


def _listen_inode(port: int) -> str | None:
    target: Final = f"{port:04X}"
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if len(cols) > 9 and cols[3] == "0A" and cols[1].rsplit(":", 1)[-1] == target:
                return cols[9]
    return None


def _ancestors(pid: int) -> frozenset[int]:
    chain: Final[set[int]] = set()
    pending: Final[list[int]] = [pid]
    while pending:
        current = pending.pop()
        if current <= 0 or current in chain:
            continue
        chain.add(current)
        try:
            stat = Path(f"/proc/{current}/stat").read_text()
        except OSError:
            continue
        pending.append(int(stat.rpartition(")")[2].split()[1]))
    return frozenset(chain)


def _socket_owner_pid(inode: str) -> int | None:
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        fd_dir = proc_dir / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    if os.readlink(fd) == f"socket:[{inode}]":
                        return int(proc_dir.name)
                except OSError:
                    continue
        except OSError:
            continue
    return None


def _verify_port_owner(port: int, proc: subprocess.Popen[bytes]) -> CheckResult:
    inode: Final = _listen_inode(port)
    if inode is None:
        return CheckResult(ok=False, detail=f"no LISTEN socket found for port {port} in /proc/net/tcp")
    owner: Final = _socket_owner_pid(inode)
    if owner is None:
        return CheckResult(ok=False, detail=f"no process owns the listen socket inode {inode} for port {port}")
    if owner != proc.pid and proc.pid not in _ancestors(owner):
        return CheckResult(
            ok=False, detail=f"port {port} owned by pid {owner} outside the launched process group {proc.pid}"
        )
    if proc.poll() is not None:
        return CheckResult(ok=False, detail=f"proxy exited with code {proc.returncode} after readiness")
    return CheckResult(ok=True)


def cmd_proxy_startup(args: _Args) -> int:
    diagnostics: Final = Path(args.diagnostics_dir)
    diagnostics.mkdir(parents=True, exist_ok=True)
    venv_bin: Final = Path(sys.executable).parent
    litellm_bin: Final = Path(args.litellm_bin) if args.litellm_bin else venv_bin / "litellm"
    port: Final = _free_port()
    master_key: Final = "sk-smoke-" + secrets.token_hex(16)
    config_path: Final = diagnostics / "config.yaml"
    config_path.write_text(_CONFIG_TEMPLATE)
    log_path: Final = diagnostics / "proxy.log"
    result_path: Final = diagnostics / "result.json"
    outcome: Final[dict[str, object]] = {
        "port": port,
        "time_to_ready_s": None,
        "shutdown_s": None,
        "readiness": None,
        "outcome": "failed",
    }
    log_file: Final = log_path.open("w")
    env: Final = {
        **os.environ,
        "LITELLM_MASTER_KEY": master_key,
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    started: Final = time.monotonic()
    proc: Final = subprocess.Popen(
        [str(litellm_bin), "--config", str(config_path), "--host", "127.0.0.1", "--port", str(port)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    body: str | None = None
    last_status: int | None = None
    while time.monotonic() - started < args.ready_deadline:
        if proc.poll() is not None:
            log_file.close()
            result_path.write_text(json.dumps(outcome))
            fail(f"proxy exited early with code {proc.returncode}\n{tail(log_path)}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health/readiness")
            resp = conn.getresponse()
            last_status = resp.status
            candidate = resp.read().decode()
            conn.close()
        except (http.client.HTTPException, ConnectionError, OSError):
            time.sleep(args.poll_interval)
            continue
        if last_status == 200:
            body = candidate
            break
        time.sleep(args.poll_interval)
    outcome["time_to_ready_s"] = round(time.monotonic() - started, 3)
    if body is None:
        _terminate(proc, log_file)
        result_path.write_text(json.dumps(outcome))
        detail = f"last status {last_status}" if last_status is not None else "no response"
        fail(f"readiness not reached within {args.ready_deadline}s ({detail})\n{tail(log_path)}")
    outcome["readiness"] = body
    try:
        readiness = cast(object, json.loads(body))
    except json.JSONDecodeError:
        readiness = None
    if readiness != {"status": "healthy", "db": "Not connected"}:
        _terminate(proc, log_file)
        result_path.write_text(json.dumps(outcome))
        fail(f"unexpected readiness body: {body}")
    owner_check: Final = _verify_port_owner(port, proc)
    if not owner_check.ok:
        _terminate(proc, log_file)
        result_path.write_text(json.dumps(outcome))
        fail(owner_check.detail)
    shutdown_started: Final = time.monotonic()
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=args.shutdown_deadline)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        outcome["shutdown_s"] = round(time.monotonic() - shutdown_started, 3)
        log_file.close()
        result_path.write_text(json.dumps(outcome))
        fail(f"forced kill after {args.shutdown_deadline}s\n{tail(log_path)}")
    outcome["shutdown_s"] = round(time.monotonic() - shutdown_started, 3)
    try:
        os.killpg(proc.pid, 0)
    except ProcessLookupError:
        pass
    else:
        os.killpg(proc.pid, signal.SIGKILL)
        log_file.close()
        result_path.write_text(json.dumps(outcome))
        fail("process group survived SIGTERM")
    log_file.close()
    outcome["outcome"] = "ok"
    result_path.write_text(json.dumps(outcome))
    ok(f"proxy-startup ready={outcome['time_to_ready_s']}s shutdown={outcome['shutdown_s']}s")
    return 0


def _terminate(proc: subprocess.Popen[bytes], log_file: TextIO) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
    log_file.close()


def _load_manifest(path: Path) -> MappingProxyType[str, str]:
    def no_duplicates(pairs: list[tuple[object, object]]) -> dict[object, object]:
        seen: dict[object, object] = {}
        for key, value in pairs:
            if key in seen:
                raise ValueError(f"duplicate key in manifest: {key}")
            seen[key] = value
        return seen

    raw_value: object = cast(object, json.loads(path.read_text(), object_pairs_hook=no_duplicates))
    if not isinstance(raw_value, dict):
        raise ValueError("manifest must be an object")
    loaded: Final = cast(dict[object, object], raw_value)
    cases_value: object = loaded.get("cases")
    if not isinstance(cases_value, dict):
        raise ValueError("manifest must be an object with a 'cases' object")
    cases_any: Final = cast(dict[object, object], cases_value)
    cases: Final = {k: v for k, v in cases_any.items() if isinstance(k, str) and isinstance(v, str)}
    if len(cases) != len(cases_any):
        raise ValueError("manifest 'cases' must map string ids to string node ids")
    return MappingProxyType(cases)


@dataclass(slots=True, eq=False)
class _Recorder:
    collect_failed: list[str] = field(default_factory=list)
    collected: tuple[str, ...] = ()
    reports: dict[str, list[tuple[str, str, bool]]] = field(default_factory=dict)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collect_failed.append(report.nodeid)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collected = tuple(item.nodeid for item in session.items)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.reports.setdefault(report.nodeid, []).append((report.when, report.outcome, hasattr(report, "wasxfail")))


def cmd_pytest(args: _Args) -> int:
    try:
        cases: Final = _load_manifest(Path(args.manifest))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        fail(f"manifest invalid: {exc}")
    if tuple(cases) != EXPECTED_CASES:
        fail(f"manifest case ids must be exactly {list(EXPECTED_CASES)} in order, got {list(cases)}")
    node_ids: Final = tuple(cases.values())
    if len(set(node_ids)) != len(node_ids):
        fail("manifest node ids are not unique")
    argv: Final = [
        *node_ids,
        "-p",
        "no:cacheprovider",
        "-p",
        "no:xdist",
        "-p",
        "no:rerunfailures",
        "-p",
        "no:randomly",
        "-rA",
        "-q",
        *(["--rootdir", args.rootdir] if args.rootdir else []),
    ]

    recorder: Final = _Recorder()
    code: Final = pytest.main(argv, plugins=[recorder])
    name_of: Final = MappingProxyType({node_id: case_id for case_id, node_id in cases.items()})
    problems: Final[list[str]] = []
    if code != 0:
        problems.append(f"pytest exit code {code}")
    for failed_id in recorder.collect_failed:
        problems.append(f"collection failed: {name_of.get(failed_id, failed_id)}")
    expected: Final = Counter(node_ids)
    collected: Final = Counter(recorder.collected)
    for node_id in expected - collected:
        problems.append(f"missing case {name_of[node_id]} ({node_id})")
    for node_id in collected - expected:
        problems.append(f"unexpected test collected: {node_id}")
    for node_id, count in collected.items():
        if count > 1:
            problems.append(f"duplicated test id: {node_id}")
    if len(recorder.collected) != len(EXPECTED_CASES):
        problems.append(f"collected {len(recorder.collected)} tests, expected {len(EXPECTED_CASES)}")
    rows: Final[list[tuple[str, bool]]] = []
    for case_id, node_id in cases.items():
        reports = recorder.reports.get(node_id, [])
        case_ok = (
            bool(reports)
            and all(outcome == "passed" and not wasxfail for _, outcome, wasxfail in reports)
            and {when for when, _, _ in reports} >= {"setup", "call", "teardown"}
        )
        rows.append((case_id, case_ok))
        if not reports:
            problems.append(f"{case_id} ({node_id}) produced no runtest reports")
            continue
        for when, outcome, wasxfail in reports:
            if outcome != "passed":
                problems.append(f"{case_id} ({node_id}) {when} outcome={outcome}")
            if wasxfail:
                problems.append(f"{case_id} ({node_id}) {when} was xfail/xpass")
        missing_phases = {"setup", "call", "teardown"} - {when for when, _, _ in reports}
        for phase in sorted(missing_phases):
            problems.append(f"{case_id} ({node_id}) missing {phase} report")
    for case_id, passed in rows:
        print(f"{case_id}  {'PASS' if passed else 'FAIL'}  {cases[case_id]}")
    if problems:
        for problem in problems:
            print(f"merge-smoke: {problem}", file=sys.stderr)
        fail("pytest verdict failed")
    ok("pytest 11 cases")
    return 0


def main() -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    subs: Final = parser.add_subparsers(dest="command", required=True)
    p_iso: Final = subs.add_parser("verify-isolation")
    p_iso.add_argument("--no-child", action="store_true")
    p_interp: Final = subs.add_parser("interpreter")
    p_interp.add_argument("--expect", required=True)
    p_cli: Final = subs.add_parser("cli")
    p_cli.add_argument("--litellm-bin", default=None)
    p_cli.add_argument("--lite-bin", default=None)
    p_proxy: Final = subs.add_parser("proxy-startup")
    p_proxy.add_argument("--diagnostics-dir", required=True)
    p_proxy.add_argument("--litellm-bin", default=None)
    p_proxy.add_argument("--ready-deadline", type=float, default=120)
    p_proxy.add_argument("--shutdown-deadline", type=float, default=20)
    p_proxy.add_argument("--poll-interval", type=float, default=0.5)
    p_test: Final = subs.add_parser("pytest")
    p_test.add_argument("--manifest", required=True)
    p_test.add_argument("--rootdir", default=None)
    args: Final = parser.parse_args(namespace=_Args())
    handlers: Final = {
        "verify-isolation": cmd_verify_isolation,
        "interpreter": cmd_interpreter,
        "cli": cmd_cli,
        "proxy-startup": cmd_proxy_startup,
        "pytest": cmd_pytest,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
