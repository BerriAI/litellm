"""Prove an e2e replay run makes zero outbound provider calls, by counting them.

`serve` pins each provider host (`--host`) to a local sink address in the hosts
file and binds a counting listener on that address, so any connection the proxy
or the record/replay edge opens to a real provider is redirected to the sink,
recorded as one line in `--hits-file`, and never leaves the box. The record and
replay edge only ever dials `127.0.0.1:<edge-port>` (a different host than the
pinned provider names), so in a clean replay the sink sees nothing; a single hit
means a provider call escaped the bundle. `assert-empty` turns that hit file into
the pass/fail check.

Stdlib only, so CI runs it under the system interpreter as root (binding :443 and
editing the hosts file both need root); `--sink-address`, `--port`, and
`--hosts-file` are injectable so it runs unprivileged against a temp hosts file on
a high port under test.
"""

# ruff: noqa: T201  # CLI script: its stdout/stderr progress and results are the interface
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Final

_BLOCK_BEGIN: Final = "# BEGIN e2e-egress-sentinel"
_BLOCK_END: Final = "# END e2e-egress-sentinel"


@dataclass(frozen=True, slots=True)
class ServeConfig:
    hosts: tuple[str, ...]
    sink_address: str
    ports: tuple[int, ...]
    hits_file: Path
    hosts_file: Path
    ready_file: Path | None
    pid_file: Path | None


def _pin_block(sink_address: str, hosts: tuple[str, ...]) -> str:
    lines = "\n".join(f"{sink_address}\t{host}" for host in hosts)
    return f"\n{_BLOCK_BEGIN}\n{lines}\n{_BLOCK_END}\n"


def _install_pins(hosts_file: Path, sink_address: str, hosts: tuple[str, ...]) -> bytes:
    original = hosts_file.read_bytes() if hosts_file.exists() else b""
    hosts_file.write_bytes(original + _pin_block(sink_address, hosts).encode())
    return original


def _restore_pins(hosts_file: Path, original: bytes) -> None:
    hosts_file.write_bytes(original)


def _bind(sink_address: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((sink_address, port))
    listener.listen(128)
    return listener


@dataclass(frozen=True, slots=True)
class _HitLog:
    path: Path
    _lock: threading.Lock

    def record(self, *, port: int, peer: tuple[str, int]) -> None:
        entry = json.dumps({"ts": time.time(), "port": port, "peer": list(peer)})
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(entry + "\n")


def _serve_socket(listener: socket.socket, port: int, hits: _HitLog, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            conn, peer = listener.accept()
        except OSError:
            return
        hits.record(port=port, peer=(peer[0], peer[1]))
        try:
            conn.close()
        except OSError:
            pass


def serve(config: ServeConfig) -> int:
    config.hits_file.write_text("", encoding="utf-8")
    original_hosts = _install_pins(config.hosts_file, config.sink_address, config.hosts)
    try:
        listeners = tuple(_bind(config.sink_address, port) for port in config.ports)
    except OSError as exc:
        _restore_pins(config.hosts_file, original_hosts)
        print(f"egress sentinel could not bind a sink: {exc}", file=sys.stderr)
        return 1

    stop = threading.Event()
    hits = _HitLog(path=config.hits_file, _lock=threading.Lock())
    threads = tuple(
        threading.Thread(target=_serve_socket, args=(listener, port, hits, stop), daemon=True)
        for listener, port in zip(listeners, config.ports)
    )
    for thread in threads:
        thread.start()

    def _handle(_signum: int, _frame: FrameType | None) -> None:
        stop.set()
        for listener in listeners:
            try:
                listener.close()
            except OSError:
                pass

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    if config.pid_file is not None:
        config.pid_file.write_text(str(os.getpid()), encoding="utf-8")
    if config.ready_file is not None:
        config.ready_file.write_text("ready", encoding="utf-8")
    print(
        f"egress sentinel up: pinned {', '.join(config.hosts)} to {config.sink_address} "
        f"on port(s) {', '.join(str(p) for p in config.ports)}",
        flush=True,
    )

    stop.wait()
    _restore_pins(config.hosts_file, original_hosts)
    if config.ready_file is not None and config.ready_file.exists():
        config.ready_file.unlink()
    if config.pid_file is not None and config.pid_file.exists():
        config.pid_file.unlink()
    return 0


def assert_empty(hits_file: Path) -> int:
    if not hits_file.exists():
        print(f"egress sentinel recorded no provider calls ({hits_file} absent): zero egress")
        return 0
    hits = [line for line in hits_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not hits:
        print("egress sentinel recorded no provider calls: zero egress")
        return 0
    print(f"egress sentinel recorded {len(hits)} provider call(s); replay was not hermetic:", file=sys.stderr)
    for line in hits:
        print(f"  {line}", file=sys.stderr)
    return 1


def _serve_from_args(args: argparse.Namespace) -> int:
    config = ServeConfig(
        hosts=tuple(args.host),
        sink_address=args.sink_address,
        ports=tuple(args.port),
        hits_file=Path(args.hits_file),
        hosts_file=Path(args.hosts_file),
        ready_file=Path(args.ready_file) if args.ready_file else None,
        pid_file=Path(args.pid_file) if args.pid_file else None,
    )
    return serve(config)


def main(argv: tuple[str, ...]) -> int:
    parser = argparse.ArgumentParser(description="count outbound provider calls during an e2e replay")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", help="pin provider hosts and count connection attempts")
    serve_parser.add_argument("--host", action="append", required=True, help="provider host to pin and watch")
    serve_parser.add_argument("--sink-address", default="127.0.0.1")
    serve_parser.add_argument("--port", action="append", type=int, default=None)
    serve_parser.add_argument("--hits-file", required=True)
    serve_parser.add_argument("--hosts-file", default="/etc/hosts")
    serve_parser.add_argument("--ready-file", default=None)
    serve_parser.add_argument("--pid-file", default=None)

    assert_parser = sub.add_parser("assert-empty", help="exit non-zero if any provider call was recorded")
    assert_parser.add_argument("--hits-file", required=True)

    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.port is None:
            args.port = [443]
        return _serve_from_args(args)
    return assert_empty(Path(args.hits_file))


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
