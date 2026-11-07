from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import pytest

from tests.integration._support.process import kill_stray_process, stop_child

EXPIRED_KEY: Final = "fixture-expired-key"
FRESH_KEY: Final = "fixture-fresh-key"
REFRESH_TOKEN: Final = "fixture-single-use-refresh"
ROTATED_REFRESH_TOKEN: Final = "fixture-rotated-refresh"

_REFRESH: Final = textwrap.dedent(
    """
    import sys, requests
    from litellm.proxy.client.cli.commands.auth import load_token, save_token
    from litellm.proxy.client.cli.commands.pkce_login import fresh_api_key
    record = load_token()
    assert record is not None
    print("ready", flush=True)
    sys.stdin.readline()
    key = fresh_api_key(record, save_token, requests.Session(), reload=load_token)
    if key is None:
        raise SystemExit(1)
    print(key, flush=True)
    """
)
_HOLD_LOCK: Final = textwrap.dedent(
    """
    import sys
    from litellm.litellm_core_utils.cli_token_utils import cli_token_lock
    with cli_token_lock():
        print("ready", flush=True)
        sys.stdin.readline()
    """
)
_CLI: Final = "from litellm_cli.main import cli; cli()"
_PRINT_TOKEN: Final = textwrap.dedent(
    """
    import sys
    from litellm.proxy.client.cli.main import cli
    print("ready", flush=True)
    sys.stdin.readline()
    cli(["--base-url", sys.argv[1], "auth", "print-token"])
    """
)
_SHORT_WAIT: Final = textwrap.dedent(
    """
    from unittest.mock import patch
    from filelock import FileLock
    def bounded_lock(path, **options):
        return FileLock(path, **{**options, "timeout": 0.1})
    with patch("litellm.litellm_core_utils.cli_token_utils.FileLock", side_effect=bounded_lock):
    """
)


class _TokenEndpoint(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _TokenHandler)
        self.requests: list[dict[str, list[str]]] = []
        self.guard = threading.Lock()
        self.response_delay = 0.0
        self.request_started = threading.Event()

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def redeem(self, form: dict[str, list[str]]) -> bool:
        with self.guard:
            self.requests.append(form)
            return len(self.requests) == 1 and form.get("refresh_token") == [REFRESH_TOKEN]


class _TokenHandler(BaseHTTPRequestHandler):
    server: _TokenEndpoint

    def do_POST(self) -> None:
        form: Final = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
        accepted: Final = self.path == "/token" and self.server.redeem(form)
        payload: Final = (
            {
                "access_token": FRESH_KEY,
                "refresh_token": ROTATED_REFRESH_TOKEN,
                "expires_in": 3600,
                "user_id": "fixture-user",
                "team_id": "fixture-team",
            }
            if accepted
            else {"error": "invalid_grant", "error_description": "the refresh token was already used"}
        )
        self.server.request_started.set()
        threading.Event().wait(self.server.response_delay)
        body: Final = json.dumps(payload).encode()
        self.send_response(200 if accepted else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def token_endpoint() -> Iterator[_TokenEndpoint]:
    server: Final = _TokenEndpoint()
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _environment(home: Path) -> dict[str, str]:
    return {
        **{
            name: os.environ[name]
            for name in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")
            if name in os.environ
        },
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "PYTHON_DOTENV_DISABLED": "1",
        "LITELLM_CLI_DISABLE_KEYRING": "1",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_API_KEY": "",
        "LITELLM_PROXY_URL": "",
    }


def _seed(home: Path, origin: str, *, fresh: bool = False) -> None:
    directory: Final = home / ".litellm"
    directory.mkdir(mode=0o700)
    path: Final = directory / "token.json"
    path.write_text(
        json.dumps(
            {
                "base_url": origin,
                "key": FRESH_KEY if fresh else EXPIRED_KEY,
                "refresh_token": ROTATED_REFRESH_TOKEN if fresh else REFRESH_TOKEN,
                "timestamp": 1.0,
                "expires_at": 1e20 if fresh else 0.0,
                "client_id": "fixture-client",
                "token_endpoint": f"{origin}/token",
                "revocation_endpoint": f"{origin}/revoke",
                "resource": origin,
                "user_id": "fixture-user",
                "team_id": "fixture-team",
            }
        )
    )
    path.chmod(0o600)


@contextmanager
def _worker(script: str, home: Path, args: tuple[str, ...] = ()) -> Generator[subprocess.Popen[str], None, None]:
    process: Final = subprocess.Popen(
        (sys.executable, "-P", "-c", script, *args),
        env=_environment(home),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline() == "ready\n"
        yield process
    finally:
        stop_child(process)
        process.communicate(timeout=20)


def _print_token(home: Path, origin: str, script: str = _CLI) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-P", "-c", script, "--base-url", origin, "auth", "print-token"),
        env=_environment(home),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.timeout(60)
def test_two_processes_with_the_same_stale_snapshot_spend_the_refresh_token_once(
    tmp_path: Path,
    token_endpoint: _TokenEndpoint,
) -> None:
    _seed(tmp_path, token_endpoint.origin)
    with _worker(_REFRESH, tmp_path) as first, _worker(_REFRESH, tmp_path) as second:
        assert first.stdin is not None and second.stdin is not None
        first.stdin.write("refresh\n")
        first.stdin.flush()
        second.stdin.write("refresh\n")
        second.stdin.flush()
        first_output: Final = first.communicate(timeout=20)
        second_output: Final = second.communicate(timeout=20)
        assert first.returncode == second.returncode == 0, (first_output, second_output)
        assert first_output == second_output == (f"{FRESH_KEY}\n", "")
    result: Final = _print_token(tmp_path, token_endpoint.origin)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{FRESH_KEY}\n"
    assert token_endpoint.requests == [
        {
            "grant_type": ["refresh_token"],
            "refresh_token": [REFRESH_TOKEN],
            "client_id": ["fixture-client"],
            "resource": [token_endpoint.origin],
        }
    ]


@pytest.mark.timeout(60)
def test_a_killed_credential_writer_releases_the_lock_for_the_next_cli(
    tmp_path: Path,
    token_endpoint: _TokenEndpoint,
) -> None:
    _seed(tmp_path, token_endpoint.origin, fresh=True)
    with _worker(_HOLD_LOCK, tmp_path) as holder:
        kill_stray_process(holder.pid)
        holder.communicate(timeout=20)
        result: Final = _print_token(tmp_path, token_endpoint.origin)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{FRESH_KEY}\n"
    assert token_endpoint.requests == []


@pytest.mark.timeout(60)
def test_lock_contention_never_sends_an_unlocked_refresh(
    tmp_path: Path,
    token_endpoint: _TokenEndpoint,
) -> None:
    _seed(tmp_path, token_endpoint.origin)
    bounded_refresh: Final = _SHORT_WAIT + textwrap.indent(_REFRESH, "    ")
    with _worker(bounded_refresh, tmp_path) as reader, _worker(_HOLD_LOCK, tmp_path):
        output: Final = reader.communicate(input="refresh\n", timeout=20)
        assert reader.returncode == 1, output
        assert output == ("", "")
    assert token_endpoint.requests == []


@pytest.mark.timeout(60)
@pytest.mark.parametrize("reader_script", (_REFRESH, _PRINT_TOKEN), ids=("renewal-helper", "cli"))
def test_helpers_wait_for_a_slow_sibling_renewal_without_logging_in_again(
    tmp_path: Path,
    token_endpoint: _TokenEndpoint,
    reader_script: str,
) -> None:
    _seed(tmp_path, token_endpoint.origin)
    token_endpoint.response_delay = 11.0
    with (
        _worker(_REFRESH, tmp_path) as renewer,
        _worker(reader_script, tmp_path, (token_endpoint.origin,)) as reader,
    ):
        assert renewer.stdin is not None
        renewer.stdin.write("refresh\n")
        renewer.stdin.flush()
        assert token_endpoint.request_started.wait(timeout=10)
        reader_output: Final = reader.communicate(input="read\n", timeout=25)
        renewer_output: Final = renewer.communicate(timeout=20)
        assert reader.returncode == renewer.returncode == 0, (reader_output, renewer_output)
        assert reader_output == renewer_output == (f"{FRESH_KEY}\n", "")
    assert len(token_endpoint.requests) == 1


@pytest.mark.timeout(60)
def test_print_token_reports_lock_contention_and_can_retry_without_logging_in_again(
    tmp_path: Path,
    token_endpoint: _TokenEndpoint,
) -> None:
    _seed(tmp_path, token_endpoint.origin)
    path: Final = tmp_path / ".litellm" / "token.json"
    original: Final = path.read_bytes()
    with _worker(_HOLD_LOCK, tmp_path):
        result: Final = _print_token(tmp_path, token_endpoint.origin, _SHORT_WAIT + textwrap.indent(_CLI, "    "))
        assert result.returncode == 1, result.stderr
        assert result.stdout == ""
        assert "Retry this command" in result.stderr
        assert "lite login" not in result.stderr
        assert "Not authenticated" not in result.stderr
        assert path.read_bytes() == original
        assert token_endpoint.requests == []
    retried: Final = _print_token(tmp_path, token_endpoint.origin)
    assert retried.returncode == 0, retried.stderr
    assert retried.stdout == f"{FRESH_KEY}\n"
    assert len(token_endpoint.requests) == 1
