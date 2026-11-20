import json
import os
import shlex
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from litellm.proxy.client.cli.commands import claude_settings
from litellm.proxy.client.cli.commands.agents import run_agent

REAL_CLAUDE_SETTINGS: Final = Path(os.path.expanduser("~")) / ".claude" / "settings.json"
LAUNCH_PROXY: Final = "http://localhost:4000"


def _current_bytes() -> bytes | None:
    return REAL_CLAUDE_SETTINGS.read_bytes() if REAL_CLAUDE_SETTINGS.exists() else None


@pytest.fixture(autouse=True)
def _statusline_script_under_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(claude_settings, "STATUSLINE_SCRIPT_PATH", tmp_path / "litellm-home" / "statusline.py")


@pytest.fixture
def fake_codex_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[str | None, int], Path]:
    directory: Final = tmp_path / "codex-bin"
    directory.mkdir()
    binary: Final = directory / ("codex.cmd" if os.name == "nt" else "codex")
    version_output: Final = directory / "version-output.txt"
    monkeypatch.setenv("PATH", str(directory))

    def install(output: str | None, returncode: int = 0) -> Path:
        if output is None:
            binary.unlink(missing_ok=True)
            return binary
        version_output.write_text(output)
        if os.name == "nt":
            binary.write_text(
                '@echo off\nif not "%~1"=="--version" exit /b 2\n'
                'if not "%~2"=="" exit /b 2\ntype "%~dp0version-output.txt"\n'
                f'exit /b {returncode}\n'
            )
        else:
            binary.write_text(
                '#!/bin/sh\nif [ "$#" -ne 1 ] || [ "$1" != "--version" ]; then\n    exit 2\nfi\n'
                f'/bin/cat {shlex.quote(str(version_output))}\nexit {returncode}\n'
            )
        binary.chmod(0o700)
        return binary

    install("codex-cli 0.129.0\n")
    return install


@pytest.fixture(autouse=True)
def _isolated_codex_version_for_configure_tests(request: pytest.FixtureRequest) -> None:
    if request.node.path.name in ("test_codex_settings.py", "test_configure_commands.py"):
        request.getfixturevalue("fake_codex_version")


@pytest.fixture(autouse=True)
def isolated_claude_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    before: Final = _current_bytes()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    yield tmp_path
    after: Final = _current_bytes()
    if after == before:
        return
    if before is None:
        REAL_CLAUDE_SETTINGS.unlink()
    else:
        REAL_CLAUDE_SETTINGS.write_bytes(before)
    pytest.fail(
        f"this test wrote the developer's real {REAL_CLAUDE_SETTINGS}; the original bytes were restored. "
        "Resolve the Claude settings path at call time (never Path.home() at import) and point the test at tmp_path"
    )


@pytest.fixture
def launch_claude(isolated_claude_home: Path) -> Callable[[bool], Path]:
    """Launch `lite claude` over a `lite login --pkce --config-claude` setup and return the `--settings` file it
    wrote, which holds the launch key. Claude Code's config dir is read at call time, so a test can move it first."""

    def launch(subscription: bool) -> Path:
        launched: Final[dict[str, tuple[str, ...]]] = {}
        run_agent(
            LAUNCH_PROXY,
            "sk-launch",
            ["claude"],
            subscription=subscription,
            base_env={"CLAUDE_CONFIG_DIR": os.environ["CLAUDE_CONFIG_DIR"]},
            which=lambda name: f"/usr/local/bin/{name}",
            verify=lambda *args: None,
            verify_subscription=lambda *args: None,
            launcher=lambda path, args, env: launched.update(args=tuple(args)),
            read_settings=lambda path: {
                "apiKeyHelper": f"/usr/local/bin/lite --base-url {LAUNCH_PROXY} auth print-token",
                "env": {"ANTHROPIC_AUTH_TOKEN": ""},
            },
            project_dir=isolated_claude_home / "repo",
        )
        args: Final = launched["args"]
        return Path(args[args.index("--settings") + 1])

    return launch


class _ProxyResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code: Final = status_code
        self.headers: Final = {"content-type": "application/json"}
        self.text: Final = json.dumps(payload)
        self.content: Final = self.text.encode()
        self._payload: Final = payload

    def json(self) -> dict[str, object]:
        return self._payload


def _cli_auth_contract(base_url: str) -> dict[str, object]:
    return {
        "contract_version": 1,
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "registration_endpoint": f"{base_url}/register",
        "revocation_endpoint": f"{base_url}/revoke",
        "resource": base_url,
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "revocation_endpoint_auth_methods_supported": ["none"],
    }


def _query_value(url: str, name: str) -> str | None:
    return parse_qs(urlparse(url).query).get(name, [None])[0]


class LoginEdges:
    teams: Final = ({"team_id": "team-a", "team_alias": "Team A"}, {"team_id": "team-b", "team_alias": "Team B"})

    def __init__(self) -> None:
        self.opened: Final[list[str]] = []
        self.polled: Final[list[str]] = []

    def open(self, url: str, new: int = 0, autoraise: bool = True) -> bool:
        self.opened.append(url)
        query: Final = parse_qs(urlparse(url).query)
        if "redirect_uri" in query:
            try:
                urllib.request.urlopen(
                    f"{query['redirect_uri'][0]}?state={query['state'][0]}&error=access_denied", timeout=5
                ).close()
            except urllib.error.HTTPError:
                pass
        return True

    def session_get(self, url: str, *, timeout: float) -> _ProxyResponse:
        return _ProxyResponse(200, _cli_auth_contract(url.removesuffix("/.well-known/litellm-cli-auth")))

    def session_post(self, url: str, *, data=None, json=None, timeout: float, allow_redirects: bool) -> _ProxyResponse:
        return _ProxyResponse(201, {"client_id": "llm_dcrc_abc"})

    def start_classic(self, url: str, *, timeout: float) -> _ProxyResponse:
        return _ProxyResponse(200, {"login_id": "cli-1", "poll_secret": "secret", "user_code": "ABCD-EFGH"})

    def poll(self, url: str, *, headers=None, timeout: float) -> _ProxyResponse:
        self.polled.append(url)
        if "team_id=" in url:
            return _ProxyResponse(403, {"detail": "no key from this proxy"})
        return _ProxyResponse(
            200, {"status": "ready", "requires_team_selection": True, "user_id": "u1", "team_details": list(self.teams)}
        )

    @property
    def login_reached(self) -> list[tuple[str, str | None]]:
        authorize_urls: Final = [url for url in self.opened if urlparse(url).path == "/authorize"]
        team_polls: Final = [url for url in self.polled if "team_id=" in url]
        return [
            *(("pkce", _query_value(url, "team")) for url in authorize_urls),
            *(("classic", _query_value(url, "team_id")) for url in (team_polls or self.polled[:1])),
        ]


@pytest.fixture
def login_edges(monkeypatch: pytest.MonkeyPatch) -> LoginEdges:
    edges: Final = LoginEdges()

    class _Session:
        get = staticmethod(edges.session_get)
        post = staticmethod(edges.session_post)

    monkeypatch.setattr(requests, "Session", _Session)
    monkeypatch.setattr(requests, "post", edges.start_classic)
    monkeypatch.setattr(requests, "get", edges.poll)
    # run_pkce_login binds webbrowser.open as a default argument, so the browser is registered, not patched
    monkeypatch.setattr(webbrowser, "_tryorder", [])
    monkeypatch.setattr(webbrowser, "_browsers", {})
    webbrowser.register("lite-login-test", None, edges, preferred=True)
    return edges
