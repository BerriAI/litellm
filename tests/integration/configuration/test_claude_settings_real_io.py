import os
import pathlib
import shlex
import stat
import subprocess
import sys
import time

import pytest
from click.testing import CliRunner
from litellm.proxy.client.cli.commands.claude_settings import (
    ApiKeyHelper,
    KeepModel,
    StartOn,
    StaticToken,
    UnpinModel,
    install_statusline_script,
    installed_executable,
    lite_api_key_helper,
    statusline_command,
)


class TestLiteApiKeyHelper:
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="runs the command through sh, as Claude Code does on POSIX")
    def test_sh_runs_the_lite_on_path_against_this_proxy(self, tmp_path):
        lite = tmp_path / "my tools" / "lite"
        lite.parent.mkdir()
        lite.write_text('#!/bin/sh\nprintf "%s\\n" "$0" "$@"\n')
        lite.chmod(0o700)

        helper = lite_api_key_helper("https://proxy.example.com/", platform="linux", which=lambda name: str(lite))

        assert helper is not None
        ran = subprocess.run(["sh", "-c", helper.command], capture_output=True, text=True, check=True)
        assert ran.stdout.splitlines() == [str(lite), "--base-url", "https://proxy.example.com", "auth", "print-token"]

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="runs the command through sh, as Claude Code does on POSIX")
    def test_a_lite_found_through_a_relative_path_entry_keeps_running_from_any_directory(self, tmp_path, monkeypatch):
        for repo, name in ((tmp_path / "login-dir", "mine"), (tmp_path / "other-repo", "planted")):
            lite = repo / "bin" / "lite"
            lite.parent.mkdir(parents=True)
            lite.write_text(f"#!/bin/sh\necho {name}\n")
            lite.chmod(0o700)
        monkeypatch.chdir(tmp_path / "login-dir")

        helper = lite_api_key_helper("https://proxy.example.com", platform="linux", which=lambda name: "bin/lite")

        assert helper is not None
        ran = subprocess.run(
            ["sh", "-c", helper.command], cwd=tmp_path / "other-repo", capture_output=True, text=True, check=True
        )
        assert ran.stdout == "mine\n"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="runs the command through sh, as Claude Code does on POSIX")
class TestInstalledExecutable:
    @staticmethod
    def _script(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o700)

    def test_the_helper_runs_the_lite_installed_beside_this_python_over_a_shim_earlier_on_path(
        self, tmp_path, monkeypatch
    ):
        self._script(tmp_path / "shims" / "lite", "echo 'pyenv: lite: command not found' >&2; exit 127")
        self._script(tmp_path / "venv" / "bin" / "lite", "echo real")
        monkeypatch.setenv("PATH", f"{tmp_path / 'shims'}{os.pathsep}{os.environ['PATH']}")

        helper = lite_api_key_helper(
            "https://proxy.example.com",
            platform="linux",
            which=lambda name: installed_executable(name, scripts_dir=str(tmp_path / "venv" / "bin")),
        )

        assert helper is not None
        ran = subprocess.run(["sh", "-c", helper.command], capture_output=True, text=True)
        assert (ran.returncode, ran.stdout) == (0, "real\n")


PROXY = "http://127.0.0.1:4000"
ANTHROPIC = "https://api.anthropic.com"
RELOGIN = StaticToken("sk-fresh-login")
HELPER = ApiKeyHelper("/usr/local/bin/lite --base-url http://127.0.0.1:4000 auth print-token")
ORIGINAL = {
    "theme": "dark",
    "permissions": {"allow": ["Bash"]},
    "env": {"KEEP_ME": "1", "ANTHROPIC_API_KEY": "sk-ant-mine", "ANTHROPIC_BASE_URL": ANTHROPIC},
    "apiKeyHelper": "/usr/local/bin/lite auth print-token",
    "model": "claude-opus-5",
}


def _set(path, value):
    """A user edit: set (or with `_ABSENT`, remove) the key at a dotted path in the settings file."""

    def edit(settings):
        section, _, key = path.rpartition(".")
        container = settings.setdefault(section, {}) if section else settings
        if value is _ABSENT:
            container.pop(key, None)
        else:
            container[key] = value
        return settings

    return edit


_ABSENT = object()


# Each row: initial file, steps (configure kwargs dicts or edit callables) between the first configure
# and unconfigure, the expected file afterwards, and the expected outcome fields. Sequences that used
# to be one test each; the receipt's rules are what make them all come out right.
UNDO_SCENARIOS = {
    "plain round trip": (ORIGINAL, [], ORIGINAL, {"kept": ()}),
    "helper round trip": (ORIGINAL, [], ORIGINAL, {"kept": ()}, {"credential": HELPER, "model": KeepModel()}),
    "helper with no file before": (None, [], None, {"file_removed": True}, {"credential": HELPER}),
    "re-logins switching between the token and the helper keep the first snapshot": (
        ORIGINAL,
        [{"credential": HELPER, "model": KeepModel()}, {"credential": RELOGIN, "model": KeepModel()}],
        ORIGINAL,
        {"kept": ()},
        {"credential": HELPER, "model": KeepModel()},
    ),
    "no file before": (None, [], None, {"file_removed": True}),
    "no env before": ({"theme": "dark"}, [], {"theme": "dark"}, {}),
    "null env before": ({"theme": "dark", "env": None}, [], {"theme": "dark", "env": None}, {}),
    "empty env before": ({"theme": "dark", "env": {}}, [], {"theme": "dark", "env": {}}, {}),
    "user edits stay and are named": (
        ORIGINAL,
        [_set("env.ENABLE_TOOL_SEARCH", "false"), _set("model", "claude-sonnet-4-6")],
        {**ORIGINAL, "env": {**ORIGINAL["env"], "ENABLE_TOOL_SEARCH": "false"}, "model": "claude-sonnet-4-6"},
        {"kept": {"env.ENABLE_TOOL_SEARCH", "model"}, "withheld": ()},
    ),
    "user filled an env configure created": (None, [_set("env.MY_VAR", "mine")], {"env": {"MY_VAR": "mine"}}, {}),
    "user deleted the file": (None, [lambda s: None], None, {"file_removed": True, "restored": (), "kept": ()}),
    "user removed our key: neither restored nor kept": (
        ORIGINAL,
        [_set("env.ANTHROPIC_AUTH_TOKEN", _ABSENT)],
        ORIGINAL,
        {"not_restored": {"env.ANTHROPIC_AUTH_TOKEN"}, "kept": ()},
    ),
    "restored names only what changed": (
        {"model": "claude-opus-5"},
        [],
        {"model": "claude-opus-5"},
        {
            "restored": {
                "env.ANTHROPIC_BASE_URL",
                "env.ANTHROPIC_AUTH_TOKEN",
                "env.ENABLE_TOOL_SEARCH",
                "env.CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
                "statusLine",
            },
            "kept": (),
        },
        {"credential": RELOGIN, "model": KeepModel()},
    ),
    "repeat across credential kinds keeps the first snapshot": (
        ORIGINAL,
        [
            {"credential": RELOGIN, "model": UnpinModel()},
            {"credential": StaticToken("sk-rotated"), "model": StartOn("claude-sonnet-4-6")},
        ],
        ORIGINAL,
        {},
    ),
    "repeat without a model lets go of our pin, user had none": ({}, [{"model": UnpinModel()}], {}, {}),
    "repeat without a model lets go of our pin, user had one": (
        {"model": "claude-opus-5"},
        [{"model": UnpinModel()}],
        {"model": "claude-opus-5"},
        {},
    ),
    "re-login keeps our pin": (None, [{"credential": RELOGIN, "model": KeepModel()}], None, {"file_removed": True}),
    "edit between configures survives an unpin repeat": (
        ORIGINAL,
        [
            _set("model", "my-favourite"),
            _set("env.ENABLE_TOOL_SEARCH", "false"),
            {"credential": RELOGIN, "model": UnpinModel()},
        ],
        {**ORIGINAL, "env": {**ORIGINAL["env"], "ENABLE_TOOL_SEARCH": "false"}, "model": "my-favourite"},
        {"kept": {"env.ENABLE_TOOL_SEARCH", "model"}},
    ),
    "edit between configures survives a re-login": (
        ORIGINAL,
        [
            _set("model", "my-favourite"),
            _set("env.ENABLE_TOOL_SEARCH", "false"),
            {"credential": RELOGIN, "model": KeepModel()},
        ],
        {**ORIGINAL, "env": {**ORIGINAL["env"], "ENABLE_TOOL_SEARCH": "false"}, "model": "my-favourite"},
        {"kept": {"env.ENABLE_TOOL_SEARCH", "model"}},
    ),
    "edit between configures: a same-model repeat displaces it, so it is what comes back": (
        ORIGINAL,
        [_set("model", "my-favourite"), _set("env.ENABLE_TOOL_SEARCH", "false"), {"credential": RELOGIN}],
        {**ORIGINAL, "env": {**ORIGINAL["env"], "ENABLE_TOOL_SEARCH": "false"}, "model": "my-favourite"},
        {"kept": {"env.ENABLE_TOOL_SEARCH"}, "restored_includes": {"model"}},
    ),
    "base URL changed since: credentials withheld, receipt kept": (
        ORIGINAL,
        [_set("env.ANTHROPIC_BASE_URL", "http://other-proxy:4000")],
        {**ORIGINAL, "env": {"KEEP_ME": "1", "ANTHROPIC_BASE_URL": "http://other-proxy:4000"}, "apiKeyHelper": _ABSENT},
        {
            "withheld": {("env.ANTHROPIC_API_KEY", ANTHROPIC), ("apiKeyHelper", ANTHROPIC)},
            "kept": {"env.ANTHROPIC_BASE_URL"},
            "receipt_kept": True,
        },
    ),
    "base URL changed and back: judged against the URL the restored file holds": (
        ORIGINAL,
        [_set("env.ANTHROPIC_BASE_URL", ANTHROPIC)],
        ORIGINAL,
        {"withheld": ()},
    ),
    "credential captured beside no URL goes back only beside no URL": (
        {"env": {"ANTHROPIC_API_KEY": "sk-default-endpoint"}},
        [_set("env.ANTHROPIC_BASE_URL", "http://other-proxy:4000")],
        {"env": {"ANTHROPIC_BASE_URL": "http://other-proxy:4000"}},
        {
            "withheld": {("env.ANTHROPIC_API_KEY", "no ANTHROPIC_BASE_URL (Anthropic's default endpoint)")},
            "receipt_kept": True,
        },
    ),
    "restored document empty while a credential is withheld: file goes, receipt stays": (
        None,
        [
            _set("env.ANTHROPIC_API_KEY", "sk-user"),
            {"credential": RELOGIN, "model": KeepModel()},
            _set("env.ANTHROPIC_BASE_URL", _ABSENT),
        ],
        None,
        {"withheld": {("env.ANTHROPIC_API_KEY", PROXY)}, "file_removed": True, "receipt_kept": True},
        {"credential": RELOGIN, "model": KeepModel()},
    ),
    "a credential the user changed is kept, never also withheld": (
        ORIGINAL,
        [_set("env.ANTHROPIC_BASE_URL", "http://other-proxy:4000"), _set("apiKeyHelper", "/opt/mine/helper")],
        {
            **ORIGINAL,
            "env": {"KEEP_ME": "1", "ANTHROPIC_BASE_URL": "http://other-proxy:4000"},
            "apiKeyHelper": "/opt/mine/helper",
        },
        {
            "withheld": {("env.ANTHROPIC_API_KEY", ANTHROPIC)},
            "kept": {"env.ANTHROPIC_BASE_URL", "apiKeyHelper"},
            "receipt_kept": True,
        },
    ),
}


class TestStatusLine:
    """Every configure registers the status line, and only while the slot is empty or already ours."""

    COMMAND = "/opt/lite/bin/python /Users/me/.litellm/statusline.py"

    def test_the_installed_script_is_the_bundled_one_and_the_command_runs_this_interpreter(self, tmp_path):
        from litellm.proxy.client.cli.commands import statusline_script

        script = tmp_path / "lite" / "statusline.py"
        command = install_statusline_script(script)
        assert script.read_bytes().split(b"\n", 1)[1] == pathlib.Path(statusline_script.__file__).read_bytes()
        assert shlex.split(command) == [sys.executable, str(script)]
        assert command == statusline_command(script)
        assert stat.S_IMODE(script.stat().st_mode) == 0o600
        assert stat.S_IMODE(script.parent.stat().st_mode) == 0o700
        assert install_statusline_script(script) == command
