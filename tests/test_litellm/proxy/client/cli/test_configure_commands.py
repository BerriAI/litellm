import io
import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Final, Literal

import click
import pytest
import requests
import responses
import tomlkit
from click.testing import CliRunner
from InquirerPy.base.control import Choice
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands import claude_settings as claude_settings_module
from litellm.proxy.client.cli.commands import configure as configure_module
from litellm.proxy.client.cli.commands.claude_settings import SettingsFileOwner
from litellm.proxy.client.cli.commands.configure import configure_claude, configure_group, interactive_configure

PROXY = "http://proxy.test:4000"
VALID_KEY = "sk-virtual-key"
LISTED_MODELS = ("claude-auto", "gpt-5.6-luna")


def _mock_models():
    responses.get(
        f"{PROXY}/v1/models",
        json={"data": [{"id": model, "object": "model"} for model in LISTED_MODELS]},
        match=[responses.matchers.header_matcher({"Authorization": f"Bearer {VALID_KEY}"})],
    )
    responses.get(f"{PROXY}/v1/models", status=401)


@pytest.fixture
def paths(monkeypatch, tmp_path):
    """The default settings file, reached the way Claude Code reaches it: CLAUDE_CONFIG_DIR names its directory."""
    settings_path = tmp_path / "claude" / "settings.json"
    state_path = tmp_path / "litellm" / "claude_configure_state.json"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(settings_path.parent))
    monkeypatch.setattr(claude_settings_module, "CLAUDE_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(claude_settings_module, "CONFIGURE_STATE_PATH", state_path)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
    return settings_path, state_path


@pytest.fixture
def lite_on_path(monkeypatch, tmp_path):
    """A real `lite` executable on PATH, so the apiKeyHelper command resolves without patching."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    lite = bin_dir / "lite"
    lite.write_text("#!/bin/sh\nexit 0\n")
    lite.chmod(lite.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return str(lite)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def codex_path():
    return Path(os.environ["CODEX_HOME"]) / "config.toml"


class _TerminalInput(io.BytesIO):
    def isatty(self):
        return True


def _mock_agent_models():
    def listing(request):
        assert request.headers["Authorization"] == f"Bearer {VALID_KEY}"
        rows = (
            [{"id": "claude-router-6175746f", "source_model": "auto"}]
            if request.headers.get("x-gateway-client") == "claude-code"
            else [{"id": "auto"}]
        )
        return 200, {"Content-Type": "application/json"}, json.dumps({"data": rows})

    responses.add_callback(responses.GET, f"{PROXY}/v1/models", callback=listing)


@pytest.fixture
def lite_up_backup(monkeypatch, tmp_path):
    """A `lite up` session holding its backup, the local precondition every settings write refuses on."""
    backup = tmp_path / "claude_settings_backup.json"
    backup.write_text("{}")
    monkeypatch.setattr(
        claude_settings_module, "SETTINGS_FILE_OWNERS", (SettingsFileOwner(backup, "lite up", "lite down"),)
    )
    return backup


def _configure(runner, *args):
    return runner.invoke(cli, ["--base-url", PROXY, "configure", "claude", *args])


class TestConfigureClaudeWithAVirtualKey:
    @responses.activate
    def test_writes_settings_and_reports_without_echoing_the_key(self, runner, paths):
        _mock_models()
        settings_path, state_path = paths
        result = _configure(runner, "--api-key", VALID_KEY, "--model", "claude-auto")
        assert result.exit_code == 0, result.output
        written = json.loads(settings_path.read_text())
        assert written["env"]["ANTHROPIC_BASE_URL"] == PROXY
        assert written["env"]["ANTHROPIC_AUTH_TOKEN"] == VALID_KEY
        assert written["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
        assert written["model"] == "claude-auto"
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in written["env"]
        assert state_path.exists()
        assert VALID_KEY not in result.output
        assert written["env"]["ANTHROPIC_MODEL"] == "claude-auto"
        assert "Starting model: claude-auto" in result.output
        assert "1 of the proxy's 2 models" in result.output
        assert "lite unconfigure claude" in result.output
        assert [call.request.headers.get("x-gateway-client") for call in responses.calls] == ["claude-code"]

    @responses.activate
    def test_takes_the_key_from_the_global_option_and_keeps_claude_codes_default(self, runner, paths):
        _mock_models()
        settings_path, _ = paths
        result = runner.invoke(cli, ["--base-url", PROXY, "--api-key", VALID_KEY, "configure", "claude"])
        assert result.exit_code == 0, result.output
        written = json.loads(settings_path.read_text())
        assert written["env"]["ANTHROPIC_AUTH_TOKEN"] == VALID_KEY
        assert "model" not in written and "ANTHROPIC_MODEL" not in written["env"]
        assert "Starting model: not pinned" in result.output

    @responses.activate
    def test_refuses_a_model_the_proxy_does_not_list(self, runner, paths):
        _mock_models()
        settings_path, _ = paths
        result = _configure(runner, "--api-key", VALID_KEY, "--model", "claude-nope")
        assert result.exit_code != 0
        assert "'claude-nope' is not served" in result.output
        assert "claude-auto, gpt-5.6-luna" in result.output
        assert not settings_path.exists()

    @responses.activate
    def test_refuses_a_key_the_proxy_rejects(self, runner, paths):
        _mock_models()
        settings_path, _ = paths
        result = _configure(runner, "--api-key", "sk-wrong")
        assert result.exit_code != 0
        assert "rejected your key (HTTP 401)" in result.output
        assert not settings_path.exists()

    @responses.activate
    @pytest.mark.parametrize(
        ("mock", "expected", "unexpected"),
        [
            (
                lambda: responses.get(f"{PROXY}/v1/models", body=requests.ConnectionError("refused")),
                "Is the proxy at",
                "answered",
            ),
            (
                lambda: responses.get(f"{PROXY}/v1/models", status=500),
                "The proxy at http://proxy.test:4000 answered",
                "Is the proxy at",
            ),
            (
                lambda: responses.get(f"{PROXY}/v1/models", body="<html>not json</html>"),
                "answered, so check that it is a LiteLLM proxy",
                "Is the proxy at",
            ),
            (
                lambda: responses.get(f"{PROXY}/v1/models", json={"data": []}),
                "Claude Code would have nothing to run",
                "Is the proxy at",
            ),
        ],
        ids=["unreachable", "http-500", "non-json-body", "empty-list"],
    )
    def test_the_listing_hint_matches_how_the_listing_failed(self, runner, paths, mock, expected, unexpected):
        # Only a proxy that never answered gets the "is it running" question; a 500, a non-JSON body or an
        # empty list prove it is up, and the hint says so instead.
        mock()
        settings_path, _ = paths
        result = _configure(runner, "--api-key", VALID_KEY)
        assert result.exit_code != 0
        assert expected in result.output and unexpected not in result.output
        assert not settings_path.exists()

    @responses.activate
    @pytest.mark.parametrize("entry", ["virtual-key", "no-key", "interactive"])
    def test_refuses_while_lite_up_holds_a_backup_before_any_request(self, runner, paths, lite_up_backup, entry):
        _mock_models()
        if entry == "interactive":
            ctx = click.Context(configure_group, obj={"base_url": PROXY, "api_key": VALID_KEY})
            with pytest.raises(click.ClickException, match="lite down"):
                interactive_configure(ctx, pick_targets=lambda: ("claude",), pick_model=lambda listed: None)
        else:
            args = ["--api-key", VALID_KEY] if entry == "virtual-key" else []
            result = runner.invoke(configure_claude, args, obj={"base_url": PROXY, "api_key": None})
            assert result.exit_code != 0 and "lite down" in result.output
        assert len(responses.calls) == 0
        assert not paths[0].exists()

    @responses.activate
    def test_says_so_when_the_key_is_written_through_a_symlink(self, runner, paths, tmp_path):
        _mock_models()
        settings_path, _ = paths
        target = tmp_path / "dotfiles" / "settings.json"
        target.parent.mkdir()
        target.write_text("{}")
        settings_path.parent.mkdir(parents=True)
        settings_path.symlink_to(target)
        result = _configure(runner, "--api-key", VALID_KEY)
        assert result.exit_code == 0, result.output
        assert "keep it out of version control" in result.output
        assert json.loads(target.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == VALID_KEY


class TestConfigureClaudeWithoutAKey:
    @responses.activate
    def test_refuses_and_names_the_ways_to_pass_a_key_without_writing_or_logging_in(self, runner, paths):
        # A `lite login` credential expires within a day; the old fallback wrote an apiKeyHelper that made
        # Claude Code spawn `lite` (and its keychain probe) on every credential refresh.
        _mock_models()
        settings_path, state_path = paths
        result = runner.invoke(
            configure_claude,
            ["--model", "claude-auto"],
            obj={"base_url": PROXY, "api_key": "sk-login-jwt", "api_key_from_token_file": True},
        )
        assert result.exit_code != 0
        assert "--api-key" in result.output and "LITELLM_PROXY_API_KEY" in result.output
        assert "apiKeyHelper" not in result.output
        assert not settings_path.exists() and not state_path.exists()
        assert len(responses.calls) == 0

    @responses.activate
    def test_an_explicit_key_still_wins_over_a_stored_login(self, runner, paths):
        _mock_models()
        settings_path, _ = paths
        result = runner.invoke(
            configure_claude,
            ["--api-key", VALID_KEY],
            obj={"base_url": PROXY, "api_key": "sk-login-jwt", "api_key_from_token_file": True},
        )
        assert result.exit_code == 0, result.output
        written = json.loads(settings_path.read_text())
        assert written["env"]["ANTHROPIC_AUTH_TOKEN"] == VALID_KEY and "apiKeyHelper" not in written


class TestInteractiveConfigure:
    @responses.activate
    def test_asks_for_targets_and_a_starting_model_then_configures(self, paths):
        _mock_models()
        settings_path, _ = paths
        asked = {}

        def pick_model(listed):
            asked["listed"] = tuple(listed)
            return "claude-auto"

        ctx = click.Context(
            configure_group, obj={"base_url": PROXY, "api_key": VALID_KEY, "api_key_from_token_file": False}
        )
        interactive_configure(ctx, pick_targets=lambda: ("claude",), pick_model=pick_model)
        assert asked["listed"] == LISTED_MODELS
        assert json.loads(settings_path.read_text())["model"] == "claude-auto"

    def test_does_nothing_when_claude_code_is_not_picked(self, paths):
        settings_path, _ = paths
        ctx = click.Context(
            configure_group, obj={"base_url": PROXY, "api_key": VALID_KEY, "api_key_from_token_file": False}
        )
        interactive_configure(ctx, pick_targets=lambda: (), pick_model=lambda listed: None)
        assert not settings_path.exists()

    def test_bare_configure_without_a_terminal_names_the_non_interactive_command(self, runner, paths):
        result = runner.invoke(cli, ["--base-url", PROXY, "configure"])
        assert result.exit_code != 0
        assert "lite configure claude --api-key" in result.output


class TestConfigureAgents:
    @responses.activate
    @pytest.mark.parametrize("targets", [("claude",), ("codex",), ("claude", "codex")])
    def test_group_options_drive_the_agent_picker_and_write_only_selected_agents(
        self, runner, paths, codex_path, monkeypatch, targets
    ):
        _mock_agent_models()
        asked = []

        def checkbox(**kwargs):
            assert tuple(choice.value for choice in kwargs["choices"]) == ("claude", "codex")
            return SimpleNamespace(execute=lambda: targets)

        def fuzzy(**kwargs):
            assert "auto" in kwargs["choices"]
            assert "claude-router-6175746f" not in kwargs["choices"]
            assert not paths[0].exists() and not codex_path.exists()
            asked.append(kwargs["message"])
            return SimpleNamespace(execute=lambda: "auto")

        monkeypatch.setattr(configure_module.inquirer, "checkbox", checkbox)
        monkeypatch.setattr(configure_module.inquirer, "fuzzy", fuzzy)
        result = runner.invoke(
            cli,
            ["configure", "--api-key", VALID_KEY, "--gateway-url", f"{PROXY}/v1/"],
            input=_TerminalInput(),
        )
        assert result.exit_code == 0, result.output
        assert VALID_KEY not in result.output
        assert len(asked) == len(targets)
        assert paths[0].exists() == ("claude" in targets)
        assert codex_path.exists() == ("codex" in targets)
        if "claude" in targets:
            claude = json.loads(paths[0].read_text())
            assert claude["model"] == "claude-router-6175746f"
            assert claude["env"]["ANTHROPIC_BASE_URL"] == PROXY
            assert claude["env"]["ANTHROPIC_AUTH_TOKEN"] == VALID_KEY
        if "codex" in targets:
            codex = tomlkit.parse(codex_path.read_text())
            assert codex["model"] == "auto"
            assert codex["model_provider"] == "litellm"
            provider = codex["model_providers"]["litellm"]
            assert provider["base_url"] == f"{PROXY}/v1"
            assert provider["http_headers"]["Authorization"] == f"Bearer {VALID_KEY}"
            assert "env_key" not in provider
        assert [call.request.headers.get("x-gateway-client") for call in responses.calls] == [
            "claude-code" if target == "claude" else None for target in targets
        ]

    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    @pytest.mark.parametrize("leaf_override", [False, True], ids=["inherit-group", "leaf-wins"])
    def test_group_connection_options_are_inherited_and_leaf_options_take_precedence(
        self, runner, paths, codex_path, target, leaf_override
    ):
        _mock_agent_models()
        group_url = "http://group.test" if leaf_override else PROXY
        group_key = "sk-group" if leaf_override else VALID_KEY
        args = [
            "--base-url", "http://global.test", "--api-key", "sk-global", "configure",
            "--gateway-url", group_url, "--api-key", group_key, target, "--model", "auto",
        ]
        if leaf_override:
            args.extend(["--base-url", f"{PROXY}/v1/", "--api-key", VALID_KEY])
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
        assert all(key not in result.output for key in (VALID_KEY, group_key, "sk-global"))
        if target == "claude":
            written = json.loads(paths[0].read_text())
            assert written["env"]["ANTHROPIC_AUTH_TOKEN"] == VALID_KEY
            assert written["env"]["ANTHROPIC_BASE_URL"] == PROXY
            assert not codex_path.exists()
        else:
            provider = tomlkit.parse(codex_path.read_text())["model_providers"]["litellm"]
            assert provider["http_headers"]["Authorization"] == f"Bearer {VALID_KEY}"
            assert provider["base_url"] == f"{PROXY}/v1"
            assert not paths[0].exists()
        assert len(responses.calls) == 1

    @responses.activate
    @pytest.mark.parametrize("failure", ["invalid-model", "cancel"])
    def test_both_model_choices_complete_before_either_configuration_changes(
        self, paths, codex_path, failure
    ):
        _mock_agent_models()
        settings_path, state_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text('{"theme": "dark"}')
        codex_path.parent.mkdir(parents=True)
        codex_path.write_text('model = "original"\n')
        before = (settings_path.read_bytes(), codex_path.read_bytes())

        def pick_codex_model(listed):
            assert listed == ("auto",)
            assert (settings_path.read_bytes(), codex_path.read_bytes()) == before
            if failure == "cancel":
                raise KeyboardInterrupt()
            return "not-listed"

        ctx = click.Context(configure_group, obj={"base_url": PROXY, "api_key": VALID_KEY})
        expected = KeyboardInterrupt if failure == "cancel" else click.ClickException
        with pytest.raises(expected):
            interactive_configure(
                ctx,
                pick_targets=lambda: ("claude", "codex"),
                pick_model=lambda listed: "auto",
                pick_codex_model=pick_codex_model,
            )
        assert (settings_path.read_bytes(), codex_path.read_bytes()) == before
        assert not state_path.exists()
        assert not tuple((codex_path.parent / ".litellm").glob("*.json"))

    @responses.activate
    def test_both_configs_are_preflighted_before_fetching_models_or_writing(
        self, paths, codex_path
    ):
        _mock_agent_models()
        codex_path.parent.mkdir(parents=True)
        codex_path.write_text("[invalid")
        ctx = click.Context(configure_group, obj={"base_url": PROXY, "api_key": VALID_KEY})
        with pytest.raises(click.ClickException, match="Could not read Codex settings"):
            interactive_configure(
                ctx,
                pick_targets=lambda: ("claude", "codex"),
                pick_model=lambda listed: "auto",
                pick_codex_model=lambda listed: "auto",
            )
        assert not paths[0].exists() and not paths[1].exists()
        assert codex_path.read_text() == "[invalid"
        assert len(responses.calls) == 0

    @responses.activate
    @pytest.mark.parametrize("targets", [("claude", "codex"), ("codex", "claude")])
    @pytest.mark.parametrize("version", [None, "codex-cli 0.128.0\n"])
    def test_unsafe_codex_blocks_both_targets_before_requests_or_writes(
        self, paths, codex_path, fake_codex_version, targets, version
    ):
        _mock_agent_models()
        fake_codex_version(version, 0)
        ctx = click.Context(configure_group, obj={"base_url": PROXY, "api_key": VALID_KEY})
        with pytest.raises(click.ClickException, match=r"0\.129\.0") as caught:
            interactive_configure(
                ctx,
                pick_targets=lambda: targets,
                pick_model=lambda listed: "auto",
                pick_codex_model=lambda listed: "auto",
            )
        assert VALID_KEY not in str(caught.value)
        assert len(responses.calls) == 0
        assert not paths[0].exists() and not paths[1].exists()
        assert not codex_path.exists() and not tuple((codex_path.parent / ".litellm").glob("*.json"))

    @responses.activate
    def test_claude_only_configuration_does_not_require_codex(
        self, runner, paths, codex_path, fake_codex_version
    ):
        _mock_agent_models()
        fake_codex_version(None, 0)
        result = runner.invoke(
            cli, ["configure", "--api-key", VALID_KEY, "--gateway-url", PROXY, "claude", "--model", "auto"]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(paths[0].read_text())["model"] == "claude-router-6175746f"
        assert not codex_path.exists()

    @responses.activate
    def test_codex_only_ignores_claudes_temporary_owner(
        self, runner, paths, codex_path, lite_up_backup
    ):
        _mock_agent_models()
        result = runner.invoke(
            cli, ["configure", "--api-key", VALID_KEY, "--gateway-url", PROXY, "codex", "--model", "auto"]
        )
        assert result.exit_code == 0, result.output
        assert tomlkit.parse(codex_path.read_text())["model"] == "auto"
        assert not paths[0].exists() and not paths[1].exists()
        assert lite_up_backup.exists()

    def test_noninteractive_codex_requires_a_model(self, runner, paths, codex_path):
        result = runner.invoke(cli, ["configure", "--api-key", VALID_KEY, "--gateway-url", PROXY, "codex"])
        assert result.exit_code != 0 and "Missing option '--model'" in result.output
        assert not paths[0].exists() and not codex_path.exists()

    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    @pytest.mark.parametrize(
        "option, value, expected",
        [
            ("--api-key", "sk-secret\ninvalid", "must not be blank"),
            ("--gateway-url", "https://user:sk-secret@proxy.test", "must not contain credentials"),
            ("--gateway-url", "https://proxy.test?key=sk-secret", "must not include a query"),
            ("--gateway-url", "file:///sk-secret", "must be a full http:// or https:// URL"),
        ],
    )
    def test_invalid_connection_input_never_writes_requests_or_echoes_secrets(
        self, runner, paths, codex_path, target, option, value, expected
    ):
        result = runner.invoke(
            cli,
            [
                "configure", "--api-key", VALID_KEY, "--gateway-url", PROXY,
                target, "--model", "auto", option, value,
            ],
        )
        assert result.exit_code != 0 and expected in result.output
        assert "sk-secret" not in result.output and VALID_KEY not in result.output
        assert not paths[0].exists() and not codex_path.exists()
        assert len(responses.calls) == 0

    @responses.activate
    @pytest.mark.parametrize("failure", ["rejected", "connection", "response-body"])
    def test_gateway_failures_never_echo_the_key(self, runner, paths, codex_path, failure):
        if failure == "rejected":
            responses.get(f"{PROXY}/v1/models", status=401)
        elif failure == "connection":
            responses.get(f"{PROXY}/v1/models", body=requests.ConnectionError(VALID_KEY))
        else:
            responses.get(f"{PROXY}/v1/models", json={"data": VALID_KEY})
        result = runner.invoke(
            cli, ["configure", "--api-key", VALID_KEY, "--gateway-url", PROXY, "codex", "--model", "auto"]
        )
        assert result.exit_code != 0 and "Error:" in result.output
        assert VALID_KEY not in result.output
        assert not paths[0].exists() and not codex_path.exists()

    @responses.activate
    def test_configure_reconfigure_and_unconfigure_do_not_read_a_stored_login(
        self, runner, paths, codex_path, tmp_path, secret_vault_factory, fake_codex_version
    ):
        _mock_agent_models()
        token_path = tmp_path / ".litellm" / "token.json"
        token_path.parent.mkdir()
        token_path.write_text(json.dumps({"base_url": PROXY, "timestamp": time.time()}))
        vault = secret_vault_factory(json.dumps({"base_url": PROXY, "key": "sk-login", "jwt_token": ""}))
        missing = runner.invoke(
            cli, ["configure", "--gateway-url", PROXY, "codex", "--model", "auto"], obj={"secret_vault": vault}
        )
        assert missing.exit_code != 0 and "needs a long-lived virtual key" in missing.output
        assert len(responses.calls) == 0 and not codex_path.exists()
        configured = runner.invoke(
            cli,
            ["configure", "--api-key", VALID_KEY, "--gateway-url", PROXY, "codex", "--model", "auto"],
            obj={"secret_vault": vault},
        )
        assert configured.exit_code == 0, configured.output
        reconfigured: Final = runner.invoke(
            cli, ["reconfigure", "codex", "--model", "auto"], obj={"secret_vault": vault}
        )
        assert reconfigured.exit_code == 0, reconfigured.output
        fake_codex_version(None, 0)
        undone = runner.invoke(cli, ["unconfigure", "codex"], obj={"secret_vault": vault})
        assert undone.exit_code == 0, undone.output
        assert vault.reads == 0 and vault.writes == [] and vault.erases == 0
        assert not codex_path.exists() and not paths[0].exists()
        assert "Removed" in undone.output
        assert "sk-login" not in missing.output + configured.output + reconfigured.output + undone.output


class TestUnconfigureClaude:
    @responses.activate
    def test_restores_the_original_file_and_removes_the_receipt(self, runner, paths):
        _mock_models()
        settings_path, state_path = paths
        settings_path.parent.mkdir(parents=True)
        original = {"theme": "dark", "model": "claude-opus-5"}
        settings_path.write_text(json.dumps(original))
        assert _configure(runner, "--api-key", VALID_KEY, "--model", "claude-auto").exit_code == 0

        result = runner.invoke(cli, ["unconfigure", "claude"])
        assert result.exit_code == 0, result.output
        assert json.loads(settings_path.read_text()) == original
        assert not state_path.exists()
        assert "Restored in" in result.output and "model" in result.output
        assert "ANTHROPIC_API_KEY" not in result.output, "a key that never existed was not restored"

    @responses.activate
    def test_a_file_only_configure_created_is_reported_removed_not_restored(self, runner, paths):
        _mock_models()
        settings_path, _ = paths
        assert _configure(runner, "--api-key", VALID_KEY).exit_code == 0
        result = runner.invoke(cli, ["unconfigure", "claude"])
        assert result.exit_code == 0, result.output
        assert not settings_path.exists()
        assert "No settings file remains" in result.output and "Restored" not in result.output

    @responses.activate
    def test_says_when_nothing_was_still_ours_and_names_what_it_kept(self, runner, paths):
        _mock_models()
        settings_path, _ = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({"theme": "dark"}))
        assert _configure(runner, "--api-key", VALID_KEY, "--model", "claude-auto").exit_code == 0
        edited = json.loads(settings_path.read_text())
        edited["env"] = {key: f"{value}-edited" for key, value in edited["env"].items()}
        edited["model"] = "mine"
        edited["statusLine"] = {"type": "command", "command": "~/.claude/my-statusline.sh"}
        settings_path.write_text(json.dumps(edited))
        result = runner.invoke(cli, ["unconfigure", "claude"])
        assert result.exit_code == 0, result.output
        assert "Nothing in" in result.output and "was still ours to restore" in result.output
        assert "Left as you changed them since:" in result.output and "model" in result.output
        assert "statusLine" in result.output

    @responses.activate
    def test_names_the_server_a_withheld_credential_was_captured_with_and_keeps_the_receipt(self, runner, paths):
        _mock_models()
        settings_path, state_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://api.anthropic.com", "ANTHROPIC_API_KEY": "sk-ant"}})
        )
        assert _configure(runner, "--api-key", VALID_KEY).exit_code == 0
        edited = json.loads(settings_path.read_text())
        edited["env"]["ANTHROPIC_BASE_URL"] = "http://other-proxy:4000"
        settings_path.write_text(json.dumps(edited))
        result = runner.invoke(cli, ["unconfigure", "claude"])
        assert result.exit_code == 0, result.output
        assert "env.ANTHROPIC_API_KEY (captured with https://api.anthropic.com)" in result.output
        assert str(state_path) in result.output and state_path.exists()
        assert "sk-ant" not in result.output

    def test_disconnected_unconfigure_does_not_touch_lite_up_backup(
        self, runner: CliRunner, paths: tuple[Path, Path], lite_up_backup: Path
    ) -> None:
        result: Final = runner.invoke(cli, ["unconfigure", "claude"])
        assert result.exit_code == 0, result.output
        assert "nothing to undo" in result.output
        assert "Agent settings were not changed" in result.output
        assert lite_up_backup.read_text() == "{}"

    @responses.activate
    def test_a_config_dir_is_configured_and_undone_apart_from_the_default_file(
        self, runner, paths, monkeypatch, tmp_path, lite_up_backup
    ):
        _mock_models()
        default_settings, default_state = paths
        work_dir = tmp_path / "claude-work"
        work_dir.mkdir()
        original = {"theme": "dark"}
        (work_dir / "settings.json").write_text(json.dumps(original))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(work_dir))

        configured = _configure(runner, "--api-key", VALID_KEY, "--model", "claude-auto")
        assert configured.exit_code == 0, configured.output
        assert f"Configured Claude Code: {work_dir / 'settings.json'}" in configured.output
        assert json.loads((work_dir / "settings.json").read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == VALID_KEY
        assert not default_settings.exists() and not default_state.exists()

        undone = runner.invoke(cli, ["unconfigure", "claude"])
        assert undone.exit_code == 0, undone.output
        assert json.loads((work_dir / "settings.json").read_text()) == original
        assert not default_settings.exists() and not default_state.exists()
        assert runner.invoke(cli, ["unconfigure", "claude"]).exit_code == 0

    def test_without_a_receipt_it_reports_nothing_to_undo(self, runner, paths):
        result = runner.invoke(cli, ["unconfigure", "claude"])
        assert result.exit_code == 0, result.output
        assert "nothing to undo" in result.output.lower()


class TestClaudeCodeView:
    VIEW = {"anthropic-version": "2023-06-01", "x-gateway-client": "claude-code"}

    def _mock(self, rows):
        responses.get(
            f"{PROXY}/v1/models",
            json={"data": rows},
            match=[responses.matchers.header_matcher({"Authorization": f"Bearer {VALID_KEY}", **self.VIEW})],
        )

    @responses.activate
    @pytest.mark.parametrize(
        "model, pinned",
        [
            ("literal-claude-router-source", "emitted-literal"),
            ("marked-sibling", "emitted-marked[1m]"),
            ("emitted-collision", "emitted-source-priority"),
            ("emitted-only", "emitted-only"),
        ],
    )
    def test_pins_source_identity_before_emitted_id(self, runner, paths, model, pinned):
        self._mock(
            [
                {"id": "emitted-collision", "source_model": "other-source"},
                {"id": "emitted-source-priority", "source_model": "emitted-collision"},
                {"id": "emitted-marked[1m]", "source_model": "marked-sibling"},
                {"id": "emitted-literal", "source_model": "literal-claude-router-source"},
                {"id": "emitted-only"},
            ]
        )
        settings_path, _ = paths
        result = _configure(runner, "--api-key", VALID_KEY, "--model", model)
        assert result.exit_code == 0, result.output
        assert json.loads(settings_path.read_text())["model"] == pinned
        assert f"Starting model: {pinned}" in result.output
        assert len(responses.calls) == 1

    @responses.activate
    def test_refuses_unknown_short_suffix(self, runner, paths):
        self._mock([{"id": "emitted-router-source", "source_model": "literal-router-source"}])
        settings_path, _ = paths
        result = _configure(runner, "--api-key", VALID_KEY, "--model", "source")
        assert result.exit_code != 0
        assert "'source' is not served" in result.output
        assert not settings_path.exists()

    @responses.activate
    def test_interactive_picker_uses_source_names(self, paths):
        self._mock([{"id": "emitted", "source_model": "source"}])
        settings_path, _ = paths
        asked = {}
        ctx = click.Context(
            configure_group, obj={"base_url": PROXY, "api_key": VALID_KEY, "api_key_from_token_file": False}
        )

        def pick_model(listed):
            asked["listed"] = tuple(listed)
            return "source"

        interactive_configure(ctx, pick_targets=lambda: ("claude",), pick_model=pick_model)
        assert asked["listed"] == ("source",)
        assert json.loads(settings_path.read_text())["model"] == "emitted"

    @responses.activate
    def test_counts_what_an_older_proxy_lets_the_picker_show(self, runner, paths):
        _mock_models()
        result = _configure(runner, "--api-key", VALID_KEY)
        assert result.exit_code == 0, result.output
        assert "/model will list 1 of the proxy's 2 models: Claude Code shows only ids containing" in result.output


def _saved_profile_path(target: Literal["claude", "codex"], settings_path: Path) -> Path:
    from litellm.proxy.client.cli.commands.configure_profiles import setup_profile_path

    return setup_profile_path(target, settings_path)


def _configure_saved_agent(runner: CliRunner, target: Literal["claude", "codex"]) -> None:
    result: Final = runner.invoke(
        cli,
        ["configure", "--gateway-url", PROXY, "--api-key", VALID_KEY, target, "--model", "auto"],
    )
    assert result.exit_code == 0, result.output


def _agent_document(settings_path: Path) -> dict[str, JsonValue]:
    adapter: Final = TypeAdapter(dict[str, JsonValue])
    if settings_path.suffix == ".json":
        return adapter.validate_json(settings_path.read_text())
    return adapter.validate_python(tomlkit.parse(settings_path.read_text()).unwrap())


def _prompt_answer(answer: str | tuple[str, ...]) -> SimpleNamespace:
    def execute() -> str | tuple[str, ...]:
        return answer

    return SimpleNamespace(execute=execute)


class TestSavedAgentSetup:
    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    @pytest.mark.parametrize("resume", [("configure",), None], ids=["all", "target"])
    def test_disconnect_then_configure_reuses_connection_and_model_without_prompts(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        target: Literal["claude", "codex"],
        resume: tuple[str, ...] | None,
    ) -> None:
        _mock_agent_models()
        settings_path: Final = paths[0] if target == "claude" else codex_path
        _configure_saved_agent(runner, target)
        configured: Final = _agent_document(settings_path)
        undone: Final = runner.invoke(cli, ["unconfigure", target])
        assert undone.exit_code == 0, undone.output
        assert not settings_path.exists()

        resumed: Final = runner.invoke(cli, list(resume or ("configure", target)))
        assert resumed.exit_code == 0, resumed.output
        assert _agent_document(settings_path) == configured
        assert "lite configure" in undone.output and "saved" in undone.output.lower()
        repeated: Final = runner.invoke(cli, ["configure", target])
        assert repeated.exit_code == 0, repeated.output
        assert _agent_document(settings_path) == configured
        restored: Final = runner.invoke(cli, ["unconfigure", target])
        assert restored.exit_code == 0, restored.output
        assert not settings_path.exists()
        assert VALID_KEY not in resumed.output + repeated.output + restored.output

    @responses.activate
    def test_resume_both_agents_captures_the_settings_changed_while_disconnected(
        self, runner: CliRunner, paths: tuple[Path, Path], codex_path: Path
    ) -> None:
        _mock_agent_models()
        _configure_saved_agent(runner, "claude")
        codex_url: Final = "https://codex-gateway.test/prefix"
        responses.get(
            f"{codex_url}/v1/models",
            json={"data": [{"id": "auto"}]},
            match=[responses.matchers.header_matcher({"Authorization": "Bearer sk-codex"})],
        )
        codex_setup: Final = runner.invoke(
            cli,
            ["configure", "codex", "--gateway-url", codex_url, "--api-key", "sk-codex", "--model", "auto"],
        )
        assert codex_setup.exit_code == 0, codex_setup.output
        undone: Final = runner.invoke(cli, ["unconfigure"])
        assert undone.exit_code == 0, undone.output
        paths[0].write_text('{"theme": "light", "model": "personal-claude"}')
        codex_path.write_text('model = "personal-codex"\napproval_policy = "on-request"\n')

        resumed: Final = runner.invoke(cli, ["configure"])
        assert resumed.exit_code == 0, resumed.output
        assert json.loads(paths[0].read_text())["model"] == "claude-router-6175746f"
        assert tomlkit.parse(codex_path.read_text())["model"] == "auto"
        assert responses.calls[-1].request.url == f"{codex_url}/v1/models"
        restored: Final = runner.invoke(cli, ["unconfigure"])
        assert restored.exit_code == 0, restored.output
        assert json.loads(paths[0].read_text()) == {"theme": "light", "model": "personal-claude"}
        assert tomlkit.parse(codex_path.read_text()) == {
            "model": "personal-codex", "approval_policy": "on-request"
        }

    @responses.activate
    @pytest.mark.parametrize("disconnected", [False, True], ids=["active", "disconnected"])
    @pytest.mark.parametrize(
        "forget, forgotten",
        [
            (("unconfigure", "--forget", "claude"), ("claude",)),
            (("unconfigure", "codex", "--forget"), ("codex",)),
            (("unconfigure", "--forget"), ("claude", "codex")),
        ],
        ids=["group-option-target", "leaf-option", "all"],
    )
    def test_forget_removes_only_selected_saved_setups_even_after_disconnect(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        disconnected: bool,
        forget: tuple[str, ...],
        forgotten: tuple[str, ...],
    ) -> None:
        _mock_agent_models()
        _configure_saved_agent(runner, "claude")
        _configure_saved_agent(runner, "codex")
        if disconnected:
            undone: Final = runner.invoke(cli, ["unconfigure"])
            assert undone.exit_code == 0, undone.output
        result: Final = runner.invoke(cli, list(forget))
        assert result.exit_code == 0, result.output
        for target, settings_path in (("claude", paths[0]), ("codex", codex_path)):
            assert _saved_profile_path(target, settings_path).exists() == (target not in forgotten)
            resumed: Final = runner.invoke(cli, ["configure", target])
            assert (resumed.exit_code == 0) == (target not in forgotten), resumed.output
            assert settings_path.exists() == (target not in forgotten)

    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    @pytest.mark.parametrize("source", ["leaf", "global", "environment"])
    def test_saved_key_never_follows_a_gateway_override_without_a_replacement(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        target: Literal["claude", "codex"],
        source: str,
    ) -> None:
        _mock_agent_models()
        _configure_saved_agent(runner, target)
        undone: Final = runner.invoke(cli, ["unconfigure", target])
        assert undone.exit_code == 0, undone.output
        replacement_url: Final = "https://replacement.test/gateway"
        if source == "environment":
            monkeypatch.setenv("LITELLM_PROXY_URL", replacement_url)
        args: Final = (
            ["--base-url", replacement_url, "configure", target]
            if source == "global"
            else ["configure", target, "--gateway-url", replacement_url]
            if source == "leaf"
            else ["configure", target]
        )
        refused: Final = runner.invoke(cli, args)
        assert refused.exit_code != 0, refused.output
        assert "--api-key" in refused.output and VALID_KEY not in refused.output
        assert len(responses.calls) == 1
        assert not paths[0].exists() and not codex_path.exists()

        responses.get(
            f"{replacement_url}/v1/models",
            json={"data": [{"id": "auto"}]},
            match=[responses.matchers.header_matcher({"Authorization": "Bearer sk-replacement"})],
        )
        replaced: Final = runner.invoke(cli, [*args, "--api-key", "sk-replacement"])
        assert replaced.exit_code == 0, replaced.output
        assert len(responses.calls) == 2
        assert responses.calls[-1].request.url == f"{replacement_url}/v1/models"
        assert VALID_KEY not in replaced.output and "sk-replacement" not in replaced.output

    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    def test_saved_setup_is_private_and_scoped_to_the_resolved_agent_home(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        target: Literal["claude", "codex"],
    ) -> None:
        _mock_agent_models()
        _configure_saved_agent(runner, target)
        settings_path: Final = paths[0] if target == "claude" else codex_path
        profile_path: Final = _saved_profile_path(target, settings_path)
        assert stat.S_IMODE(profile_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(profile_path.parent.stat().st_mode) & 0o077 == 0
        undone: Final = runner.invoke(cli, ["unconfigure", target])
        assert undone.exit_code == 0, undone.output
        alternate_home: Final = tmp_path / f"other-{target}"
        alternate_settings: Final = alternate_home / settings_path.name
        environment: Final = "CLAUDE_CONFIG_DIR" if target == "claude" else "CODEX_HOME"
        monkeypatch.setenv(environment, str(alternate_home))
        missing: Final = runner.invoke(cli, ["configure", target])
        assert missing.exit_code != 0, missing.output
        assert not alternate_settings.exists() and len(responses.calls) == 1
        assert profile_path.exists()
        stored_url: Final = runner.invoke(cli, ["config", "set", "base_url", "https://other-default.test"])
        assert stored_url.exit_code == 0, stored_url.output
        monkeypatch.setenv(environment, str(settings_path.parent))
        resumed: Final = runner.invoke(cli, ["configure", target])
        assert resumed.exit_code == 0, resumed.output
        assert settings_path.exists() and not alternate_settings.exists()

    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    @pytest.mark.parametrize("fault", ["json", "version", "target", "path"])
    def test_invalid_saved_setup_fails_without_network_or_secret_output_and_can_be_forgotten(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        target: Literal["claude", "codex"],
        fault: str,
    ) -> None:
        _mock_agent_models()
        _configure_saved_agent(runner, target)
        settings_path: Final = paths[0] if target == "claude" else codex_path
        profile_path: Final = _saved_profile_path(target, settings_path)
        profile: Final = TypeAdapter(dict[str, JsonValue]).validate_json(profile_path.read_text())
        corrupted: Final = (
            "{ " + VALID_KEY
            if fault == "json"
            else json.dumps({**profile, "version": 999})
            if fault == "version"
            else json.dumps({**profile, "target": "codex" if target == "claude" else "claude"})
            if fault == "target"
            else json.dumps({**profile, "settings_path": str(settings_path.parent / "another-file")})
        )
        undone: Final = runner.invoke(cli, ["unconfigure", target])
        assert undone.exit_code == 0, undone.output
        profile_path.write_text(corrupted)
        failed: Final = runner.invoke(cli, ["configure", target])
        assert failed.exit_code != 0, failed.output
        assert "saved" in failed.output.lower() and "--forget" in failed.output
        assert VALID_KEY not in failed.output
        assert not settings_path.exists() and len(responses.calls) == 1
        forgotten: Final = runner.invoke(cli, ["unconfigure", "--forget", target])
        assert forgotten.exit_code == 0, forgotten.output
        assert not profile_path.exists() and not settings_path.exists()

    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    def test_reconfigure_prefills_saved_choices_and_changes_only_the_selected_agent(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        target: Literal["claude", "codex"],
    ) -> None:
        _mock_agent_models()
        _configure_saved_agent(runner, "claude")
        _configure_saved_agent(runner, "codex")
        untouched: Final = codex_path if target == "claude" else paths[0]
        before: Final = untouched.read_bytes()
        responses.replace(
            responses.GET, f"{PROXY}/v1/models", json={"data": [{"id": "auto"}, {"id": "replacement"}]}
        )

        def checkbox(**kwargs: object) -> SimpleNamespace:
            choices: Final = kwargs["choices"]
            assert isinstance(choices, list) and len(choices) == 2
            for choice in choices:
                assert isinstance(choice, Choice) and choice.enabled
            return _prompt_answer((target,))

        def fuzzy(**kwargs: object) -> SimpleNamespace:
            assert kwargs["default"] == "auto"
            return _prompt_answer("replacement")

        monkeypatch.setattr(configure_module.inquirer, "checkbox", checkbox)
        monkeypatch.setattr(configure_module.inquirer, "fuzzy", fuzzy)
        changed: Final = runner.invoke(cli, ["reconfigure"], input=_TerminalInput(b"\n\n"))
        assert changed.exit_code == 0, changed.output
        assert PROXY in changed.output and VALID_KEY not in changed.output
        assert untouched.read_bytes() == before
        undone: Final = runner.invoke(cli, ["unconfigure", target])
        assert undone.exit_code == 0, undone.output
        resumed: Final = runner.invoke(cli, ["configure", target])
        assert resumed.exit_code == 0, resumed.output
        if target == "claude":
            assert json.loads(paths[0].read_text())["model"] == "replacement"
        else:
            assert tomlkit.parse(codex_path.read_text())["model"] == "replacement"
        assert untouched.read_bytes() == before

    @responses.activate
    def test_reconfigure_cancel_preserves_every_agents_settings_and_saved_choices(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _mock_agent_models()
        _configure_saved_agent(runner, "claude")
        _configure_saved_agent(runner, "codex")
        files: Final = (
            paths[0], codex_path, _saved_profile_path("claude", paths[0]), _saved_profile_path("codex", codex_path)
        )
        before: Final = tuple(path.read_bytes() for path in files)

        def checkbox(**kwargs: object) -> SimpleNamespace:
            return _prompt_answer(("claude", "codex"))

        def fuzzy(**kwargs: object) -> SimpleNamespace:
            assert tuple(path.read_bytes() for path in files) == before
            if "Codex" in str(kwargs["message"]):
                raise KeyboardInterrupt()
            return _prompt_answer("Keep Claude Code's own default")

        monkeypatch.setattr(configure_module.inquirer, "checkbox", checkbox)
        monkeypatch.setattr(configure_module.inquirer, "fuzzy", fuzzy)
        cancelled: Final = runner.invoke(cli, ["reconfigure"], input=_TerminalInput(b"\n\n\n\n"))
        assert cancelled.exit_code != 0, cancelled.output
        assert "Aborted" in cancelled.output
        assert tuple(path.read_bytes() for path in files) == before

    @responses.activate
    def test_explicit_default_model_unpins_claude_and_remains_the_saved_choice(
        self, runner: CliRunner, paths: tuple[Path, Path]
    ) -> None:
        _mock_agent_models()
        _configure_saved_agent(runner, "claude")
        changed: Final = runner.invoke(cli, ["reconfigure", "claude", "--default-model"])
        assert changed.exit_code == 0, changed.output
        assert "model" not in json.loads(paths[0].read_text())
        undone: Final = runner.invoke(cli, ["unconfigure", "claude"])
        assert undone.exit_code == 0, undone.output
        resumed: Final = runner.invoke(cli, ["configure", "claude"])
        assert resumed.exit_code == 0, resumed.output
        assert "model" not in json.loads(paths[0].read_text())

    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    @pytest.mark.parametrize("fault", ["key", "model"])
    def test_terminal_resume_repairs_only_the_rejected_saved_choice(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        target: Literal["claude", "codex"],
        fault: str,
    ) -> None:
        _mock_agent_models()
        _configure_saved_agent(runner, target)
        settings_path: Final = paths[0] if target == "claude" else codex_path
        profile_path: Final = _saved_profile_path(target, settings_path)
        before: Final = profile_path.read_bytes()
        undone: Final = runner.invoke(cli, ["unconfigure", target])
        assert undone.exit_code == 0, undone.output
        responses.reset()
        if fault == "key":
            responses.get(
                f"{PROXY}/v1/models", status=401,
                match=[responses.matchers.header_matcher({"Authorization": f"Bearer {VALID_KEY}"})],
            )
        responses.get(
            f"{PROXY}/v1/models",
            json={"data": [{"id": "auto" if fault == "key" else "replacement"}]},
            match=[responses.matchers.header_matcher({
                "Authorization": "Bearer sk-repaired" if fault == "key" else f"Bearer {VALID_KEY}"
            })],
        )
        failed: Final = runner.invoke(cli, ["configure", target])
        assert failed.exit_code != 0, failed.output
        assert not settings_path.exists() and profile_path.read_bytes() == before
        assert len(responses.calls) == 1

        def checkbox(**kwargs: object) -> SimpleNamespace:
            raise AssertionError("Saved resume must not ask which agents to configure")

        def fuzzy(**kwargs: object) -> SimpleNamespace:
            assert fault == "model", "A rejected key must not discard the saved model"
            return _prompt_answer("replacement")

        monkeypatch.setattr(configure_module.inquirer, "checkbox", checkbox)
        monkeypatch.setattr(configure_module.inquirer, "fuzzy", fuzzy)
        resumed: Final = runner.invoke(
            cli, ["configure"], input=_TerminalInput(b"sk-repaired\n" if fault == "key" else b"")
        )
        assert resumed.exit_code == 0, resumed.output
        assert "gateway URL" not in resumed.output
        assert VALID_KEY not in resumed.output and "sk-repaired" not in resumed.output
        assert settings_path.exists()
        saved: Final = TypeAdapter(dict[str, JsonValue]).validate_json(profile_path.read_text())
        assert saved["api_key"] == ("sk-repaired" if fault == "key" else VALID_KEY)
        assert saved["model"] == ("auto" if fault == "key" else "replacement")

    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    @pytest.mark.parametrize("lost_receipt", [False, True], ids=["malformed-settings", "lost-receipt"])
    def test_forget_without_receipt_preserves_agent_settings_and_reports_unknown_connection(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        target: Literal["claude", "codex"],
        lost_receipt: bool,
    ) -> None:
        from litellm.proxy.client.cli.commands.configure_profiles import receipt_path_for

        _mock_agent_models()
        _configure_saved_agent(runner, target)
        settings_path: Final = paths[0] if target == "claude" else codex_path
        profile_path: Final = _saved_profile_path(target, settings_path)
        if lost_receipt:
            receipt_path_for(target, settings_path).unlink()
        else:
            undone: Final = runner.invoke(cli, ["unconfigure", target])
            assert undone.exit_code == 0, undone.output
            settings_path.write_text("[invalid")
        before: Final = settings_path.read_bytes()
        forgotten: Final = runner.invoke(cli, ["unconfigure", target, "--forget"])
        assert forgotten.exit_code == 0, forgotten.output
        assert not profile_path.exists()
        assert settings_path.read_bytes() == before
        assert "Cannot confirm disconnection" in forgotten.output
        assert "gateway connection and key manually" in forgotten.output
        assert str(settings_path) in forgotten.output
        assert "already disconnected" not in forgotten.output and VALID_KEY not in forgotten.output
        assert len(responses.calls) == 1

    @responses.activate
    @pytest.mark.parametrize("target", ["claude", "codex"])
    @pytest.mark.parametrize("failure", ["stage_private_json", "commit_staged_json", "apply"])
    def test_failed_setup_write_preserves_saved_intent_and_plain_configure_retries_it(
        self,
        runner: CliRunner,
        paths: tuple[Path, Path],
        codex_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        target: Literal["claude", "codex"],
        failure: str,
    ) -> None:
        from litellm.proxy.client.cli.commands import configure_profiles, configure_setup

        _mock_agent_models()
        _configure_saved_agent(runner, target)
        settings_path: Final = paths[0] if target == "claude" else codex_path
        profile_path: Final = _saved_profile_path(target, settings_path)
        receipt_path: Final = configure_profiles.receipt_path_for(target, settings_path)
        before: Final = (settings_path.read_bytes(), profile_path.read_bytes(), receipt_path.read_bytes())
        original_settings: Final = _agent_document(settings_path)
        original_profile: Final = TypeAdapter(dict[str, JsonValue]).validate_json(profile_path.read_text())
        replacement_url: Final = "https://replacement.test/gateway"
        replacement_key: Final = "sk-replacement"
        responses.get(
            f"{replacement_url}/v1/models",
            json={"data": [{"id": "replacement"}]},
            match=[responses.matchers.header_matcher({"Authorization": f"Bearer {replacement_key}"})],
        )

        def fail_write(*args: object, **kwargs: object) -> str:
            raise OSError(f"simulated disk error {VALID_KEY}")

        def fail_apply(*args: object, **kwargs: object) -> None:
            error: Final = (
                configure_setup.ClaudeSettingsError if target == "claude" else configure_setup.CodexSettingsError
            )
            raise error("simulated agent settings write failure")

        with monkeypatch.context() as patch:
            if failure == "apply":
                patch.setattr(configure_setup, f"configure_{target}_settings", fail_apply)
            else:
                patch.setattr(configure_profiles, failure, fail_write)
            failed: Final = runner.invoke(
                cli,
                [
                    "reconfigure", target, "--gateway-url", replacement_url,
                    "--api-key", replacement_key, "--model", "replacement",
                ],
            )
        assert failed.exit_code != 0, failed.output
        assert VALID_KEY not in failed.output and replacement_key not in failed.output
        assert (settings_path.read_bytes(), receipt_path.read_bytes()) == (before[0], before[2])
        saved: Final = TypeAdapter(dict[str, JsonValue]).validate_json(profile_path.read_text())
        if failure == "apply":
            assert saved == {
                **original_profile, "base_url": replacement_url, "api_key": replacement_key, "model": "replacement"
            }
            assert "simulated agent settings write failure" in failed.output
            assert "setup was saved" in failed.output and f"lite configure {target}" in failed.output
        else:
            assert "could not save" in failed.output.lower()
            assert profile_path.read_bytes() == before[1]
        retried: Final = runner.invoke(cli, ["configure", target])
        assert retried.exit_code == 0, retried.output
        written: Final = _agent_document(settings_path)
        if failure != "apply":
            assert written == original_settings
        elif target == "claude":
            environment: Final = written["env"]
            assert isinstance(environment, dict)
            assert (environment["ANTHROPIC_BASE_URL"], environment["ANTHROPIC_AUTH_TOKEN"], written["model"]) == (
                replacement_url, replacement_key, "replacement"
            )
        else:
            providers: Final = written["model_providers"]
            assert isinstance(providers, dict)
            provider: Final = providers["litellm"]
            assert isinstance(provider, dict)
            headers: Final = provider["http_headers"]
            assert isinstance(headers, dict)
            assert (provider["base_url"], headers["Authorization"], written["model"]) == (
                f"{replacement_url}/v1", f"Bearer {replacement_key}", "replacement"
            )

    @responses.activate
    def test_contended_setup_lock_blocks_requests_and_agent_writes(
        self, runner: CliRunner, paths: tuple[Path, Path]
    ) -> None:
        from litellm.proxy.client.cli.commands.configure_profiles import setup_locks

        _mock_agent_models()
        with setup_locks(("claude",)):
            blocked: Final = runner.invoke(
                cli,
                ["configure", "claude", "--gateway-url", PROXY, "--api-key", VALID_KEY, "--model", "auto"],
            )
        assert blocked.exit_code != 0, blocked.output
        assert "Could not lock agent setup" in blocked.output
        assert len(responses.calls) == 0
        assert not paths[0].exists() and not paths[1].exists()
        assert not _saved_profile_path("claude", paths[0]).exists()
