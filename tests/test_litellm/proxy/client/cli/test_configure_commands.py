import json
import os
import stat

import click
import pytest
import requests
import responses
from click.testing import CliRunner

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

    def test_refuses_while_lite_up_holds_a_backup(self, runner, paths, lite_up_backup):
        result = runner.invoke(cli, ["unconfigure", "claude"])
        assert result.exit_code != 0 and "lite down" in result.output

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
        assert runner.invoke(cli, ["unconfigure", "claude"]).exit_code != 0, "the receipt is gone with the undo"

    def test_without_a_receipt_it_fails_loudly(self, runner, paths):
        result = runner.invoke(cli, ["unconfigure", "claude"])
        assert result.exit_code != 0
        assert "nothing to undo" in result.output


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
