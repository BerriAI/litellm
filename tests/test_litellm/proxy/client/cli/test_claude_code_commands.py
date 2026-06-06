import os
import sys
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from litellm.proxy.client.cli.commands.claude_code import (
    ClaudeCodeNotFoundError,
    build_claude_env,
    claude_code,
    launch_claude_code,
)

CLAUDE_CODE_MODULE = "litellm.proxy.client.cli.commands.claude_code"


class TestBuildClaudeEnv:
    def test_sets_base_url_and_bearer_token(self):
        env = build_claude_env({}, "http://localhost:4000/", "sk-litellm-123")

        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-litellm-123"

    def test_drops_existing_anthropic_api_key(self):
        env = build_claude_env(
            {"ANTHROPIC_API_KEY": "real-anthropic-key"},
            "http://localhost:4000",
            "sk-litellm-123",
        )

        assert "ANTHROPIC_API_KEY" not in env

    def test_omits_model_overrides_by_default(self):
        env = build_claude_env({}, "http://localhost:4000", "sk-litellm-123")

        assert "ANTHROPIC_MODEL" not in env
        assert "ANTHROPIC_SMALL_FAST_MODEL" not in env

    def test_sets_model_overrides_when_provided(self):
        env = build_claude_env(
            {},
            "http://localhost:4000",
            "sk-litellm-123",
            model="claude-sonnet-proxy",
            small_fast_model="claude-haiku-proxy",
        )

        assert env["ANTHROPIC_MODEL"] == "claude-sonnet-proxy"
        assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-haiku-proxy"

    def test_preserves_unrelated_env_and_does_not_mutate_input(self):
        base = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "real-anthropic-key"}

        env = build_claude_env(base, "http://localhost:4000", "sk-litellm-123")

        assert env["PATH"] == "/usr/bin"
        assert base == {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "real-anthropic-key"}


class TestLaunchClaudeCode:
    def test_launches_resolved_binary_with_wired_env(self):
        calls = {}

        def fake_launcher(path, args, env):
            calls["path"] = path
            calls["args"] = tuple(args)
            calls["env"] = dict(env)

        launch_claude_code(
            "http://localhost:4000/",
            "sk-litellm-123",
            model="claude-sonnet-proxy",
            small_fast_model="claude-haiku-proxy",
            claude_args=["--resume"],
            base_env={"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "leaked"},
            which=lambda name: "/usr/local/bin/claude",
            launcher=fake_launcher,
        )

        assert calls["path"] == "/usr/local/bin/claude"
        assert calls["args"] == ("--resume",)
        env = calls["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-litellm-123"
        assert env["ANTHROPIC_MODEL"] == "claude-sonnet-proxy"
        assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-haiku-proxy"
        assert "ANTHROPIC_API_KEY" not in env
        assert env["PATH"] == "/usr/bin"

    def test_resolves_binary_by_name(self):
        seen = {}

        def fake_which(name):
            seen["name"] = name
            return "/usr/local/bin/claude"

        launch_claude_code(
            "http://localhost:4000",
            "sk-litellm-123",
            base_env={},
            which=fake_which,
            launcher=lambda *a: None,
        )

        assert seen["name"] == "claude"

    def test_raises_when_binary_missing(self):
        launched = []

        with pytest.raises(ClaudeCodeNotFoundError):
            launch_claude_code(
                "http://localhost:4000",
                "sk-litellm-123",
                base_env={},
                which=lambda name: None,
                launcher=lambda *a: launched.append(a),
            )

        assert launched == []


class TestClaudeCodeCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def test_launches_with_stored_key_and_forwards_args(self):
        captured = {}

        def fake_launch(base_url, api_key, *, model, small_fast_model, claude_args):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["model"] = model
            captured["small_fast_model"] = small_fast_model
            captured["claude_args"] = tuple(claude_args)

        with patch(f"{CLAUDE_CODE_MODULE}.launch_claude_code", side_effect=fake_launch):
            result = self.runner.invoke(
                claude_code,
                ["--model", "claude-sonnet-proxy", "--", "--resume"],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-litellm-123"},
            )

        assert result.exit_code == 0, result.output
        assert captured["api_key"] == "sk-litellm-123"
        assert captured["base_url"] == "http://localhost:4000"
        assert captured["model"] == "claude-sonnet-proxy"
        assert captured["claude_args"] == ("--resume",)

    def test_logs_in_when_no_key_then_launches(self):
        captured = {}

        @click.command()
        def fake_login():
            pass

        def fake_launch(base_url, api_key, *, model, small_fast_model, claude_args):
            captured["api_key"] = api_key

        with (
            patch(f"{CLAUDE_CODE_MODULE}.login", fake_login),
            patch(
                f"{CLAUDE_CODE_MODULE}.get_stored_api_key",
                return_value="sk-after-login",
            ) as mock_get,
            patch(f"{CLAUDE_CODE_MODULE}.launch_claude_code", side_effect=fake_launch),
        ):
            result = self.runner.invoke(
                claude_code,
                [],
                obj={"base_url": "http://localhost:4000", "api_key": None},
            )

        assert result.exit_code == 0, result.output
        assert captured["api_key"] == "sk-after-login"
        mock_get.assert_called_once_with(expected_base_url="http://localhost:4000")

    def test_errors_when_login_does_not_yield_key(self):
        @click.command()
        def fake_login():
            pass

        with (
            patch(f"{CLAUDE_CODE_MODULE}.login", fake_login),
            patch(f"{CLAUDE_CODE_MODULE}.get_stored_api_key", return_value=None),
            patch(f"{CLAUDE_CODE_MODULE}.launch_claude_code") as mock_launch,
        ):
            result = self.runner.invoke(
                claude_code,
                [],
                obj={"base_url": "http://localhost:4000", "api_key": None},
            )

        assert result.exit_code != 0
        assert "Could not obtain a LiteLLM API key" in result.output
        mock_launch.assert_not_called()

    def test_reports_missing_binary_as_click_error(self):
        with patch(
            f"{CLAUDE_CODE_MODULE}.launch_claude_code",
            side_effect=ClaudeCodeNotFoundError("claude not on PATH"),
        ):
            result = self.runner.invoke(
                claude_code,
                [],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-litellm-123"},
            )

        assert result.exit_code != 0
        assert "claude not on PATH" in result.output
