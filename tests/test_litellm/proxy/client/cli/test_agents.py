import os
import sys
from unittest.mock import patch

import click
import pytest
import requests
from click.testing import CliRunner

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from litellm.proxy.client.cli.commands.agents import (
    AgentRunError,
    agent_commands,
    agent_launch_args,
    agent_profile,
    build_agent_env,
    run_agent,
    verify_proxy_key,
)

AGENTS_MODULE = "litellm.proxy.client.cli.commands.agents"


def _agent_command(name):
    return next(c for c in agent_commands() if c.name == name)


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class TestAgentProfile:
    def test_claude_is_anthropic(self):
        name, profiles = agent_profile("claude")
        assert name == "Claude Code"
        assert profiles == frozenset({"anthropic"})

    def test_claude_full_path_uses_basename(self):
        name, profiles = agent_profile("/usr/local/bin/claude")
        assert name == "Claude Code"
        assert profiles == frozenset({"anthropic"})

    def test_codex_and_opencode_are_openai(self):
        assert agent_profile("codex") == ("Codex", frozenset({"openai"}))
        assert agent_profile("opencode") == ("OpenCode", frozenset({"openai"}))

    def test_unknown_command_gets_both_profiles(self):
        name, profiles = agent_profile("mytool")
        assert name == "mytool"
        assert profiles == frozenset({"anthropic", "openai"})


class TestBuildAgentEnv:
    def test_anthropic_profile_uses_bare_root_and_bearer(self):
        env = build_agent_env(
            {}, "http://localhost:4000/", "sk-key", frozenset({"anthropic"})
        )
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-key"
        assert "OPENAI_BASE_URL" not in env
        assert "OPENAI_API_KEY" not in env

    def test_anthropic_profile_drops_existing_api_key(self):
        env = build_agent_env(
            {"ANTHROPIC_API_KEY": "real-key"},
            "http://localhost:4000",
            "sk-key",
            frozenset({"anthropic"}),
        )
        assert "ANTHROPIC_API_KEY" not in env

    def test_openai_profile_appends_v1(self):
        env = build_agent_env(
            {}, "http://localhost:4000/", "sk-key", frozenset({"openai"})
        )
        assert env["OPENAI_BASE_URL"] == "http://localhost:4000/v1"
        assert env["OPENAI_API_KEY"] == "sk-key"
        assert "ANTHROPIC_BASE_URL" not in env

    def test_both_profiles_set_everything(self):
        env = build_agent_env(
            {}, "http://localhost:4000", "sk-key", frozenset({"anthropic", "openai"})
        )
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert env["OPENAI_BASE_URL"] == "http://localhost:4000/v1"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-key"
        assert env["OPENAI_API_KEY"] == "sk-key"

    def test_preserves_unrelated_env_and_does_not_mutate_input(self):
        base = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "real-key"}
        env = build_agent_env(
            base, "http://localhost:4000", "sk-key", frozenset({"anthropic"})
        )
        assert env["PATH"] == "/usr/bin"
        assert base == {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "real-key"}


class TestAgentLaunchArgs:
    def test_claude_and_opencode_get_no_extra_args(self):
        assert agent_launch_args("claude", "http://localhost:4000") == []
        assert agent_launch_args("opencode", "http://localhost:4000") == []

    def test_unknown_agent_gets_no_extra_args(self):
        assert agent_launch_args("mytool", "http://localhost:4000") == []

    def test_codex_points_provider_at_proxy_over_http(self):
        args = agent_launch_args("codex", "http://localhost:4000/")
        joined = " ".join(args)
        assert 'model_provider="litellm"' in args
        assert 'model_providers.litellm.base_url="http://localhost:4000/v1"' in args
        assert 'model_providers.litellm.env_key="OPENAI_API_KEY"' in args
        assert 'model_providers.litellm.wire_api="responses"' in args
        assert "model_providers.litellm.supports_websockets=false" in args
        assert joined.count("-c") == 6

    def test_codex_uses_basename(self):
        assert agent_launch_args("/usr/local/bin/codex", "http://localhost:4000") == (
            agent_launch_args("codex", "http://localhost:4000")
        )


class TestVerifyProxyKey:
    def test_ok_status_passes_and_uses_models_endpoint(self):
        captured = {}

        def fake_get(url, headers, timeout):
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse(200)

        verify_proxy_key("http://localhost:4000/", "sk-key", get=fake_get)

        assert captured["url"] == "http://localhost:4000/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer sk-key"}

    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_key_raises(self, status):
        with pytest.raises(AgentRunError, match="rejected your key"):
            verify_proxy_key(
                "http://localhost:4000",
                "sk-key",
                get=lambda *a, **k: _FakeResponse(status),
            )

    def test_unreachable_proxy_raises(self):
        def boom(*a, **k):
            raise requests.ConnectionError("refused")

        with pytest.raises(AgentRunError, match="Could not reach"):
            verify_proxy_key("http://localhost:4000", "sk-key", get=boom)

    def test_other_non_2xx_is_tolerated(self):
        verify_proxy_key(
            "http://localhost:4000",
            "sk-key",
            get=lambda *a, **k: _FakeResponse(500),
        )


class TestRunAgent:
    def test_wires_env_and_launches_resolved_binary(self):
        calls = {}

        def fake_launcher(path, args, env):
            calls["path"] = path
            calls["args"] = tuple(args)
            calls["env"] = dict(env)

        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["claude", "--resume"],
            base_env={"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "leaked"},
            which=lambda name: "/usr/local/bin/claude",
            verify=lambda *a: None,
            launcher=fake_launcher,
        )

        assert calls["path"] == "/usr/local/bin/claude"
        assert calls["args"] == ("claude", "--resume")
        env = calls["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-key"
        assert "ANTHROPIC_API_KEY" not in env
        assert "OPENAI_BASE_URL" not in env

    def test_codex_gets_openai_env(self):
        calls = {}
        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["codex"],
            base_env={},
            which=lambda name: "/usr/local/bin/codex",
            verify=lambda *a: None,
            launcher=lambda p, a, e: calls.update(env=dict(e)),
        )
        assert calls["env"]["OPENAI_BASE_URL"] == "http://localhost:4000/v1"
        assert calls["env"]["OPENAI_API_KEY"] == "sk-key"
        assert "ANTHROPIC_BASE_URL" not in calls["env"]

    def test_codex_injects_proxy_provider_args_before_user_args(self):
        calls = {}
        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["codex", "exec", "do a thing"],
            base_env={},
            which=lambda name: "/usr/local/bin/codex",
            verify=lambda *a: None,
            launcher=lambda p, a, e: calls.update(args=tuple(a)),
        )
        args = calls["args"]
        assert args[0] == "codex"
        assert args[-2:] == ("exec", "do a thing")
        assert 'model_provider="litellm"' in args
        assert 'model_providers.litellm.base_url="http://localhost:4000/v1"' in args
        # overrides must precede the codex subcommand so codex parses them
        assert args.index('model_provider="litellm"') < args.index("exec")

    def test_claude_launches_without_injected_args(self):
        calls = {}
        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["claude", "--resume"],
            base_env={},
            which=lambda name: "/usr/local/bin/claude",
            verify=lambda *a: None,
            launcher=lambda p, a, e: calls.update(args=tuple(a)),
        )
        assert calls["args"] == ("claude", "--resume")

    def test_missing_binary_raises_with_install_hint(self):
        with pytest.raises(AgentRunError, match="claude.*Install it first"):
            run_agent(
                "http://localhost:4000",
                "sk-key",
                ["claude"],
                base_env={},
                which=lambda name: None,
                verify=lambda *a: None,
                launcher=lambda *a: None,
            )

    def test_skip_verify_does_not_call_verify(self):
        verified = []
        launched = []
        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["claude"],
            skip_verify=True,
            base_env={},
            which=lambda name: "/usr/local/bin/claude",
            verify=lambda *a: verified.append(a),
            launcher=lambda *a: launched.append(a),
        )
        assert verified == []
        assert len(launched) == 1

    def test_verify_failure_aborts_before_launch(self):
        launched = []

        def boom(*a):
            raise AgentRunError("rejected")

        with pytest.raises(AgentRunError):
            run_agent(
                "http://localhost:4000",
                "sk-key",
                ["claude"],
                base_env={},
                which=lambda name: "/usr/local/bin/claude",
                verify=boom,
                launcher=lambda *a: launched.append(a),
            )
        assert launched == []

    def test_empty_command_raises(self):
        with pytest.raises(AgentRunError):
            run_agent("http://localhost:4000", "sk-key", [])

    def test_reattach_terminal_runs_just_before_launch(self):
        order = []
        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["claude"],
            skip_verify=True,
            base_env={},
            which=lambda name: "/usr/local/bin/claude",
            launcher=lambda *a: order.append("launch"),
            reattach_terminal=lambda: order.append("reattach"),
        )
        assert order == ["reattach", "launch"]

    def test_no_reattach_terminal_by_default(self):
        order = []
        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["claude"],
            skip_verify=True,
            base_env={},
            which=lambda name: "/usr/local/bin/claude",
            launcher=lambda *a: order.append("launch"),
        )
        assert order == ["launch"]


class TestAgentCommands:
    def setup_method(self):
        self.runner = CliRunner()

    def test_one_command_per_known_agent(self):
        assert {c.name for c in agent_commands()} == {"claude", "codex", "opencode"}

    def test_claude_launches_with_stored_key_and_forwards_args(self):
        captured = {}

        def fake_run_agent(base_url, api_key, command, **kwargs):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["command"] = list(command)
            captured["skip_verify"] = kwargs.get("skip_verify")

        with patch(f"{AGENTS_MODULE}.run_agent", side_effect=fake_run_agent):
            result = self.runner.invoke(
                _agent_command("claude"),
                ["--resume", "-p", "hi"],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-key"},
            )

        assert result.exit_code == 0, result.output
        assert captured["api_key"] == "sk-key"
        assert captured["command"] == ["claude", "--resume", "-p", "hi"]
        assert captured["skip_verify"] is False
        assert (
            "routing Claude Code through proxy at http://localhost:4000"
            in result.output
        )

    def test_codex_shows_friendly_name(self):
        captured = {}
        with patch(
            f"{AGENTS_MODULE}.run_agent",
            side_effect=lambda b, k, c, **kw: captured.update(command=list(c)),
        ):
            result = self.runner.invoke(
                _agent_command("codex"),
                ["exec", "do a thing"],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-key"},
            )
        assert result.exit_code == 0, result.output
        assert captured["command"] == ["codex", "exec", "do a thing"]
        assert "routing Codex through proxy" in result.output

    def test_skip_verify_is_consumed_not_forwarded(self):
        captured = {}

        def fake_run_agent(base_url, api_key, command, **kwargs):
            captured["command"] = list(command)
            captured["skip_verify"] = kwargs.get("skip_verify")

        with patch(f"{AGENTS_MODULE}.run_agent", side_effect=fake_run_agent):
            result = self.runner.invoke(
                _agent_command("claude"),
                ["--skip-verify", "--resume"],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-key"},
            )

        assert result.exit_code == 0, result.output
        assert captured["skip_verify"] is True
        assert captured["command"] == ["claude", "--resume"]

    def test_non_interactive_without_key_errors_clearly(self):
        with (
            patch(f"{AGENTS_MODULE}._is_interactive", return_value=False),
            patch(f"{AGENTS_MODULE}.run_agent") as mock_run,
        ):
            result = self.runner.invoke(
                _agent_command("claude"),
                [],
                obj={"base_url": "http://localhost:4000", "api_key": None},
            )
        assert result.exit_code != 0
        assert "LITELLM_PROXY_API_KEY" in result.output
        mock_run.assert_not_called()

    def test_interactive_without_key_logs_in_then_launches(self):
        captured = {}

        @click.command()
        def fake_login():
            pass

        with (
            patch(f"{AGENTS_MODULE}._is_interactive", return_value=True),
            patch(f"{AGENTS_MODULE}.login", fake_login),
            patch(
                f"{AGENTS_MODULE}.get_stored_api_key", return_value="sk-after-login"
            ) as mock_get,
            patch(
                f"{AGENTS_MODULE}.run_agent",
                side_effect=lambda base_url, api_key, command, **k: captured.update(
                    api_key=api_key
                ),
            ),
        ):
            result = self.runner.invoke(
                _agent_command("claude"),
                [],
                obj={"base_url": "http://localhost:4000", "api_key": None},
            )

        assert result.exit_code == 0, result.output
        assert captured["api_key"] == "sk-after-login"
        mock_get.assert_called_once_with(expected_base_url="http://localhost:4000")

    def test_agent_run_error_becomes_click_error(self):
        with patch(
            f"{AGENTS_MODULE}.run_agent",
            side_effect=AgentRunError("could not reach proxy"),
        ):
            result = self.runner.invoke(
                _agent_command("claude"),
                [],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-key"},
            )
        assert result.exit_code != 0
        assert "could not reach proxy" in result.output

    def test_interactive_session_reattaches_terminal_before_handoff(self):
        from litellm.proxy.client.cli.commands.agents import (
            _restore_controlling_terminal,
        )

        captured = {}
        with (
            patch(f"{AGENTS_MODULE}._is_interactive", return_value=True),
            patch(
                f"{AGENTS_MODULE}.run_agent",
                side_effect=lambda b, k, c, **kw: captured.update(kw),
            ),
        ):
            result = self.runner.invoke(
                _agent_command("claude"),
                [],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-key"},
            )
        assert result.exit_code == 0, result.output
        assert captured["reattach_terminal"] is _restore_controlling_terminal

    def test_non_interactive_agent_mode_leaves_stdin_alone(self):
        captured = {}
        with (
            patch(f"{AGENTS_MODULE}._is_interactive", return_value=False),
            patch(
                f"{AGENTS_MODULE}.run_agent",
                side_effect=lambda b, k, c, **kw: captured.update(kw),
            ),
        ):
            result = self.runner.invoke(
                _agent_command("claude"),
                [],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-key"},
            )
        assert result.exit_code == 0, result.output
        assert captured["reattach_terminal"] is None
