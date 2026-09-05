import inspect
import json
import os
import sys
from unittest.mock import patch

import click
import pytest
import requests
from click.testing import CliRunner



from litellm.proxy.client.cli.commands.agents import (
    AgentRunError,
    ModelSyncSkipped,
    _hand_off,
    _replace_process,
    _spawn_and_wait,
    agent_commands,
    agent_launch_args,
    agent_model_sync_env,
    agent_profile,
    build_agent_env,
    opencode_model_sync_env,
    run_agent,
    verify_proxy_key,
)

AGENTS_MODULE = "litellm.proxy.client.cli.commands.agents"


def _agent_command(name):
    return next(c for c in agent_commands() if c.name == name)


def _default_of(func, param):
    return inspect.signature(func).parameters[param].default


class _FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self.content = json.dumps(body).encode() if body is not None else b""


class _Recorder:
    def __init__(self, returns=None):
        self.returns = returns
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.returns


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
        assert env["ENABLE_TOOL_SEARCH"] == "true"
        assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
        assert "OPENAI_BASE_URL" not in env
        assert "OPENAI_API_KEY" not in env

    def test_anthropic_profile_preserves_existing_gateway_model_discovery(self):
        env = build_agent_env(
            {"CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "0"},
            "http://localhost:4000",
            "sk-key",
            frozenset({"anthropic"}),
        )
        assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "0"

    def test_anthropic_profile_preserves_existing_tool_search(self):
        env = build_agent_env(
            {"ENABLE_TOOL_SEARCH": "false"},
            "http://localhost:4000",
            "sk-key",
            frozenset({"anthropic"}),
        )
        assert env["ENABLE_TOOL_SEARCH"] == "false"

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
        assert "ENABLE_TOOL_SEARCH" not in env
        assert "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY" not in env

    def test_both_profiles_set_everything(self):
        env = build_agent_env(
            {}, "http://localhost:4000", "sk-key", frozenset({"anthropic", "openai"})
        )
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert env["OPENAI_BASE_URL"] == "http://localhost:4000/v1"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-key"
        assert env["OPENAI_API_KEY"] == "sk-key"
        assert env["ENABLE_TOOL_SEARCH"] == "true"

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


class TestOpencodeModelSync:
    @staticmethod
    def _listing(*models):
        return {"object": "list", "data": list(models)}

    def _sync(self, listing, base_env=None, base_url="http://localhost:4000/"):
        captured = {}

        def fake_get(url, headers, timeout):
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse(200, listing)

        env = opencode_model_sync_env(base_env or {}, base_url, "sk-key", get=fake_get)
        return captured, env

    def test_declares_proxy_as_litellm_provider_with_listed_models(self):
        listing = self._listing(
            {"id": "gpt-5.5", "object": "model", "created": 1, "owned_by": "openai", "mode": "chat"},
            {"id": "claude-opus-4-7", "object": "model", "created": 1, "owned_by": "openai"},
        )
        captured, env = self._sync(listing)

        assert captured["url"] == "http://localhost:4000/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer sk-key"}
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        provider = config["provider"]["litellm"]
        assert provider["npm"] == "@ai-sdk/openai-compatible"
        assert provider["name"] == "LiteLLM"
        assert provider["options"] == {
            "baseURL": "http://localhost:4000/v1",
            "apiKey": "{env:OPENAI_API_KEY}",
        }
        assert provider["models"] == {
            "gpt-5.5": {"name": "gpt-5.5"},
            "claude-opus-4-7": {"name": "claude-opus-4-7"},
        }
        assert "sk-key" not in env["OPENCODE_CONFIG_CONTENT"]

    def test_token_limits_become_opencode_limits(self):
        listing = self._listing(
            {
                "id": "gpt-5.5",
                "object": "model",
                "created": 1,
                "owned_by": "openai",
                "max_input_tokens": 400000,
                "max_output_tokens": 128000,
            },
            {"id": "half", "object": "model", "created": 1, "owned_by": "openai", "max_input_tokens": 8192},
        )
        _, env = self._sync(listing)
        models = json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["litellm"]["models"]
        assert models["gpt-5.5"]["limit"] == {"context": 400000, "output": 128000}
        assert "limit" not in models["half"]

    def test_non_chat_models_are_left_out(self):
        listing = self._listing(
            {"id": "chat", "object": "model", "created": 1, "owned_by": "openai", "mode": "chat"},
            {"id": "resp", "object": "model", "created": 1, "owned_by": "openai", "mode": "responses"},
            {"id": "embed", "object": "model", "created": 1, "owned_by": "openai", "mode": "embedding"},
            {"id": "img", "object": "model", "created": 1, "owned_by": "openai", "mode": "image_generation"},
        )
        _, env = self._sync(listing)
        models = json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["litellm"]["models"]
        assert set(models) == {"chat", "resp"}

    def test_existing_config_content_is_left_alone(self):
        calls = []

        def fake_get(*a, **k):
            calls.append(a)
            return _FakeResponse(200, self._listing())

        result = opencode_model_sync_env(
            {"OPENCODE_CONFIG_CONTENT": "{}"}, "http://localhost:4000", "sk-key", get=fake_get
        )
        assert isinstance(result, ModelSyncSkipped)
        assert "OPENCODE_CONFIG_CONTENT" in result.reason
        assert calls == []

    def test_unreachable_proxy_is_reported_not_raised(self):
        def boom(*a, **k):
            raise requests.ConnectionError("refused")

        result = opencode_model_sync_env({}, "http://localhost:4000", "sk-key", get=boom)
        assert isinstance(result, ModelSyncSkipped)
        assert "refused" in result.reason

    def test_non_200_is_reported(self):
        result = opencode_model_sync_env(
            {}, "http://localhost:4000", "sk-key", get=lambda *a, **k: _FakeResponse(500)
        )
        assert isinstance(result, ModelSyncSkipped)
        assert "HTTP 500" in result.reason

    def test_unexpected_body_is_reported(self):
        result = opencode_model_sync_env(
            {}, "http://localhost:4000", "sk-key", get=lambda *a, **k: _FakeResponse(200, {"data": "nope"})
        )
        assert isinstance(result, ModelSyncSkipped)
        assert "unexpected body" in result.reason

    @pytest.mark.parametrize("command", ["claude", "codex", "/usr/bin/claude"])
    def test_only_opencode_syncs(self, command):
        def boom(*a, **k):
            raise AssertionError("no agent other than opencode should call the proxy")

        assert agent_model_sync_env(command, {}, "http://localhost:4000", "sk-key", False, get=boom) == {}

    def test_skip_verify_keeps_the_launch_offline(self):
        def boom(*a, **k):
            raise AssertionError("--skip-verify must not touch the proxy")

        result = agent_model_sync_env("opencode", {}, "http://localhost:4000", "sk-key", True, get=boom)
        assert isinstance(result, ModelSyncSkipped)
        assert "--skip-verify" in result.reason

    def test_full_path_opencode_syncs(self):
        listing = self._listing({"id": "m", "object": "model", "created": 1, "owned_by": "x"})
        env = agent_model_sync_env(
            "/opt/bin/opencode",
            {},
            "http://localhost:4000",
            "sk-key",
            False,
            get=lambda *a, **k: _FakeResponse(200, listing),
        )
        assert "m" in json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["litellm"]["models"]

    def test_default_http_client_is_requests_get(self):
        assert _default_of(agent_model_sync_env, "get") is requests.get
        assert _default_of(opencode_model_sync_env, "get") is requests.get


class TestRunAgent:
    def test_synced_model_config_reaches_the_agent_alongside_profile_env(self):
        calls = {}
        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["opencode"],
            base_env={"HOME": "/home/me"},
            sync_models=lambda *a: {"OPENCODE_CONFIG_CONTENT": '{"provider":{}}'},
            which=lambda name: "/usr/local/bin/opencode",
            verify=lambda *a: None,
            launcher=lambda p, a, e: calls.update(env=dict(e)),
        )
        assert calls["env"]["OPENCODE_CONFIG_CONTENT"] == '{"provider":{}}'
        assert calls["env"]["OPENAI_BASE_URL"] == "http://localhost:4000/v1"
        assert calls["env"]["OPENAI_API_KEY"] == "sk-key"
        assert calls["env"]["HOME"] == "/home/me"

    def test_sync_gets_the_launch_inputs_and_runs_after_verify(self):
        order = []
        calls = {}

        def fake_sync(command, base_env, base_url, api_key, skip_verify):
            order.append("sync")
            calls["args"] = (command, dict(base_env), base_url, api_key, skip_verify)
            return {"OPENCODE_CONFIG_CONTENT": '{"provider":{"litellm":{}}}'}

        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["opencode"],
            base_env={"HOME": "/home/me"},
            sync_models=fake_sync,
            which=lambda name: "/usr/local/bin/opencode",
            verify=lambda *a: order.append("verify"),
            launcher=lambda p, a, e: order.append("launch"),
        )
        assert order == ["verify", "sync", "launch"]
        assert calls["args"] == ("opencode", {"HOME": "/home/me"}, "http://localhost:4000", "sk-key", False)

    def test_unreachable_proxy_is_not_asked_for_models(self):
        def failing_verify(*a):
            raise AgentRunError("Could not reach the LiteLLM proxy")

        def boom(*a):
            raise AssertionError("a failed key check must not be followed by a model fetch")

        with pytest.raises(AgentRunError):
            run_agent(
                "http://localhost:4000",
                "sk-key",
                ["opencode"],
                base_env={},
                sync_models=boom,
                which=lambda name: "/usr/local/bin/opencode",
                verify=failing_verify,
                launcher=lambda *a: None,
            )

    def test_skip_verify_reaches_the_sync_which_reports_the_skip(self):
        warnings = []
        calls = {}

        def fake_sync(command, base_env, base_url, api_key, skip_verify):
            calls["skip_verify"] = skip_verify
            return ModelSyncSkipped("offline")

        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["opencode"],
            skip_verify=True,
            base_env={},
            sync_models=fake_sync,
            warn=warnings.append,
            which=lambda name: "/usr/local/bin/opencode",
            verify=lambda *a: pytest.fail("--skip-verify must not verify"),
            launcher=lambda p, a, e: calls.update(env=dict(e)),
        )
        assert calls["skip_verify"] is True
        assert "OPENCODE_CONFIG_CONTENT" not in calls["env"]
        assert warnings == ["litellm: not syncing OpenCode models from the proxy: offline"]

    def test_skipped_sync_still_launches_with_plain_openai_env(self):
        calls = {}
        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["opencode"],
            base_env={},
            sync_models=lambda *a: ModelSyncSkipped("proxy said no"),
            warn=lambda message: calls.setdefault("warned", message),
            which=lambda name: "/usr/local/bin/opencode",
            verify=lambda *a: None,
            launcher=lambda p, a, e: calls.update(env=dict(e)),
        )
        assert calls["env"]["OPENAI_BASE_URL"] == "http://localhost:4000/v1"
        assert "OPENCODE_CONFIG_CONTENT" not in calls["env"]
        assert "proxy said no" in calls["warned"]

    def test_non_opencode_agent_is_not_warned_about_model_sync(self):
        warnings = []
        run_agent(
            "http://localhost:4000",
            "sk-key",
            ["claude"],
            base_env={},
            warn=warnings.append,
            which=lambda name: "/usr/local/bin/claude",
            verify=lambda *a: None,
            launcher=lambda *a: None,
            sync_models=agent_model_sync_env,
        )
        assert warnings == []

    def test_default_sync_is_the_agent_model_sync(self):
        assert _default_of(run_agent, "sync_models") is agent_model_sync_env

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
        assert env["ENABLE_TOOL_SEARCH"] == "true"
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
        assert "ENABLE_TOOL_SEARCH" not in calls["env"]

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
        with pytest.raises(AgentRunError, match=r"claude.*Install it first"):
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


_WINDOWS_CLAUDE_EXE = "C:\\Program Files\\Claude\\claude.exe"
_WINDOWS_CLAUDE_CMD = "C:\\Users\\dev\\AppData\\Roaming\\npm\\claude.cmd"
_AGENT_ENV = {"ANTHROPIC_BASE_URL": "http://localhost:4000"}
_CMD_PREFIX = "cmd.exe /d /e:on /v:off /s /c "


def _shim_command_line(*args):
    spawn = _Recorder(returns=0)
    with pytest.raises(SystemExit):
        _hand_off(
            _WINDOWS_CLAUDE_CMD,
            ["claude", *args],
            _AGENT_ENV,
            platform="win32",
            replace=_Recorder(),
            spawn=spawn,
        )
    return spawn.calls[0][0]


class TestHandOff:
    def test_windows_spawns_child_instead_of_exec(self):
        replace = _Recorder()
        spawn = _Recorder(returns=0)

        with pytest.raises(SystemExit) as excinfo:
            _hand_off(
                _WINDOWS_CLAUDE_EXE,
                ["claude", "--resume"],
                _AGENT_ENV,
                platform="win32",
                replace=replace,
                spawn=spawn,
            )

        assert excinfo.value.code == 0
        assert replace.calls == []
        assert spawn.calls == [
            ((_WINDOWS_CLAUDE_EXE, "--resume"), _AGENT_ENV),
        ]

    @pytest.mark.parametrize("code", [1, 42, 130])
    def test_windows_propagates_child_exit_code(self, code):
        with pytest.raises(SystemExit) as excinfo:
            _hand_off(
                _WINDOWS_CLAUDE_EXE,
                ["claude"],
                _AGENT_ENV,
                platform="win32",
                replace=_Recorder(),
                spawn=_Recorder(returns=code),
            )
        assert excinfo.value.code == code

    @pytest.mark.parametrize(
        "path",
        [
            _WINDOWS_CLAUDE_CMD,
            "C:\\shims\\claude.CMD",
            "C:\\shims\\claude.bat",
        ],
    )
    def test_windows_batch_shim_goes_through_cmd_exe(self, path):
        spawn = _Recorder(returns=0)

        with pytest.raises(SystemExit):
            _hand_off(
                path,
                ["claude", "--resume"],
                _AGENT_ENV,
                platform="win32",
                replace=_Recorder(),
                spawn=spawn,
            )

        assert spawn.calls[0][0] == f'{_CMD_PREFIX}""{path}" "--resume""'

    def test_windows_shim_quotes_a_path_containing_spaces(self):
        spawn = _Recorder(returns=0)
        path = "C:\\Program Files\\npm\\claude.cmd"

        with pytest.raises(SystemExit):
            _hand_off(
                path,
                ["claude", "-p", "hello world"],
                _AGENT_ENV,
                platform="win32",
                replace=_Recorder(),
                spawn=spawn,
            )

        expected = f'{_CMD_PREFIX}""C:\\Program Files\\npm\\claude.cmd" "-p" "hello world""'
        assert spawn.calls[0][0] == expected

    @pytest.mark.parametrize("payload", ["a&calc", "a|calc", "a>out", "a^b", "a&&calc"])
    def test_windows_shim_never_leaves_a_metacharacter_unquoted(self, payload):
        expected = f'{_CMD_PREFIX}""{_WINDOWS_CLAUDE_CMD}" "-p" "{payload}""'
        assert _shim_command_line("-p", payload) == expected

    def test_windows_shim_doubles_an_embedded_quote(self):
        assert _shim_command_line("-p", 'say "hi"').endswith('"-p" "say ""hi""""')

    @pytest.mark.parametrize(
        "payload, quoted",
        [
            ("%PATH%", "%%cd:~,%PATH%%cd:~,%"),
            ("100%", "100%%cd:~,%"),
            ("%OS%%CD%", "%%cd:~,%OS%%cd:~,%%%cd:~,%CD%%cd:~,%"),
        ],
    )
    def test_windows_shim_stops_cmd_expanding_a_percent_variable(self, payload, quoted):
        assert _shim_command_line("-p", payload).endswith(f'"-p" "{quoted}""')

    def test_windows_shim_guards_a_percent_in_the_shim_path(self):
        spawn = _Recorder(returns=0)
        path = "C:\\dev%HOME%\\claude.cmd"

        with pytest.raises(SystemExit):
            _hand_off(
                path,
                ["claude"],
                _AGENT_ENV,
                platform="win32",
                replace=_Recorder(),
                spawn=spawn,
            )

        assert spawn.calls[0][0] == f'{_CMD_PREFIX}""C:\\dev%%cd:~,%HOME%%cd:~,%\\claude.cmd""'

    @pytest.mark.parametrize(
        "payload, quoted",
        [
            ("C:\\dir\\", "C:\\dir\\\\"),
            ('say \\"hi', 'say \\\\""hi'),
            ('a\\\\"b', 'a\\\\\\\\""b'),
        ],
    )
    def test_windows_shim_doubles_backslashes_that_precede_a_quote(self, payload, quoted):
        assert _shim_command_line("-p", payload).endswith(f'"-p" "{quoted}""')

    @pytest.mark.parametrize("payload", ["one\ntwo", "one\r\ntwo", "trailing\r"])
    def test_windows_shim_refuses_an_argument_holding_a_line_break(self, payload):
        with pytest.raises(AgentRunError, match="line break"):
            _hand_off(
                _WINDOWS_CLAUDE_CMD,
                ["claude", "-p", payload],
                _AGENT_ENV,
                platform="win32",
                replace=_Recorder(),
                spawn=_Recorder(returns=0),
            )

    def test_windows_shim_keeps_the_switches_the_quoting_depends_on(self):
        command = _shim_command_line("-p", "hi")
        assert command.startswith("cmd.exe ")
        switches = command.split(" /c ")[0].split()[1:]
        assert switches == ["/d", "/e:on", "/v:off", "/s"]

    def test_windows_exe_is_not_wrapped_in_cmd_exe(self):
        spawn = _Recorder(returns=0)
        with pytest.raises(SystemExit):
            _hand_off(
                _WINDOWS_CLAUDE_EXE,
                ["claude"],
                _AGENT_ENV,
                platform="win32",
                replace=_Recorder(),
                spawn=spawn,
            )
        assert spawn.calls[0][0] == (_WINDOWS_CLAUDE_EXE,)

    @pytest.mark.parametrize("platform", ["darwin", "linux", "freebsd8"])
    def test_posix_still_replaces_the_process(self, platform):
        replace = _Recorder()
        spawn = _Recorder(returns=0)

        _hand_off(
            "/usr/local/bin/claude",
            ["claude", "--resume"],
            _AGENT_ENV,
            platform=platform,
            replace=replace,
            spawn=spawn,
        )

        assert spawn.calls == []
        assert replace.calls == [
            ("/usr/local/bin/claude", ["claude", "--resume"], _AGENT_ENV),
        ]
        path, args, env = replace.calls[0]
        assert isinstance(args, list)
        assert isinstance(env, dict)

    def test_replace_process_calls_execvpe_with_argv_and_env(self):
        execvpe = _Recorder()

        _replace_process(
            "/usr/local/bin/claude",
            ("claude", "--resume"),
            _AGENT_ENV,
            execvpe=execvpe,
        )

        assert execvpe.calls == [
            ("/usr/local/bin/claude", ["claude", "--resume"], _AGENT_ENV),
        ]
        _path, argv, env = execvpe.calls[0]
        assert isinstance(argv, list)
        assert isinstance(env, dict)

    def test_posix_default_replacement_is_execvpe(self):
        assert _default_of(run_agent, "launcher") is _hand_off
        assert _default_of(_hand_off, "replace") is _replace_process
        assert _default_of(_replace_process, "execvpe") is os.execvpe
        assert _default_of(_hand_off, "spawn") is _spawn_and_wait
        assert _default_of(_hand_off, "platform") == sys.platform

    def test_spawn_and_wait_blocks_until_the_child_is_done(self, tmp_path):
        marker = tmp_path / "child-finished"
        script = (
            "import os, pathlib, time; time.sleep(0.5); "
            "pathlib.Path(os.environ['MARKER']).write_text('done'); "
            "raise SystemExit(int(os.environ['RC']))"
        )

        code = _spawn_and_wait(
            [sys.executable, "-c", script],
            {"RC": "7", "MARKER": str(marker), "PATH": os.environ.get("PATH", "")},
        )

        assert marker.read_text() == "done"
        assert code == 7

    def test_windows_run_agent_spawns_resolved_binary_with_proxy_args(self):
        spawn = _Recorder(returns=3)
        replace = _Recorder()

        def launcher(path, args, env):
            _hand_off(path, args, env, platform="win32", replace=replace, spawn=spawn)

        with pytest.raises(SystemExit) as excinfo:
            run_agent(
                "http://localhost:4000",
                "sk-key",
                ["codex", "exec", "do a thing"],
                skip_verify=True,
                base_env={},
                which=lambda name: _WINDOWS_CLAUDE_CMD.replace("claude", "codex"),
                launcher=launcher,
            )

        assert excinfo.value.code == 3
        assert replace.calls == []
        command, env = spawn.calls[0]
        shim = _WINDOWS_CLAUDE_CMD.replace("claude", "codex")
        assert command.startswith(f'{_CMD_PREFIX}""{shim}" ')
        assert command.endswith('"exec" "do a thing""')
        assert '"model_provider=""litellm"""' in command
        assert env["OPENAI_API_KEY"] == "sk-key"


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

    def test_opencode_launches_through_the_proxy(self):
        captured = {}
        with patch(f"{AGENTS_MODULE}.run_agent", side_effect=lambda b, k, c, **kw: captured.update(command=list(c))):
            result = self.runner.invoke(
                _agent_command("opencode"),
                [],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-key"},
            )

        assert result.exit_code == 0, result.output
        assert captured["command"] == ["opencode"]
        assert "routing OpenCode through proxy at http://localhost:4000" in result.output

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

    def test_interactive_without_key_logs_in_then_launches(self, secret_vault_factory):
        captured = {}
        vault = secret_vault_factory()

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
                obj={"base_url": "http://localhost:4000", "api_key": None, "secret_vault": vault},
            )

        assert result.exit_code == 0, result.output
        assert captured["api_key"] == "sk-after-login"
        mock_get.assert_called_once_with(expected_base_url="http://localhost:4000", vault=vault)

    def test_child_exit_code_reaches_the_shell(self):
        with patch(f"{AGENTS_MODULE}.run_agent", side_effect=SystemExit(42)):
            result = self.runner.invoke(
                _agent_command("claude"),
                [],
                obj={"base_url": "http://localhost:4000", "api_key": "sk-key"},
            )
        assert result.exit_code == 42

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
