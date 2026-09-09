import stat
from pathlib import Path

import pytest
import tomlkit

from litellm.proxy.client.cli.commands.agent_config import AgentConfigError, ConfigureReceipt
from litellm.proxy.client.cli.commands.claude_settings import ApiKeyHelper, KeepModel, StartOn, StaticToken, UnpinModel
from litellm.proxy.client.cli.commands.codex_settings import (
    OWNED_ROOT_KEYS,
    codex_configure_state_path,
    codex_home,
    configure_codex_config,
    unconfigure_codex_config,
)

BASE_URL = "http://127.0.0.1:4000"
PRINT_TOKEN = ("/usr/local/bin/lite", "--base-url", BASE_URL, "auth", "print-token")
ORIGINAL = """# my codex config
model = "gpt-6-astra"   # pinned by hand
model_reasoning_effort = "xhigh"

[model_providers.other]
name = "Other"
base_url = "https://other.example.com/v1"

[projects."/Users/me"]
trust_level = "trusted"
"""


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "codex" / "config.toml", tmp_path / "state" / "codex_configure_state.json"


def _configure(paths, model=StartOn("claude-auto"), credential=StaticToken("sk-virtual-key"), context_window=200000):
    config_path, state_path = paths
    configure_codex_config(BASE_URL, credential, lambda: PRINT_TOKEN, model, context_window, config_path, state_path)


def _no_lite_on_path():
    raise AgentConfigError("Could not find `lite` on your PATH.")


def _failing_commit(config_path, state_path, break_restore):
    landed_receipt = False

    def commit(staged, path):
        nonlocal landed_receipt
        from litellm.litellm_core_utils.private_json import commit_staged_json

        if path == str(config_path):
            Path(staged).unlink()
            if break_restore:
                state_path.unlink()
                state_path.mkdir()
            raise OSError("config rename failed")
        if break_restore and path == str(state_path) and landed_receipt:
            Path(staged).unlink()
            raise OSError("receipt rollback failed")
        commit_staged_json(staged, path)
        if path == str(state_path):
            landed_receipt = True

    return commit


class TestConfigureCodex:
    def test_wires_the_provider_and_pins_the_model_keeping_comments_and_other_tables(self, paths):
        config_path, state_path = paths
        config_path.parent.mkdir(parents=True)
        config_path.write_text(ORIGINAL)

        _configure(paths)

        text = config_path.read_text()
        doc = tomlkit.parse(text).unwrap()
        assert doc["model_provider"] == "litellm"
        assert doc["model"] == "claude-auto"
        assert doc["model_context_window"] == 200000
        assert doc["model_providers"]["litellm"] == {
            "name": "LiteLLM proxy",
            "base_url": f"{BASE_URL}/v1",
            "wire_api": "responses",
            "supports_websockets": False,
            "experimental_bearer_token": "sk-virtual-key",
        }
        assert doc["model_providers"]["other"] == {"name": "Other", "base_url": "https://other.example.com/v1"}
        assert doc["projects"]["/Users/me"] == {"trust_level": "trusted"}
        assert "# my codex config" in text and "# pinned by hand" in text
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600

    def test_the_login_credential_becomes_a_command_codex_runs_itself(self, paths):
        config_path, _ = paths
        _configure(paths, credential=ApiKeyHelper("ignored for codex"), model=UnpinModel(), context_window=None)
        provider = tomlkit.parse(config_path.read_text()).unwrap()["model_providers"]["litellm"]
        assert provider["auth"] == {"command": PRINT_TOKEN[0], "args": list(PRINT_TOKEN[1:]), "timeout_ms": 5000}
        assert "experimental_bearer_token" not in provider
        assert "model" not in tomlkit.parse(config_path.read_text()).unwrap()

    def test_the_receipt_never_holds_the_key(self, paths):
        _, state_path = paths
        _configure(paths, credential=StaticToken("sk-never-on-disk-twice"))
        assert "sk-never-on-disk-twice" not in state_path.read_text()
        assert set(ConfigureReceipt.model_validate_json(state_path.read_bytes()).sections[""].written) == set(
            OWNED_ROOT_KEYS
        )

    def test_reports_invalid_toml_without_touching_the_file(self, paths):
        config_path, _ = paths
        config_path.parent.mkdir(parents=True)
        config_path.write_text("model = [unterminated\n")
        with pytest.raises(AgentConfigError, match="not valid TOML"):
            _configure(paths)
        assert config_path.read_text() == "model = [unterminated\n"

    def test_a_static_key_never_needs_lite_on_path(self, paths):
        # A desktop-launched Codex reads the key from the file; only the login credential runs lite.
        config_path, state_path = paths
        configure_codex_config(
            BASE_URL,
            StaticToken("sk-virtual-key"),
            _no_lite_on_path,
            StartOn("claude-auto"),
            None,
            config_path,
            state_path,
        )
        assert tomlkit.parse(config_path.read_text()).unwrap()["model_providers"]["litellm"][
            "experimental_bearer_token"
        ]
        with pytest.raises(AgentConfigError, match="lite"):
            configure_codex_config(
                BASE_URL,
                ApiKeyHelper("ignored for codex"),
                _no_lite_on_path,
                StartOn("claude-auto"),
                None,
                config_path,
                state_path,
            )

    def test_a_repin_to_a_model_with_no_known_window_drops_the_old_window(self, paths):
        # Codex sizes its context from model_context_window; a stale 200k from the previous pin would be wrong.
        config_path, _ = paths
        _configure(paths, model=StartOn("claude-auto"), context_window=200000)
        _configure(paths, model=StartOn("gpt-5.6-luna"), context_window=None)
        after = tomlkit.parse(config_path.read_text()).unwrap()
        assert after["model"] == "gpt-5.6-luna" and "model_context_window" not in after

    @pytest.mark.parametrize(
        ("shape", "text"),
        [
            ("dotted", 'model_providers.other.name = "Other"\nmodel = "gpt-6-astra"\n'),
            ("inline", 'model_providers = { other = { name = "Other" } }\nmodel = "gpt-6-astra"\n'),
            (
                "split",
                '[model_providers.a]\nname = "a"\n\n[projects."/Users/me"]\ntrust_level = "trusted"\n\n[model_providers.b]\nname = "b"\n',
            ),
        ],
    )
    def test_every_toml_object_shape_of_model_providers_round_trips(self, paths, shape, text):
        # TOML spells an object as a standard table, dotted keys, an inline table, or fragments split by
        # other tables; configure must extend all of them and unconfigure must restore all of them.
        config_path, state_path = paths
        config_path.parent.mkdir(parents=True)
        config_path.write_text(text)
        _configure(paths)
        configured = tomlkit.parse(config_path.read_text()).unwrap()
        assert configured["model_providers"]["litellm"]["wire_api"] == "responses"
        assert configured["model_provider"] == "litellm"
        original_providers = tomlkit.parse(text).unwrap()["model_providers"]
        assert {k: v for k, v in configured["model_providers"].items() if k != "litellm"} == original_providers
        unconfigure_codex_config(config_path, state_path)
        assert tomlkit.parse(config_path.read_text()).unwrap() == tomlkit.parse(text).unwrap()

    def test_toml_dates_in_owned_keys_are_fingerprinted_not_rejected(self, paths):
        config_path, state_path = paths
        config_path.parent.mkdir(parents=True)
        config_path.write_text("model = 2024-01-01\n[model_providers.litellm]\nsince = 1979-05-27T07:32:00Z\n")
        _configure(paths)
        unconfigure_codex_config(config_path, state_path)
        after = tomlkit.parse(config_path.read_text()).unwrap()
        assert str(after["model"]) == "2024-01-01" and "since" in after["model_providers"]["litellm"]


class TestConfigureRollback:
    @pytest.mark.parametrize("repeat", [False, True], ids=["first", "repeat"])
    def test_config_failure_restores_the_previous_receipt(self, paths, repeat):
        config_path, state_path = paths
        if repeat:
            _configure(paths)
            receipt_before = state_path.read_bytes()
            config_before = config_path.read_bytes()
        else:
            receipt_before = None
            config_before = None
        with pytest.raises(AgentConfigError, match="config rename failed"):
            configure_codex_config(
                BASE_URL,
                StaticToken("sk-rotated"),
                lambda: PRINT_TOKEN,
                StartOn("claude-auto"),
                200000,
                config_path,
                state_path,
                commit=_failing_commit(config_path, state_path, False),
            )
        assert (state_path.read_bytes() if state_path.exists() else None) == receipt_before
        assert (config_path.read_bytes() if config_path.exists() else None) == config_before
        assert not list(state_path.parent.glob(".tmp-*")) and not list(config_path.parent.glob(".tmp-*"))

    @pytest.mark.parametrize("repeat", [False, True], ids=["first", "repeat"])
    def test_double_failure_names_the_stale_receipt_and_keeps_original_cause(self, paths, repeat):
        config_path, state_path = paths
        config_before = None
        if repeat:
            _configure(paths)
            config_before = config_path.read_bytes()
        with pytest.raises(AgentConfigError, match="could not be put back") as exc_info:
            configure_codex_config(
                BASE_URL,
                StaticToken("sk-rotated"),
                lambda: PRINT_TOKEN,
                StartOn("claude-auto"),
                200000,
                config_path,
                state_path,
                commit=_failing_commit(config_path, state_path, True),
            )
        assert str(state_path) in str(exc_info.value)
        assert "remove it before retrying" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, OSError)
        assert str(exc_info.value.__cause__) == "config rename failed"
        assert state_path.exists()
        assert (config_path.read_bytes() if config_path.exists() else None) == config_before
        assert not list(config_path.parent.glob(".tmp-*"))


class TestUnconfigureCodex:
    def test_returns_the_file_to_its_original_bytes(self, paths):
        config_path, state_path = paths
        config_path.parent.mkdir(parents=True)
        config_path.write_text(ORIGINAL)
        _configure(paths)

        outcome = unconfigure_codex_config(config_path, state_path)

        assert tomlkit.parse(config_path.read_text()).unwrap() == tomlkit.parse(ORIGINAL).unwrap()
        assert "# pinned by hand" in config_path.read_text()
        assert not state_path.exists()
        assert outcome.kept == ()
        assert set(outcome.restored) == {"model_provider", "model", "model_context_window", "model_providers.litellm"}

    def test_removes_a_config_that_only_configure_created(self, paths):
        config_path, state_path = paths
        _configure(paths)
        unconfigure_codex_config(config_path, state_path)
        assert not config_path.exists()

    def test_leaves_keys_the_user_changed_since(self, paths):
        config_path, state_path = paths
        config_path.parent.mkdir(parents=True)
        config_path.write_text(ORIGINAL)
        _configure(paths)
        doc = tomlkit.parse(config_path.read_text())
        doc["model"] = "claude-sonnet-4-6"
        config_path.write_text(tomlkit.dumps(doc))

        outcome = unconfigure_codex_config(config_path, state_path)
        after = tomlkit.parse(config_path.read_text()).unwrap()
        assert after["model"] == "claude-sonnet-4-6"
        assert "model_provider" not in after and "litellm" not in after["model_providers"]
        assert outcome.kept == ("model",)

    def test_a_key_the_user_edited_between_two_configures_stays_theirs(self, paths):
        # A repeat configure (a re-login is one) must not adopt the user's edit as its own write and
        # then delete it on unconfigure.
        config_path, state_path = paths
        config_path.parent.mkdir(parents=True)
        config_path.write_text(ORIGINAL)
        _configure(paths, model=StartOn("claude-auto"))
        doc = tomlkit.parse(config_path.read_text())
        doc["model"] = "my-favourite"
        config_path.write_text(tomlkit.dumps(doc))
        _configure(paths, model=KeepModel(), credential=StaticToken("sk-rotated"), context_window=None)
        assert tomlkit.parse(config_path.read_text()).unwrap()["model"] == "my-favourite"
        outcome = unconfigure_codex_config(config_path, state_path)
        after = tomlkit.parse(config_path.read_text()).unwrap()
        assert after["model"] == "my-favourite"
        assert "litellm" not in after["model_providers"] and "model_provider" not in after
        assert "model" in outcome.kept

    def test_refuses_when_the_provider_section_became_a_scalar(self, paths):
        config_path, state_path = paths
        _configure(paths)
        config_path.write_text('model_providers = "oops"\nmodel_provider = "litellm"\n')
        with pytest.raises(AgentConfigError, match="non-object"):
            unconfigure_codex_config(config_path, state_path)
        assert state_path.exists()

    def test_a_relogin_keeps_the_pin_and_a_repeat_without_a_model_releases_it(self, paths):
        config_path, state_path = paths
        config_path.parent.mkdir(parents=True)
        config_path.write_text(ORIGINAL)
        _configure(paths, model=StartOn("claude-auto"))
        _configure(paths, model=KeepModel(), credential=ApiKeyHelper("ignored for codex"), context_window=None)
        assert tomlkit.parse(config_path.read_text()).unwrap()["model"] == "claude-auto"
        _configure(paths, model=UnpinModel(), context_window=None)
        after = tomlkit.parse(config_path.read_text()).unwrap()
        assert after["model"] == "gpt-6-astra" and "model_context_window" not in after
        unconfigure_codex_config(config_path, state_path)
        assert tomlkit.parse(config_path.read_text()).unwrap() == tomlkit.parse(ORIGINAL).unwrap()


def test_relocated_codex_receipt_stays_under_litellm_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_path = tmp_path / "codex-home" / "config.toml"
    receipt = codex_configure_state_path(config_path)
    assert receipt.parent.parent == Path.home() / ".litellm"
    assert receipt.parent.name == "codex_configure_state"
    assert receipt.name.endswith(".json")
    assert receipt != config_path.parent / ".litellm-configure-state.json"


def test_codex_home_honors_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "elsewhere"))
    assert codex_home() == tmp_path / "elsewhere"
    monkeypatch.delenv("CODEX_HOME")
    assert codex_home() == Path.home() / ".codex"
