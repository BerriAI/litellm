import json
import os
import pathlib
import shlex
import stat
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from litellm.litellm_core_utils.private_json import commit_staged_json
from litellm.proxy.client.cli.commands.claude_settings import (
    ANTHROPIC_DEFAULT_MODEL_ENV_KEYS,
    AUTOROUTE_BACKUP_PATH,
    BACKUP_PATH,
    CLAUDE_SETTINGS_PATH,
    CONFIGURE_STATE_PATH,
    OWNED_ENV_KEYS,
    OWNED_TOP_LEVEL_KEYS,
    SETTINGS_FILE_OWNERS,
    ClaudeSettingsError,
    KeepModel,
    SettingsFileOwner,
    StartOn,
    StaticToken,
    UnpinModel,
    claude_settings_path,
    configure_claude_settings,
    install_statusline_script,
    configure_state_path,
    merge_claude_settings,
    statusline_command,
    unconfigure_claude_settings,
    with_status_line,
)


def _owners(*backup_paths):
    """Stand-in owners for the real `lite up` / `lite autoroute up` registry."""
    return tuple(SettingsFileOwner(path, "lite up", "lite down") for path in backup_paths)


CLAUDE_SETTINGS_MODULE = "litellm.proxy.client.cli.commands.claude_settings"
AUTH_MODULE = "litellm.proxy.client.cli.commands.auth"
@pytest.fixture
def paths(tmp_path):
    return tmp_path / "claude" / "settings.json", tmp_path / "backup.json"


def _static_configure(base_url, settings_path, owners, state_path=None):
    """`lite configure claude --api-key`'s shape: a virtual key as a static token, no pinned model."""
    state = state_path if state_path is not None else settings_path.parent.parent / "state.json"
    configure_claude_settings(base_url.rstrip("/"), StaticToken("sk-virtual-key"), KeepModel(), settings_path, state, owners)


class TestConfigureClaudeSettings:
    def test_creates_the_file_and_its_parent_when_missing(self, paths):
        settings_path, backup_path = paths
        assert not settings_path.parent.exists()

        _static_configure("https://proxy.example.com/", settings_path, _owners(backup_path))

        written = json.loads(settings_path.read_text())
        assert written["env"]["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"
        assert written["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-virtual-key"
        assert written["env"]["ENABLE_TOOL_SEARCH"] == "true"
        assert written["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
        assert "apiKeyHelper" not in written
        assert "model" not in written and "ANTHROPIC_MODEL" not in written["env"]

    def test_updates_an_existing_file_preserving_unrelated_settings(self, paths):
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "permissions": {"allow": ["Bash"]},
                    "env": {"SOME_OTHER_VAR": "keep-me", "ANTHROPIC_BASE_URL": "https://old.example.com"},
                }
            )
        )

        _static_configure("https://proxy.example.com", settings_path, _owners(backup_path))

        written = json.loads(settings_path.read_text())
        assert written["theme"] == "dark"
        assert written["permissions"] == {"allow": ["Bash"]}
        assert written["env"]["SOME_OTHER_VAR"] == "keep-me"
        assert written["env"]["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"

    def test_a_helper_left_by_an_older_lite_is_stripped_so_only_the_static_token_is_sent(self, paths):
        # Older `lite` versions wrote `apiKeyHelper: lite auth print-token`; Claude Code would keep spawning
        # `lite` (and its keychain probe) on every credential refresh, so configure takes the slot over.
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps({"apiKeyHelper": "/usr/local/bin/lite auth print-token", "env": {"ANTHROPIC_API_KEY": "sk-leaked"}})
        )

        _static_configure("https://proxy.example.com", settings_path, _owners(backup_path))

        written = json.loads(settings_path.read_text())
        assert "apiKeyHelper" not in written
        assert "ANTHROPIC_API_KEY" not in written["env"]
        assert written["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-virtual-key"

    def test_written_file_is_owner_only(self, paths):
        settings_path, backup_path = paths
        _static_configure("https://proxy.example.com", settings_path, _owners(backup_path))
        assert stat.S_IMODE(settings_path.stat().st_mode) == 0o600

    def test_refuses_while_lite_up_holds_a_backup(self, paths):
        settings_path, backup_path = paths
        backup_path.write_text("{}")

        with pytest.raises(ClaudeSettingsError, match="lite down"):
            _static_configure("https://proxy.example.com", settings_path, _owners(backup_path))

        assert not settings_path.exists()

    def test_refuses_on_corrupt_existing_settings_without_touching_the_file(self, paths):
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text("not json at all {{{")

        with pytest.raises(ClaudeSettingsError, match="invalid JSON"):
            _static_configure("https://proxy.example.com", settings_path, _owners(backup_path))

        assert settings_path.read_text() == "not json at all {{{"

    def test_reports_an_actionable_error_on_a_non_utf8_file(self, paths):
        # UnicodeDecodeError is a ValueError, not an OSError, so a decode-side catch is easy to miss.
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_bytes(b'{"theme": "\xff\xfe"}')

        with pytest.raises(ClaudeSettingsError, match="invalid JSON"):
            _static_configure("https://proxy.example.com", settings_path, _owners(backup_path))

    def test_reports_an_actionable_error_when_the_file_cannot_be_read(self, paths):
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.mkdir()

        with pytest.raises(ClaudeSettingsError, match="Could not read"):
            _static_configure("https://proxy.example.com", settings_path, _owners(backup_path))

    def test_reports_an_actionable_error_when_the_file_cannot_be_written(self, paths):
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.parent.chmod(0o500)
        try:
            with pytest.raises(ClaudeSettingsError, match="Could not write"):
                _static_configure("https://proxy.example.com", settings_path, _owners(backup_path))
        finally:
            settings_path.parent.chmod(0o700)
        assert not settings_path.exists()


class TestConflictingOwnersOfTheSettingsFile:
    """Both `lite up` and `lite autoroute up` restore a backup when they stop.

    Guarding only one of them leaves the other free to silently revert this
    write, which is the exact hazard the guard exists to prevent.
    """

    def test_any_owner_holding_a_backup_blocks_the_write(self, tmp_path):
        settings_path = tmp_path / "claude" / "settings.json"

        for index, owner in enumerate(SETTINGS_FILE_OWNERS):
            backup = tmp_path / f"backup-{index}.json"
            backup.write_text("{}")
            stand_in = SettingsFileOwner(backup, owner.start_command, owner.stop_command)
            with pytest.raises(ClaudeSettingsError, match="currently managing"):
                _static_configure("https://proxy.example.com", settings_path, (stand_in,))
            backup.unlink()
            assert not settings_path.exists()

    def test_the_error_names_the_owner_that_actually_holds_the_file(self, tmp_path):
        settings_path = tmp_path / "claude" / "settings.json"
        backup = tmp_path / "auto.json"
        backup.write_text("{}")
        autoroute = SettingsFileOwner(backup, "lite autoroute up", "lite autoroute down")

        with pytest.raises(ClaudeSettingsError, match="`lite autoroute up` is currently managing"):
            _static_configure("https://proxy.example.com", settings_path, (autoroute,))
        with pytest.raises(ClaudeSettingsError, match="Run `lite autoroute down` first"):
            _static_configure("https://proxy.example.com", settings_path, (autoroute,))

    def test_the_registry_matches_the_paths_the_commands_actually_use(self):
        """A second definition of the autoroute dir must not drift from this one."""
        from litellm.proxy.client.cli.commands.autoroute.process import AUTOROUTE_DIR

        assert AUTOROUTE_BACKUP_PATH == AUTOROUTE_DIR / "claude_settings_backup.json"
        assert {o.backup_path for o in SETTINGS_FILE_OWNERS} == {BACKUP_PATH, AUTOROUTE_BACKUP_PATH}
        assert {o.stop_command for o in SETTINGS_FILE_OWNERS} == {"lite down", "lite autoroute down"}


class TestDoesNotDestroyUserOwnedStructure:
    @pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs elevation or Developer Mode on Windows")
    def test_writes_through_a_symlinked_settings_file(self, tmp_path):
        """os.replace() swaps the symlink for a regular file, detaching a dotfiles repo.

        There is no backup here to undo that, so the link must survive and its
        target must be the thing that gets updated.
        """
        real = tmp_path / "dotfiles" / "settings.json"
        real.parent.mkdir()
        real.write_text(json.dumps({"theme": "dark"}))
        link = tmp_path / "claude" / "settings.json"
        link.parent.mkdir()
        link.symlink_to(real)

        _static_configure("https://proxy.example.com", link, ())

        assert link.is_symlink()
        assert json.loads(real.read_text())["env"]["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"
        assert json.loads(real.read_text())["theme"] == "dark"

    def test_refuses_rather_than_discarding_a_non_object_env(self, paths):
        """merge coerces a non-dict env to {}; that is silent data loss on a persistent write."""
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({"theme": "dark", "env": "not-an-object"}))

        with pytest.raises(ClaudeSettingsError, match="non-object"):
            _static_configure("https://proxy.example.com", settings_path, _owners(backup_path))

        assert json.loads(settings_path.read_text())["env"] == "not-an-object"


class TestClaudeSettingsPath:
    def test_defaults_to_the_home_settings_file(self):
        assert claude_settings_path({}) == CLAUDE_SETTINGS_PATH
        assert claude_settings_path({"CLAUDE_CONFIG_DIR": ""}) == CLAUDE_SETTINGS_PATH

    def test_follows_claude_config_dir_like_claude_code_does(self, tmp_path):
        assert claude_settings_path({"CLAUDE_CONFIG_DIR": str(tmp_path)}) == tmp_path / "settings.json"

    def test_expands_a_tilde_in_claude_config_dir(self):
        assert claude_settings_path({"CLAUDE_CONFIG_DIR": "~/.claude-work"}) == (
            Path.home() / ".claude-work" / "settings.json"
        )


class TestConfigureStatePath:
    """Each settings file gets its own undo receipt: the default file keeps the long-standing path, and
    a CLAUDE_CONFIG_DIR file gets one keyed by its resolved location, so `lite unconfigure claude`
    under one config dir never restores the other file's history."""

    @pytest.fixture
    def default_paths(self, tmp_path):
        default_settings = tmp_path / "home" / ".claude" / "settings.json"
        default_state = tmp_path / "home" / ".litellm" / "claude_configure_state.json"
        with (
            patch(f"{CLAUDE_SETTINGS_MODULE}.CLAUDE_SETTINGS_PATH", default_settings),
            patch(f"{CLAUDE_SETTINGS_MODULE}.CONFIGURE_STATE_PATH", default_state),
        ):
            yield default_settings, default_state

    def test_the_default_file_keeps_the_default_receipt(self, default_paths):
        default_settings, default_state = default_paths
        assert configure_state_path(default_settings) == default_state

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs elevation or Developer Mode on Windows")
    def test_a_symlink_alias_of_the_default_file_shares_its_receipt(self, default_paths):
        default_settings, default_state = default_paths
        default_settings.parent.mkdir(parents=True)
        alias = default_settings.parent.parent / "claude-alias"
        alias.symlink_to(default_settings.parent, target_is_directory=True)
        assert configure_state_path(alias / "settings.json") == default_state

    def test_another_settings_file_gets_a_receipt_of_its_own_beside_the_default_one(self, default_paths, tmp_path):
        _default_settings, default_state = default_paths
        work_state = configure_state_path(tmp_path / "work" / "settings.json")
        play_state = configure_state_path(tmp_path / "play" / "settings.json")
        assert work_state != default_state and play_state != default_state
        assert work_state != play_state
        assert work_state.parent == play_state.parent == default_state.parent / "claude_configure_state"
        assert work_state == configure_state_path(tmp_path / "work" / "settings.json")

    def test_configure_and_unconfigure_under_a_config_dir_leave_the_default_receipt_alone(
        self, default_paths, tmp_path
    ):
        _default_settings, default_state = default_paths
        work_settings = tmp_path / "work" / "settings.json"
        work_state = configure_state_path(work_settings)
        configure_claude_settings(
            "https://proxy.example.com", StaticToken("sk-virtual-key"), KeepModel(), work_settings, work_state, ()
        )
        assert work_state.exists() and not default_state.exists()
        outcome = unconfigure_claude_settings(work_settings, work_state, ())
        assert outcome.file_removed and not work_settings.exists()
        assert not work_state.exists()


class TestMergeClaudeSettings:
    """One merge for every way Claude Code gets wired: `lite up`, `lite configure claude` and `lite autoroute up`."""

    def test_a_static_token_lands_in_env_and_the_helper_slot_is_cleared(self):
        settings = {"apiKeyHelper": "/usr/local/bin/lite auth print-token", "env": {"ANTHROPIC_API_KEY": "leaked"}}
        merged = merge_claude_settings(settings, "http://127.0.0.1:4000/", StaticToken("token-abc"))
        assert merged["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
        assert merged["env"]["ANTHROPIC_AUTH_TOKEN"] == "token-abc"
        assert merged["env"]["ENABLE_TOOL_SEARCH"] == "true"
        assert merged["env"]["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
        assert "ANTHROPIC_API_KEY" not in merged["env"]
        assert "apiKeyHelper" not in merged
        assert "model" not in merged
        assert not any(key in merged["env"] for key in ANTHROPIC_DEFAULT_MODEL_ENV_KEYS)

    def test_keeps_existing_switch_values_and_unrelated_keys_without_mutating_the_input(self):
        settings = {"theme": "dark", "env": {"SOME_OTHER_VAR": "value", "ENABLE_TOOL_SEARCH": "false"}}
        merged = merge_claude_settings(settings, "http://127.0.0.1:4000", StaticToken("token-abc"))
        assert merged["theme"] == "dark"
        assert merged["env"]["SOME_OTHER_VAR"] == "value"
        assert merged["env"]["ENABLE_TOOL_SEARCH"] == "false"
        assert settings == {"theme": "dark", "env": {"SOME_OTHER_VAR": "value", "ENABLE_TOOL_SEARCH": "false"}}

    def test_a_default_model_pins_the_starting_row_and_the_model_a_resumed_session_keeps(self):
        # `model` is only the row Claude Code starts on: a resumed session re-sends the model its transcript
        # recorded, which behind a raw-model auto-router is the tier model (403 for a key scoped to the
        # router). ANTHROPIC_MODEL outranks the transcript on resume, so the pin has to land there too.
        merged = merge_claude_settings(
            {}, "http://127.0.0.1:4000", StaticToken("token-abc"), default_model="claude-auto"
        )
        assert merged["model"] == "claude-auto"
        assert merged["env"]["ANTHROPIC_MODEL"] == "claude-auto"
        assert not any(key in merged["env"] for key in ANTHROPIC_DEFAULT_MODEL_ENV_KEYS)

    def test_without_a_default_model_neither_pin_is_written_and_a_users_own_stays(self):
        settings = {"model": "mine", "env": {"ANTHROPIC_MODEL": "mine-too"}}
        merged = merge_claude_settings(settings, "http://127.0.0.1:4000", StaticToken("token-abc"))
        assert merged["model"] == "mine" and merged["env"]["ANTHROPIC_MODEL"] == "mine-too"

    def test_a_tier_model_forces_every_claude_code_tier_as_autoroute_needs(self):
        # Router's auto-router registry is keyed by the literal requested model string with no
        # wildcard resolution, so `lite autoroute up` overrides the env var each tier reads.
        settings = {"env": {"ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-opus-4-8"}}
        merged = merge_claude_settings(
            settings, "http://127.0.0.1:4000", StaticToken("token-abc"), tier_model="autorouter"
        )
        assert {merged["env"][key] for key in ANTHROPIC_DEFAULT_MODEL_ENV_KEYS} == {"autorouter"}
        assert "model" not in merged

    def test_touches_exactly_the_declared_owned_keys(self):
        # The receipt and unconfigure restore exactly OWNED_*_KEYS, so a key the merge writes outside
        # that table would be written by configure and never undone.
        settings = {
            "theme": "dark",
            "permissions": {"allow": ["Bash"]},
            "env": {"KEEP_ME": "1", "ANTHROPIC_API_KEY": "old", "ENABLE_TOOL_SEARCH": "false"},
            "apiKeyHelper": "old-helper",
            "model": "old-model",
        }
        for credential in (StaticToken("token-abc"), StaticToken("token-rotated")):
            merged = merge_claude_settings(settings, "http://127.0.0.1:4000", credential, default_model="claude-auto")
            changed_top_level = {key for key in set(settings) | set(merged) if settings.get(key) != merged.get(key)}
            assert changed_top_level - {"env"} <= set(OWNED_TOP_LEVEL_KEYS)
            changed_env = {
                key
                for key in set(settings["env"]) | set(merged["env"])
                if settings["env"].get(key) != merged["env"].get(key)
            }
            assert changed_env <= set(OWNED_ENV_KEYS)
            assert merged["permissions"] == {"allow": ["Bash"]}
            assert merged["env"]["KEEP_ME"] == "1"


PROXY = "http://127.0.0.1:4000"
ANTHROPIC = "https://api.anthropic.com"
RELOGIN = StaticToken("sk-fresh-login")
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


class _Rig:
    """One settings file plus receipt under tmp_path, driven through the public functions only."""

    def __init__(self, tmp_path, initial):
        self.settings = tmp_path / "claude" / "settings.json"
        self.state = tmp_path / "state" / "claude_configure_state.json"
        if initial is not None:
            self.settings.parent.mkdir(parents=True)
            self.settings.write_text(json.dumps(initial))

    def read(self):
        return json.loads(self.settings.read_text()) if self.settings.exists() else None

    def configure(self, credential=StaticToken("sk-virtual-key"), model=StartOn("claude-auto"), **kwargs):
        configure_claude_settings(PROXY, credential, model, self.settings, self.state, (), **kwargs)

    def edit(self, *edits):
        settings = self.read()
        for apply in edits:
            settings = apply(settings)
        self.settings.write_text(json.dumps(settings))

    def unconfigure(self):
        return unconfigure_claude_settings(self.settings, self.state, ())


# Each row: initial file, steps (configure kwargs dicts or edit callables) between the first configure
# and unconfigure, the expected file afterwards, and the expected outcome fields. Sequences that used
# to be one test each; the receipt's rules are what make them all come out right.
UNDO_SCENARIOS = {
    "plain round trip": (ORIGINAL, [], ORIGINAL, {"kept": ()}),
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


def _expected_file(expected):
    if expected is None:
        return None
    return {k: v for k, v in expected.items() if v is not _ABSENT}


class TestConfigureAndUnconfigure:
    """`configure_claude_settings` records how to undo itself; `unconfigure_claude_settings` undoes only that."""

    @pytest.mark.parametrize("scenario", UNDO_SCENARIOS.values(), ids=UNDO_SCENARIOS.keys())
    def test_undo_matrix(self, tmp_path, scenario):
        initial, steps, expected, outcome_expectations, *first = scenario
        rig = _Rig(tmp_path, initial)
        rig.configure(**(first[0] if first else {}))
        for step in steps:
            if isinstance(step, dict):
                rig.configure(**step)
            elif rig.settings.exists() and step(json.loads(rig.settings.read_text())) is None:
                rig.settings.unlink()
            else:
                rig.edit(step)

        outcome = rig.unconfigure()

        assert rig.read() == _expected_file(expected)
        assert rig.state.exists() == outcome_expectations.get("receipt_kept", False)
        for field, want in outcome_expectations.items():
            if field == "withheld":
                assert {(item.key, item.endpoint) for item in outcome.withheld} == set(want)
            elif field == "not_restored":
                assert not set(want) & set(outcome.restored) and not set(want) & set(outcome.kept)
            elif field == "restored_includes":
                assert set(want) <= set(outcome.restored)
            elif field in ("restored", "kept"):
                assert set(getattr(outcome, field)) == set(want)
            elif field != "receipt_kept":
                assert getattr(outcome, field) == want
        assert not {item.key for item in outcome.withheld} & set(outcome.kept)

    def test_configure_writes_owner_only_and_the_receipt_never_holds_the_key(self, tmp_path):
        rig = _Rig(tmp_path, ORIGINAL)
        rig.configure(credential=StaticToken("sk-virtual-key-never-on-disk-twice"))
        configured = rig.read()
        assert configured["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-virtual-key-never-on-disk-twice"
        assert configured["env"]["ANTHROPIC_BASE_URL"] == PROXY and configured["model"] == "claude-auto"
        assert "ANTHROPIC_API_KEY" not in configured["env"] and "apiKeyHelper" not in configured
        assert stat.S_IMODE(rig.settings.stat().st_mode) == 0o600 == stat.S_IMODE(rig.state.stat().st_mode)
        assert "sk-virtual-key-never-on-disk-twice" not in rig.state.read_text()

    def test_withheld_credentials_come_back_once_the_url_points_at_their_server_again(self, tmp_path):
        # The kept receipt owns only the withheld slots: the second unconfigure restores exactly those.
        rig = _Rig(tmp_path, ORIGINAL)
        rig.configure()
        rig.edit(_set("env.ANTHROPIC_BASE_URL", "http://other-proxy:4000"))
        rig.unconfigure()
        rig.edit(_set("env.ANTHROPIC_BASE_URL", ANTHROPIC), _set("theme", "light"))
        outcome = rig.unconfigure()
        assert rig.read() == {**ORIGINAL, "theme": "light"}
        assert set(outcome.restored) == {"env.ANTHROPIC_API_KEY", "apiKeyHelper"}
        assert outcome.kept == () and outcome.withheld == () and not rig.state.exists()

    @pytest.mark.parametrize(
        ("path", "value", "repeat_credential"),
        [
            ("env.ANTHROPIC_API_KEY", "sk-user-added-later", RELOGIN),
            ("apiKeyHelper", "/opt/mine/helper", StaticToken("sk-rotated")),
        ],
        ids=["user-adds-api-key", "user-sets-own-helper"],
    )
    def test_a_credential_the_user_set_between_two_configures_is_what_comes_back(
        self, tmp_path, path, value, repeat_credential
    ):
        # The repeat's merge clears the slot, so the displaced value is snapshotted and is what returns;
        # it was set while the file pointed at the proxy, so it returns once the file points there again.
        rig = _Rig(tmp_path, {"theme": "dark"})
        rig.configure(credential=RELOGIN, model=KeepModel())
        rig.edit(_set(path, value))
        rig.configure(credential=repeat_credential, model=KeepModel())
        assert not _lookup(rig.read(), path)

        outcome = rig.unconfigure()
        assert [(item.key, item.endpoint) for item in outcome.withheld] == [(path, PROXY)]
        assert rig.read() == {"theme": "dark"} and rig.state.exists()
        rig.settings.write_text(json.dumps({"theme": "dark", "env": {"ANTHROPIC_BASE_URL": PROXY}}))
        outcome = rig.unconfigure()
        assert _lookup(rig.read(), path) == value
        assert outcome.restored == (path,) and outcome.withheld == () and not rig.state.exists()

    def test_a_receipt_commit_that_fails_leaves_no_staged_token_behind(self, tmp_path):
        rig = _Rig(tmp_path, {})

        def commit_receipt_fails(staged, path):
            if path == str(rig.state):
                os.unlink(staged)
                raise OSError("receipt rename failed")
            commit_staged_json(staged, path)

        with pytest.raises(ClaudeSettingsError, match=r"Could not write .*receipt rename failed"):
            rig.configure(credential=StaticToken("sk-never-left-in-a-temp-file"), commit=commit_receipt_fails)
        assert not list(rig.settings.parent.glob(".tmp-*")) and not list(rig.state.parent.glob(".tmp-*"))
        assert rig.read() == {} and not rig.state.exists()

    @pytest.mark.parametrize("configured_before", [False, True], ids=["first-configure", "repeat-configure"])
    def test_a_settings_commit_that_fails_after_the_receipt_landed_puts_the_receipt_back(
        self, tmp_path, configured_before
    ):
        # The two renames are not atomic: a settings rename that fails after the receipt landed must
        # not leave a receipt describing settings that were never written.
        rig = _Rig(tmp_path, ORIGINAL)
        if configured_before:
            rig.configure()
        receipt_before = rig.state.read_text() if configured_before else None
        settings_before = rig.settings.read_text()

        def commit_settings_fails(staged, path):
            if path == str(rig.settings):
                os.unlink(staged)
                raise OSError("rename failed")
            commit_staged_json(staged, path)

        with pytest.raises(ClaudeSettingsError, match="rename failed"):
            rig.configure(credential=StaticToken("sk-rotated"), commit=commit_settings_fails)
        assert rig.settings.read_text() == settings_before
        assert (rig.state.read_text() if rig.state.exists() else None) == receipt_before
        if configured_before:
            rig.unconfigure()
            assert rig.read() == ORIGINAL

    def test_a_failed_repeat_configure_leaves_the_earlier_undo_intact(self, tmp_path):
        rig = _Rig(tmp_path, ORIGINAL)
        rig.configure()
        receipt_before = rig.state.read_text()
        rig.settings.parent.chmod(0o500)
        try:
            with pytest.raises(ClaudeSettingsError, match="Could not write"):
                rig.configure(credential=StaticToken("sk-rotated"))
        finally:
            rig.settings.parent.chmod(0o700)
        assert rig.state.read_text() == receipt_before and not list(rig.state.parent.glob(".tmp-*"))
        assert rig.read()["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-virtual-key"
        rig.unconfigure()
        assert rig.read() == ORIGINAL

    def test_unconfigure_reports_a_receipt_it_cannot_remove_as_a_settings_error(self, tmp_path):
        rig = _Rig(tmp_path, ORIGINAL)
        rig.configure()
        rig.state.parent.chmod(0o500)
        try:
            with pytest.raises(ClaudeSettingsError, match="Could not remove"):
                rig.unconfigure()
        finally:
            rig.state.parent.chmod(0o700)

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs elevation or Developer Mode on Windows")
    def test_configure_writes_through_a_symlinked_settings_file(self, tmp_path):
        target = tmp_path / "dotfiles" / "settings.json"
        target.parent.mkdir()
        target.write_text(json.dumps({"theme": "dark"}))
        link = tmp_path / "settings.json"
        link.symlink_to(target)
        configure_claude_settings(PROXY, StaticToken("sk-virtual-key"), UnpinModel(), link, tmp_path / "state.json", ())
        assert link.is_symlink()
        assert json.loads(target.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-virtual-key"

    @pytest.mark.parametrize("operation", ["configure", "unconfigure"])
    def test_refuses_while_a_temporary_owner_holds_a_backup(self, paths, tmp_path, operation):
        settings_path, backup_path = paths
        backup_path.write_text("{}")
        owners = _owners(backup_path)
        state = tmp_path / "state.json"
        attempt = (
            (lambda: configure_claude_settings(PROXY, StaticToken("k"), UnpinModel(), settings_path, state, owners))
            if operation == "configure"
            else (lambda: unconfigure_claude_settings(settings_path, state, owners))
        )
        with pytest.raises(ClaudeSettingsError, match="lite down"):
            attempt()
        assert not settings_path.exists()

    def test_unconfigure_without_a_receipt_is_an_error_not_a_silent_no_op(self, tmp_path):
        with pytest.raises(ClaudeSettingsError, match="nothing to undo"):
            _Rig(tmp_path, None).unconfigure()


def _lookup(settings, path):
    section, _, key = path.rpartition(".")
    return (settings.get(section) or {}).get(key) if section else settings.get(key)


class TestStatusLine:
    """Every configure registers the status line, and only while the slot is empty or already ours."""

    COMMAND = "/opt/lite/bin/python /Users/me/.litellm/statusline.py"

    def test_an_empty_slot_gets_our_status_line(self):
        assert with_status_line({}, self.COMMAND)["statusLine"] == {"type": "command", "command": self.COMMAND}

    def test_a_users_own_status_line_is_never_replaced(self):
        theirs = {"type": "command", "command": "~/.claude/my-statusline.sh"}
        assert with_status_line({"statusLine": theirs}, self.COMMAND)["statusLine"] == theirs

    def test_ours_under_an_older_interpreter_is_refreshed(self):
        stale = {"type": "command", "command": "/old/python /Users/me/.litellm/statusline.py"}
        assert with_status_line({"statusLine": stale}, self.COMMAND)["statusLine"]["command"] == self.COMMAND

    def test_the_merge_carries_it(self):
        merged = merge_claude_settings({}, PROXY, StaticToken("tok"), status_line=self.COMMAND)
        assert merged["statusLine"] == {"type": "command", "command": self.COMMAND}

    def test_the_installed_script_is_the_bundled_one_and_the_command_runs_this_interpreter(self, tmp_path):
        from litellm.proxy.client.cli.commands import statusline_script

        script = tmp_path / "lite" / "statusline.py"
        command = install_statusline_script(script)
        assert script.read_bytes() == pathlib.Path(statusline_script.__file__).read_bytes()
        assert shlex.split(command) == [sys.executable, str(script)]
        assert command == statusline_command(script)
        assert stat.S_IMODE(script.stat().st_mode) == 0o600
        assert stat.S_IMODE(script.parent.stat().st_mode) == 0o700
        assert install_statusline_script(script) == command

    def test_a_reinstall_replaces_the_script_in_one_step_and_a_refused_one_leaves_the_old_script_whole(self, tmp_path):
        # Claude Code may be running the script at the moment `lite` reinstalls it; the file it has open
        # must stay complete, and a reinstall that cannot land must not leave a truncated script behind.
        from litellm.proxy.client.cli.commands import statusline_script

        script = tmp_path / "lite" / "statusline.py"
        install_statusline_script(script)
        bundled = pathlib.Path(statusline_script.__file__).read_bytes()
        with script.open("rb") as running:
            install_statusline_script(script)
            assert running.read() == bundled
        assert [child.name for child in script.parent.iterdir()] == ["statusline.py"]

        if os.geteuid() != 0:
            script.parent.chmod(0o500)
            try:
                with pytest.raises(ClaudeSettingsError, match="Could not install the status line script"):
                    install_statusline_script(script)
            finally:
                script.parent.chmod(0o700)
            assert script.read_bytes() == bundled

    def test_configure_installs_it_and_unconfigure_removes_only_ours(self, tmp_path):
        rig = _Rig(tmp_path, {"theme": "dark"})
        script = tmp_path / "statusline.py"
        rig.configure(script_path=script)
        assert rig.read()["statusLine"]["command"] == statusline_command(script)
        assert script.exists()

        outcome = rig.unconfigure()
        assert rig.read() == {"theme": "dark"}
        assert "statusLine" in outcome.restored

    def test_a_status_line_the_user_replaced_after_configure_survives_unconfigure(self, tmp_path):
        rig = _Rig(tmp_path, None)
        rig.configure(model=KeepModel(), script_path=tmp_path / "statusline.py")
        theirs = {"type": "command", "command": "~/.claude/my-statusline.sh"}
        rig.edit(lambda settings: {**settings, "statusLine": theirs})

        outcome = rig.unconfigure()
        assert rig.read()["statusLine"] == theirs
        assert "statusLine" in outcome.kept

    def test_a_receipt_from_before_the_status_line_existed_still_unconfigures(self, tmp_path):
        # Older receipts never claimed statusLine; a key no configure wrote is never ours, so it stays.
        rig = _Rig(tmp_path, None)
        script = tmp_path / "statusline.py"
        rig.configure(model=KeepModel(), script_path=script)
        receipt = json.loads(rig.state.read_text())
        receipt["written"].pop("statusLine")
        receipt["previous"].pop("statusLine")
        rig.state.write_text(json.dumps(receipt))

        outcome = rig.unconfigure()
        restored = rig.read()
        assert "ANTHROPIC_AUTH_TOKEN" not in restored.get("env", {})
        assert restored["statusLine"]["command"] == statusline_command(script)
        assert "statusLine" not in outcome.restored
