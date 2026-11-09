import json
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
import tomlkit

from litellm.litellm_core_utils.private_json import commit_staged_json
from litellm.proxy.client.cli.commands import codex_settings as codex_settings_module
from litellm.proxy.client.cli.commands.agents import (
    agent_launch_args,
    codex_config_path,
)
from litellm.proxy.client.cli.commands.codex_settings import (
    CodexSettingsError,
    _snapshot,
    _with,
    codex_configure_state_path,
    configure_codex_settings,
    preflight_codex_settings,
    unconfigure_codex_settings,
)

GATEWAY: Final = "https://gateway.example.com/team"
KEY: Final = "sk-test-new-gateway-key"
MODEL: Final = "gateway-codex-model"


@pytest.mark.parametrize("path,existing,first_value,second_value", [
    ("model", 'model = "original" # starting model\n', 'value = "first"\n', 'value = "second"\n'),
    ("model_providers.litellm", '', '[value]\nname = "first"\n', '[value]\nname = "second"\n'),
    ("model_providers.litellm", '[model_providers.litellm]\nname = "original" # provider\n',
     '[value]\nname = "first"\n', '[value]\nname = "second"\n'),
])
def test_toml_transitions_leave_source_and_independent_results_unchanged(
    path: str, existing: str, first_value: str, second_value: str
) -> None:
    source: Final = tomlkit.parse('# user settings\n' + existing + '[profiles.work]\nmodel = "keep" # profile\n')
    original_bytes: Final = source.as_string().encode()
    first: Final = _with(source, path, first_value)
    first_bytes: Final = first.as_string().encode()
    second: Final = _with(source, path, second_value)
    second_bytes: Final = second.as_string().encode()
    removed: Final = _with(first, path, None)
    first_snapshot: Final = _snapshot(first, path)
    second_snapshot: Final = _snapshot(second, path)
    assert source.as_string().encode() == original_bytes
    assert first.as_string().encode() == first_bytes
    assert second.as_string().encode() == second_bytes
    assert first_snapshot is not None and tomlkit.parse(first_snapshot) == tomlkit.parse(first_value)
    assert second_snapshot is not None and tomlkit.parse(second_snapshot) == tomlkit.parse(second_value)
    assert _snapshot(removed, path) is None
    for result in (first, second, removed):
        assert result["profiles"] == source["profiles"]
        assert '# user settings' in result.as_string()
        assert '# profile' in result.as_string()


def test_persistent_provider_is_complete_and_preserves_unrelated_toml(tmp_path: Path) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text(
        '# user settings\nmodel = "old-model" # starting model\n'
        'model_provider = "openai"\nprofile = "work"\n'
        '[model_providers.litellm]\nname = "old gateway"\n'
        'base_url = "https://old.example.com/v1"\nenv_key = "OLD_KEY"\n'
        'experimental_bearer_token = "sk-old"\nrequires_openai_auth = true\n'
        '[model_providers.litellm.auth]\ncommand = "old-token-helper"\n'
        '[model_providers.other]\nname = "Keep me" # other provider\n'
        '[profiles.work]\nmodel = "work-model"\n'
        '[[hooks.Stop]]\nhooks = [{type = "command", command = "echo done"}]\n'
    )
    original: Final = tomlkit.parse(path.read_text())
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    configured: Final = tomlkit.parse(path.read_text())
    assert configured["model"] == MODEL
    assert configured["model_provider"] == "litellm"
    assert "profile" not in configured
    assert configured["model_providers"]["litellm"] == {
        "name": "LiteLLM proxy",
        "base_url": GATEWAY + "/v1",
        "wire_api": "responses",
        "supports_websockets": False,
        "requires_openai_auth": False,
        "http_headers": {"Authorization": "Bearer " + KEY},
    }
    assert configured["model_providers"]["other"] == original["model_providers"]["other"]
    assert configured["profiles"] == original["profiles"]
    assert configured["hooks"] == original["hooks"]
    assert "# user settings" in path.read_text()
    assert "# other provider" in path.read_text()
    assert KEY not in codex_configure_state_path(path).read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(codex_configure_state_path(path).stat().st_mode) == 0o600
    assert stat.S_IMODE(codex_configure_state_path(path).parent.stat().st_mode) == 0o700
    outcome: Final = unconfigure_codex_settings(path)
    assert not outcome.kept and not outcome.file_removed
    assert tomlkit.parse(path.read_text()) == original
    assert "# starting model" in path.read_text()
    assert "# other provider" in path.read_text()
    assert not codex_configure_state_path(path).exists()


@pytest.mark.parametrize("original", [None, "", "# my preferences\n", '[model_providers]\n'])
def test_undo_distinguishes_missing_empty_and_existing_tables(tmp_path: Path, original: str | None) -> None:
    path: Final = tmp_path / "config.toml"
    if original is not None:
        path.write_text(original)
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    outcome: Final = unconfigure_codex_settings(path)
    assert outcome.file_removed == (original is None)
    assert path.exists() == (original is not None)
    if original is not None:
        assert tomlkit.parse(path.read_text()) == tomlkit.parse(original)
        assert original.strip() in path.read_text()


def test_repeat_setup_preserves_original_and_undo_keeps_user_edits(tmp_path: Path) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text('model = "original"\nmodel_provider = "openai"\n')
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    configure_codex_settings(GATEWAY + "/second", "sk-second", "second-model", path)
    assert tomlkit.parse(path.read_text())["model"] == "second-model"
    path.write_text(path.read_text().replace('model = "second-model"', 'model = "my-custom-model"'))
    outcome: Final = unconfigure_codex_settings(path)
    assert outcome.kept == ("model",)
    assert tomlkit.parse(path.read_text()) == {"model": "my-custom-model", "model_provider": "openai"}


def test_repeat_setup_restores_the_user_value_it_displaced(tmp_path: Path) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text('model = "original"\n')
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    path.write_text(path.read_text().replace('model = "gateway-codex-model"', 'model = "user-edited"'))
    configure_codex_settings(GATEWAY, "sk-rotated", "third-model", path)
    unconfigure_codex_settings(path)
    assert tomlkit.parse(path.read_text()) == {"model": "user-edited"}


def test_undo_keeps_provider_credentials_and_endpoint_together(tmp_path: Path) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text('[model_providers.litellm]\nbase_url = "https://old.example.com/v1"\n'
                    'http_headers = { Authorization = "Bearer old-key" }\n')
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    path.write_text(path.read_text().replace(GATEWAY, "https://user.example.com"))
    outcome: Final = unconfigure_codex_settings(path)
    provider: Final = tomlkit.parse(path.read_text())["model_providers"]["litellm"]
    assert outcome.kept == ("model_providers.litellm",)
    assert provider["base_url"] == "https://user.example.com/v1"
    assert provider["http_headers"] == {"Authorization": "Bearer " + KEY}
    assert "old-key" not in path.read_text()


def test_user_deleted_config_is_not_recreated_to_restore_profile(tmp_path: Path) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text('profile = "old-profile"\n')
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    path.unlink()
    outcome: Final = unconfigure_codex_settings(path)
    assert outcome.file_removed and outcome.restored == ()
    assert not path.exists()


def test_user_comment_in_new_config_survives_undo(tmp_path: Path) -> None:
    path: Final = tmp_path / "config.toml"
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    path.write_text("# keep my note\n" + path.read_text())
    assert not unconfigure_codex_settings(path).file_removed
    assert "# keep my note" in path.read_text()


def test_code_home_and_symlink_aliases_share_receipt_and_write_target(tmp_path: Path) -> None:
    target: Final = tmp_path / "real-config.toml"
    target.write_text('model = "old"\n')
    custom_home: Final = tmp_path / "codex-home"
    custom_home.mkdir()
    alias: Final = codex_config_path({"CODEX_HOME": str(custom_home)})
    alias.symlink_to(target)
    configure_codex_settings(GATEWAY, KEY, MODEL, alias)
    assert alias.is_symlink()
    assert codex_configure_state_path(alias) == codex_configure_state_path(target)
    assert tomlkit.parse(target.read_text())["model"] == MODEL
    unconfigure_codex_settings(target)
    assert alias.is_symlink()
    assert tomlkit.parse(alias.read_text()) == {"model": "old"}


@pytest.mark.parametrize("invalid", [
    'token = "sk-secret\n',
    'model_providers = "sk-secret"\n',
    '[model_providers]\nlitellm = "sk-secret"\n',
])
def test_invalid_settings_are_unchanged_and_errors_hide_content(tmp_path: Path, invalid: str) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text(invalid)
    with pytest.raises(CodexSettingsError) as caught:
        configure_codex_settings(GATEWAY, KEY, MODEL, path)
    assert "sk-secret" not in str(caught.value)
    assert path.read_text() == invalid
    assert not codex_configure_state_path(path).exists()


def test_invalid_receipt_fails_preflight_before_settings_change(tmp_path: Path) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text('model = "keep"\n')
    state: Final = codex_configure_state_path(path)
    state.parent.mkdir()
    state.write_text('{"previous": "sk-secret"}')
    with pytest.raises(CodexSettingsError) as caught:
        preflight_codex_settings(path)
    assert "sk-secret" not in str(caught.value)
    assert path.read_text() == 'model = "keep"\n'


@pytest.mark.parametrize("configured_before", [False, True])
@pytest.mark.parametrize("failed_target", ["receipt", "settings"])
def test_failed_commit_restores_receipt_and_cleans_private_staging(
    tmp_path: Path, configured_before: bool, failed_target: str
) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text('model = "old"\n')
    state: Final = codex_configure_state_path(path)
    if configured_before:
        configure_codex_settings(GATEWAY, KEY, MODEL, path)
    before: Final = path.read_bytes()
    receipt_before: Final = state.read_bytes() if state.exists() else None

    def failing_commit(staged: str, destination: str) -> None:
        if destination == str(state if failed_target == "receipt" else path.resolve()):
            raise OSError("sk-secret OS error")
        commit_staged_json(staged, destination)

    with pytest.raises(CodexSettingsError) as caught:
        configure_codex_settings(GATEWAY, "sk-replacement", "new-model", path, commit=failing_commit)
    assert "sk-secret" not in str(caught.value)
    assert path.read_bytes() == before
    assert (state.read_bytes() if state.exists() else None) == receipt_before
    assert not tuple(tmp_path.rglob(".tmp-*"))
    if configured_before:
        unconfigure_codex_settings(path)
        assert tomlkit.parse(path.read_text()) == {"model": "old"}


def test_failed_undo_keeps_the_settings_and_receipt_for_retry(tmp_path: Path) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text('model = "old"\n')
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    before: Final = path.read_bytes()

    def fail(staged: str, destination: str) -> None:
        raise OSError("cannot replace")

    with pytest.raises(CodexSettingsError):
        unconfigure_codex_settings(path, commit=fail)
    assert path.read_bytes() == before
    assert codex_configure_state_path(path).exists()
    assert not tuple(tmp_path.rglob(".tmp-*"))
    unconfigure_codex_settings(path)
    assert tomlkit.parse(path.read_text()) == {"model": "old"}


def test_wrapper_and_persistent_provider_agree_except_credential_source(tmp_path: Path) -> None:
    path: Final = tmp_path / "config.toml"
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    provider: Final = tomlkit.parse(path.read_text())["model_providers"]["litellm"]
    args: Final = agent_launch_args("codex", GATEWAY)
    overrides: Final = dict(argument.split("=", 1) for argument in args[1::2])
    for field in ("name", "base_url", "wire_api", "supports_websockets", "requires_openai_auth"):
        assert json.loads(overrides[f"model_providers.litellm.{field}"]) == provider[field]
    assert overrides["model_providers.litellm.http_headers"] == "{}"
    assert overrides["model_providers.litellm.env_key"] == '"OPENAI_API_KEY"'


@pytest.mark.parametrize("version", [
    "codex-cli 0.129.0", "codex-cli 0.129.1", "codex-cli 0.130.0", "codex-cli 1.0.0",
    " \ncodex-cli 0.129.0\n",
])
def test_version_guard_accepts_the_fixed_release_and_newer_stable_versions(version: str) -> None:
    assert codex_settings_module.require_safe_codex(version=lambda: version) is None


@pytest.mark.parametrize("version", [
    None, "", "codex-cli 0.99.0", "codex-cli 0.128.99", "codex-cli 0.129.0-alpha.1",
    "codex-cli 1.0.0-beta.1", "0.129.0", "codex-cli 0.129.0 extra", "unparseable-sk-version-secret",
])
def test_version_guard_refuses_missing_unsafe_or_unrecognized_versions(version: str | None) -> None:
    with pytest.raises(CodexSettingsError) as caught:
        codex_settings_module.require_safe_codex(version=lambda: version)
    assert "0.129.0" in str(caught.value)
    assert "sk-version-secret" not in str(caught.value)


@pytest.mark.parametrize("output,returncode", [
    (None, 0),
    ("codex-cli 0.129.0\n", 7),
])
def test_version_probe_handles_missing_or_failed_executable(
    fake_codex_version: Callable[[str | None, int], Path], output: str | None, returncode: int
) -> None:
    fake_codex_version(output, returncode)
    assert codex_settings_module._codex_version() is None


@pytest.mark.parametrize("output,returncode", [
    (None, 0),
    ("codex-cli 0.128.0\n", 0),
    ("codex-cli 0.129.0-alpha.1\n", 0),
    ("unparseable-sk-version-secret\n", 0),
    ("codex-cli 0.129.0\n", 7),
])
def test_writer_checks_the_installed_codex_before_replacing_a_key_or_receipt(
    tmp_path: Path, fake_codex_version: Callable[[str | None, int], Path],
    output: str | None, returncode: int,
) -> None:
    path: Final = tmp_path / "config.toml"
    path.write_text('model = "original"\n')
    configure_codex_settings(GATEWAY, "sk-existing-gateway", MODEL, path)
    state: Final = codex_configure_state_path(path)
    before: Final = (path.read_bytes(), state.read_bytes())
    fake_codex_version(output, returncode)
    with pytest.raises(CodexSettingsError) as caught:
        configure_codex_settings(GATEWAY, KEY, "replacement-model", path)
    assert "0.129.0" in str(caught.value)
    assert KEY not in str(caught.value) and "sk-version-secret" not in str(caught.value)
    assert (path.read_bytes(), state.read_bytes()) == before
    assert not tuple(tmp_path.rglob(".tmp-*"))


def test_undo_does_not_require_codex_to_remain_installed(
    tmp_path: Path, fake_codex_version: Callable[[str | None, int], Path]
) -> None:
    path: Final = tmp_path / "config.toml"
    original: Final = 'model = "original"\n'
    path.write_text(original)
    configure_codex_settings(GATEWAY, KEY, MODEL, path)
    fake_codex_version(None, 0)
    outcome: Final = unconfigure_codex_settings(path)
    assert outcome.restored and not outcome.kept
    assert tomlkit.parse(path.read_text()) == tomlkit.parse(original)
    assert not codex_configure_state_path(path).exists()
