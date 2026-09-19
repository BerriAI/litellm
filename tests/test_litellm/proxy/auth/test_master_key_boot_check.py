import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from litellm.proxy.auth.master_key_boot_check import (
    GENERATE_MASTER_KEY_COMMAND,
    MASTER_KEY_ENV_VAR,
    PRINT_NEW_MASTER_KEY_COMMAND,
    ROTATION_DOCS_URL,
    UNSAFE_PROXY_OVERRIDE_ENV_VAR,
    UNSAFE_PROXY_OVERRIDE_SETTING,
    ConfigFileSource,
    EnvironmentSource,
    MasterKeyBootVerdict,
    SafeMasterKey,
    UnsafeMasterKeyAllowed,
    UnsafeMasterKeyError,
    UnsafeMasterKeyReason,
    UnsafeMasterKeyRefused,
    announce_on_stderr_at_exit,
    enforce_master_key_boot_verdict,
    master_key_boot_verdict,
    render_refusal,
)


def _verdict(
    master_key: str | None,
    general_settings: Mapping[str, object] | None = None,
    *,
    environment_master_key: str | None = None,
    config_file_path: str | None = None,
    override_env_is_on: bool = False,
    salt_key_is_set: bool = False,
    database_is_configured: bool = False,
) -> MasterKeyBootVerdict:
    return master_key_boot_verdict(
        master_key=master_key,
        environment_master_key=environment_master_key,
        general_settings=general_settings or {},
        config_file_path=config_file_path,
        override_env_is_on=override_env_is_on,
        salt_key_is_set=salt_key_is_set,
        database_is_configured=database_is_configured,
    )


@pytest.mark.parametrize(
    ("master_key", "reason"),
    [
        (None, UnsafeMasterKeyReason.NOT_SET),
        ("", UnsafeMasterKeyReason.EMPTY),
        (" \t\n", UnsafeMasterKeyReason.EMPTY),
        ("sk-1234", UnsafeMasterKeyReason.PUBLICLY_KNOWN),
        ("  sk-1234\n", UnsafeMasterKeyReason.PUBLICLY_KNOWN),
    ],
)
def test_unsafe_master_keys_are_refused_with_their_reason(master_key: str | None, reason: UnsafeMasterKeyReason):
    verdict = _verdict(master_key)

    assert isinstance(verdict, UnsafeMasterKeyRefused)
    assert verdict.reason is reason


@pytest.mark.parametrize("master_key", ["sk-12345", "sk-1234567890", "1234", "sk-qa-9f2c1e7a44b0d3"])
def test_keys_that_only_resemble_the_known_default_are_safe(master_key: str):
    assert _verdict(master_key) == SafeMasterKey()


@pytest.mark.parametrize("master_key", [None, "", "sk-1234"])
def test_either_override_lets_an_unsafe_key_through(master_key: str | None):
    from_env = _verdict(master_key, override_env_is_on=True)
    from_yaml = _verdict(master_key, {UNSAFE_PROXY_OVERRIDE_SETTING: True})

    assert isinstance(from_env, UnsafeMasterKeyAllowed)
    assert from_env == from_yaml


def test_override_switched_off_in_yaml_still_refuses():
    assert isinstance(_verdict("sk-1234", {UNSAFE_PROXY_OVERRIDE_SETTING: False}), UnsafeMasterKeyRefused)


def test_yaml_master_key_is_the_source_even_when_it_resolved_to_nothing():
    verdict = _verdict(None, {"master_key": None}, config_file_path="/app/config.yaml")

    assert isinstance(verdict, UnsafeMasterKeyRefused)
    assert verdict.source == ConfigFileSource(config_file_path="/app/config.yaml")


def test_yaml_master_key_is_the_source_when_it_differs_from_the_environment():
    verdict = _verdict(
        "sk-1234",
        {"master_key": "sk-1234"},
        environment_master_key="sk-qa-9f2c1e7a44b0d3",
        config_file_path="/app/config.yaml",
    )

    assert isinstance(verdict, UnsafeMasterKeyRefused)
    assert verdict.source == ConfigFileSource(config_file_path="/app/config.yaml")


@pytest.mark.parametrize("unsafe_key", ["sk-1234", ""])
def test_environment_is_the_source_when_yaml_only_relays_the_environment_variable(unsafe_key: str):
    verdict = _verdict(
        unsafe_key,
        {"master_key": unsafe_key},
        environment_master_key=unsafe_key,
        config_file_path="/app/config.yaml",
    )

    assert isinstance(verdict, UnsafeMasterKeyRefused)
    assert verdict.source == EnvironmentSource()


def test_environment_is_the_source_when_yaml_does_not_set_a_master_key():
    verdict = _verdict("sk-1234", {"database_url": "postgresql://db"}, config_file_path="/app/config.yaml")

    assert isinstance(verdict, UnsafeMasterKeyRefused)
    assert verdict.source == EnvironmentSource()


@pytest.mark.parametrize(
    ("master_key", "salt_key_is_set", "database_is_configured", "needs_rotation"),
    [
        ("sk-1234", False, True, True),
        ("sk-1234", True, True, False),
        ("sk-1234", False, False, False),
        (None, False, True, False),
        ("", False, True, False),
    ],
)
def test_rotation_is_only_needed_when_the_known_key_encrypts_a_database(
    master_key: str | None, salt_key_is_set: bool, database_is_configured: bool, needs_rotation: bool
):
    verdict = _verdict(master_key, salt_key_is_set=salt_key_is_set, database_is_configured=database_is_configured)

    assert isinstance(verdict, UnsafeMasterKeyRefused)
    assert verdict.stored_credentials_need_rotation is needs_rotation


def _refusal(
    reason: UnsafeMasterKeyReason = UnsafeMasterKeyReason.PUBLICLY_KNOWN,
    source: ConfigFileSource | EnvironmentSource = EnvironmentSource(),
    environment_variable_is_set: bool = False,
    stored_credentials_need_rotation: bool = False,
) -> UnsafeMasterKeyRefused:
    return UnsafeMasterKeyRefused(
        reason=reason,
        source=source,
        environment_variable_is_set=environment_variable_is_set,
        stored_credentials_need_rotation=stored_credentials_need_rotation,
    )


def test_config_refusal_names_the_file_and_tells_it_to_read_the_environment():
    text = render_refusal(_refusal(source=ConfigFileSource(config_file_path="/app/config.yaml")))

    assert "general_settings.master_key in /app/config.yaml" in text
    assert f"master_key: os.environ/{MASTER_KEY_ENV_VAR}" in text
    assert GENERATE_MASTER_KEY_COMMAND in text


def test_environment_refusal_gives_the_command_without_a_config_step():
    text = render_refusal(_refusal(reason=UnsafeMasterKeyReason.NOT_SET, source=EnvironmentSource()))

    assert f"the {MASTER_KEY_ENV_VAR} environment variable" in text
    assert GENERATE_MASTER_KEY_COMMAND in text
    assert "os.environ/" not in text


@pytest.mark.parametrize(
    ("master_key", "general_settings", "environment_master_key", "is_set"),
    [
        (None, {}, None, False),
        ("sk-1234", {"master_key": "sk-1234"}, None, False),
        ("sk-1234", {}, "sk-1234", True),
        ("sk-1234", {"master_key": "sk-1234"}, "", True),
    ],
)
def test_refusal_records_whether_the_environment_variable_is_already_set(
    master_key: str | None, general_settings: Mapping[str, object], environment_master_key: str | None, is_set: bool
):
    refusal = _verdict(master_key, general_settings, environment_master_key=environment_master_key)

    assert isinstance(refusal, UnsafeMasterKeyRefused)
    assert refusal.environment_variable_is_set is is_set


@pytest.mark.parametrize("source", [EnvironmentSource(), ConfigFileSource(config_file_path="/app/config.yaml")])
def test_refusal_never_tells_a_user_with_an_exported_key_to_append_to_the_env_file(
    source: ConfigFileSource | EnvironmentSource,
):
    text = render_refusal(_refusal(source=source, environment_variable_is_set=True))

    assert PRINT_NEW_MASTER_KEY_COMMAND in text
    assert "tee" not in text
    assert "wins over .env" in text


def test_unset_key_refusal_says_nothing_supplied_one():
    text = render_refusal(_refusal(reason=UnsafeMasterKeyReason.NOT_SET, source=EnvironmentSource()))

    assert "Neither general_settings.master_key nor" in text


def test_rotation_guide_appears_only_when_needed():
    with_rotation = render_refusal(_refusal(stored_credentials_need_rotation=True))
    without_rotation = render_refusal(_refusal(stored_credentials_need_rotation=False))

    assert ROTATION_DOCS_URL in with_rotation
    assert ROTATION_DOCS_URL not in without_rotation


@pytest.mark.parametrize("source", [EnvironmentSource(), ConfigFileSource(config_file_path="/app/config.yaml")])
def test_refusal_never_tells_a_user_who_must_rotate_to_save_the_new_key_first(
    source: ConfigFileSource | EnvironmentSource,
):
    text = render_refusal(_refusal(source=source, stored_credentials_need_rotation=True))

    assert PRINT_NEW_MASTER_KEY_COMMAND in text
    assert ".env" not in text
    assert "os.environ/" not in text


@pytest.mark.parametrize("stored_credentials_need_rotation", [True, False])
def test_override_hint_is_the_last_paragraph(stored_credentials_need_rotation: bool):
    text = render_refusal(_refusal(stored_credentials_need_rotation=stored_credentials_need_rotation))
    last_paragraph = text.split("\n\n")[-1]

    assert UNSAFE_PROXY_OVERRIDE_ENV_VAR in last_paragraph
    assert f"general_settings.{UNSAFE_PROXY_OVERRIDE_SETTING}" in last_paragraph


@pytest.mark.skipif(shutil.which("openssl") is None, reason="the printed command shells out to openssl")
def test_printed_command_saves_a_key_the_boot_check_accepts(tmp_path: Path):
    completed = subprocess.run(
        ["bash", "-c", GENERATE_MASTER_KEY_COMMAND], cwd=tmp_path, capture_output=True, text=True, check=True
    )

    saved = (tmp_path / ".env").read_text()
    match = re.fullmatch(rf"{MASTER_KEY_ENV_VAR}=(sk-[0-9a-f]{{64}})\n", saved)
    assert match is not None
    assert completed.stdout == saved
    assert _verdict(match.group(1)) == SafeMasterKey()


@pytest.mark.skipif(shutil.which("openssl") is None, reason="the printed command shells out to openssl")
def test_rotation_command_prints_a_key_the_boot_check_accepts_and_saves_nothing(tmp_path: Path):
    completed = subprocess.run(
        ["bash", "-c", PRINT_NEW_MASTER_KEY_COMMAND], cwd=tmp_path, capture_output=True, text=True, check=True
    )

    assert re.fullmatch(r"sk-[0-9a-f]{64}\n", completed.stdout) is not None
    assert _verdict(completed.stdout.strip()) == SafeMasterKey()
    assert list(tmp_path.iterdir()) == []


def test_refusal_announces_the_fix_and_aborts_the_boot():
    announced: list[str] = []
    refusal = _refusal(source=ConfigFileSource(config_file_path="/app/config.yaml"))

    with pytest.raises(UnsafeMasterKeyError, match="refused to start"):
        enforce_master_key_boot_verdict(refusal, announce=announced.append)

    assert [message.strip() for message in announced] == [render_refusal(refusal)]


@pytest.mark.parametrize("verdict", [SafeMasterKey(), UnsafeMasterKeyAllowed(reason=UnsafeMasterKeyReason.NOT_SET)])
def test_safe_and_overridden_keys_boot_without_announcing(verdict: MasterKeyBootVerdict):
    announced: list[str] = []

    enforce_master_key_boot_verdict(verdict, announce=announced.append)

    assert announced == []


def test_announced_fix_is_the_last_thing_a_crashing_process_prints():
    crash_after_announcing = (
        "from litellm.proxy.auth.master_key_boot_check import announce_on_stderr_at_exit\n"
        "announce_on_stderr_at_exit('THE FIX')\n"
        "print('buffered stdout')\n"
        "raise RuntimeError('lifespan failed')\n"
    )

    completed = subprocess.run([sys.executable, "-c", crash_after_announcing], capture_output=True, text=True)

    assert completed.returncode != 0
    assert "RuntimeError: lifespan failed" in completed.stderr
    assert completed.stderr.endswith("THE FIX")
    assert completed.stdout == "buffered stdout\n"
