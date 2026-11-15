import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.auth.master_key_boot_check import (
    GENERATE_MASTER_KEY_COMMAND,
    MASTER_KEY_ENV_VAR,
    MIGRATE_FROM_MASTER_KEY_ENV_VAR,
    PRINT_NEW_MASTER_KEY_COMMAND,
    ConfigFileSource,
    EnvironmentSource,
    MasterKeyBootVerdict,
    SafeMasterKey,
    StoredSecretsMigration,
    UnsafeMasterKeyReason,
    UnsafeMasterKeyRefused,
    announce_on_stderr_at_exit,
    master_key_boot_verdict,
    render_refusal,
)

PUBLICLY_KNOWN_KEY: Final = "sk-" + "1234"


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


def _refusal(
    reason: UnsafeMasterKeyReason = UnsafeMasterKeyReason.PUBLICLY_KNOWN,
    source: ConfigFileSource | EnvironmentSource = EnvironmentSource(),
    environment_variable_is_set: bool = False,
    migration: StoredSecretsMigration | None = None,
) -> UnsafeMasterKeyRefused:
    return UnsafeMasterKeyRefused(
        reason=reason,
        source=source,
        environment_variable_is_set=environment_variable_is_set,
        migration=migration,
    )


_MIGRATION = StoredSecretsMigration(from_master_key=PUBLICLY_KNOWN_KEY, encrypted_value_count=3)


@pytest.mark.skipif(shutil.which("openssl") is None, reason="the printed command shells out to openssl")
@pytest.mark.parametrize("from_master_key", [PUBLICLY_KNOWN_KEY, "", " " + PUBLICLY_KNOWN_KEY])
def test_printed_migration_commands_save_both_keys_to_the_env_file(tmp_path: Path, from_master_key: str):
    from dotenv import dotenv_values

    text = render_refusal(
        _refusal(migration=StoredSecretsMigration(from_master_key=from_master_key, encrypted_value_count=1))
    )
    commands = [line.strip() for line in text.splitlines() if line.strip().startswith("echo ")]

    subprocess.run(["bash", "-c", "\n".join(commands)], cwd=tmp_path, capture_output=True, text=True, check=True)

    saved = dotenv_values(tmp_path / ".env")
    assert saved[MIGRATE_FROM_MASTER_KEY_ENV_VAR] == from_master_key
    assert _verdict(saved[MASTER_KEY_ENV_VAR]) == SafeMasterKey()
    assert len(commands) == 2


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
