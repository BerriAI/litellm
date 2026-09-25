import atexit
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Final

from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger

WEAK_OR_UNSET_MASTER_KEY_OVERRIDE_SETTING: Final = "dangerously_permit_weak_or_unset_master_key"
WEAK_OR_UNSET_MASTER_KEY_OVERRIDE_ENV_VAR: Final = "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY"
MASTER_KEY_SETTING: Final = "master_key"
MASTER_KEY_ENV_VAR: Final = "LITELLM_MASTER_KEY"
SALT_KEY_ENV_VAR: Final = "LITELLM_SALT_KEY"
MIGRATE_FROM_MASTER_KEY_ENV_VAR: Final = "LITELLM_MIGRATE_FROM_MASTER_KEY"
PUBLICLY_KNOWN_MASTER_KEYS: Final = frozenset({"sk-1234"})
ROTATION_DOCS_URL: Final = "https://docs.litellm.ai/docs/proxy/master_key_rotations#proxy-refuses-to-start"
_NEW_MASTER_KEY: Final = "sk-$(openssl rand -hex 32)"
GENERATE_MASTER_KEY_COMMAND: Final = f'echo "{MASTER_KEY_ENV_VAR}={_NEW_MASTER_KEY}" | tee -a .env'
PRINT_NEW_MASTER_KEY_COMMAND: Final = f'echo "{_NEW_MASTER_KEY}"'


class UnsafeMasterKeyReason(Enum):
    NOT_SET = "not_set"
    EMPTY = "empty"
    PUBLICLY_KNOWN = "publicly_known"


@dataclass(frozen=True, slots=True)
class ConfigFileSource:
    config_file_path: str | None


@dataclass(frozen=True, slots=True)
class EnvironmentSource:
    pass


MasterKeySource = ConfigFileSource | EnvironmentSource


@dataclass(frozen=True, slots=True)
class SafeMasterKey:
    pass


@dataclass(frozen=True, slots=True)
class UnsafeMasterKeyAllowed:
    reason: UnsafeMasterKeyReason


@dataclass(frozen=True, slots=True)
class StoredSecretsMigration:
    from_master_key: str
    encrypted_value_count: int | None


@dataclass(frozen=True, slots=True)
class UnsafeMasterKeyRefused:
    reason: UnsafeMasterKeyReason
    source: MasterKeySource
    environment_variable_is_set: bool
    migration: StoredSecretsMigration | None


MasterKeyBootVerdict = SafeMasterKey | UnsafeMasterKeyAllowed | UnsafeMasterKeyRefused


class UnsafeMasterKeyError(Exception):
    pass


def master_key_boot_verdict(
    *,
    master_key: str | None,
    environment_master_key: str | None,
    general_settings: Mapping[str, object],
    config_file_path: str | None,
    override_env_is_on: bool,
    salt_key_is_set: bool,
    database_is_configured: bool,
) -> MasterKeyBootVerdict:
    reason: Final = _unsafe_reason(master_key)
    if reason is None:
        return SafeMasterKey()
    if override_env_is_on or general_settings.get(WEAK_OR_UNSET_MASTER_KEY_OVERRIDE_SETTING) is True:
        return UnsafeMasterKeyAllowed(reason=reason)
    config_file_only_relays_the_environment: Final = master_key is not None and master_key == environment_master_key
    return UnsafeMasterKeyRefused(
        reason=reason,
        source=(
            ConfigFileSource(config_file_path=config_file_path)
            if MASTER_KEY_SETTING in general_settings and not config_file_only_relays_the_environment
            else EnvironmentSource()
        ),
        environment_variable_is_set=environment_master_key is not None,
        migration=(
            StoredSecretsMigration(from_master_key=master_key, encrypted_value_count=None)
            if master_key is not None and not salt_key_is_set and database_is_configured
            else None
        ),
    )


async def with_stored_secrets_counted(
    verdict: MasterKeyBootVerdict, count_values_encrypted_with: Callable[[str], Awaitable[int | None]]
) -> MasterKeyBootVerdict:
    if not isinstance(verdict, UnsafeMasterKeyRefused) or verdict.migration is None:
        return verdict
    count: Final = await count_values_encrypted_with(verdict.migration.from_master_key)
    return replace(verdict, migration=None if count == 0 else replace(verdict.migration, encrypted_value_count=count))


def enforce_master_key_boot_verdict(verdict: MasterKeyBootVerdict, announce: Callable[[str], object]) -> None:
    match verdict:
        case SafeMasterKey():
            return
        case UnsafeMasterKeyAllowed(reason=reason):
            verbose_proxy_logger.warning(
                "%s is on, so the proxy is starting with %s. Never run this outside local development.",
                WEAK_OR_UNSET_MASTER_KEY_OVERRIDE_SETTING,
                _UNSAFE_STATE[reason],
            )
        case UnsafeMasterKeyRefused(reason=reason):
            announce(f"\n{render_refusal(verdict)}\n\n")
            raise UnsafeMasterKeyError(
                f"LiteLLM proxy refused to start: {_REFUSAL_HEADLINE[reason]} The fix is printed once the server exits."
            )
        case _:
            assert_never(verdict)


def announce_on_stderr_at_exit(message: str) -> None:
    """A logger would redact the key-shaped command and the lifespan traceback would bury it, so print at exit."""
    atexit.register(_flush_stdout_then_write_stderr, message)


def _flush_stdout_then_write_stderr(message: str) -> None:
    sys.stdout.flush()
    sys.stderr.write(message)


def render_refusal(refusal: UnsafeMasterKeyRefused) -> str:
    return "\n\n".join(
        (
            f"LiteLLM proxy refused to start: {_REFUSAL_HEADLINE[refusal.reason]}\n{_source_line(refusal)}",
            _fix_steps(refusal),
            _OVERRIDE_HINT,
        )
    )


_UNSAFE_STATE: Final = MappingProxyType(
    {
        UnsafeMasterKeyReason.NOT_SET: "no master key, which accepts every request without authentication",
        UnsafeMasterKeyReason.EMPTY: "an empty master key",
        UnsafeMasterKeyReason.PUBLICLY_KNOWN: "a publicly known master key",
    }
)

_REFUSAL_HEADLINE: Final = MappingProxyType(
    {
        UnsafeMasterKeyReason.NOT_SET: (
            "no master key is set, so every request would be accepted without authentication."
        ),
        UnsafeMasterKeyReason.EMPTY: "the master key is empty.",
        UnsafeMasterKeyReason.PUBLICLY_KNOWN: "the master key is a publicly known default.",
    }
)

_SAVE_KEY_STEP: Final = (
    "Generate a key and save it to .env:\n"
    f"     {GENERATE_MASTER_KEY_COMMAND}\n"
    "   Not using a .env file (docker run, Kubernetes, pip install)? Pass the same value as the\n"
    f"   {MASTER_KEY_ENV_VAR} environment variable instead."
)

_REPLACE_EXPORTED_KEY_STEP: Final = (
    "Generate a key:\n"
    f"     {PRINT_NEW_MASTER_KEY_COMMAND}\n"
    f"   Put it in place of the current {MASTER_KEY_ENV_VAR} value wherever that is set: a shell export, your\n"
    "   container or deployment environment, or its line in .env. Do not just add it to .env, because a value\n"
    "   already exported in the environment wins over .env."
)

_RESTART_TO_MIGRATE_STEP: Final = (
    "Start the proxy again. It re-encrypts the stored values with the new key, then logs that\n"
    f"   {MIGRATE_FROM_MASTER_KEY_ENV_VAR} can be removed. Details: {ROTATION_DOCS_URL}"
)

_OVERRIDE_HINT: Final = (
    f"Local development only: set {WEAK_OR_UNSET_MASTER_KEY_OVERRIDE_ENV_VAR}=true, or\n"
    f"general_settings.{WEAK_OR_UNSET_MASTER_KEY_OVERRIDE_SETTING}: true, to start anyway."
)


def _unsafe_reason(master_key: str | None) -> UnsafeMasterKeyReason | None:
    if master_key is None:
        return UnsafeMasterKeyReason.NOT_SET
    stripped: Final = master_key.strip()
    if not stripped:
        return UnsafeMasterKeyReason.EMPTY
    if stripped in PUBLICLY_KNOWN_MASTER_KEYS:
        return UnsafeMasterKeyReason.PUBLICLY_KNOWN
    return None


def _config_label(source: ConfigFileSource) -> str:
    return source.config_file_path or "your config"


def _source_line(refusal: UnsafeMasterKeyRefused) -> str:
    match refusal.source:
        case ConfigFileSource() as source:
            if refusal.reason is UnsafeMasterKeyReason.NOT_SET:
                return (
                    f"general_settings.{MASTER_KEY_SETTING} in {_config_label(source)} is blank, or points at an "
                    "environment variable that is not set."
                )
            return f"It comes from general_settings.{MASTER_KEY_SETTING} in {_config_label(source)}."
        case EnvironmentSource():
            if refusal.reason is UnsafeMasterKeyReason.NOT_SET:
                return (
                    f"Neither general_settings.{MASTER_KEY_SETTING} nor the {MASTER_KEY_ENV_VAR} "
                    "environment variable is set."
                )
            return f"It comes from the {MASTER_KEY_ENV_VAR} environment variable."
        case _:
            assert_never(refusal.source)


def _fix_steps(refusal: UnsafeMasterKeyRefused) -> str:
    steps: Final = (*_config_steps(refusal.source), *_key_steps(refusal))
    numbered: Final = "\n".join(f"{number}. {step}" for number, step in enumerate(steps, start=1))
    return numbered if refusal.migration is None else f"{_migration_lead(refusal.migration)}\n{numbered}"


def _config_steps(source: MasterKeySource) -> tuple[str, ...]:
    match source:
        case ConfigFileSource():
            return (
                f"Make sure {_config_label(source)} reads the key from the environment:\n"
                f"     general_settings:\n       {MASTER_KEY_SETTING}: os.environ/{MASTER_KEY_ENV_VAR}",
            )
        case EnvironmentSource():
            return ()
        case _:
            assert_never(source)


def _key_steps(refusal: UnsafeMasterKeyRefused) -> tuple[str, ...]:
    if refusal.migration is None:
        return (_REPLACE_EXPORTED_KEY_STEP if refusal.environment_variable_is_set else _SAVE_KEY_STEP,)
    if refusal.environment_variable_is_set:
        return (
            f"Set the key to migrate from next to {MASTER_KEY_ENV_VAR}, wherever that is set (a shell export, your\n"
            "   container or deployment environment, or .env):\n"
            f"     {_migrate_from_assignment(refusal.migration)}",
            _REPLACE_EXPORTED_KEY_STEP,
            _RESTART_TO_MIGRATE_STEP,
        )
    return (
        "Save the key to migrate from and a newly generated key to .env:\n"
        f"     echo '{_migrate_from_assignment(refusal.migration)}' | tee -a .env\n"
        f"     {GENERATE_MASTER_KEY_COMMAND}\n"
        "   Not using a .env file (docker run, Kubernetes, pip install)? Pass the same two values as\n"
        "   environment variables instead.",
        _RESTART_TO_MIGRATE_STEP,
    )


def _migrate_from_assignment(migration: StoredSecretsMigration) -> str:
    key: Final = migration.from_master_key
    value: Final = key if key == key.strip() else f'"{key}"'
    return f"{MIGRATE_FROM_MASTER_KEY_ENV_VAR}={value}"


def _migration_lead(migration: StoredSecretsMigration) -> str:
    found: Final = (
        "could not be checked for values"
        if migration.encrypted_value_count is None
        else f"holds {migration.encrypted_value_count} value(s)"
    )
    return (
        f"Your database {found} encrypted with this master key,\n"
        f"which encrypts stored credentials while {SALT_KEY_ENV_VAR} is not set. Replacing the key alone makes them\n"
        "unreadable, so also tell the proxy which key to migrate from:"
    )
