import atexit
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final

from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger

UNSAFE_PROXY_OVERRIDE_SETTING: Final = "dangerously_allow_unsafe_proxy"
UNSAFE_PROXY_OVERRIDE_ENV_VAR: Final = "LITELLM_DANGEROUSLY_ALLOW_UNSAFE_PROXY"
MASTER_KEY_SETTING: Final = "master_key"
MASTER_KEY_ENV_VAR: Final = "LITELLM_MASTER_KEY"
SALT_KEY_ENV_VAR: Final = "LITELLM_SALT_KEY"
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
class UnsafeMasterKeyRefused:
    reason: UnsafeMasterKeyReason
    source: MasterKeySource
    environment_variable_is_set: bool
    stored_credentials_need_rotation: bool


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
    if override_env_is_on or general_settings.get(UNSAFE_PROXY_OVERRIDE_SETTING) is True:
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
        stored_credentials_need_rotation=(
            reason is UnsafeMasterKeyReason.PUBLICLY_KNOWN and not salt_key_is_set and database_is_configured
        ),
    )


def enforce_master_key_boot_verdict(verdict: MasterKeyBootVerdict, announce: Callable[[str], object]) -> None:
    match verdict:
        case SafeMasterKey():
            return
        case UnsafeMasterKeyAllowed(reason=reason):
            verbose_proxy_logger.warning(
                "%s is on, so the proxy is starting with %s. Never run this outside local development.",
                UNSAFE_PROXY_OVERRIDE_SETTING,
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
            _ROTATE_INSTEAD_OF_REPLACING if refusal.stored_credentials_need_rotation else _fix_steps(refusal),
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

_ROTATE_INSTEAD_OF_REPLACING: Final = (
    f"Credentials stored in your database are encrypted with this master key because {SALT_KEY_ENV_VAR} is not\n"
    "set, so replacing the key makes them undecryptable. Rotate it by following this guide, which re-encrypts them:\n"
    f"     {ROTATION_DOCS_URL}\n"
    "Generate the new key for it with (save it only once the guide says to):\n"
    f"     {PRINT_NEW_MASTER_KEY_COMMAND}"
)

_OVERRIDE_HINT: Final = (
    f"Local development only: set {UNSAFE_PROXY_OVERRIDE_ENV_VAR}=true, or\n"
    f"general_settings.{UNSAFE_PROXY_OVERRIDE_SETTING}: true, to start anyway."
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
    set_key_step: Final = _REPLACE_EXPORTED_KEY_STEP if refusal.environment_variable_is_set else _SAVE_KEY_STEP
    match refusal.source:
        case ConfigFileSource() as source:
            return (
                f"1. Make sure {_config_label(source)} reads the key from the environment:\n"
                f"     general_settings:\n       {MASTER_KEY_SETTING}: os.environ/{MASTER_KEY_ENV_VAR}\n"
                f"2. {set_key_step}"
            )
        case EnvironmentSource():
            return f"1. {set_key_step}"
        case _:
            assert_never(refusal.source)
