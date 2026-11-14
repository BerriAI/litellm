import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from pydantic import JsonValue, TypeAdapter
from typing_extensions import assert_never

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.proxy.auth.master_key_boot_check import (
    MIGRATE_FROM_MASTER_KEY_ENV_VAR,
    MIGRATE_FROM_SALT_KEY_ENV_VAR,
    SALT_KEY_ENV_VAR,
)
from litellm.proxy.common_utils.callback_utils import CALLBACK_VAR_ENCRYPTED_PREFIX
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with, encrypt_value_helper
from litellm.proxy.db.create_views import SupportsRawQueries


@dataclass(frozen=True, slots=True)
class _SecretColumn:
    table: str
    primary_key: str
    column: str
    is_json: bool = True
    only_rows_with_marked_ciphertexts: bool = False


_SECRET_COLUMNS: Final = (
    _SecretColumn("LiteLLM_ProxyModelTable", "model_id", "litellm_params"),
    _SecretColumn("LiteLLM_CredentialsTable", "credential_id", "credential_values"),
    _SecretColumn("LiteLLM_Config", "param_name", "param_value"),
    _SecretColumn("LiteLLM_GuardrailsTable", "guardrail_id", "litellm_params", only_rows_with_marked_ciphertexts=True),
    _SecretColumn("LiteLLM_SSOConfig", "id", "sso_settings"),
    _SecretColumn("LiteLLM_CacheConfig", "id", "cache_settings"),
    _SecretColumn("LiteLLM_ConfigOverrides", "config_type", "config_value"),
    _SecretColumn("LiteLLM_MCPServerTable", "server_id", "credentials"),
    _SecretColumn("LiteLLM_MCPServerTable", "server_id", "static_headers"),
    _SecretColumn("LiteLLM_MCPServerTable", "server_id", "env_vars"),
    _SecretColumn("LiteLLM_MCPServerTable", "server_id", "env"),
    _SecretColumn("LiteLLM_MCPServerOAuthClient", "server_id", "credentials"),
    _SecretColumn("LiteLLM_MCPUserCredentials", "id", "credential_b64", is_json=False),
    _SecretColumn("LiteLLM_MCPUserEnvVars", "id", "values_b64", is_json=False),
    _SecretColumn("LiteLLM_UserProviderCredentials", "id", "credential_b64", is_json=False),
    _SecretColumn("LiteLLM_SSOIdentityAssertion", "user_id", "assertion_b64", is_json=False),
    _SecretColumn("LiteLLM_SearchToolsTable", "search_tool_id", "litellm_params"),
    _SecretColumn("LiteLLM_TeamTable", "team_id", "metadata", only_rows_with_marked_ciphertexts=True),
    _SecretColumn("LiteLLM_VerificationToken", "token", "metadata", only_rows_with_marked_ciphertexts=True),
    _SecretColumn("LiteLLM_UserTable", "user_id", "metadata", only_rows_with_marked_ciphertexts=True),
    _SecretColumn("LiteLLM_DeletedTeamTable", "id", "metadata", only_rows_with_marked_ciphertexts=True),
    _SecretColumn("LiteLLM_DeletedVerificationToken", "id", "metadata", only_rows_with_marked_ciphertexts=True),
)

_STORED_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_PRIMARY_KEY: Final = TypeAdapter(str)

ReplaceCiphertext = Callable[[str], str | None]


def replace_ciphertexts(value: JsonValue, replacement_for: ReplaceCiphertext, depth: int = 0) -> tuple[JsonValue, int]:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return value, 0
    match value:
        case str():
            marker: Final = CALLBACK_VAR_ENCRYPTED_PREFIX if value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX) else ""
            replacement: Final = replacement_for(value.removeprefix(marker))
            return (value, 0) if replacement is None else (marker + replacement, 1)
        case list():
            items: Final = tuple(replace_ciphertexts(item, replacement_for, depth + 1) for item in value)
            return [item for item, _ in items], sum(count for _, count in items)
        case dict():
            fields: Final = {key: replace_ciphertexts(item, replacement_for, depth + 1) for key, item in value.items()}
            return {key: item for key, (item, _) in fields.items()}, sum(count for _, count in fields.values())
        case _:
            return value, 0


async def count_values_encrypted_with(database: SupportsRawQueries, signing_key: str) -> int:
    def keep(value: str) -> str | None:
        return None if decrypt_if_encrypted_with(value, signing_key) is None else value

    return sum(
        [
            replace_ciphertexts(_STORED_VALUE.validate_python(row[secret_column.column]), keep)[1]
            for secret_column in await _secret_columns_in(database)
            for row in await _rows_of(database, secret_column)
        ]
    )


async def count_values_encrypted_with_or_none(
    connect: Callable[[], Awaitable[SupportsRawQueries]], signing_key: str
) -> int | None:
    try:
        return await count_values_encrypted_with(await connect(), signing_key)
    except Exception:  # noqa: BLE001  # an unreadable database must not replace the boot refusal with a traceback
        return None


async def reencrypt_stored_values(database: SupportsRawQueries, *, from_key: str, to_key: str) -> int:
    def reencrypted(value: str) -> str | None:
        plaintext: Final = decrypt_if_encrypted_with(value, from_key)
        return None if plaintext is None else _CIPHERTEXT.validate_python(encrypt_value_helper(plaintext, to_key))

    return sum(
        [
            await _reencrypt_row(database, secret_column, row, reencrypted)
            for secret_column in await _secret_columns_in(database)
            for row in await _rows_of(database, secret_column)
        ]
    )


_CIPHERTEXT: Final = TypeAdapter(str)


async def _secret_columns_in(database: SupportsRawQueries) -> tuple[_SecretColumn, ...]:
    existing: Final = frozenset(
        (row["table_name"], row["column_name"])
        for row in await database.query_raw(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = ANY (current_schemas(false))"
        )
    )
    return tuple(
        secret_column for secret_column in _SECRET_COLUMNS if (secret_column.table, secret_column.column) in existing
    )


async def _rows_of(database: SupportsRawQueries, secret_column: _SecretColumn) -> tuple[Mapping[str, object], ...]:
    marked_only: Final = (
        f" AND \"{secret_column.column}\"::text LIKE '%{CALLBACK_VAR_ENCRYPTED_PREFIX}%'"
        if secret_column.only_rows_with_marked_ciphertexts
        else ""
    )
    return tuple(
        await database.query_raw(
            f'SELECT "{secret_column.primary_key}", "{secret_column.column}" FROM "{secret_column.table}" '
            f'WHERE "{secret_column.column}" IS NOT NULL{marked_only}'
        )
    )


async def _reencrypt_row(
    database: SupportsRawQueries,
    secret_column: _SecretColumn,
    row: Mapping[str, object],
    reencrypted: ReplaceCiphertext,
) -> int:
    stored: Final = _STORED_VALUE.validate_python(row[secret_column.column])
    migrated, count = replace_ciphertexts(stored, reencrypted)
    if count == 0:
        return 0
    cast_to: Final = "::jsonb" if secret_column.is_json else ""
    rows_updated: Final = await database.execute_raw(
        f'UPDATE "{secret_column.table}" SET "{secret_column.column}" = $1{cast_to} '
        f'WHERE "{secret_column.primary_key}" = $2 AND "{secret_column.column}" = $3{cast_to}',
        _as_sql_parameter(migrated, secret_column),
        _PRIMARY_KEY.validate_python(row[secret_column.primary_key]),
        _as_sql_parameter(stored, secret_column),
    )
    return count if rows_updated else 0


def _as_sql_parameter(value: JsonValue, secret_column: _SecretColumn) -> str:
    return json.dumps(value) if secret_column.is_json else _CIPHERTEXT.validate_python(value)


class NothingToMigrate(Enum):
    SALT_KEY_ENCRYPTS_STORED_VALUES = "salt_key_encrypts_stored_values"
    SALT_KEY_NOT_SET = "salt_key_not_set"
    NO_DATABASE = "no_database"
    NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY = "nothing_encrypted_with_previous_key"


@dataclass(frozen=True, slots=True)
class Migrated:
    migrated: int
    remaining: int


@dataclass(frozen=True, slots=True)
class MigrationFailed:
    error: Exception


MigrationOutcome = NothingToMigrate | Migrated | MigrationFailed


@dataclass(frozen=True, slots=True)
class KeyRotation:
    previous_key_variable: str
    key_name: str


MASTER_KEY_ROTATION: Final = KeyRotation(previous_key_variable=MIGRATE_FROM_MASTER_KEY_ENV_VAR, key_name="master key")
SALT_KEY_ROTATION: Final = KeyRotation(previous_key_variable=MIGRATE_FROM_SALT_KEY_ENV_VAR, key_name="salt key")


async def migrate_stored_values_if_requested(
    *,
    environ: Mapping[str, str],
    master_key: str | None,
    connected_database: Callable[[], SupportsRawQueries | None],
    log: Callable[[str], None],
    raise_unless_tolerated: Callable[[Exception], None],
) -> None:
    await migrate_if_requested(
        environ=environ,
        master_key=master_key,
        connected_database=connected_database,
        log=log,
        raise_unless_tolerated=raise_unless_tolerated,
    )
    await migrate_salt_key_if_requested(
        environ=environ,
        connected_database=connected_database,
        log=log,
        raise_unless_tolerated=raise_unless_tolerated,
    )


async def migrate_if_requested(
    *,
    environ: Mapping[str, str],
    master_key: str | None,
    connected_database: Callable[[], SupportsRawQueries | None],
    log: Callable[[str], None],
    raise_unless_tolerated: Callable[[Exception], None],
) -> MigrationOutcome | None:
    previous_master_key: Final = environ.get(MIGRATE_FROM_MASTER_KEY_ENV_VAR)
    if previous_master_key is None or master_key is None:
        return None
    return _raised_unless_tolerated(
        await migrate_from_previous_master_key(
            previous_master_key=previous_master_key,
            master_key=master_key,
            salt_key_is_set=SALT_KEY_ENV_VAR in environ,
            database=connected_database(),
            log=log,
        ),
        raise_unless_tolerated,
    )


async def migrate_salt_key_if_requested(
    *,
    environ: Mapping[str, str],
    connected_database: Callable[[], SupportsRawQueries | None],
    log: Callable[[str], None],
    raise_unless_tolerated: Callable[[Exception], None],
) -> MigrationOutcome | None:
    previous_salt_key: Final = environ.get(MIGRATE_FROM_SALT_KEY_ENV_VAR)
    if previous_salt_key is None:
        return None
    return _raised_unless_tolerated(
        await migrate_from_previous_salt_key(
            previous_salt_key=previous_salt_key,
            salt_key=environ.get(SALT_KEY_ENV_VAR),
            database=connected_database(),
            log=log,
        ),
        raise_unless_tolerated,
    )


def _raised_unless_tolerated(
    outcome: MigrationOutcome, raise_unless_tolerated: Callable[[Exception], None]
) -> MigrationOutcome:
    if isinstance(outcome, MigrationFailed):
        raise_unless_tolerated(outcome.error)
    return outcome


async def migrate_from_previous_master_key(
    *,
    previous_master_key: str,
    master_key: str,
    salt_key_is_set: bool,
    database: SupportsRawQueries | None,
    log: Callable[[str], None],
) -> MigrationOutcome:
    outcome: Final = (
        NothingToMigrate.SALT_KEY_ENCRYPTS_STORED_VALUES
        if salt_key_is_set
        else await _migrate_or_failure(
            MASTER_KEY_ROTATION, previous_key=previous_master_key, key=master_key, database=database, log=log
        )
    )
    log(describe_outcome(outcome, MASTER_KEY_ROTATION))
    return outcome


async def migrate_from_previous_salt_key(
    *,
    previous_salt_key: str,
    salt_key: str | None,
    database: SupportsRawQueries | None,
    log: Callable[[str], None],
) -> MigrationOutcome:
    outcome: Final = (
        NothingToMigrate.SALT_KEY_NOT_SET
        if salt_key is None
        else await _migrate_or_failure(
            SALT_KEY_ROTATION, previous_key=previous_salt_key, key=salt_key, database=database, log=log
        )
    )
    log(describe_outcome(outcome, SALT_KEY_ROTATION))
    return outcome


async def _migrate_or_failure(
    rotation: KeyRotation,
    *,
    previous_key: str,
    key: str,
    database: SupportsRawQueries | None,
    log: Callable[[str], None],
) -> MigrationOutcome:
    try:
        return await _migrate(rotation, previous_key=previous_key, key=key, database=database, log=log)
    except Exception as error:  # noqa: BLE001  # a value, so the boot applies its own database outage rule to it
        return MigrationFailed(error=error)


async def _migrate(
    rotation: KeyRotation,
    *,
    previous_key: str,
    key: str,
    database: SupportsRawQueries | None,
    log: Callable[[str], None],
) -> MigrationOutcome:
    if database is None:
        return NothingToMigrate.NO_DATABASE
    if previous_key == key:
        return NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY
    found: Final = await count_values_encrypted_with(database, previous_key)
    if found == 0:
        return NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY
    log(
        f"Re-encrypting {found} stored value(s) from the {rotation.previous_key_variable} key to the new "
        f"{rotation.key_name}."
    )
    migrated: Final = Migrated(
        migrated=await reencrypt_stored_values(database, from_key=previous_key, to_key=key),
        remaining=await count_values_encrypted_with(database, previous_key),
    )
    another_worker_migrated_everything: Final = migrated == Migrated(migrated=0, remaining=0)
    return NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY if another_worker_migrated_everything else migrated


def describe_outcome(outcome: MigrationOutcome, rotation: KeyRotation) -> str:
    variable: Final = rotation.previous_key_variable
    match outcome:
        case NothingToMigrate.SALT_KEY_ENCRYPTS_STORED_VALUES:
            return (
                f"{variable} is set, but {SALT_KEY_ENV_VAR} is what encrypts your stored "
                f"values, so there is nothing to migrate. You may now delete {variable}."
            )
        case NothingToMigrate.SALT_KEY_NOT_SET:
            return (
                f"{variable} is set, but {SALT_KEY_ENV_VAR} is not, so nothing was migrated. Set {SALT_KEY_ENV_VAR} "
                f"to the new salt key, keep {variable} set to the previous one, and restart the proxy."
            )
        case NothingToMigrate.NO_DATABASE:
            return (
                f"{variable} is set, but no database is connected, so nothing was migrated. If "
                f"this proxy has no database, you may now delete {variable}."
            )
        case NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY:
            return (
                f"{variable} is still set, but nothing in the database is left to migrate "
                f"from that key. You may now delete {variable}."
            )
        case Migrated(migrated=migrated, remaining=0):
            return (
                f"Done re-encrypting {migrated} stored value(s) with the new {rotation.key_name}. You may now delete "
                f"the {variable} environment variable."
            )
        case Migrated(migrated=migrated, remaining=remaining):
            return (
                f"Re-encrypted {migrated} stored value(s), but {remaining} are still encrypted with the previous key "
                f"because they changed during the migration. Keep {variable} set and restart "
                "the proxy to migrate them."
            )
        case MigrationFailed(error=error):
            cause: Final = f"{type(error).__name__}: {error}"[:300]
            return (
                f"Could not migrate stored values from the {variable} key ({cause}). Values "
                "still encrypted with the previous key cannot be read until the migration succeeds. Keep "
                f"{variable} set and restart the proxy once the database is reachable."
            )
        case _:
            assert_never(outcome)
