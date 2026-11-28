import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.proxy.auth.master_key_boot_check import MIGRATE_FROM_MASTER_KEY_ENV_VAR, SALT_KEY_ENV_VAR
from litellm.proxy.common_utils.callback_utils import CALLBACK_VAR_ENCRYPTED_PREFIX
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    V2_GCM_PREFIX,
    decode_legacy_mcp_user_credential,
    decrypt_if_encrypted_with,
    encrypt_secret_map,
    encrypt_value_helper,
)
from litellm.proxy.db.create_views import SupportsRawQueries


@dataclass(frozen=True, slots=True)
class _SecretColumn:
    table: str
    primary_key: str
    column: str
    is_json: bool = True
    only_rows_with_marked_ciphertexts: bool = False
    legacy_mcp_format: Literal["credential", "secret_map"] | None = None
    in_lock_order: bool = False


_AUTH_PROFILE_COLUMN: Final = _SecretColumn(
    "LiteLLM_MCPServerTable", "server_id", "additional_auth", in_lock_order=True
)
_SECRET_COLUMNS: Final = (
    _SecretColumn("LiteLLM_ProxyModelTable", "model_id", "litellm_params"),
    _SecretColumn("LiteLLM_CredentialsTable", "credential_id", "credential_values"),
    _SecretColumn("LiteLLM_Config", "param_name", "param_value"),
    _SecretColumn("LiteLLM_GuardrailsTable", "guardrail_id", "litellm_params", only_rows_with_marked_ciphertexts=True),
    _SecretColumn("LiteLLM_SSOConfig", "id", "sso_settings"),
    _SecretColumn("LiteLLM_CacheConfig", "id", "cache_settings"),
    _SecretColumn("LiteLLM_ConfigOverrides", "config_type", "config_value"),
    _SecretColumn("LiteLLM_MCPServerTable", "server_id", "credentials"),
    _AUTH_PROFILE_COLUMN,
    _SecretColumn("LiteLLM_MCPServerTable", "server_id", "static_headers", legacy_mcp_format="secret_map"),
    _SecretColumn("LiteLLM_MCPServerTable", "server_id", "env_vars"),
    _SecretColumn("LiteLLM_MCPServerTable", "server_id", "env", legacy_mcp_format="secret_map"),
    _SecretColumn("LiteLLM_MCPServerOAuthClient", "server_id", "credentials"),
    _SecretColumn("LiteLLM_MCPUserCredentials", "id", "credential_b64", is_json=False, legacy_mcp_format="credential"),
    _SecretColumn("LiteLLM_UserProviderCredentials", "id", "credential_b64", is_json=False),
    _SecretColumn("LiteLLM_MCPUserEnvVars", "id", "values_b64", is_json=False),
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
_CIPHERTEXT: Final = TypeAdapter(str)
_MCP_SECRET_MAP: Final = TypeAdapter(dict[str, str])

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
            sum(
                _replace_column_ciphertexts(
                    secret_column, _STORED_VALUE.validate_python(row[secret_column.column]), keep
                )[1]
                for row in await _rows_of(database, secret_column)
            )
            for secret_column in await _secret_columns_in(database)
        ]
    )


async def count_values_encrypted_with_or_none(
    connect: Callable[[], Awaitable[SupportsRawQueries]], signing_key: str
) -> int | None:
    try:
        return await count_values_encrypted_with(await connect(), signing_key)
    except Exception:  # noqa: BLE001  # an unreadable database must not replace the boot refusal with a traceback
        return None


def _reencrypting(from_key: str, to_key: str) -> ReplaceCiphertext:
    def reencrypted(value: str) -> str | None:
        if from_key == to_key and value.startswith(V2_GCM_PREFIX):
            return None
        plaintext: Final = decrypt_if_encrypted_with(value, from_key)
        return None if plaintext is None else _CIPHERTEXT.validate_python(encrypt_value_helper(plaintext, to_key))

    return reencrypted


async def reencrypt_stored_values(
    database: SupportsRawQueries,
    *,
    from_key: str,
    to_key: str,
    require_unchanged: bool = False,
    upgrade_legacy_mcp: bool = False,
) -> int | None:
    reencrypted: Final = _reencrypting(from_key, to_key)
    migrated = 0  # rebind-ok: count successful writes while stopping on the first conflict during rotation
    for secret_column in await _secret_columns_in(database):
        for row in await _rows_of(database, secret_column):
            count = await _reencrypt_row(
                database, secret_column, row, reencrypted, to_key if upgrade_legacy_mcp else None
            )
            if count is None and require_unchanged:
                return None
            migrated += count or 0
    return migrated


async def _has_auth_profile_column(database: SupportsRawQueries) -> bool:
    return _AUTH_PROFILE_COLUMN in await _secret_columns_in(database)


async def stored_auth_profile_sets(database: SupportsRawQueries) -> tuple[JsonValue, ...]:
    if not await _has_auth_profile_column(database):
        return ()
    return tuple(
        _STORED_VALUE.validate_python(row[_AUTH_PROFILE_COLUMN.column])
        for row in await _rows_of(database, _AUTH_PROFILE_COLUMN)
    )


@dataclass(frozen=True, slots=True)
class ProfileRotation:
    rotated: int
    conflicts: int


async def reencrypt_auth_profile_secrets(
    database: SupportsRawQueries, *, from_key: str, to_key: str
) -> ProfileRotation:
    if not await _has_auth_profile_column(database):
        return ProfileRotation(rotated=0, conflicts=0)
    reencrypted: Final = _reencrypting(from_key, to_key)
    counts: Final = tuple(
        [
            await _reencrypt_row(database, _AUTH_PROFILE_COLUMN, row, reencrypted, None)
            for row in await _rows_of(database, _AUTH_PROFILE_COLUMN)
        ]
    )
    return ProfileRotation(
        rotated=sum(count or 0 for count in counts), conflicts=sum(count is None for count in counts)
    )


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
    lock_order: Final = f' ORDER BY "{secret_column.primary_key}" COLLATE "C"' if secret_column.in_lock_order else ""
    return tuple(
        await database.query_raw(
            f'SELECT "{secret_column.primary_key}", "{secret_column.column}" FROM "{secret_column.table}" '
            f'WHERE "{secret_column.column}" IS NOT NULL{marked_only}{lock_order}'
        )
    )


async def _reencrypt_row(
    database: SupportsRawQueries,
    secret_column: _SecretColumn,
    row: Mapping[str, object],
    reencrypted: ReplaceCiphertext,
    legacy_mcp_key: str | None,
) -> int | None:
    stored: Final = _STORED_VALUE.validate_python(row[secret_column.column])
    migrated, count = _replace_column_ciphertexts(secret_column, stored, reencrypted, legacy_mcp_key)
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
    return count if rows_updated else None


def _encrypt_legacy_mcp_map(stored: JsonValue, to_key: str) -> str | None:
    if not (isinstance(stored, dict) or (isinstance(stored, str) and stored.lstrip().startswith("{"))):
        return None
    try:
        plaintext: Final = (
            _MCP_SECRET_MAP.validate_json(stored, strict=True)
            if isinstance(stored, str)
            else _MCP_SECRET_MAP.validate_python(stored, strict=True)
        )
    except ValidationError:
        return None
    return encrypt_secret_map(plaintext, new_encryption_key=to_key) if plaintext else None


def _replace_column_ciphertexts(
    column: _SecretColumn, stored: JsonValue, replacement_for: ReplaceCiphertext, legacy_mcp_key: str | None = None
) -> tuple[JsonValue, int]:
    if legacy_mcp_key is not None and column.legacy_mcp_format == "secret_map":
        encrypted_map: Final = _encrypt_legacy_mcp_map(stored, legacy_mcp_key)
        if encrypted_map is not None:
            return encrypted_map, 1
    if column.is_json and isinstance(stored, str) and stored.lstrip().startswith(("{", "[", '"')):
        try:
            parsed: Final = _STORED_VALUE.validate_json(stored)
        except ValueError:
            return stored, 0
        replaced, count = replace_ciphertexts(parsed, replacement_for)
        return (json.dumps(replaced), count) if count else (stored, 0)
    migrated, migrated_count = replace_ciphertexts(stored, replacement_for)
    if migrated_count or legacy_mcp_key is None:
        return migrated, migrated_count
    if column.legacy_mcp_format == "credential" and isinstance(stored, str) and not stored.startswith(V2_GCM_PREFIX):
        plaintext: Final = decode_legacy_mcp_user_credential(stored)
        if plaintext is not None:
            return encrypt_value_helper(plaintext, legacy_mcp_key), 1
    return stored, 0


def _as_sql_parameter(value: JsonValue, secret_column: _SecretColumn) -> str:
    return json.dumps(value) if secret_column.is_json else _CIPHERTEXT.validate_python(value)


class NothingToMigrate(Enum):
    SALT_KEY_ENCRYPTS_STORED_VALUES = "salt_key_encrypts_stored_values"
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
    outcome: Final = await migrate_from_previous_master_key(
        previous_master_key=previous_master_key,
        master_key=master_key,
        salt_key_is_set=SALT_KEY_ENV_VAR in environ,
        database=connected_database(),
        log=log,
    )
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
    outcome: Final = await _migrate_or_failure(
        previous_master_key=previous_master_key,
        master_key=master_key,
        salt_key_is_set=salt_key_is_set,
        database=database,
        log=log,
    )
    log(describe_outcome(outcome))
    return outcome


async def _migrate_or_failure(
    *,
    previous_master_key: str,
    master_key: str,
    salt_key_is_set: bool,
    database: SupportsRawQueries | None,
    log: Callable[[str], None],
) -> MigrationOutcome:
    try:
        return await _migrate(
            previous_master_key=previous_master_key,
            master_key=master_key,
            salt_key_is_set=salt_key_is_set,
            database=database,
            log=log,
        )
    except Exception as error:  # noqa: BLE001  # a value, so the boot applies its own database outage rule to it
        return MigrationFailed(error=error)


async def _migrate(
    *,
    previous_master_key: str,
    master_key: str,
    salt_key_is_set: bool,
    database: SupportsRawQueries | None,
    log: Callable[[str], None],
) -> MigrationOutcome:
    if salt_key_is_set:
        return NothingToMigrate.SALT_KEY_ENCRYPTS_STORED_VALUES
    if database is None:
        return NothingToMigrate.NO_DATABASE
    if previous_master_key == master_key:
        return NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY
    found: Final = await count_values_encrypted_with(database, previous_master_key)
    if found == 0:
        return NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY
    log(f"Re-encrypting {found} stored value(s) from the {MIGRATE_FROM_MASTER_KEY_ENV_VAR} key to the new master key.")
    migrated: Final = Migrated(
        migrated=await reencrypt_stored_values(database, from_key=previous_master_key, to_key=master_key) or 0,
        remaining=await count_values_encrypted_with(database, previous_master_key),
    )
    another_worker_migrated_everything: Final = migrated == Migrated(migrated=0, remaining=0)
    return NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY if another_worker_migrated_everything else migrated


def describe_outcome(outcome: MigrationOutcome) -> str:
    match outcome:
        case NothingToMigrate.SALT_KEY_ENCRYPTS_STORED_VALUES:
            return (
                f"{MIGRATE_FROM_MASTER_KEY_ENV_VAR} is set, but {SALT_KEY_ENV_VAR} is what encrypts your stored "
                f"values, so there is nothing to migrate. You may now delete {MIGRATE_FROM_MASTER_KEY_ENV_VAR}."
            )
        case NothingToMigrate.NO_DATABASE:
            return (
                f"{MIGRATE_FROM_MASTER_KEY_ENV_VAR} is set, but no database is connected, so nothing was migrated. If "
                f"this proxy has no database, you may now delete {MIGRATE_FROM_MASTER_KEY_ENV_VAR}."
            )
        case NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY:
            return (
                f"{MIGRATE_FROM_MASTER_KEY_ENV_VAR} is still set, but nothing in the database is left to migrate "
                f"from that key. You may now delete {MIGRATE_FROM_MASTER_KEY_ENV_VAR}."
            )
        case Migrated(migrated=migrated, remaining=0):
            return (
                f"Done re-encrypting {migrated} stored value(s) with the new master key. You may now delete the "
                f"{MIGRATE_FROM_MASTER_KEY_ENV_VAR} environment variable."
            )
        case Migrated(migrated=migrated, remaining=remaining):
            return (
                f"Re-encrypted {migrated} stored value(s), but {remaining} are still encrypted with the previous key "
                f"because they changed during the migration. Keep {MIGRATE_FROM_MASTER_KEY_ENV_VAR} set and restart "
                "the proxy to migrate them."
            )
        case MigrationFailed(error=error):
            cause: Final = f"{type(error).__name__}: {error}"[:300]
            return (
                f"Could not migrate stored values from the {MIGRATE_FROM_MASTER_KEY_ENV_VAR} key ({cause}). Values "
                "still encrypted with the previous key cannot be read until the migration succeeds. Keep "
                f"{MIGRATE_FROM_MASTER_KEY_ENV_VAR} set and restart the proxy once the database is reachable."
            )
        case _:
            assert_never(outcome)
