import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, cast

from pydantic import JsonValue, TypeAdapter
from typing_extensions import assert_never

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.proxy.auth.master_key_boot_check import MIGRATE_FROM_MASTER_KEY_ENV_VAR, SALT_KEY_ENV_VAR
from litellm.proxy.common_utils.callback_utils import CALLBACK_VAR_ENCRYPTED_PREFIX
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_if_encrypted_with,
    encrypt_value_helper,
    legacy_encryption_available,
    legacy_unreadable,
)
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
        decrypted: Final = decrypt_if_encrypted_with(value, signing_key)
        if decrypted is not None:
            return value
        # A legacy value PyNaCl is needed to read still counts: without this, the boot would
        # report nothing encrypted with the previous key and tell the operator to delete
        # LITELLM_MIGRATE_FROM_MASTER_KEY while values under that key remain, unreadable.
        return value if legacy_unreadable(value) else None

    return await _count_secret_leaves(database, lambda leaf: keep(leaf) is not None)


async def count_values_encrypted_with_or_none(
    connect: Callable[[], Awaitable[SupportsRawQueries]], signing_key: str
) -> int | None:
    try:
        return await count_values_encrypted_with(await connect(), signing_key)
    except Exception:  # noqa: BLE001  # an unreadable database must not replace the boot refusal with a traceback
        return None


async def _count_values_legacy_unreadable(database: SupportsRawQueries) -> int:
    """How many stored strings only PyNaCl could read — 0 without even walking when it is installed.

    Without PyNaCl an unprefixed value cannot be told apart from legacy ciphertext, so the
    encrypted-with-previous-key count stays conservative and this walk explains the residual."""
    if legacy_encryption_available():
        return 0
    return await _count_secret_leaves(database, legacy_unreadable)


async def _count_secret_leaves(database: SupportsRawQueries, counted: Callable[[str], bool]) -> int:
    """Count the writer-encrypted secret leaves across every secret column that satisfy ``counted``."""

    def counted_in(secret_column: _SecretColumn, row: Mapping[str, object]) -> int:
        stored: Final = _STORED_VALUE.validate_python(row[secret_column.column])
        leaves: Final = _counted_secret_leaves(secret_column, row[secret_column.primary_key], stored)
        return sum(1 for leaf in leaves if counted(leaf))

    return sum(
        [
            counted_in(secret_column, row)
            for secret_column in await _secret_columns_in(database)
            for row in await _rows_of(database, secret_column)
        ]
    )


# Columns whose writers encrypt each top-level string param and leave everything nested
# inside a list/dict param (tags, routing metadata, …) stored as plaintext.
_LITELLM_PARAMS_SECRET_COLUMNS: Final = frozenset(
    (
        ("LiteLLM_ProxyModelTable", "litellm_params"),
        ("LiteLLM_SearchToolsTable", "litellm_params"),
    )
)
_ENV_VARS_CONFIG_ROW: Final = "environment_variables"
_MCP_CREDENTIAL_TABLES: Final = frozenset(("LiteLLM_MCPServerTable", "LiteLLM_MCPServerOAuthClient"))
# Config rows whose writers encrypt selected top-level fields (the same fields the
# credential-migration walkers re-encrypt); the rows' other fields are plaintext metadata.
_CONFIG_ROW_SENSITIVE_FIELDS: Final[Mapping[str, tuple[str, ...]]] = {
    "vantage_settings": ("api_key", "integration_token"),
    "cloudzero_settings": ("api_key",),
}


def _counted_secret_leaves(secret_column: _SecretColumn, primary_key: object, stored: JsonValue) -> Iterator[str]:
    """Yield the string leaves of one stored value the writers of this column actually encrypt.

    Telling ciphertext from plaintext needs PyNaCl for unprefixed values, so a count over
    every string leaf counts plaintext noise (config booleans, model tags, guardrail modes)
    as possibly-encrypted-with-the-previous-key and the boot migration then never finishes
    on an install that cannot install PyNaCl (a FIPS image). Restricting the candidates to
    the leaves each column's writers encrypt keeps that noise out — the same per-column
    selection the credential-migration scanner applies. Columns without a known writer
    shape keep the conservative every-string-leaf walk. The callback-encryption marker is
    stripped exactly as ``replace_ciphertexts`` strips it before the decrypt attempt."""
    if secret_column.table == "LiteLLM_Config":
        yield from _config_row_leaves(primary_key, stored)
        return
    if (secret_column.table, secret_column.column) in _LITELLM_PARAMS_SECRET_COLUMNS:
        params: Final[JsonValue] = stored
        if isinstance(params, dict):
            for value in params.values():
                if isinstance(value, str):
                    yield value.removeprefix(CALLBACK_VAR_ENCRYPTED_PREFIX)
        return
    if secret_column.table == "LiteLLM_GuardrailsTable":
        yield from _sensitive_prefixed_leaves(stored)
        return
    if secret_column.column == "credentials" and secret_column.table in _MCP_CREDENTIAL_TABLES:
        yield from _mcp_credential_leaves(stored)
        return
    if secret_column.table == "LiteLLM_MCPServerTable" and secret_column.column == "env_vars":
        yield from _mcp_global_env_var_leaves(stored)
        return
    yield from _string_leaves(stored)


def _config_row_leaves(primary_key: object, stored: JsonValue) -> Iterator[str]:
    param_name: Final[str] = _PRIMARY_KEY.validate_python(primary_key)
    if param_name == _ENV_VARS_CONFIG_ROW:
        yield from _string_leaves(stored)
        return
    sensitive_fields: Final[tuple[str, ...] | None] = _CONFIG_ROW_SENSITIVE_FIELDS.get(param_name)
    if sensitive_fields is None:
        # Every other config row (general_settings, …) stores plaintext by
        # construction — counting its strings would read them as ciphertext.
        return
    if not isinstance(stored, dict):
        return
    for field in sensitive_fields:
        value = stored.get(field)
        if isinstance(value, str):
            yield value.removeprefix(CALLBACK_VAR_ENCRYPTED_PREFIX)


def _mcp_credential_leaves(stored: JsonValue) -> Iterator[str]:
    # The MCP credentials writer encrypts only its secret fields; region / service
    # names and scopes next to them are stored plaintext by construction.
    from litellm.proxy._experimental.mcp_server.db import MCP_CREDENTIAL_SECRET_FIELDS

    if not isinstance(stored, dict):
        return
    for field in MCP_CREDENTIAL_SECRET_FIELDS:
        value = stored.get(field)
        if isinstance(value, str):
            yield value.removeprefix(CALLBACK_VAR_ENCRYPTED_PREFIX)


def _mcp_global_env_var_leaves(stored: JsonValue) -> Iterator[str]:
    # Only global-scope entries carry an encrypted value; user-scope placeholders
    # hold a name and scope in plaintext by construction.
    from litellm.proxy._experimental.mcp_server.db import is_global_env_var_scope

    if not isinstance(stored, list):
        return
    for entry in stored:
        if not isinstance(entry, dict) or not is_global_env_var_scope(entry.get("scope")):
            continue
        value = entry.get("value")
        if isinstance(value, str):
            yield value.removeprefix(CALLBACK_VAR_ENCRYPTED_PREFIX)


def _string_leaves(value: JsonValue) -> Iterator[str]:
    """Every string leaf of a nested structure, marker-stripped, via an explicit stack."""
    stack: Final[list[JsonValue]] = [value]  # mutable-ok: walker stack
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            yield cur.removeprefix(CALLBACK_VAR_ENCRYPTED_PREFIX)
        elif isinstance(cur, (list, dict)):
            stack.extend(cur.values() if isinstance(cur, dict) else cur)


def _sensitive_prefixed_leaves(value: JsonValue) -> Iterator[str]:
    """Sensitive-keyed, marker-prefixed string leaves at any depth of a guardrails row.

    The guardrails writer stores every sensitive-keyed string with the callback
    encryption marker — including strings inside lists under the sensitive key — and
    everything else as plaintext by construction, so only marked leaves are candidates."""
    from litellm.proxy.common_utils.callback_utils import is_sensitive_callback_key

    stack: Final[list[tuple[str | None, JsonValue]]] = [(None, value)]  # mutable-ok: walker stack
    while stack:
        key, cur = stack.pop()
        if isinstance(cur, str):
            if key is not None and is_sensitive_callback_key(key) and cur.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX):
                yield cur.removeprefix(CALLBACK_VAR_ENCRYPTED_PREFIX)
        elif isinstance(cur, dict):
            entries = cast("dict[str, JsonValue]", cur)  # cast-ok: JSON object keys are strings
            stack.extend(entries.items())
        elif isinstance(cur, list):
            items = cast("list[JsonValue]", cur)  # cast-ok: JSON arrays hold JSON values
            stack.extend((key, item) for item in items)


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
    NO_DATABASE = "no_database"
    NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY = "nothing_encrypted_with_previous_key"


@dataclass(frozen=True, slots=True)
class Migrated:
    migrated: int
    remaining: int
    unreadable: int = 0  # values only PyNaCl could read; this build can neither migrate nor clear them


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
        migrated=await reencrypt_stored_values(database, from_key=previous_master_key, to_key=master_key),
        remaining=await count_values_encrypted_with(database, previous_master_key),
        unreadable=await _count_values_legacy_unreadable(database),
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
        case Migrated(migrated=migrated, remaining=0, unreadable=0):
            return (
                f"Done re-encrypting {migrated} stored value(s) with the new master key. You may now delete the "
                f"{MIGRATE_FROM_MASTER_KEY_ENV_VAR} environment variable."
            )
        case Migrated(migrated=migrated, remaining=remaining, unreadable=0):
            return (
                f"Re-encrypted {migrated} stored value(s), but {remaining} are still encrypted with the previous key "
                f"because they changed during the migration. Keep {MIGRATE_FROM_MASTER_KEY_ENV_VAR} set and restart "
                "the proxy to migrate them."
            )
        case Migrated(migrated=migrated, remaining=remaining, unreadable=unreadable):
            changed: Final = remaining - unreadable
            return (
                f"Re-encrypted {migrated} stored value(s), but {unreadable} cannot be read without the "
                "legacy-encryption extra (pip install 'litellm[legacy-encryption]')"
                + (f" and {changed} changed during the migration" if changed > 0 else "")
                + f". They may still be encrypted with the previous key, so keep {MIGRATE_FROM_MASTER_KEY_ENV_VAR} "
                "set, install the extra, and restart the proxy to finish."
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
