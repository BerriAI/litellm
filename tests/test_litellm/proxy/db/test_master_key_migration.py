import json
import re
from collections.abc import Mapping, Sequence
from functools import reduce

import pytest
from pydantic import JsonValue

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.proxy import proxy_server
from litellm.proxy.auth.master_key_boot_check import MIGRATE_FROM_MASTER_KEY_ENV_VAR, SALT_KEY_ENV_VAR
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with, encrypt_value_helper
from litellm.proxy.db.master_key_migration import (
    _SECRET_COLUMNS,
    Migrated,
    MigrationFailed,
    NothingToMigrate,
    count_values_encrypted_with,
    describe_outcome,
    migrate_from_previous_master_key,
    migrate_if_requested,
    reencrypt_stored_values,
    replace_ciphertexts,
)

PREVIOUS_KEY = "sk-1234"
NEW_KEY = "sk-qa-9f2c1e7a44b0d3"
UNRELATED_KEY = "sk-some-other-deployment"

Tables = dict[str, list[dict[str, object]]]


@pytest.fixture(autouse=True)
def _legacy_algorithm_and_no_salt_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(SALT_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(proxy_server, "general_settings", {})


def _encrypted(plaintext: str, key: str = PREVIOUS_KEY) -> str:
    return str(encrypt_value_helper(plaintext, new_encryption_key=key))


class _FakeDatabase:
    def __init__(self, tables: Tables, tables_missing_from_the_schema: frozenset[str] = frozenset()) -> None:
        self.tables = tables
        self.tables_missing_from_the_schema = tables_missing_from_the_schema
        self.writes: list[tuple[str, str, str]] = []

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if "information_schema.columns" in query:
            assert "table_schema = ANY (current_schemas(false))" in query
            return [
                {"table_name": secret_column.table, "column_name": secret_column.column}
                for secret_column in _SECRET_COLUMNS
                if secret_column.table not in self.tables_missing_from_the_schema
            ]
        select = re.fullmatch(r'SELECT "(\w+)", "(\w+)" FROM "(\w+)" WHERE "\2" IS NOT NULL(.*)', query)
        assert select is not None, query
        primary_key, column, table, row_filter = select.groups()
        assert row_filter in ("", f" AND \"{column}\"::text LIKE '%litellm_enc::%'")
        assert table not in self.tables_missing_from_the_schema, f'relation "{table}" does not exist'
        return [
            {primary_key: row[primary_key], column: json.loads(json.dumps(row[column]))}
            for row in self.tables.get(table, [])
            if row.get(column) is not None and (not row_filter or "litellm_enc::" in json.dumps(row[column]))
        ]

    async def execute_raw(self, query: str, *args: object) -> int:
        update = re.fullmatch(r'UPDATE "(\w+)" SET "(\w+)" = \$1(::jsonb|) WHERE "(\w+)" = \$2 AND "\2" = \$3\3', query)
        assert update is not None, query
        table, column, json_cast, primary_key = update.groups()
        new_value, row_id, expected = (
            json.loads(str(arg)) if json_cast and index != 1 else arg for index, arg in enumerate(args)
        )
        matching = [row for row in self.tables[table] if row[primary_key] == row_id and row[column] == expected]
        for row in matching:
            row[column] = new_value
            self.writes.append((table, column, str(row_id)))
        return len(matching)


class _DatabaseThatMustNotBeTouched:
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        raise AssertionError(f"unexpected read: {query}")

    async def execute_raw(self, query: str, *args: object) -> int:
        raise AssertionError(f"unexpected write: {query}")


def _seeded_tables() -> Tables:
    return {
        "LiteLLM_ProxyModelTable": [
            {
                "model_id": "model-1",
                "litellm_params": {
                    "api_key": _encrypted("provider-key"),
                    "model": _encrypted("openai/gpt-5.4-mini"),
                    "rpm": 10,
                    "use_in_pass_through": False,
                    "api_base": None,
                    "extra_headers": {"Authorization": _encrypted("Bearer nested-secret"), "X-Trace": "plain"},
                },
            },
            {"model_id": "model-2", "litellm_params": {"api_key": _encrypted("other-deployment", UNRELATED_KEY)}},
        ],
        "LiteLLM_GuardrailsTable": [
            {
                "guardrail_id": "guardrail-1",
                "litellm_params": {
                    "guardrail": _encrypted("openai_moderation"),
                    "api_key": _encrypted("guardrail-key"),
                    "default_on": False,
                },
            },
            {"guardrail_id": "guardrail-plain", "litellm_params": {"guardrail": "bedrock", "mode": "pre_call"}},
        ],
        "LiteLLM_Config": [
            {"param_name": "environment_variables", "param_value": {"LANGFUSE_SECRET_KEY": _encrypted("env-secret")}},
            {"param_name": "general_settings", "param_value": {"proxy_batch_write_at": 10, "ui_name": "plain text"}},
            {"param_name": "cleared", "param_value": None},
        ],
        "LiteLLM_MCPServerTable": [
            {
                "server_id": "mcp-1",
                "credentials": {"auth_value": _encrypted("mcp-token"), "aws_region_name": "us-east-1"},
                "static_headers": _encrypted('{"X-Api-Key": "header-secret"}'),
                "env_vars": [
                    {"name": "GLOBAL", "scope": "global", "value": _encrypted("global-env")},
                    {"name": "PER_USER", "scope": "user", "value": ""},
                ],
                "env": {},
            }
        ],
        "LiteLLM_MCPUserCredentials": [{"id": "cred-row-1", "credential_b64": _encrypted("byok-secret")}],
        "LiteLLM_TeamTable": [
            {
                "team_id": "team-1",
                "metadata": {
                    "logging": [
                        {
                            "callback_name": "langfuse",
                            "callback_vars": {
                                "langfuse_host": "https://example.invalid",
                                "langfuse_secret_key": "litellm_enc::" + _encrypted("team-callback-secret"),
                            },
                        }
                    ]
                },
            },
            {"team_id": "team-without-callbacks", "metadata": {"note": _encrypted("unmarked, so never selected")}},
        ],
    }


_VALUES_UNDER_THE_PREVIOUS_KEY = 11


@pytest.mark.asyncio
async def test_reencryption_moves_every_stored_shape_to_the_new_key_and_nothing_else():
    tables = _seeded_tables()
    untouched_before = json.dumps(
        [
            tables["LiteLLM_ProxyModelTable"][1],
            tables["LiteLLM_Config"][1:],
            tables["LiteLLM_TeamTable"][1],
            tables["LiteLLM_GuardrailsTable"][1],
        ]
    )

    migrated = await reencrypt_stored_values(_FakeDatabase(tables), from_key=PREVIOUS_KEY, to_key=NEW_KEY)

    assert migrated == _VALUES_UNDER_THE_PREVIOUS_KEY
    model_params = tables["LiteLLM_ProxyModelTable"][0]["litellm_params"]
    assert isinstance(model_params, dict)
    assert decrypt_if_encrypted_with(model_params["api_key"], NEW_KEY) == "provider-key"
    assert decrypt_if_encrypted_with(model_params["model"], NEW_KEY) == "openai/gpt-5.4-mini"
    assert decrypt_if_encrypted_with(model_params["api_key"], PREVIOUS_KEY) is None
    assert (model_params["rpm"], model_params["use_in_pass_through"], model_params["api_base"]) == (10, False, None)
    assert decrypt_if_encrypted_with(model_params["extra_headers"]["Authorization"], NEW_KEY) == "Bearer nested-secret"
    assert model_params["extra_headers"]["X-Trace"] == "plain"
    guardrail_params = tables["LiteLLM_GuardrailsTable"][0]["litellm_params"]
    assert isinstance(guardrail_params, dict)
    assert decrypt_if_encrypted_with(guardrail_params["api_key"], NEW_KEY) == "guardrail-key"
    assert decrypt_if_encrypted_with(guardrail_params["guardrail"], NEW_KEY) == "openai_moderation"
    assert guardrail_params["default_on"] is False
    mcp_server = tables["LiteLLM_MCPServerTable"][0]
    assert decrypt_if_encrypted_with(mcp_server["static_headers"], NEW_KEY) == '{"X-Api-Key": "header-secret"}'
    assert mcp_server["credentials"]["aws_region_name"] == "us-east-1"
    assert decrypt_if_encrypted_with(mcp_server["env_vars"][0]["value"], NEW_KEY) == "global-env"
    assert mcp_server["env_vars"][1] == {"name": "PER_USER", "scope": "user", "value": ""}
    assert (
        decrypt_if_encrypted_with(tables["LiteLLM_MCPUserCredentials"][0]["credential_b64"], NEW_KEY) == "byok-secret"
    )
    callback_secret = tables["LiteLLM_TeamTable"][0]["metadata"]["logging"][0]["callback_vars"]["langfuse_secret_key"]
    assert callback_secret.startswith("litellm_enc::")
    assert decrypt_if_encrypted_with(callback_secret.removeprefix("litellm_enc::"), NEW_KEY) == "team-callback-secret"
    assert untouched_before == json.dumps(
        [
            tables["LiteLLM_ProxyModelTable"][1],
            tables["LiteLLM_Config"][1:],
            tables["LiteLLM_TeamTable"][1],
            tables["LiteLLM_GuardrailsTable"][1],
        ]
    )


@pytest.mark.asyncio
async def test_count_follows_the_values_from_the_previous_key_to_the_new_one():
    database = _FakeDatabase(_seeded_tables())

    assert await count_values_encrypted_with(database, PREVIOUS_KEY) == _VALUES_UNDER_THE_PREVIOUS_KEY
    assert await count_values_encrypted_with(database, NEW_KEY) == 0

    await reencrypt_stored_values(database, from_key=PREVIOUS_KEY, to_key=NEW_KEY)

    assert await count_values_encrypted_with(database, PREVIOUS_KEY) == 0
    assert await count_values_encrypted_with(database, NEW_KEY) == _VALUES_UNDER_THE_PREVIOUS_KEY
    assert await count_values_encrypted_with(database, UNRELATED_KEY) == 1


@pytest.mark.asyncio
async def test_only_rows_holding_values_under_the_previous_key_are_written():
    database = _FakeDatabase(_seeded_tables())

    await reencrypt_stored_values(database, from_key=PREVIOUS_KEY, to_key=NEW_KEY)

    assert sorted(database.writes) == [
        ("LiteLLM_Config", "param_value", "environment_variables"),
        ("LiteLLM_GuardrailsTable", "litellm_params", "guardrail-1"),
        ("LiteLLM_MCPServerTable", "credentials", "mcp-1"),
        ("LiteLLM_MCPServerTable", "env_vars", "mcp-1"),
        ("LiteLLM_MCPServerTable", "static_headers", "mcp-1"),
        ("LiteLLM_MCPUserCredentials", "credential_b64", "cred-row-1"),
        ("LiteLLM_ProxyModelTable", "litellm_params", "model-1"),
        ("LiteLLM_TeamTable", "metadata", "team-1"),
    ]


@pytest.mark.asyncio
async def test_plaintext_that_base64_decodes_to_nothing_is_neither_counted_nor_rewritten():
    settings = {"allowed_routes": ["*"], "ui_name": "-", "separator": "...", "blank": " ", "shape": "{}"}
    tables: Tables = {
        "LiteLLM_Config": [{"param_name": "general_settings", "param_value": dict(settings)}],
        "LiteLLM_VerificationToken": [
            {"token": "hashed", "metadata": {"notes": "...", "secret": "litellm_enc::" + _encrypted("callback-secret")}}
        ],
    }
    database = _FakeDatabase(tables)

    found = await count_values_encrypted_with(database, PREVIOUS_KEY)
    migrated = await reencrypt_stored_values(database, from_key=PREVIOUS_KEY, to_key=NEW_KEY)

    assert found == migrated == 1
    assert tables["LiteLLM_Config"][0]["param_value"] == settings
    assert tables["LiteLLM_VerificationToken"][0]["metadata"]["notes"] == "..."
    assert database.writes == [("LiteLLM_VerificationToken", "metadata", "hashed")]


@pytest.mark.asyncio
async def test_schema_without_some_of_the_tables_is_migrated_for_the_tables_it_has():
    missing = frozenset({"LiteLLM_MCPUserCredentials", "LiteLLM_SSOIdentityAssertion"})
    tables = _seeded_tables()
    database = _FakeDatabase(tables, tables_missing_from_the_schema=missing)

    found = await count_values_encrypted_with(database, PREVIOUS_KEY)
    migrated = await reencrypt_stored_values(database, from_key=PREVIOUS_KEY, to_key=NEW_KEY)

    assert found == migrated == _VALUES_UNDER_THE_PREVIOUS_KEY - 1
    assert decrypt_if_encrypted_with(str(tables["LiteLLM_MCPUserCredentials"][0]["credential_b64"]), PREVIOUS_KEY)


class _SomeoneEditsEachRowAfterItIsRead(_FakeDatabase):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        rows = await super().query_raw(query, *args)
        for row in self.tables.get("LiteLLM_MCPUserCredentials", []):
            row["credential_b64"] = "edited-by-an-admin"
        return rows


@pytest.mark.asyncio
async def test_value_edited_while_the_migration_runs_is_not_overwritten():
    tables: Tables = {"LiteLLM_MCPUserCredentials": [{"id": "cred-row-1", "credential_b64": _encrypted("byok-secret")}]}

    migrated = await reencrypt_stored_values(
        _SomeoneEditsEachRowAfterItIsRead(tables), from_key=PREVIOUS_KEY, to_key=NEW_KEY
    )

    assert migrated == 0
    assert tables["LiteLLM_MCPUserCredentials"][0]["credential_b64"] == "edited-by-an-admin"


def test_replacing_ciphertexts_keeps_structure_markers_and_non_strings():
    value = {"keep": [1, True, None, "plain"], "swap": ["old", {"nested": "litellm_enc::old"}]}

    replaced, count = replace_ciphertexts(value, lambda text: "new" if text == "old" else None)

    assert replaced == {"keep": [1, True, None, "plain"], "swap": ["new", {"nested": "litellm_enc::new"}]}
    assert count == 2
    assert value["swap"] == ["old", {"nested": "litellm_enc::old"}]


def _nested(levels: int, leaf: str) -> JsonValue:
    return reduce(lambda inner, _: [inner], range(levels), leaf)


@pytest.mark.parametrize("levels_past_the_cap, replaced_count", [(0, 1), (1, 0), (50, 0)])
def test_walk_stops_at_the_recursion_cap_and_leaves_deeper_values_as_they_were(
    levels_past_the_cap: int, replaced_count: int
):
    value = _nested(DEFAULT_MAX_RECURSE_DEPTH + levels_past_the_cap, "old")

    replaced, count = replace_ciphertexts(value, lambda text: "new")

    assert count == replaced_count
    assert replaced == _nested(DEFAULT_MAX_RECURSE_DEPTH + levels_past_the_cap, "new" if replaced_count else "old")


async def _run(
    database: _FakeDatabase | _DatabaseThatMustNotBeTouched | None,
    *,
    previous_master_key: str = PREVIOUS_KEY,
    master_key: str = NEW_KEY,
    salt_key_is_set: bool = False,
) -> tuple[object, list[str]]:
    logged: list[str] = []
    outcome = await migrate_from_previous_master_key(
        previous_master_key=previous_master_key,
        master_key=master_key,
        salt_key_is_set=salt_key_is_set,
        database=database,
        log=logged.append,
    )
    return outcome, logged


@pytest.mark.asyncio
async def test_migration_announces_itself_then_says_the_variable_can_go():
    database = _FakeDatabase(_seeded_tables())

    outcome, logged = await _run(database)

    assert outcome == Migrated(migrated=_VALUES_UNDER_THE_PREVIOUS_KEY, remaining=0)
    assert len(logged) == 2
    assert logged[0].startswith(f"Re-encrypting {_VALUES_UNDER_THE_PREVIOUS_KEY} stored value(s)")
    assert logged[1].startswith(f"Done re-encrypting {_VALUES_UNDER_THE_PREVIOUS_KEY} stored value(s)")
    assert f"You may now delete the {MIGRATE_FROM_MASTER_KEY_ENV_VAR} environment variable" in logged[1]


@pytest.mark.asyncio
async def test_variable_left_set_after_the_migration_is_a_no_op_with_one_notice():
    database = _FakeDatabase(_seeded_tables())
    await _run(database)
    writes_after_the_migration = list(database.writes)
    stored_after_the_migration = json.dumps(database.tables)

    outcome, logged = await _run(database)

    assert outcome is NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY
    assert logged == [describe_outcome(NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY)]
    assert database.writes == writes_after_the_migration
    assert json.dumps(database.tables) == stored_after_the_migration


@pytest.mark.asyncio
async def test_empty_previous_master_key_migrates_like_any_other():
    tables: Tables = {"LiteLLM_CredentialsTable": [{"credential_id": "c1", "credential_values": {"api_key": "x"}}]}
    tables["LiteLLM_CredentialsTable"][0]["credential_values"] = {"api_key": _encrypted_with_empty_key("cred-secret")}

    outcome, _ = await _run(_FakeDatabase(tables), previous_master_key="")

    assert outcome == Migrated(migrated=1, remaining=0)
    stored = tables["LiteLLM_CredentialsTable"][0]["credential_values"]
    assert isinstance(stored, dict)
    assert decrypt_if_encrypted_with(stored["api_key"], NEW_KEY) == "cred-secret"


def _encrypted_with_empty_key(plaintext: str) -> str:
    import base64

    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value

    return base64.urlsafe_b64encode(encrypt_value(value=plaintext, signing_key="")).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("salt_key_is_set", "has_database", "master_key", "expected"),
    [
        (True, True, NEW_KEY, NothingToMigrate.SALT_KEY_ENCRYPTS_STORED_VALUES),
        (False, False, NEW_KEY, NothingToMigrate.NO_DATABASE),
        (False, True, PREVIOUS_KEY, NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY),
    ],
)
async def test_database_is_left_alone_when_the_master_key_cannot_have_encrypted_it(
    salt_key_is_set: bool, has_database: bool, master_key: str, expected: NothingToMigrate
):
    outcome, logged = await _run(
        _DatabaseThatMustNotBeTouched() if has_database else None,
        master_key=master_key,
        salt_key_is_set=salt_key_is_set,
    )

    assert outcome is expected
    assert logged == [describe_outcome(expected)]


class _AnotherWorkerMigratesRightAfterTheFirstCount(_FakeDatabase):
    def __init__(self, tables: Tables) -> None:
        super().__init__(tables)
        self.reads = 0

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        rows = await super().query_raw(query, *args)
        self.reads += 1
        if self.reads == len(_SECRET_COLUMNS):
            await reencrypt_stored_values(_FakeDatabase(self.tables), from_key=PREVIOUS_KEY, to_key=NEW_KEY)
        return rows


@pytest.mark.asyncio
async def test_worker_that_loses_the_race_reports_nothing_left_instead_of_zero_values_done():
    database = _AnotherWorkerMigratesRightAfterTheFirstCount(_seeded_tables())

    outcome, logged = await _run(database)

    assert outcome is NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY
    assert database.writes == []
    assert logged[-1] == describe_outcome(NothingToMigrate.NOTHING_ENCRYPTED_WITH_PREVIOUS_KEY)


@pytest.mark.asyncio
async def test_values_that_could_not_be_written_keep_the_variable_in_place():
    tables: Tables = {"LiteLLM_MCPUserCredentials": [{"id": "cred-row-1", "credential_b64": _encrypted("byok-secret")}]}

    class _EveryWriteLosesToACurrentEdit(_FakeDatabase):
        async def execute_raw(self, query: str, *args: object) -> int:
            return 0

    outcome, logged = await _run(_EveryWriteLosesToACurrentEdit(tables))

    assert outcome == Migrated(migrated=0, remaining=1)
    assert "1 are still encrypted with the previous key" in logged[-1]
    assert f"Keep {MIGRATE_FROM_MASTER_KEY_ENV_VAR} set" in logged[-1]
    assert "You may now delete" not in logged[-1]


@pytest.mark.parametrize(
    "outcome", [*NothingToMigrate, Migrated(migrated=5, remaining=0)], ids=lambda outcome: str(outcome)
)
def test_every_finished_outcome_tells_the_user_the_variable_can_be_deleted(outcome: NothingToMigrate | Migrated):
    message = describe_outcome(outcome)

    assert "ou may now delete" in message
    assert MIGRATE_FROM_MASTER_KEY_ENV_VAR in message


def test_salt_key_outcome_names_the_salt_key_as_the_reason():
    assert SALT_KEY_ENV_VAR in describe_outcome(NothingToMigrate.SALT_KEY_ENCRYPTS_STORED_VALUES)
    assert SALT_KEY_ENV_VAR not in describe_outcome(NothingToMigrate.NO_DATABASE)


@pytest.mark.asyncio
async def test_encrypted_empty_string_is_migrated_like_any_other_value():
    tables: Tables = {"LiteLLM_MCPUserCredentials": [{"id": "cred-row-1", "credential_b64": _encrypted("")}]}

    migrated = await reencrypt_stored_values(_FakeDatabase(tables), from_key=PREVIOUS_KEY, to_key=NEW_KEY)

    assert migrated == 1
    assert decrypt_if_encrypted_with(str(tables["LiteLLM_MCPUserCredentials"][0]["credential_b64"]), NEW_KEY) == ""


class _DatabaseIsDown(_FakeDatabase):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        raise ConnectionError("Can't reach database server")


@pytest.mark.asyncio
async def test_database_error_during_the_migration_comes_back_as_a_value_and_is_logged():
    outcome, logged = await _run(_DatabaseIsDown(_seeded_tables()))

    assert isinstance(outcome, MigrationFailed)
    assert isinstance(outcome.error, ConnectionError)
    assert len(logged) == 1
    assert "ConnectionError: Can't reach database server" in logged[0]
    assert f"Keep {MIGRATE_FROM_MASTER_KEY_ENV_VAR} set" in logged[0]
    assert "ou may now delete" not in logged[0]


def _raise(error: Exception) -> None:
    raise error


def _tolerate(error: Exception) -> None:
    return None


@pytest.mark.asyncio
async def test_boot_stops_on_a_failed_migration_when_the_outage_is_not_tolerated():
    logged: list[str] = []

    with pytest.raises(ConnectionError, match="Can't reach database server"):
        await migrate_if_requested(
            environ={MIGRATE_FROM_MASTER_KEY_ENV_VAR: PREVIOUS_KEY},
            master_key=NEW_KEY,
            connected_database=lambda: _DatabaseIsDown(_seeded_tables()),
            log=logged.append,
            raise_unless_tolerated=_raise,
        )

    assert len(logged) == 1
    assert f"Keep {MIGRATE_FROM_MASTER_KEY_ENV_VAR} set" in logged[0]


@pytest.mark.asyncio
async def test_boot_continues_past_a_failed_migration_when_the_outage_is_tolerated():
    outcome = await migrate_if_requested(
        environ={MIGRATE_FROM_MASTER_KEY_ENV_VAR: PREVIOUS_KEY},
        master_key=NEW_KEY,
        connected_database=lambda: _DatabaseIsDown(_seeded_tables()),
        log=lambda line: None,
        raise_unless_tolerated=_tolerate,
    )

    assert isinstance(outcome, MigrationFailed)


@pytest.mark.asyncio
@pytest.mark.parametrize("previous_master_key", [PREVIOUS_KEY, ""])
async def test_boot_migrates_from_the_environment_variable_to_the_running_master_key(previous_master_key: str):
    tables: Tables = {
        "LiteLLM_CredentialsTable": [
            {
                "credential_id": "cred-1",
                "credential_values": {
                    "api_key": _encrypted("sk-provider", previous_master_key)
                    if previous_master_key
                    else _encrypted_with_empty_key("sk-provider")
                },
            }
        ]
    }
    logged: list[str] = []

    outcome = await migrate_if_requested(
        environ={MIGRATE_FROM_MASTER_KEY_ENV_VAR: previous_master_key},
        master_key=NEW_KEY,
        connected_database=lambda: _FakeDatabase(tables),
        log=logged.append,
        raise_unless_tolerated=_raise,
    )

    assert outcome == Migrated(migrated=1, remaining=0)
    stored = tables["LiteLLM_CredentialsTable"][0]["credential_values"]
    assert isinstance(stored, dict)
    assert decrypt_if_encrypted_with(stored["api_key"], NEW_KEY) == "sk-provider"
    assert "Done re-encrypting 1 stored value(s)" in logged[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "environ, master_key, outcome",
    [
        ({}, NEW_KEY, None),
        ({MIGRATE_FROM_MASTER_KEY_ENV_VAR: PREVIOUS_KEY}, None, None),
        (
            {MIGRATE_FROM_MASTER_KEY_ENV_VAR: PREVIOUS_KEY, SALT_KEY_ENV_VAR: "a-salt-key"},
            NEW_KEY,
            NothingToMigrate.SALT_KEY_ENCRYPTS_STORED_VALUES,
        ),
    ],
    ids=["variable-not-set", "no-master-key", "salt-key-set"],
)
async def test_boot_leaves_the_database_alone_unless_a_migration_was_requested_and_can_apply(
    environ: dict[str, str], master_key: str | None, outcome: NothingToMigrate | None
):
    logged: list[str] = []
    database_handles_taken: list[str] = []

    def connected_database() -> _DatabaseThatMustNotBeTouched:
        database_handles_taken.append("taken")
        return _DatabaseThatMustNotBeTouched()

    result = await migrate_if_requested(
        environ=environ,
        master_key=master_key,
        connected_database=connected_database,
        log=logged.append,
        raise_unless_tolerated=_raise,
    )

    assert result is outcome
    assert len(database_handles_taken) == (0 if outcome is None else 1)
    assert len(logged) == (0 if outcome is None else 1)
