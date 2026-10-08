from itertools import product
from typing import Final

import psycopg
from psycopg import sql

_TABLE_NAMES: Final = (
    "LiteLLM_DailyUserSpend",
    "LiteLLM_DailyTeamSpend",
    "LiteLLM_VerificationToken",
    "LiteLLM_DeletedVerificationToken",
    "LiteLLM_UserTable",
    "LiteLLM_TeamTable",
)

_TAG_KEY_MEMBERSHIPS: Final = (
    ("tag-a", "entity-key-0"),
    ("tag-a", "entity-key-1"),
    ("tag-a", "entity-key-2"),
    ("tag-a", "entity-key-3"),
    ("tag-a", "entity-key-4"),
    ("tag-b", "entity-key-0"),
    ("tag-b", "entity-key-1"),
    ("tag-b", "entity-key-5"),
    ("tag-c", "entity-key-2"),
    ("tag-c", "entity-key-3"),
    ("tag-c", "entity-key-6"),
    ("tag-c", "entity-key-7"),
    ("tag-d", "entity-key-4"),
    ("tag-d", "entity-key-5"),
    ("tag-d", "entity-key-6"),
    ("tag-d", "entity-key-7"),
)
_TAG_ACTIVITY_DATES: Final = ("2026-06-01", "2026-06-02")


def seed_daily_activity_fixture(connection: psycopg.Connection, *, schema: str, ptu_sentinel_api_key: str) -> None:
    daily_user_table: Final = sql.Identifier(schema, "LiteLLM_DailyUserSpend")
    daily_team_table: Final = sql.Identifier(schema, "LiteLLM_DailyTeamSpend")
    verification_token_table: Final = sql.Identifier(schema, "LiteLLM_VerificationToken")
    deleted_token_table: Final = sql.Identifier(schema, "LiteLLM_DeletedVerificationToken")
    user_table: Final = sql.Identifier(schema, "LiteLLM_UserTable")
    team_table: Final = sql.Identifier(schema, "LiteLLM_TeamTable")
    keys: Final = (
        ("key-a", "model-popular", 100.0, 2, 1),
        ("key-b", "model-popular", 90.0, 3, 1),
        ("key-c", "model-popular", 80.0, 4, 1),
        ("key-target", "model-target", 1.0, 5, 2),
        ("key-cache", "model-cache", 2.0, 1000, 1),
    )
    user_rows: Final = tuple(
        (
            f"user-row-{index}",
            "user-1",
            "2026-06-01",
            api_key,
            model,
            "",
            "provider-a",
            None,
            "/v1/chat/completions",
            prompt_tokens,
            2,
            cache_read_tokens,
            0,
            spend,
            1,
            1,
            0,
            "2026-06-01 12:00:00",
        )
        for index, (api_key, model, spend, prompt_tokens, cache_read_tokens) in enumerate(keys)
    )
    team_rows: Final = tuple(
        (
            f"team-row-{index}",
            "team-1",
            "2026-06-01",
            api_key,
            model,
            "",
            "provider-a",
            None,
            "/v1/chat/completions",
            prompt_tokens,
            2,
            cache_read_tokens,
            0,
            spend,
            1,
            1,
            0,
            0.0,
            "2026-06-01 12:00:00",
        )
        for index, (api_key, model, spend, prompt_tokens, cache_read_tokens) in enumerate(keys)
    )
    sentinel_user_row: Final = (
        "user-row-ptu",
        "user-1",
        "2026-06-01",
        ptu_sentinel_api_key,
        "model-ptu",
        "",
        "provider-a",
        None,
        "/v1/chat/completions",
        0,
        0,
        0,
        0,
        1000.0,
        0,
        0,
        0,
        "2026-06-01 12:00:00",
    )
    sentinel_team_row: Final = (
        "team-row-ptu",
        "team-1",
        "2026-06-01",
        ptu_sentinel_api_key,
        "model-ptu",
        "",
        "provider-a",
        None,
        "/v1/chat/completions",
        0,
        0,
        0,
        0,
        1000.0,
        0,
        0,
        0,
        42.0,
        "2026-06-01 12:00:00",
    )
    with connection.cursor() as cursor:
        for table_name in _TABLE_NAMES:
            cursor.execute(
                sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS INCLUDING CONSTRAINTS)").format(
                    sql.Identifier(schema, table_name),
                    sql.Identifier(table_name),
                )
            )
        cursor.executemany(
            sql.SQL("""
            INSERT INTO {}
                (id, user_id, date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint, prompt_tokens, completion_tokens,
                 cache_read_input_tokens, cache_creation_input_tokens, spend, api_requests,
                 successful_requests, failed_requests, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """).format(daily_user_table),
            (*user_rows, sentinel_user_row),
        )
        cursor.executemany(
            sql.SQL("""
            INSERT INTO {}
                (id, team_id, date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint, prompt_tokens, completion_tokens,
                 cache_read_input_tokens, cache_creation_input_tokens, spend, api_requests,
                 successful_requests, failed_requests, ptu_flat_cost, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """).format(daily_team_table),
            (*team_rows, sentinel_team_row),
        )
        cursor.executemany(
            sql.SQL(
                "INSERT INTO {} (token, key_alias, team_id, user_id, metadata, models) VALUES (%s, %s, %s, %s, %s, %s)"
            ).format(verification_token_table),
            (
                ("key-a", "alias-a", "team-1", "user-1", '{"tags": ["blue", "gold"]}', []),
                ("key-b", "alias-b", "team-1", "user-1", '{"tags": []}', []),
                ("key-c", "alias-c", "team-1", "user-1", '{"tags": []}', []),
                ("key-cache", "alias-cache", "team-1", "user-1", '{"tags": []}', []),
            ),
        )
        cursor.executemany(
            sql.SQL("""
            INSERT INTO {}
                (id, token, key_alias, team_id, user_id, metadata, models, deleted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """).format(deleted_token_table),
            (
                ("deleted-old", "key-target", "older-target", "team-1", "user-1", '{"tags": []}', [], "2026-06-01"),
                (
                    "deleted-new",
                    "key-target",
                    "deleted-target",
                    "team-1",
                    "user-1",
                    '{"tags": ["archived"]}',
                    [],
                    "2026-06-02",
                ),
            ),
        )
        cursor.execute(
            sql.SQL("INSERT INTO {} (user_id, user_email, models) VALUES (%s, %s, %s)").format(user_table),
            ("user-1", "user@example.com", []),
        )
        cursor.execute(
            sql.SQL("INSERT INTO {} (team_id, team_alias, admins, members, models) VALUES (%s, %s, %s, %s, %s)").format(
                team_table
            ),
            ("team-1", "Usage Team", [], [], []),
        )
    connection.commit()


def seed_daily_tag_activity_fixture(connection: psycopg.Connection, *, schema: str) -> None:
    tag_table: Final = sql.Identifier(schema, "LiteLLM_DailyTagSpend")
    rows: Final = tuple(
        (
            f"tag-rollup-{row_index}",
            tag,
            date,
            api_key,
            "entity-rollup-model",
            "",
            "provider-a",
            None,
            "/v1/chat/completions",
            row_index + 1,
            float(row_index + 1),
            row_index % 5 + 1,
            f"{date} 12:00:00",
        )
        for row_index, (date, (tag, api_key)) in enumerate(product(_TAG_ACTIVITY_DATES, _TAG_KEY_MEMBERSHIPS))
    )
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS INCLUDING CONSTRAINTS)").format(
                tag_table,
                sql.Identifier("LiteLLM_DailyTagSpend"),
            )
        )
        cursor.executemany(
            sql.SQL("""
            INSERT INTO {}
                (id, tag, date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint, prompt_tokens, spend, api_requests, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """).format(tag_table),
            rows,
        )
    connection.commit()


def seed_daily_tag_float_tie_fixture(connection: psycopg.Connection, *, schema: str) -> None:
    tag_table: Final = sql.Identifier(schema, "LiteLLM_DailyTagSpend")
    key_spends: Final = (
        ("key-z", 0.1),
        ("key-z", 0.2),
        ("key-z", 0.3),
        ("key-a", 0.3),
        ("key-a", 0.2),
        ("key-a", 0.1),
    )
    rows: Final = (
        (
            f"float-tie-{row_index}",
            "tag-float-tie",
            "2026-06-01",
            api_key,
            "float-tie-model",
            "",
            "provider-a",
            None,
            "/v1/chat/completions",
            1,
            spend,
            1,
            "2026-06-01 12:00:00",
        )
        for row_index, (api_key, spend) in enumerate(key_spends, start=1)
    )
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS INCLUDING CONSTRAINTS)").format(
                tag_table,
                sql.Identifier("LiteLLM_DailyTagSpend"),
            )
        )
        cursor.executemany(
            sql.SQL("""
            INSERT INTO {}
                (id, tag, date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint, prompt_tokens, spend, api_requests, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """).format(tag_table),
            rows,
        )
    connection.commit()


def seed_daily_team_unassigned_fixture(
    connection: psycopg.Connection, *, schema: str, ptu_sentinel_api_key: str
) -> None:
    team_table: Final = sql.Identifier(schema, "LiteLLM_DailyTeamSpend")
    rows: Final = (
        ("unassigned-null", None, "key-unassigned-null", 3.0, 0.0),
        ("unassigned-empty", "", "key-unassigned-empty", 7.0, 0.0),
        ("unassigned-ptu", None, ptu_sentinel_api_key, 13.0, 13.0),
    )
    with connection.cursor() as cursor:
        cursor.executemany(
            sql.SQL("""
            INSERT INTO {}
                (id, team_id, date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint, prompt_tokens, spend, api_requests, ptu_flat_cost, updated_at)
            VALUES (%s, %s, '2026-06-03', %s, 'model-a', '', 'provider-a', NULL, '/v1/chat/completions',
                    1, %s, 1, %s, '2026-06-03 12:00:00')
            """).format(team_table),
            rows,
        )
    connection.commit()


def seed_daily_team_tag_shared_fixture(connection: psycopg.Connection, *, schema: str) -> None:
    tag_table: Final = sql.Identifier(schema, "LiteLLM_DailyTagSpend")
    stored_tag_table: Final = sql.Identifier(schema, "LiteLLM_TagTable")
    team_table: Final = sql.Identifier(schema, "LiteLLM_TeamTable")
    tag_rows: Final = (
        ("shared-a-1", "shared", "team-a", "2026-06-10", "key-a", "model-a", 1.0),
        ("shared-a-2", "shared", "team-a", "2026-06-11", "key-b", "model-a", 4.0),
        ("shared-b-1", "shared", "team-b", "2026-06-10", "key-a", "model-a", 8.0),
        ("other-a-1", "other", "team-a", "2026-06-10", "key-a", "model-a", 1.0),
    )
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS INCLUDING CONSTRAINTS)").format(
                tag_table,
                sql.Identifier("LiteLLM_DailyTagSpend"),
            )
        )
        cursor.execute(
            sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS INCLUDING CONSTRAINTS)").format(
                stored_tag_table,
                sql.Identifier("LiteLLM_TagTable"),
            )
        )
        cursor.executemany(
            sql.SQL("""
            INSERT INTO {}
                (id, tag, team_id, date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint, prompt_tokens, completion_tokens, spend,
                 api_requests, successful_requests, failed_requests, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, '', 'provider-a', NULL, '/v1/chat/completions',
                    10, 10, %s, 1, 1, 0, NOW())
            """).format(tag_table),
            tag_rows,
        )
        cursor.executemany(
            sql.SQL(
                "INSERT INTO {} (team_id, team_alias, admins, members, models) VALUES (%s, %s, %s, %s, %s)"
            ).format(team_table),
            (
                ("team-a", "Team A", [], [], []),
                ("team-b", "Team B", [], [], []),
            ),
        )
        cursor.executemany(
            sql.SQL(
                "INSERT INTO {} (tag_name, description, models, model_info, spend) VALUES (%s, %s, %s, %s::jsonb, %s)"
            ).format(stored_tag_table),
            (
                ("shared", "stored shared tag", ["model-a"], "{}", 0.0),
                ("stored-no-usage", "stored tag without usage", [], "{}", 0.0),
            ),
        )
    connection.commit()


def seed_daily_team_exclusion_fixture(connection: psycopg.Connection, *, schema: str) -> None:
    team_table: Final = sql.Identifier(schema, "LiteLLM_DailyTeamSpend")
    rows: Final = (
        ("exclusion-null", None, "key-excluded-null", 3.0),
        ("exclusion-empty", "", "key-excluded-empty", 7.0),
        ("exclusion-dashboard", "litellm-dashboard", "key-excluded-dashboard", 11.0),
        ("exclusion-normal", "team-normal", "key-excluded-normal", 13.0),
    )
    with connection.cursor() as cursor:
        cursor.executemany(
            sql.SQL("""
            INSERT INTO {}
                (id, team_id, date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint, prompt_tokens, spend, api_requests, updated_at)
            VALUES (%s, %s, '2026-06-04', %s, 'model-a', '', 'provider-a', NULL, '/v1/chat/completions',
                    1, %s, 1, '2026-06-04 12:00:00')
            """).format(team_table),
            rows,
        )
    connection.commit()
