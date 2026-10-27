"""Tests for the single-statement LiteLLM_ToolTable upsert (LIT-9997)."""

from litellm.proxy.db.tool_table_bulk_upsert import build_tool_bulk_upsert


def test_one_row_per_item_and_the_conflict_update_only_moves_the_counters():
    sql, params = build_tool_bulk_upsert(
        [
            {"tool_name": "tool_a", "origin": "mcp_server", "created_by": None, "call_count": 4},
            {
                "tool_name": "tool_b",
                "origin": "user_defined",
                "created_by": "alice",
                "call_count": 1,
                "key_hash": "hash",
                "team_id": "team",
                "key_alias": "alias",
                "user_agent": "agent",
            },
        ]
    )
    assert sql.startswith('INSERT INTO "LiteLLM_ToolTable"')
    assert sql.count("$1::text") == 1 and "$11::text" in sql and "$21::text" not in sql
    assert "$4::integer" in sql and "$14::integer" in sql
    assert 'ON CONFLICT ("tool_name") DO UPDATE SET' in sql
    assert '"call_count" = "LiteLLM_ToolTable"."call_count" + EXCLUDED."call_count"' in sql
    assert '"last_used_at" = EXCLUDED."last_used_at"' in sql
    assert '"input_policy"' not in sql and '"output_policy"' not in sql, "policies stay on the table defaults"
    assert '"origin" =' not in sql and '"key_hash" =' not in sql, "an existing row keeps who discovered it"
    assert len(params) == 20
    assert params[1:10] == ("tool_a", "mcp_server", 4, "system", "system", None, None, None, None)
    assert params[11:20] == ("tool_b", "user_defined", 1, "alice", "alice", "hash", "team", "alias", "agent")
    assert params[0] != params[10], "each row gets its own tool_id"


def test_defaults_fill_origin_created_by_and_call_count():
    _, params = build_tool_bulk_upsert([{"tool_name": "tool_a"}])
    assert params[1:6] == ("tool_a", "user_defined", 1, "system", "system")
