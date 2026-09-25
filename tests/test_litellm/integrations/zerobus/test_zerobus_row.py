import json

from litellm.integrations.zerobus.row import TRACE_TABLE_COLUMNS, create_table_sql, trace_row


def _payload() -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "trace_id": "trace-1",
        "session_id": "session-1",
        "litellm_call_id": "call-1",
        "call_type": "acompletion",
        "status": "success",
        "model": "gpt-4o",
        "model_group": "gpt-4o-group",
        "custom_llm_provider": "openai",
        "api_base": "https://api.openai.com",
        "stream": False,
        "cache_hit": None,
        "startTime": 1_700_000_000.25,
        "endTime": 1_700_000_001.5,
        "completionStartTime": 1_700_000_000.75,
        "response_time": 1.25,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "response_cost": 0.0015,
        "saved_cache_cost": 0.0,
        "end_user": "end-user-1",
        "requester_ip_address": "10.0.0.1",
        "user_agent": "curl/8",
        "request_tags": ["prod"],
        "messages": [{"role": "user", "content": "hi"}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        "error_str": None,
        "error_information": None,
        "metadata": {
            "user_api_key_hash": "hash-1",
            "user_api_key_alias": "alias-1",
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "team-alias-1",
            "user_api_key_user_id": "user-1",
            "user_api_key_org_id": "org-1",
        },
        "model_parameters": {"temperature": 0.2},
        "hidden_params": {"response_cost": 0.0015},
        "guardrail_information": None,
        "cost_breakdown": {"input_cost": 0.001, "output_cost": 0.0005},
    }


def test_every_row_has_exactly_the_documented_columns():
    """Zerobus rejects a record naming a column the table lacks, so the row and the DDL must agree."""
    assert tuple(trace_row(_payload())) == tuple(TRACE_TABLE_COLUMNS)
    assert tuple(trace_row({})) == tuple(TRACE_TABLE_COLUMNS)


def test_scalars_land_in_their_columns():
    row = trace_row(_payload())

    assert row["id"] == "chatcmpl-1"
    assert row["trace_id"] == "trace-1"
    assert row["status"] == "success"
    assert row["model"] == "gpt-4o"
    assert row["stream"] is False
    assert row["prompt_tokens"] == 10
    assert row["total_tokens"] == 15
    assert row["response_cost"] == 0.0015
    assert row["end_user"] == "end-user-1"


def test_key_and_team_identity_is_lifted_out_of_metadata():
    """Filtering spend by team or key is the main query, so those live in their own columns."""
    row = trace_row(_payload())

    assert row["api_key_hash"] == "hash-1"
    assert row["api_key_alias"] == "alias-1"
    assert row["team_id"] == "team-1"
    assert row["team_alias"] == "team-alias-1"
    assert row["user_id"] == "user-1"
    assert row["org_id"] == "org-1"


def test_timestamps_become_epoch_microseconds():
    row = trace_row(_payload())

    assert row["start_time"] == 1_700_000_000_250_000
    assert row["end_time"] == 1_700_000_001_500_000
    assert row["completion_start_time"] == 1_700_000_000_750_000


def test_a_zero_timestamp_is_null_rather_than_1970():
    """LiteLLM leaves completionStartTime at 0 when there is no first token, which is not a real time."""
    row = trace_row({**_payload(), "completionStartTime": 0})

    assert row["completion_start_time"] is None


def test_nested_fields_are_json_text_for_the_variant_columns():
    row = trace_row(_payload())

    assert json.loads(str(row["messages"])) == [{"role": "user", "content": "hi"}]
    assert json.loads(str(row["metadata"]))["user_api_key_team_id"] == "team-1"
    assert json.loads(str(row["request_tags"])) == ["prod"]
    assert json.loads(str(row["cost_breakdown"])) == {"input_cost": 0.001, "output_cost": 0.0005}


def test_missing_and_null_fields_are_null():
    row = trace_row({**_payload(), "messages": None, "guardrail_information": None})

    assert row["messages"] is None
    assert row["guardrail_information"] is None
    assert row["error_str"] is None
    assert row["cache_hit"] is None


def test_a_wrongly_typed_field_is_null_instead_of_a_rejected_record():
    """One odd payload must not poison the whole batch: the table type wins."""
    row = trace_row({**_payload(), "prompt_tokens": "ten", "stream": "yes", "startTime": "now"})

    assert row["prompt_tokens"] is None
    assert row["stream"] is None
    assert row["start_time"] is None


def test_the_row_is_json_serializable():
    json.dumps(dict(trace_row(_payload())))


def test_create_table_sql_declares_every_column_with_its_type():
    sql = create_table_sql("main.litellm.traces")

    assert sql.startswith("CREATE TABLE main.litellm.traces (")
    assert "  start_time TIMESTAMP," in sql
    assert "  messages VARIANT," in sql
    assert "  cost_breakdown VARIANT\n);" in sql
    assert sql.count(",") == len(TRACE_TABLE_COLUMNS) - 1
