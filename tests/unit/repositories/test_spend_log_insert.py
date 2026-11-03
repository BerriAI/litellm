from datetime import datetime, timezone
import json

import pytest

from litellm.repositories.spend_log_insert import spend_log_insert_statement, spend_log_takeover_statement


@pytest.mark.parametrize(
    "rows",
    [(), ({"request_id": "request", "unknown_column": 1},), ({"request_id": "first"}, {"spend": 1})],
)
def test_insert_rejects_empty_unknown_or_inconsistent_columns(rows):
    with pytest.raises(ValueError, match="Spend log insert requires"):
        spend_log_insert_statement(rows)


def test_takeover_rejects_unknown_columns():
    with pytest.raises(ValueError, match="Batch takeover requires known SpendLogs columns"):
        spend_log_takeover_statement({"request_id": "request", "untrusted_column": "value"})


def test_insert_keeps_json_and_timestamps_in_bound_data():
    timestamp = datetime(2026, 11, 2, tzinfo=timezone.utc)
    request_id = "request'); DROP TABLE unrelated; --"
    statement, encoded = spend_log_insert_statement(({
        "request_id": request_id, "startTime": timestamp, "metadata": '{"project":"project"}',
        "request_tags": '["tag"]', "messages": '[{"role":"user","content":"hello"}]',
    },))
    row = json.loads(encoded)[0]
    assert request_id not in statement
    assert row["request_id"] == request_id
    assert row["startTime"] == timestamp.isoformat()
    assert row["metadata"] == {"project": "project"}
    assert row["request_tags"] == ["tag"]
    assert row["messages"] == [{"role": "user", "content": "hello"}]
