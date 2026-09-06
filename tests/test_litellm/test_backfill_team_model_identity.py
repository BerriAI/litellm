import argparse
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.backfill_team_model_identity import (
    Snapshot,
    batch_size,
    classify,
    connection_parameters,
    positive_int,
    report_batch,
)


@pytest.mark.parametrize("raw", [None, "null", "{}", '{"team_id": null}', '"{}"', '"null"'])
def test_global_models_are_unchanged(raw):
    decision = classify(Snapshot("global-id", "model_name_not_an_owner", raw, None))
    assert decision.status == "global"
    assert decision.proposed_team_id is None


@pytest.mark.parametrize("raw", ["{", "[]", "1", "true", '"not json"', '"[]"', '"123"', '"\\"{}\\""'])
def test_malformed_metadata_is_reported_without_guessing(raw):
    assert classify(Snapshot("id", "model_name_team-a_uuid", raw, None)).status == "invalid_metadata"


@pytest.mark.parametrize(
    "owner", ["", " ", "\t\n", 0, 1, True, False, [], {}, ["team"], "bad\x00id", "high\ud800id", "low\udfffid"]
)
def test_invalid_ownership_is_not_coerced(owner):
    row = Snapshot("id", "public", json.dumps({"team_id": owner}), None)
    assert classify(row).status == "invalid_team_id"


@pytest.mark.parametrize("owner", ["a", "team with spaces", " 团队 ", "a'b\"c\\d", "a\n\tb", "x" * 10000])
@pytest.mark.parametrize("encoded", [False, True])
def test_ownership_is_preserved_exactly_including_legacy_json_strings(owner, encoded):
    raw = json.dumps({"team_id": owner, "other": {"nested": "unchanged"}})
    row = Snapshot("id", "name", json.dumps(raw) if encoded else raw, None)
    assert classify(row).status == "pending"
    assert classify(row).proposed_team_id == owner
    assert classify(Snapshot("id", "name", row.raw_info, owner)).status == "already_set"


@pytest.mark.parametrize("raw", [None, "null", "{}", '{"team_id":null}', '{"team_id":"different"}'])
def test_existing_conflicting_column_is_never_overwritten(raw):
    decision = classify(Snapshot("id", "model", raw, "existing"))
    assert decision.status == "conflict"
    events = []
    assert report_batch((decision,), frozenset(), True, events.append) == 1
    assert events[0]["action"] == "none"


@pytest.mark.parametrize("public_name", [None, "public", ""])
def test_team_marker_without_owner_is_ambiguous(public_name):
    assert (
        classify(Snapshot("id", "model", json.dumps({"team_public_model_name": public_name}), None)).status
        == "invalid_team_id"
    )


@given(st.text(min_size=1).filter(lambda text: bool(text.strip()) and "\x00" not in text))
@settings(max_examples=300)
def test_classification_is_idempotent_and_does_not_normalize_owner(owner):
    row = Snapshot("id", "name", json.dumps({"team_id": owner}), None)
    pending = classify(row)
    assert pending.status == "pending"
    assert pending.proposed_team_id == owner
    applied = classify(Snapshot(row.model_id, row.model_name, row.raw_info, pending.proposed_team_id))
    assert applied.status == "already_set"
    assert applied.row.raw_info == row.raw_info


@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(),
        lambda children: st.lists(children) | st.dictionaries(st.text(), children),
        max_leaves=30,
    )
)
@settings(max_examples=300)
def test_arbitrary_json_never_creates_ownership_from_an_unrelated_field(payload):
    decision = classify(Snapshot("id", "model_name_team-hidden_uuid", json.dumps({"unrelated": payload}), None))
    assert decision.status == "global"


def test_deeply_nested_unreadable_metadata_is_a_row_error():
    decision = classify(Snapshot("id", "name", "[" * 10000 + "0" + "]" * 10000, None))
    assert decision.status == "invalid_metadata"


@pytest.mark.parametrize("encoded", [False, True])
def test_duplicate_ownership_keys_are_ambiguous(encoded):
    raw = '{"team_id":"team-a", "team_id":"team-b"}'
    assert classify(Snapshot("id", "name", json.dumps(raw) if encoded else raw, None)).status == "invalid_metadata"


def test_preview_contains_exact_transformation_without_provider_data():
    row = Snapshot("id\n1", 'public"name', json.dumps({"team_id": "team-a", "api_key": "NEVER-PRINT"}), None)
    events = []
    assert report_batch((classify(row),), frozenset(), False, events.append) == 0
    assert events == [
        {
            "event": "model",
            "model_id": "id\n1",
            "model_name": 'public"name',
            "status": "pending",
            "from_team_id": None,
            "to_team_id": "team-a",
            "action": "set_team_id",
        }
    ]
    assert "NEVER-PRINT" not in json.dumps(events)


def test_only_returned_updates_are_reported_as_committed():
    rows = tuple(classify(Snapshot(str(n), "name", '{"team_id":"a"}', None)) for n in range(2))
    events = []
    assert report_batch(rows, frozenset({"0"}), True, events.append) == 1
    assert [event["status"] for event in events] == ["updated", "concurrent_change"]
    assert [event["action"] for event in events] == ["set_team_id", "none"]


def test_prisma_url_parameters_are_removed_without_losing_tls_or_credentials():
    dsn, schema = connection_parameters(
        "postgresql://u:p%40ss@db:5432/app?schema=team%20models&sslmode=verify-full&sslrootcert=%2Fcerts%2Fca.pem&connection_limit=5&pool_timeout=10&pgbouncer=true",
        None,
    )
    assert schema == "team models"
    assert dsn == "postgresql://u:p%40ss@db:5432/app?sslmode=verify-full&sslrootcert=%2Fcerts%2Fca.pem"


@pytest.mark.parametrize(
    "dsn, schema",
    [
        ("postgresql://db/app?schema=a&schema=b", None),
        ("postgresql://db/app?schema=", None),
        ("postgresql://db/app?schema=a", "b"),
    ],
)
def test_conflicting_schema_configuration_fails(dsn, schema):
    with pytest.raises(ValueError, match="Conflicting or empty schema settings"):
        connection_parameters(dsn, schema)


@pytest.mark.parametrize(
    "dsn", ["host=/tmp dbname=app", "postgresql://db/app", "postgresql://db1,db2/app?target_session_attrs=read-write"]
)
def test_non_prisma_connection_settings_are_preserved(dsn):
    assert connection_parameters(dsn, "custom") == (dsn, "custom")


@pytest.mark.parametrize("value", ["0", "-1", "2147483648"])
def test_timeouts_cannot_be_disabled_or_overflow_postgres(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


@pytest.mark.parametrize("value", ["0", "-1", "10001"])
def test_batches_are_bounded(value):
    with pytest.raises(argparse.ArgumentTypeError):
        batch_size(value)


def test_boundary_option_values_are_accepted():
    assert batch_size("1") == 1
    assert batch_size("10000") == 10000
    assert positive_int("2147483647") == 2147483647
