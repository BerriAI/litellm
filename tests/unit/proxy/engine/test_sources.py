import base64
import json

from litellm.proxy.engine.sources import execution_id, parse_execution


def test_same_trace_id_from_different_keys_is_a_distinct_execution() -> None:
    assert execution_id("traces", "team", "trace", "key-one-ref") != execution_id(
        "traces", "team", "trace", "key-two-ref"
    )
    assert parse_execution(execution_id("traces", "team", "trace", "key-one-ref")) == (
        "traces",
        "team",
        "trace",
        "key-one-ref",
    )


def test_previous_saved_findings_keep_their_execution_links() -> None:
    assert parse_execution(base64.urlsafe_b64encode(json.dumps(("traces", "team", "trace")).encode()).decode()) == (
        "traces",
        "team",
        "trace",
        "",
    )
