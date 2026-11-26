"""
Unit tests for spend_logs_metadata inclusion in Prometheus custom labels.

Verifies that metadata from x-litellm-spend-logs-metadata header is available
in Prometheus custom labels via combined_metadata.
"""

from typing import Final

import pytest

from litellm.integrations.prometheus import (
    _get_combined_custom_metadata_from_standard_logging_payload,
    get_custom_labels_from_metadata,
)
from litellm.types.utils import PROMPT_QUOTING_ROUTING_DECISION_FIELDS


def test_get_custom_labels_includes_spend_logs_metadata(monkeypatch):
    """
    Test that get_custom_labels_from_metadata extracts fields from
    spend_logs_metadata when it is merged into combined_metadata.
    """
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels",
        ["metadata.department", "metadata.env"],
    )

    # Simulate combined_metadata after merging all three sources
    combined_metadata = {
        "request_key": "from_requester",  # from requester_metadata
        "auth_key": "from_auth",  # from user_api_key_auth_metadata
        "department": "engineering",  # from spend_logs_metadata
        "env": "production",  # from spend_logs_metadata
    }

    result = get_custom_labels_from_metadata(combined_metadata)
    assert result == {
        "metadata_department": "engineering",
        "metadata_env": "production",
    }


def test_spend_logs_metadata_overrides_earlier_sources(monkeypatch):
    """
    Test that spend_logs_metadata values take precedence over
    requester_metadata and user_api_key_auth_metadata when keys overlap,
    since it is spread last in combined_metadata.
    """
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels",
        ["metadata.team"],
    )

    # Simulate the dict spread order: requester -> auth -> spend_logs
    requester_metadata = {"team": "old_team"}
    user_api_key_auth_metadata = {"team": "auth_team"}
    spend_logs_metadata = {"team": "spend_team"}

    combined_metadata = {
        **requester_metadata,
        **user_api_key_auth_metadata,
        **spend_logs_metadata,
    }

    result = get_custom_labels_from_metadata(combined_metadata)
    assert result == {"metadata_team": "spend_team"}


def test_combined_metadata_with_all_three_sources(monkeypatch):
    """
    Test that combined_metadata correctly merges requester_metadata,
    user_api_key_auth_metadata, and spend_logs_metadata.
    """
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels",
        ["metadata.from_requester", "metadata.from_auth", "metadata.from_spend"],
    )

    # Reproduce the exact spread pattern from prometheus.py
    _requester_metadata = {"from_requester": "val1"}
    user_api_key_auth_metadata = {"from_auth": "val2"}
    spend_logs_metadata = {"from_spend": "val3"}

    combined_metadata = {
        **(_requester_metadata if _requester_metadata else {}),
        **(user_api_key_auth_metadata if user_api_key_auth_metadata else {}),
        **(spend_logs_metadata if spend_logs_metadata else {}),
    }

    result = get_custom_labels_from_metadata(combined_metadata)
    assert result == {
        "metadata_from_requester": "val1",
        "metadata_from_auth": "val2",
        "metadata_from_spend": "val3",
    }


def test_combined_metadata_with_none_spend_logs(monkeypatch):
    """
    Test that combined_metadata works when spend_logs_metadata is None.
    """
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels",
        ["metadata.foo"],
    )

    _requester_metadata = {"foo": "bar"}
    user_api_key_auth_metadata = None
    spend_logs_metadata = None

    combined_metadata = {
        **(_requester_metadata if _requester_metadata else {}),
        **(user_api_key_auth_metadata if user_api_key_auth_metadata else {}),
        **(spend_logs_metadata if spend_logs_metadata else {}),
    }

    result = get_custom_labels_from_metadata(combined_metadata)
    assert result == {"metadata_foo": "bar"}


def test_combined_metadata_includes_top_level_fields():
    """
    Regression test for LIT-3741: user_api_key_project_alias (and other
    top-level metadata fields) must be included in the combined metadata
    so they can be referenced via custom_prometheus_metadata_labels.
    """
    standard_logging_payload = {
        "metadata": {
            "user_api_key_hash": "sk-abc123",
            "user_api_key_alias": "hotel-key",
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "hotel-team",
            "user_api_key_project_id": "proj-1",
            "user_api_key_project_alias": "hotel-recommendations",
            "user_api_key_user_id": "user-1",
            "user_api_key_user_email": "user@example.com",
            "user_api_key_end_user_id": None,
            "user_api_key_org_id": None,
            "user_api_key_org_alias": None,
            "user_api_key_request_route": "/v1/chat/completions",
            "requester_metadata": {"custom_field": "custom_value"},
            "user_api_key_auth_metadata": {"auth_field": "auth_value"},
            "spend_logs_metadata": None,
        }
    }

    combined = _get_combined_custom_metadata_from_standard_logging_payload(
        standard_logging_payload
    )

    assert combined["user_api_key_project_alias"] == "hotel-recommendations"
    assert combined["user_api_key_project_id"] == "proj-1"
    assert combined["user_api_key_team_alias"] == "hotel-team"
    assert combined["user_api_key_request_route"] == "/v1/chat/completions"
    assert combined["custom_field"] == "custom_value"
    assert combined["auth_field"] == "auth_value"


def test_project_alias_accessible_via_custom_prometheus_labels(monkeypatch):
    """
    Regression test for LIT-3741: configuring
    custom_prometheus_metadata_labels with "metadata.user_api_key_project_alias"
    should produce a label with the project's alias value.
    """
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels",
        ["metadata.user_api_key_project_alias"],
    )

    standard_logging_payload = {
        "metadata": {
            "user_api_key_project_alias": "hotel-recommendations",
            "requester_metadata": None,
            "user_api_key_auth_metadata": None,
            "spend_logs_metadata": None,
        }
    }

    combined = _get_combined_custom_metadata_from_standard_logging_payload(
        standard_logging_payload
    )
    result = get_custom_labels_from_metadata(combined)

    assert result == {"metadata_user_api_key_project_alias": "hotel-recommendations"}


def test_project_alias_accessible_without_prefix(monkeypatch):
    """
    user_api_key_project_alias should also be accessible without
    the "metadata." prefix in custom_prometheus_metadata_labels config.
    """
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels",
        ["user_api_key_project_alias"],
    )

    standard_logging_payload = {
        "metadata": {
            "user_api_key_project_alias": "hotel-recommendations",
            "requester_metadata": None,
            "user_api_key_auth_metadata": None,
            "spend_logs_metadata": None,
        }
    }

    combined = _get_combined_custom_metadata_from_standard_logging_payload(
        standard_logging_payload
    )
    result = get_custom_labels_from_metadata(combined)

    assert result == {"user_api_key_project_alias": "hotel-recommendations"}


@pytest.mark.parametrize(
    ("standard_logging_metadata", "expected_labels"),
    [
        (
            {"is_silent_experiment": True, "requester_metadata": {"is_silent_experiment": False}},
            {"metadata_is_silent_experiment": "true"},
        ),
        (
            {
                "requester_metadata": {"is_silent_experiment": True},
                "spend_logs_metadata": {"is_silent_experiment": "true"},
            },
            {},
        ),
    ],
    ids=["mirrored call", "direct call with a client flag"],
)
def test_silent_experiment_label_comes_only_from_the_router_flag(
    monkeypatch: pytest.MonkeyPatch,
    standard_logging_metadata: dict[str, object],
    expected_labels: dict[str, str],
):
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", ["metadata.is_silent_experiment"])

    combined: Final = _get_combined_custom_metadata_from_standard_logging_payload(
        {"metadata": standard_logging_metadata}
    )

    assert get_custom_labels_from_metadata(combined) == expected_labels


_ROUTING_LABELS: Final = (
    "routing_decision.tier",
    "metadata.routing_decision.cause",
    "routing_decision.routed_model",
)
_ROUTER_RECORD: Final = {"tier": "SIMPLE", "cause": "heuristic_scorer", "routed_model": "tier-small"}
_ROUTER_LABELS: Final = {
    "routing_decision_tier": "SIMPLE",
    "metadata_routing_decision_cause": "heuristic_scorer",
    "routing_decision_routed_model": "tier-small",
}
_CALLER_RECORD: Final = {"tier": "CALLER", "cause": "caller", "routed_model": "caller-model", "signals": "caller"}
_KEY_RECORD: Final = {"tier": "KEY", "cause": "key", "routed_model": "key-model"}
_KEY_LABELS: Final = {
    "routing_decision_tier": "KEY",
    "metadata_routing_decision_cause": "key",
    "routing_decision_routed_model": "key-model",
}
_AUTO_ROUTED: Final = {"routing_decision": _ROUTER_RECORD}
_RECORD_WITHOUT_A_TIER: Final = {"routing_decision": {"cause": "heuristic_scorer", "tier": None}}
_NULL_RECORD: Final = {"routing_decision": None}
_DIRECT_CALL: Final[dict[str, object]] = {}
_ROUTER_SOURCE_IDS: Final = ("auto-routed call", "record without a tier", "null record", "direct call")


@pytest.mark.parametrize(
    "caller_source",
    [
        {},
        {"requester_metadata": {"routing_decision": _CALLER_RECORD}},
        {"spend_logs_metadata": {"routing_decision": _CALLER_RECORD}},
        {"spend_logs_metadata": {"routing_decision": "CALLER"}},
    ],
    ids=["no caller value", "requester metadata", "spend logs header", "scalar value"],
)
@pytest.mark.parametrize(
    ("router_source", "expected_labels"),
    [
        (_AUTO_ROUTED, _ROUTER_LABELS),
        (_RECORD_WITHOUT_A_TIER, {"metadata_routing_decision_cause": "heuristic_scorer"}),
        (_NULL_RECORD, {}),
        (_DIRECT_CALL, {}),
    ],
    ids=_ROUTER_SOURCE_IDS,
)
def test_routing_decision_labels_never_come_from_the_caller(
    monkeypatch: pytest.MonkeyPatch,
    router_source: dict[str, object],
    caller_source: dict[str, object],
    expected_labels: dict[str, str],
):
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", list(_ROUTING_LABELS))

    combined: Final = _get_combined_custom_metadata_from_standard_logging_payload(
        {"metadata": {**caller_source, **router_source}}
    )

    assert get_custom_labels_from_metadata(combined) == expected_labels


@pytest.mark.parametrize(
    ("router_source", "request_labels"),
    [
        (_AUTO_ROUTED, _ROUTER_LABELS),
        (_RECORD_WITHOUT_A_TIER, {"metadata_routing_decision_cause": "heuristic_scorer"}),
        (_NULL_RECORD, _KEY_LABELS),
        (_DIRECT_CALL, _KEY_LABELS),
    ],
    ids=_ROUTER_SOURCE_IDS,
)
def test_key_or_team_routing_decision_labels_what_the_router_did_not_record_and_every_gauge(
    monkeypatch: pytest.MonkeyPatch, router_source: dict[str, object], request_labels: dict[str, str]
):
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", list(_ROUTING_LABELS))
    payload: Final = {
        "metadata": {
            "user_api_key_auth_metadata": {"routing_decision": _KEY_RECORD},
            "spend_logs_metadata": {"routing_decision": _CALLER_RECORD},
            **router_source,
        }
    }

    labels: Final = {
        include_router_record: get_custom_labels_from_metadata(
            _get_combined_custom_metadata_from_standard_logging_payload(
                payload, include_router_record=include_router_record
            )
        )
        for include_router_record in (True, False)
    }

    assert labels == {True: request_labels, False: _KEY_LABELS}


@pytest.mark.parametrize(
    "router_source", (_AUTO_ROUTED, _RECORD_WITHOUT_A_TIER, _NULL_RECORD, _DIRECT_CALL), ids=_ROUTER_SOURCE_IDS
)
def test_gauges_never_take_the_routers_per_request_record(
    monkeypatch: pytest.MonkeyPatch, router_source: dict[str, object]
):
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", list(_ROUTING_LABELS))

    combined: Final = _get_combined_custom_metadata_from_standard_logging_payload(
        {"metadata": router_source}, include_router_record=False
    )

    assert get_custom_labels_from_metadata(combined) == {}


@pytest.mark.parametrize("field", sorted(PROMPT_QUOTING_ROUTING_DECISION_FIELDS))
def test_routing_decision_fields_that_quote_the_prompt_never_become_labels(monkeypatch: pytest.MonkeyPatch, field: str):
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels", [f"routing_decision.{field}", "routing_decision.tier"]
    )

    combined: Final = _get_combined_custom_metadata_from_standard_logging_payload(
        {"metadata": {"routing_decision": {**_ROUTER_RECORD, field: "text from the prompt"}}}
    )

    assert get_custom_labels_from_metadata(combined) == {"routing_decision_tier": "SIMPLE"}


def test_other_nested_logging_metadata_stays_out_of_the_labels(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "litellm.custom_prometheus_metadata_labels",
        ["requester_custom_headers.x-team", "mcp_tool_call_metadata.name", "routing_decision.tier"],
    )

    combined: Final = _get_combined_custom_metadata_from_standard_logging_payload(
        {
            "metadata": {
                "requester_custom_headers": {"x-team": "from-a-header"},
                "mcp_tool_call_metadata": {"name": "search"},
                "routing_decision": _ROUTER_RECORD,
            }
        }
    )

    assert get_custom_labels_from_metadata(combined) == {"routing_decision_tier": "SIMPLE"}


def test_combining_the_label_metadata_leaves_the_logged_routing_decision_untouched(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", ["routing_decision.tier"])
    logged_record: Final = {**_ROUTER_RECORD, "matched_keyword": "text from the prompt"}

    _get_combined_custom_metadata_from_standard_logging_payload({"metadata": {"routing_decision": logged_record}})

    assert logged_record == {**_ROUTER_RECORD, "matched_keyword": "text from the prompt"}
