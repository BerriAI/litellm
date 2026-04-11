"""
Unit tests for spend_logs_metadata inclusion in Prometheus custom labels.

Verifies that metadata from x-litellm-spend-logs-metadata header is available
in Prometheus custom labels via combined_metadata.
"""
from litellm.integrations.prometheus import get_custom_labels_from_metadata


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
