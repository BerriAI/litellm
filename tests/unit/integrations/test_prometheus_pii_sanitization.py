import warnings
import pytest
from unittest.mock import MagicMock, patch

import litellm
from litellm.integrations.prometheus_helpers import (
    PrometheusLabelFactoryContext,
    sanitize_pii_label_value,
)
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


def test_sanitize_pii_label_value_user_email():
    """Verify that user identifiers containing email addresses are sanitized to a deterministic hash."""
    email_user = "employee@internal.company.com"
    sanitized = sanitize_pii_label_value("user", email_user)
    assert sanitized is not None
    assert "internal.company.com" not in sanitized
    assert "employee" not in sanitized
    assert sanitized.startswith("redacted_user_")

    sanitized_email = sanitize_pii_label_value("user_email", email_user)
    assert sanitized_email is not None
    assert "internal.company.com" not in sanitized_email
    assert sanitized_email.startswith("redacted_user_")


def test_sanitize_pii_label_value_team_alias():
    """Verify that employee emails in team_alias are masked."""
    team_alias = "Enterprise Corp (POC: admin@corp.example.com)"
    sanitized = sanitize_pii_label_value("team_alias", team_alias)
    assert sanitized is not None
    assert "admin@corp.example.com" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized
    assert "Enterprise Corp" in sanitized


def test_sanitize_pii_label_value_client_ip():
    """Verify that IPv4 client IPs mask host topology and IPv6 IPs are hashed."""
    ipv4 = "198.51.100.42"
    sanitized_ipv4 = sanitize_pii_label_value("client_ip", ipv4)
    assert sanitized_ipv4 == "198.51.0.0/16"

    ipv6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    sanitized_ipv6 = sanitize_pii_label_value("client_ip", ipv6)
    assert sanitized_ipv6 is not None
    assert sanitized_ipv6.startswith("ipv6_")


def test_sanitize_pii_label_value_user_agent_uuid():
    """Verify that UUIDs (e.g. Azure Logic Apps workflow IDs) in user-agent strings are redacted."""
    ua = "AzureLogicApps/1.0 workflow/d9b4b726-1e6f-4f3c-8a5e-2f190e3fbb61"
    sanitized = sanitize_pii_label_value("user_agent", ua)
    assert "d9b4b726-1e6f-4f3c-8a5e-2f190e3fbb61" not in sanitized
    assert "AzureLogicApps/1.0 workflow/[UUID]" in sanitized


def test_prometheus_label_context_auto_sanitization_when_unauthenticated(monkeypatch):
    """
    When /metrics endpoint is exposed unauthenticated (via allow_unauthenticated_metrics=True
    or require_auth_for_metrics_endpoint=False), PII fields must automatically be sanitized
    in PrometheusLabelFactoryContext to protect multi-tenant privacy.
    """
    monkeypatch.setattr(litellm, "allow_unauthenticated_metrics", True)
    monkeypatch.setattr(litellm, "require_auth_for_metrics_endpoint", False)
    monkeypatch.setattr(litellm, "prometheus_mask_pii_labels", None)

    enum_values = UserAPIKeyLabelValues(
        user="john.doe@example.com",
        user_email="john.doe@example.com",
        team_alias="Finance Dept (lead: jane@example.com)",
        client_ip="203.0.113.195",
        user_agent="AzureLogicApps/1.0 workflow/12345678-1234-1234-1234-123456789abc",
    )

    ctx = PrometheusLabelFactoryContext(enum_values)

    assert "john.doe@example.com" not in str(ctx.sanitized_enum.get("user"))
    assert ctx.sanitized_enum["user"].startswith("redacted_user_")
    assert "john.doe@example.com" not in str(ctx.sanitized_enum.get("user_email"))
    assert "[REDACTED_EMAIL]" in ctx.sanitized_enum["team_alias"]
    assert ctx.sanitized_enum["client_ip"] == "203.0.0.0/16"
    assert "12345678-1234-1234-1234-123456789abc" not in ctx.sanitized_enum["user_agent"]
    assert "[UUID]" in ctx.sanitized_enum["user_agent"]


def test_prometheus_label_context_preserves_raw_when_authenticated(monkeypatch):
    """
    When /metrics endpoint is properly authenticated, raw label values are preserved
    for internal dashboard backward-compatibility.
    """
    monkeypatch.setattr(litellm, "allow_unauthenticated_metrics", None)
    monkeypatch.setattr(litellm, "require_auth_for_metrics_endpoint", True)
    monkeypatch.setattr(litellm, "prometheus_mask_pii_labels", None)

    enum_values = UserAPIKeyLabelValues(
        user="john.doe@example.com",
        user_email="john.doe@example.com",
        team_alias="Finance Dept",
        client_ip="203.0.113.195",
    )

    ctx = PrometheusLabelFactoryContext(enum_values)

    assert ctx.sanitized_enum["user"] == "john.doe@example.com"
    assert ctx.sanitized_enum["user_email"] == "john.doe@example.com"
    assert ctx.sanitized_enum["team_alias"] == "Finance Dept"
    assert ctx.sanitized_enum["client_ip"] == "203.0.113.195"


def test_prometheus_label_context_explicit_mask_pii_flag(monkeypatch):
    """
    When prometheus_mask_pii_labels=True is explicitly set, PII is masked even when authenticated.
    """
    monkeypatch.setattr(litellm, "allow_unauthenticated_metrics", None)
    monkeypatch.setattr(litellm, "require_auth_for_metrics_endpoint", True)
    monkeypatch.setattr(litellm, "prometheus_mask_pii_labels", True)

    enum_values = UserAPIKeyLabelValues(
        user="john.doe@example.com",
        client_ip="203.0.113.195",
    )

    ctx = PrometheusLabelFactoryContext(enum_values)
    assert ctx.sanitized_enum["user"].startswith("redacted_user_")
    assert ctx.sanitized_enum["client_ip"] == "203.0.0.0/16"


def test_prometheus_mount_warning_when_unauthenticated(monkeypatch):
    """
    Verify that mounting /metrics when unauthenticated emits a UserWarning and logs a warning.
    """
    from litellm.integrations.prometheus import PrometheusLogger

    monkeypatch.setattr(litellm, "allow_unauthenticated_metrics", True)
    monkeypatch.setattr(litellm, "require_auth_for_metrics_endpoint", False)

    mock_app = MagicMock()
    with patch("litellm.proxy.proxy_server.app", mock_app), \
         patch("litellm.integrations.prometheus_metrics_endpoint.make_metrics_asgi_app", return_value=MagicMock()):
        with pytest.warns(UserWarning, match="without authentication"):
            PrometheusLogger._mount_metrics_endpoint()

        mock_app.mount.assert_called_once()
