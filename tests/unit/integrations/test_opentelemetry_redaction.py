"""Regression tests for ``redact_user_api_key_info`` in OTEL spans (#36758)."""

from unittest.mock import MagicMock

import pytest

import litellm
from litellm.integrations.opentelemetry import OpenTelemetry

_USER_API_KEY_METADATA = {
    "user_api_key_hash": "hashed-secret",
    "user_api_key_user_id": "uid-123",
    "user_api_key_user_email": "user@example.com",
    "user_api_key_team_id": "team-456",
}


def _metadata_attribute_keys(redact: bool, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setattr(litellm, "redact_user_api_key_info", redact)
    otel_logger = OpenTelemetry()
    otel_logger.safe_set_attribute = MagicMock()
    otel_logger._set_inference_identity_attributes = MagicMock()
    otel_logger._set_service_tier_attributes = MagicMock()
    otel_logger._capture_in_span = MagicMock(return_value=False)

    otel_logger.set_attributes(
        span=MagicMock(),
        kwargs={
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "call_type": "completion",
                "metadata": {**_USER_API_KEY_METADATA, "generation_name": "test-gen"},
            },
        },
        response_obj=None,
    )
    return [call.kwargs["key"] for call in otel_logger.safe_set_attribute.call_args_list]


def test_set_attributes_drops_user_api_key_metadata_when_redaction_enabled(monkeypatch):
    attribute_keys = _metadata_attribute_keys(redact=True, monkeypatch=monkeypatch)

    assert "metadata.generation_name" in attribute_keys
    for key in _USER_API_KEY_METADATA:
        assert f"metadata.{key}" not in attribute_keys


def test_set_attributes_keeps_user_api_key_metadata_when_redaction_disabled(monkeypatch):
    attribute_keys = _metadata_attribute_keys(redact=False, monkeypatch=monkeypatch)

    for key in _USER_API_KEY_METADATA:
        assert f"metadata.{key}" in attribute_keys


@pytest.mark.parametrize("redact, expected_calls", [(True, 0), (False, 2)])
def test_team_attributes_on_child_spans_respect_redaction(monkeypatch, redact, expected_calls):
    monkeypatch.setattr(litellm, "redact_user_api_key_info", redact)
    otel_logger = OpenTelemetry()
    otel_logger.safe_set_attribute = MagicMock()

    otel_logger._set_team_attributes_on_span(span=MagicMock(), team_id="team-456", team_alias="team-alias")

    assert otel_logger.safe_set_attribute.call_count == expected_calls
