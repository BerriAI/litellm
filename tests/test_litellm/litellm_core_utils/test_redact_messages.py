"""
Tests for litellm.litellm_core_utils.redact_messages.should_redact_message_logging

Covers the proxy flow where headers arrive in litellm_params["metadata"]["headers"]
but litellm_params["litellm_metadata"] is None.
"""

import pytest

import litellm
from litellm.litellm_core_utils.redact_messages import should_redact_message_logging


@pytest.fixture(autouse=True)
def _reset_global_redaction():
    """Ensure the global setting is off for every test."""
    original = litellm.turn_off_message_logging
    litellm.turn_off_message_logging = False
    yield
    litellm.turn_off_message_logging = original


def _make_model_call_details(
    metadata_headers=None,
    litellm_metadata=None,
    metadata=None,
    standard_callback_dynamic_params=None,
):
    """Build a model_call_details dict that mimics real proxy/SDK flows."""
    litellm_params = {}
    if metadata is not None:
        litellm_params["metadata"] = metadata
    elif metadata_headers is not None:
        litellm_params["metadata"] = {"headers": metadata_headers}
    else:
        litellm_params["metadata"] = {}

    # get_litellm_params always sets this key (even when value is None)
    litellm_params["litellm_metadata"] = litellm_metadata

    details = {"litellm_params": litellm_params}
    if standard_callback_dynamic_params is not None:
        details["standard_callback_dynamic_params"] = standard_callback_dynamic_params
    return details


class TestShouldRedactMessageLogging:
    """Unit tests for should_redact_message_logging()."""

    # ---- proxy flow: headers in metadata, litellm_metadata is None ----

    def test_enable_redaction_via_x_header_proxy_flow(self):
        """x-litellm-enable-message-redaction header should enable redaction
        even when litellm_metadata is None (proxy path)."""
        details = _make_model_call_details(
            metadata_headers={"x-litellm-enable-message-redaction": "true"},
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is True

    def test_enable_redaction_via_old_header_proxy_flow(self):
        """litellm-enable-message-redaction header should enable redaction
        even when litellm_metadata is None (proxy path)."""
        details = _make_model_call_details(
            metadata_headers={"litellm-enable-message-redaction": "true"},
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is True

    def test_disable_redaction_via_header_proxy_flow(self):
        """litellm-disable-message-redaction should suppress redaction
        even when global setting is on, and litellm_metadata is None."""
        litellm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata_headers={"litellm-disable-message-redaction": "true"},
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    # ---- SDK direct-call flow: headers in litellm_metadata ----

    def test_enable_redaction_via_header_in_litellm_metadata(self):
        """Headers inside litellm_metadata (SDK direct call) should work."""
        details = _make_model_call_details(
            litellm_metadata={"headers": {"x-litellm-enable-message-redaction": "true"}},
        )
        assert should_redact_message_logging(details) is True

    # ---- no headers at all ----

    def test_no_headers_defaults_to_global_off(self):
        """Without headers, falls back to global setting (False)."""
        details = _make_model_call_details(
            metadata_headers=None,
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    def test_no_headers_global_on(self):
        """Without headers, respects global turn_off_message_logging=True."""
        litellm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata_headers=None,
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is True

    # ---- dynamic params take precedence ----

    def test_dynamic_param_enables_redaction(self):
        """Dynamic turn_off_message_logging=True should enable redaction."""
        details = _make_model_call_details(
            metadata_headers={},
            litellm_metadata=None,
            standard_callback_dynamic_params={"turn_off_message_logging": True},
        )
        assert should_redact_message_logging(details) is True

    def test_dynamic_param_false_overrides_header(self):
        """Dynamic turn_off_message_logging=False should take precedence over enable header."""
        details = _make_model_call_details(
            metadata_headers={"x-litellm-enable-message-redaction": "true"},
            litellm_metadata=None,
            standard_callback_dynamic_params={"turn_off_message_logging": False},
        )
        assert should_redact_message_logging(details) is False

    # ---- non-dict metadata safety ----

    def test_both_metadata_fields_none(self):
        """When both litellm_metadata and metadata are None, should not raise."""
        details = _make_model_call_details(
            metadata=None,
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is False

    def test_both_metadata_fields_none_global_on(self):
        """When both metadata fields are None but global is on, should still return True."""
        litellm.turn_off_message_logging = True
        details = _make_model_call_details(
            metadata=None,
            litellm_metadata=None,
        )
        assert should_redact_message_logging(details) is True
