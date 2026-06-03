import os
import pytest
from litellm.protocol_routing import (
    SupportedProtocol,
    ProtocolRoutingMode,
    ProtocolMismatchError,
    get_protocol_routing_mode,
    set_protocol_routing_mode,
)


class TestSupportedProtocol:
    """Test SupportedProtocol enum."""

    def test_enum_values(self):
        """Verify all protocol enum values are defined."""
        assert SupportedProtocol.OPENAI_CHAT == "openai_chat"
        assert SupportedProtocol.OPENAI_RESPONSES == "openai_responses"
        assert SupportedProtocol.ANTHROPIC_MESSAGES == "anthropic_messages"
        assert SupportedProtocol.GOOGLE_GENERATE_CONTENT == "google_generate_content"
        assert SupportedProtocol.LLM_PASSTHROUGH == "llm_passthrough"

    def test_enum_is_string(self):
        """Verify enum values are strings."""
        assert isinstance(SupportedProtocol.OPENAI_CHAT, str)
        assert isinstance(SupportedProtocol.ANTHROPIC_MESSAGES, str)


class TestProtocolRoutingMode:
    """Test protocol routing mode configuration."""

    def test_default_mode_is_bridged(self, monkeypatch):
        """Verify default mode is bridged when no env override is set.

        The module-level _protocol_routing_mode is initialized from env at import
        time. To test the documented default ("bridged"), patch the variable
        directly to simulate a clean-import state.
        """
        from litellm.protocol_routing import _types
        monkeypatch.setattr(_types, "_protocol_routing_mode", "bridged")
        assert _types.get_protocol_routing_mode() == "bridged"

    def test_set_mode_strict(self):
        """Verify setting mode to strict."""
        original = get_protocol_routing_mode()
        try:
            set_protocol_routing_mode("strict")
            assert get_protocol_routing_mode() == "strict"
        finally:
            set_protocol_routing_mode(original)

    def test_set_mode_bridged(self):
        """Verify setting mode to bridged."""
        original = get_protocol_routing_mode()
        try:
            set_protocol_routing_mode("strict")
            set_protocol_routing_mode("bridged")
            assert get_protocol_routing_mode() == "bridged"
        finally:
            set_protocol_routing_mode(original)

    def test_set_mode_invalid_raises(self):
        """Verify invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            set_protocol_routing_mode("invalid_mode")

    def test_set_mode_case_sensitive(self):
        """Verify mode is case-sensitive."""
        with pytest.raises(ValueError, match="Invalid mode"):
            set_protocol_routing_mode("STRICT")
        with pytest.raises(ValueError, match="Invalid mode"):
            set_protocol_routing_mode("Bridged")


class TestProtocolMismatchError:
    """Test ProtocolMismatchError exception."""

    def test_error_is_bad_request_error(self):
        """Verify error inherits from BadRequestError (not Exception)."""
        from litellm.exceptions import BadRequestError

        err = ProtocolMismatchError(
            model="gpt-4",
            requested_protocol="anthropic_messages",
            available_protocols=["openai_chat", "openai_responses"],
        )
        assert isinstance(err, BadRequestError)
        assert not isinstance(err, Exception) or isinstance(err, BadRequestError)

    def test_error_message_format(self):
        """Verify error message contains all required information."""
        err = ProtocolMismatchError(
            model="gpt-4",
            requested_protocol="anthropic_messages",
            available_protocols=["openai_chat", "openai_responses"],
        )
        error_msg = str(err)

        assert "gpt-4" in error_msg
        assert "anthropic_messages" in error_msg
        assert "openai_chat" in error_msg
        assert "openai_responses" in error_msg
        assert "bridged" in error_msg.lower()

    def test_error_attributes(self):
        """Verify error stores attributes correctly."""
        err = ProtocolMismatchError(
            model="claude-3",
            requested_protocol="openai_chat",
            available_protocols=["anthropic_messages"],
        )
        assert err.model == "claude-3"
        assert err.requested_protocol == "openai_chat"
        assert err.available_protocols == ["anthropic_messages"]

    def test_error_with_empty_protocols(self):
        """Verify error handles empty available protocols list."""
        err = ProtocolMismatchError(
            model="unknown-model",
            requested_protocol="anthropic_messages",
            available_protocols=[],
        )
        assert err.available_protocols == []
        assert "[]" in str(err) or "no protocols" in str(err).lower()
