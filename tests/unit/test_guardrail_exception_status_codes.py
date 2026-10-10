"""
Tests for guardrail exception status codes.

GuardrailRaisedException and BlockedPiiEntityError must carry
``status_code = 400`` so the proxy exception handler
(``getattr(e, "status_code", 500)``) returns HTTP 400 instead of 500
for intentional guardrail blocks.
"""

from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException


class TestGuardrailRaisedExceptionStatusCode:
    """GuardrailRaisedException should default to status_code=400."""

    def test_default_status_code(self):
        exc = GuardrailRaisedException(
            guardrail_name="test_guardrail",
            message="blocked",
        )
        assert exc.status_code == 400

    def test_custom_status_code(self):
        exc = GuardrailRaisedException(
            guardrail_name="test_guardrail",
            message="rate limited",
            status_code=429,
        )
        assert exc.status_code == 429

    def test_getattr_fallback_resolves_to_400(self):
        """The proxy uses ``getattr(e, 'status_code', 500)`` — verify it
        resolves to 400, not the 500 default."""
        exc = GuardrailRaisedException(
            guardrail_name="test_guardrail",
            message="blocked",
        )
        assert getattr(exc, "status_code", 500) == 400


class TestBlockedPiiEntityErrorStatusCode:
    """BlockedPiiEntityError should default to status_code=400."""

    def test_default_status_code(self):
        exc = BlockedPiiEntityError(
            entity_type="CREDIT_CARD",
            guardrail_name="presidio",
        )
        assert exc.status_code == 400

    def test_custom_status_code(self):
        exc = BlockedPiiEntityError(
            entity_type="SSN",
            guardrail_name="presidio",
            status_code=403,
        )
        assert exc.status_code == 403

    def test_getattr_fallback_resolves_to_400(self):
        """The proxy uses ``getattr(e, 'status_code', 500)`` — verify it
        resolves to 400, not the 500 default."""
        exc = BlockedPiiEntityError(
            entity_type="PHONE_NUMBER",
            guardrail_name="presidio",
        )
        assert getattr(exc, "status_code", 500) == 400
