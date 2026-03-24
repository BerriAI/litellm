"""
Tests that guardrail exceptions return proper HTTP status codes.

GuardrailRaisedException and BlockedPiiEntityError should return 400,
not 500, since guardrail blocks are client-triggered errors.
"""

from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException


class TestGuardrailRaisedExceptionStatusCode:
    def test_default_status_code_is_400(self):
        exc = GuardrailRaisedException(
            guardrail_name="test-guardrail",
            message="blocked",
        )
        assert exc.status_code == 400

    def test_custom_status_code(self):
        exc = GuardrailRaisedException(
            guardrail_name="test-guardrail",
            message="rate limited",
            status_code=429,
        )
        assert exc.status_code == 429

    def test_getattr_returns_status_code(self):
        """Mirrors the pattern used in _handle_llm_api_exception"""
        exc = GuardrailRaisedException(
            guardrail_name="test-guardrail",
            message="blocked",
        )
        assert getattr(exc, "status_code", 500) == 400


class TestBlockedPiiEntityErrorStatusCode:
    def test_default_status_code_is_400(self):
        exc = BlockedPiiEntityError(
            entity_type="CREDIT_CARD",
            guardrail_name="presidio",
        )
        assert exc.status_code == 400

    def test_custom_status_code(self):
        exc = BlockedPiiEntityError(
            entity_type="CREDIT_CARD",
            guardrail_name="presidio",
            status_code=403,
        )
        assert exc.status_code == 403

    def test_getattr_returns_status_code(self):
        """Mirrors the pattern used in _handle_llm_api_exception"""
        exc = BlockedPiiEntityError(
            entity_type="CREDIT_CARD",
            guardrail_name="presidio",
        )
        assert getattr(exc, "status_code", 500) == 400
