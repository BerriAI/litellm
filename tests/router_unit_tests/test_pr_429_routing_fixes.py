"""
Regression tests for LiteLLM 429 routing fixes.

Covers three bugs:

1. (cooldown_handlers.py) APIConnectionError wrapping a 429 bypassed cooldown.
   `_is_cooldown_required()` skipped cooldown for any exception containing
   "APIConnectionError" — but providers.json providers have their 429s wrapped
   as APIConnectionError by the catch-all mapper, so those deployments were
   never cooled down. (Issue #24366)

2. (constants.py) providers.json providers absent from openai_compatible_providers.
   Without being in that list, their exceptions fell through to the catch-all
   mapper which wraps everything as APIConnectionError regardless of status.
   (Issue #24366)

3. (exception_mapping_utils.py) Anthropic 400 "credit balance too low" did not
   trigger fallback routing because it was mapped to BadRequestError.
   The request itself is valid — the failure is a billing/quota issue that
   should be treated like a rate limit so the router tries fallbacks.
   (Issue #24320)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_router():
    """Return a minimal mock that satisfies _is_cooldown_required()."""
    router = MagicMock()
    router.disable_cooldowns = False
    router.provider_default_deployment_ids = []
    router.get_model_group.return_value = "test-group"
    return router


def _make_anthropic_exc(status_code: int, message: str) -> MagicMock:
    exc = MagicMock()
    exc.status_code = status_code
    exc.message = message
    exc.__str__ = lambda self: self.message
    return exc


# ---------------------------------------------------------------------------
# Fix 1 — APIConnectionError wrapping 429 must still trigger cooldown
# Issue: https://github.com/BerriAI/litellm/issues/24366
# ---------------------------------------------------------------------------

class TestCooldownAPIConnectionError429:
    """
    When a providers.json provider returns HTTP 429, the exception falls through
    the catch-all mapper and is wrapped as APIConnectionError.  The cooldown
    handler must *not* ignore such exceptions — it must detect the embedded 429
    signal and still cool down the deployment.
    """

    def _is_cooldown(self, exception_str: str) -> bool:
        from litellm.router_utils.cooldown_handlers import _is_cooldown_required
        return _is_cooldown_required(
            litellm_router_instance=_make_mock_router(),
            model_id="test-deployment",
            exception_status="APIConnectionError",
            exception_str=exception_str,
        )

    def test_pure_connection_error_no_cooldown(self):
        """A genuine connection failure must NOT trigger cooldown."""
        result = self._is_cooldown(
            "litellm.APIConnectionError: Connection refused to api.example.com"
        )
        assert result is False, (
            "Pure APIConnectionError (no rate-limit signal) must not trigger cooldown"
        )

    def test_apiconnectionerror_with_429_triggers_cooldown(self):
        """
        Regression: before fix, this was False (no cooldown), causing the router
        to keep sending traffic to the rate-limited deployment.
        """
        result = self._is_cooldown(
            "litellm.APIConnectionError: Error code: 429 - "
            "{'error': {'message': 'Rate limit exceeded. Retry in 6s.', "
            "'type': 'rate_limit'}}"
        )
        assert result is True, (
            "APIConnectionError wrapping 429 must trigger cooldown so the router "
            "can pick a healthy deployment.  Regression of issue #24366."
        )

    def test_apiconnectionerror_with_rate_limit_text_triggers_cooldown(self):
        """'rate limit' text in an APIConnectionError must trigger cooldown."""
        result = self._is_cooldown(
            "litellm.APIConnectionError: Rate limit exceeded. Retry in 6s."
        )
        assert result is True

    def test_apiconnectionerror_with_ratelimit_underscore_triggers_cooldown(self):
        """'rate_limit' (underscore variant) must also trigger cooldown."""
        result = self._is_cooldown(
            "litellm.APIConnectionError: error type: rate_limit"
        )
        assert result is True

    def test_normal_429_status_code_still_cools_down(self):
        """Standard numeric 429 status code must still trigger cooldown."""
        from litellm.router_utils.cooldown_handlers import _is_cooldown_required
        result = _is_cooldown_required(
            litellm_router_instance=_make_mock_router(),
            model_id="test-deployment",
            exception_status=429,
            exception_str=None,
        )
        assert result is True

    def test_401_without_rate_limit_no_cooldown(self):
        """
        An APIConnectionError that is NOT a rate limit must still be exempt
        from cooldown (original behaviour preserved).
        """
        result = self._is_cooldown(
            "litellm.APIConnectionError: DNS resolution failed"
        )
        assert result is False


# ---------------------------------------------------------------------------
# Fix 2 — providers.json providers in openai_compatible_providers
# Issue: https://github.com/BerriAI/litellm/issues/24366
# ---------------------------------------------------------------------------

class TestProvidersJsonInOpenaiCompatible:
    """
    Providers registered via providers.json must be in openai_compatible_providers
    so that their HTTP exceptions enter the status-code-aware mapping path
    (which raises RateLimitError for 429) instead of the catch-all path
    (which raises APIConnectionError regardless of status code).
    """

    @pytest.fixture(autouse=True)
    def _import_litellm(self):
        import litellm
        self.litellm = litellm

    @pytest.mark.parametrize("provider", [
        "veniceai",
        "xiaomi_mimo",
        "scaleway",
        "abliteration",
        "llamagate",
        "gmi",
        "sarvam",
        "assemblyai",
        "charity_engine",
    ])
    def test_provider_in_openai_compatible_providers(self, provider: str):
        """
        Regression: providers.json providers were missing from this list, causing
        their 429s to fall through to the APIConnectionError catch-all.
        """
        assert provider in self.litellm.openai_compatible_providers, (
            f"Provider '{provider}' from providers.json must be in "
            "litellm.openai_compatible_providers so that its 429 errors are "
            "correctly mapped to RateLimitError.  Regression of issue #24366."
        )

    def test_existing_providers_still_present(self):
        """Ensure we didn't accidentally remove providers already in the list."""
        for p in ["groq", "deepseek", "perplexity", "together_ai", "fireworks_ai"]:
            assert p in self.litellm.openai_compatible_providers, (
                f"Pre-existing provider '{p}' must still be in openai_compatible_providers"
            )


# ---------------------------------------------------------------------------
# Fix 3 — Anthropic 400 "credit balance too low" → RateLimitError
# Issue: https://github.com/BerriAI/litellm/issues/24320
# ---------------------------------------------------------------------------

class TestAnthropicCreditBalanceMapping:
    """
    When Anthropic returns HTTP 400 with a message about insufficient credit
    balance, LiteLLM must raise RateLimitError (not BadRequestError) so that
    the router attempts fallback routing to alternative deployments.

    The request itself is valid — the failure is a billing/quota issue that
    should be treated exactly like a rate limit from the router's perspective.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import litellm
        litellm.suppress_debug_info = True
        self.litellm = litellm

    def _map(self, message: str):
        from litellm.litellm_core_utils.exception_mapping_utils import exception_type
        exc = _make_anthropic_exc(400, message)
        exception_type("claude-3-opus-20240229", exc, "anthropic")

    def test_credit_balance_too_low_raises_rate_limit_error(self):
        """
        Regression: before fix, 'credit balance too low' raised BadRequestError,
        preventing router fallback.  Now it must raise RateLimitError.
        """
        with pytest.raises(self.litellm.RateLimitError):
            self._map(
                "Your credit balance is too low to access the Anthropic API. "
                "Please go to Plans & Billing to upgrade or purchase credits."
            )

    def test_credit_balance_message_variant_rate_limit_error(self):
        """Variant phrasing 'balance is too low' must also raise RateLimitError."""
        with pytest.raises(self.litellm.RateLimitError):
            self._map(
                "AnthropicException - Your credit balance is too low."
            )

    def test_normal_400_bad_request_still_raises_bad_request_error(self):
        """
        A genuine bad request (e.g. invalid parameter) must still raise
        BadRequestError — the fix must not affect legitimate 400 errors.
        """
        with pytest.raises(self.litellm.BadRequestError):
            self._map("Invalid value for 'temperature': must be between 0 and 1.")

    def test_anthropic_400_invalid_request_error_stays_bad_request(self):
        """Anthropic 'invalid_request_error' type must stay BadRequestError."""
        with pytest.raises(self.litellm.BadRequestError):
            self._map(
                "{'type': 'error', 'error': {'type': 'invalid_request_error', "
                "'message': \"messages: roles must alternate between 'user' and 'assistant'\"}}"
            )

    def test_anthropic_429_still_raises_rate_limit_error(self):
        """HTTP 429 from Anthropic must continue to raise RateLimitError."""
        from litellm.litellm_core_utils.exception_mapping_utils import exception_type
        exc = _make_anthropic_exc(429, "Rate limit exceeded.")
        with pytest.raises(self.litellm.RateLimitError):
            exception_type("claude-3-opus-20240229", exc, "anthropic")
