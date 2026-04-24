import litellm
from litellm.router_utils.provider_account_fallback_errors import (
    is_provider_account_fallback_eligible_error,
)


def test_credit_balance_message_detected():
    err = litellm.BadRequestError(
        message="AnthropicException - Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.",
        model="claude-opus-4-6",
        llm_provider="anthropic",
    )
    assert is_provider_account_fallback_eligible_error(err) is True


def test_openai_billing_hard_limit_detected():
    err = litellm.BadRequestError(
        message='{"error":{"code":"billing_hard_limit_reached","message":"Billing"}}',
        model="gpt-4",
        llm_provider="openai",
    )
    assert is_provider_account_fallback_eligible_error(err) is True


def test_generic_validation_400_not_detected():
    err = litellm.BadRequestError(
        message="invalid maxOutputTokens",
        model="gemini-pro",
        llm_provider="vertex_ai",
    )
    assert is_provider_account_fallback_eligible_error(err) is False
