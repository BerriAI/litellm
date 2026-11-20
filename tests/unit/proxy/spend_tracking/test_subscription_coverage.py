import pytest

from litellm.constants import SUBSCRIPTION_BUCKET_SUFFIX
from litellm.proxy.spend_tracking.subscription_coverage import daily_spend_provider

ALLOWANCE = {"claude_subscription_usage": {"is_overage": False, "status": "allowed", "overage_status": None}}
OVERAGE = {"claude_subscription_usage": {"is_overage": True, "status": "rejected", "overage_status": "allowed"}}


@pytest.mark.parametrize(
    ("provider", "metadata", "expected"),
    [
        ("anthropic", ALLOWANCE, f"anthropic{SUBSCRIPTION_BUCKET_SUFFIX}"),
        ("anthropic", OVERAGE, "anthropic"),
        ("anthropic", {"claude_subscription_usage": None}, "anthropic"),
        ("anthropic", {}, "anthropic"),
        ("bedrock", ALLOWANCE, "bedrock"),
        ("chatgpt", ALLOWANCE, "chatgpt"),
        (None, ALLOWANCE, None),
    ],
    ids=["allowance", "overage", "no_header", "pre_existing_row", "other_provider", "provider_level", "no_provider"],
)
def test_only_a_covered_client_credential_request_moves_to_the_subscription_bucket(provider, metadata, expected):
    assert daily_spend_provider(provider, metadata) == expected
