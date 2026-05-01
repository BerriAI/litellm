"""
Unit tests for the Bedrock Mantle (Claude Mythos Preview) integration.

Tests cover route detection, URL construction, and config dispatch for both
the /chat/completions and /messages endpoints.
"""

from litellm.llms.bedrock.common_utils import BedrockModelInfo, get_bedrock_chat_config
from litellm.llms.bedrock.chat.mantle.transformation import AmazonMantleConfig
from litellm.llms.bedrock.messages.mantle_transformation import (
    AmazonMantleMessagesConfig,
)


def test_get_bedrock_route_mantle():
    assert (
        BedrockModelInfo.get_bedrock_route("mantle/anthropic.claude-mythos-preview")
        == "mantle"
    )


def test_get_bedrock_route_mantle_does_not_match_other_routes():
    assert (
        BedrockModelInfo.get_bedrock_route("anthropic.claude-3-sonnet-20240229-v1:0")
        != "mantle"
    )
    assert (
        BedrockModelInfo.get_bedrock_route("converse/anthropic.claude-3-sonnet")
        != "mantle"
    )


def test_explicit_mantle_route_flag():
    assert (
        BedrockModelInfo._explicit_mantle_route(
            "mantle/anthropic.claude-mythos-preview"
        )
        is True
    )
    assert BedrockModelInfo._explicit_mantle_route("anthropic.claude-3-sonnet") is False
    assert (
        BedrockModelInfo._explicit_mantle_route("converse/anthropic.claude-3-sonnet")
        is False
    )


def test_mantle_url_construction():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-east-1"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-east-1.api.aws/v1/messages"


def test_mantle_url_construction_different_region():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-west-2"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-west-2.api.aws/v1/messages"


def test_get_bedrock_chat_config_returns_mantle_config():
    config = get_bedrock_chat_config("mantle/anthropic.claude-mythos-preview")
    assert isinstance(config, AmazonMantleConfig)


def test_get_bedrock_provider_config_for_messages_api_mantle():
    config = BedrockModelInfo.get_bedrock_provider_config_for_messages_api(
        "mantle/anthropic.claude-mythos-preview"
    )
    assert isinstance(config, AmazonMantleMessagesConfig)


def test_mantle_messages_url_construction():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-east-1"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-east-1.api.aws/v1/messages"


def test_mantle_transform_request_strips_prefix_and_adds_model():
    config = AmazonMantleConfig()
    request = config.transform_request(
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "anthropic.claude-mythos-preview"
    assert "mantle/" not in request["model"]


# ── VERIA-88: aws_region_name SSRF guard ──────────────────────────────────────


import pytest


@pytest.mark.parametrize(
    "malicious_region",
    [
        # ``@`` makes the URL parser treat ``bedrock-mantle.us-east-1`` as
        # basic-auth userinfo and ``attacker.example`` as the target host.
        "us-east-1@attacker.example",
        # ``/`` injects path components (and could break out of the
        # authority entirely with a leading ``//``).
        "us-east-1/foo",
        "us-east-1//attacker.example",
        # Percent-encoded ``@``/``/`` reach ``httpx``'s URL parser intact.
        "us-east-1%40attacker.example",
        # Subdomain hop — ``.api.aws`` tail in the template is a single
        # label, not the full registered domain. ``foo.attacker.example``
        # rewrites the host completely.
        "us-east-1.attacker",
        # Userinfo with port + path.
        "us-east-1:80@attacker.example/foo",
        # Whitespace / control chars.
        " us-east-1",
        "us-east-1 ",
        "us-east-1\nHost: attacker.example",
        # Empty / missing.
        "",
    ],
)
def test_mantle_messages_url_rejects_malicious_region(malicious_region):
    """``aws_region_name`` is interpolated directly into the URL authority.
    Any character that lets the value escape the ``{region}`` slot would
    redirect the SigV4-signed request (with the operator's AWS access-key
    in the headers) to the attacker. VERIA-88."""
    config = AmazonMantleMessagesConfig()
    with pytest.raises(ValueError, match="Invalid AWS region"):
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="mantle/anthropic.claude-mythos-preview",
            optional_params={"aws_region_name": malicious_region},
            litellm_params={},
        )


@pytest.mark.parametrize(
    "valid_region",
    [
        "us-east-1",
        "us-west-2",
        "eu-west-3",
        "ap-northeast-1",
        # AWS GovCloud and ISO partitions.
        "us-gov-east-1",
        "us-gov-west-1",
        "us-isob-east-1",
        # Future regions with longer numeric suffixes.
        "us-east-10",
    ],
)
def test_mantle_messages_url_accepts_valid_region(valid_region):
    """Every documented AWS region shape must continue to work."""
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": valid_region},
        litellm_params={},
    )
    assert url == f"https://bedrock-mantle.{valid_region}.api.aws/v1/messages"
