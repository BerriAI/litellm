"""
Unit tests for Bedrock Guardrails
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../../../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
    _redact_pii_matches,
)
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test__redact_pii_matches_function():
    """Test the _redact_pii_matches function directly"""

    # Test case 1: Response with PII entities
    response_with_pii = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {"type": "NAME", "match": "John Smith", "action": "BLOCKED"},
                        {
                            "type": "US_SOCIAL_SECURITY_NUMBER",
                            "match": "324-12-3212",
                            "action": "BLOCKED",
                        },
                        {"type": "PHONE", "match": "607-456-7890", "action": "BLOCKED"},
                    ]
                }
            }
        ],
        "outputs": [{"text": "Input blocked by PII policy"}],
    }

    # Call the redaction function
    redacted_response = _redact_pii_matches(response_with_pii)

    # Verify that PII matches are redacted
    pii_entities = redacted_response["assessments"][0]["sensitiveInformationPolicy"][
        "piiEntities"
    ]

    assert pii_entities[0]["match"] == "[REDACTED]", "Name should be redacted"
    assert pii_entities[1]["match"] == "[REDACTED]", "SSN should be redacted"
    assert pii_entities[2]["match"] == "[REDACTED]", "Phone should be redacted"

    # Verify other fields remain unchanged
    assert pii_entities[0]["type"] == "NAME"
    assert pii_entities[1]["type"] == "US_SOCIAL_SECURITY_NUMBER"
    assert pii_entities[2]["type"] == "PHONE"
    assert redacted_response["action"] == "GUARDRAIL_INTERVENED"
    assert redacted_response["outputs"][0]["text"] == "Input blocked by PII policy"

    print("PII redaction function test passed")


@pytest.mark.asyncio
async def test__redact_pii_matches_no_pii():
    """Test _redact_pii_matches with response that has no PII"""

    response_no_pii = {"action": "NONE", "assessments": [], "outputs": []}

    # Call the redaction function
    redacted_response = _redact_pii_matches(response_no_pii)

    # Should return the same response unchanged
    assert redacted_response == response_no_pii
    print("No PII redaction test passed")


@pytest.mark.asyncio
async def test__redact_pii_matches_empty_assessments():
    """Test _redact_pii_matches with empty assessments"""

    response_empty_assessments = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{"sensitiveInformationPolicy": {"piiEntities": []}}],
        "outputs": [{"text": "Some output"}],
    }

    # Call the redaction function
    redacted_response = _redact_pii_matches(response_empty_assessments)

    # Should return the same response unchanged
    assert redacted_response == response_empty_assessments
    print("Empty assessments redaction test passed")


@pytest.mark.asyncio
async def test__redact_pii_matches_malformed_response():
    """Test _redact_pii_matches with malformed response (should not crash)"""

    # Test with completely malformed response
    malformed_response = {
        "action": "GUARDRAIL_INTERVENED",
        # Wrong type for assessments; redact_nested_match_and_regex_keys walks dict
        # values and skips non-dict/list nodes, so this must not raise.
        "assessments": "not_a_list",
    }

    # Should not crash (deep copy + walk skips the string value under assessments)
    redacted_response = _redact_pii_matches(malformed_response)
    assert redacted_response == malformed_response

    # Test with missing keys
    missing_keys_response = {
        "action": "GUARDRAIL_INTERVENED"
        # Missing assessments key
    }

    redacted_response = _redact_pii_matches(missing_keys_response)
    assert redacted_response == missing_keys_response

    print("Malformed response redaction test passed")


@pytest.mark.asyncio
async def test__redact_pii_matches_multiple_assessments():
    """Test _redact_pii_matches with multiple assessments containing PII"""

    response_multiple_assessments = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "EMAIL",
                            "match": "john@example.com",
                            "action": "ANONYMIZED",
                        }
                    ]
                }
            },
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "CREDIT_DEBIT_CARD_NUMBER",
                            "match": "1234-5678-9012-3456",
                            "action": "BLOCKED",
                        },
                        {
                            "type": "ADDRESS",
                            "match": "123 Main St, Anytown USA",
                            "action": "ANONYMIZED",
                        },
                    ]
                }
            },
        ],
        "outputs": [{"text": "Multiple PII detected"}],
    }

    # Call the redaction function
    redacted_response = _redact_pii_matches(response_multiple_assessments)

    # Verify all PII in all assessments are redacted
    assessment1_pii = redacted_response["assessments"][0]["sensitiveInformationPolicy"][
        "piiEntities"
    ]
    assessment2_pii = redacted_response["assessments"][1]["sensitiveInformationPolicy"][
        "piiEntities"
    ]

    assert assessment1_pii[0]["match"] == "[REDACTED]", "Email should be redacted"
    assert assessment2_pii[0]["match"] == "[REDACTED]", "Credit card should be redacted"
    assert assessment2_pii[1]["match"] == "[REDACTED]", "Address should be redacted"

    # Verify types remain unchanged
    assert assessment1_pii[0]["type"] == "EMAIL"
    assert assessment2_pii[0]["type"] == "CREDIT_DEBIT_CARD_NUMBER"
    assert assessment2_pii[1]["type"] == "ADDRESS"

    print("Multiple assessments redaction test passed")


@pytest.mark.asyncio
async def test_bedrock_guardrail_logging_uses_redacted_response():
    """Debug logs and standard_logging payloads must not include raw match values."""

    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )

    # Mock the Bedrock API response with PII
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Hello, my phone number is {PHONE}"}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "PHONE",
                            "match": "+1 412 555 1212",  # This should be redacted in logs
                            "action": "ANONYMIZED",
                        }
                    ]
                }
            }
        ],
    }

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello, my phone number is +1 412 555 1212"},
        ],
    }

    # Mock AWS credentials to avoid credential loading issues in CI
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    # Mock AWS-related methods to ensure test runs without external dependencies
    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch(
            "litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.verbose_proxy_logger.debug"
        ) as mock_debug,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ) as mock_load_creds,
        patch.object(
            guardrail, "_prepare_request", return_value=MagicMock()
        ) as mock_prepare_request,
    ):

        mock_post.return_value = mock_bedrock_response

        # Call the method that should log the redacted response
        await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=request_data.get("messages"),
            request_data=request_data,
        )

        # Verify that debug logging was called
        mock_debug.assert_called()

        # Get the logged response (second argument to debug call)
        logged_calls = mock_debug.call_args_list
        bedrock_response_log_call = None

        for call in logged_calls:
            args, kwargs = call
            if len(args) >= 2 and "Bedrock AI response" in str(args[0]):
                bedrock_response_log_call = call
                break

        assert (
            bedrock_response_log_call is not None
        ), "Should have logged Bedrock AI response"

        # Extract the logged response data
        logged_response = bedrock_response_log_call[0][
            1
        ]  # Second argument to debug call

        # Verify that the logged response has redacted PII
        assert (
            logged_response["assessments"][0]["sensitiveInformationPolicy"][
                "piiEntities"
            ][0]["match"]
            == "[REDACTED]"
        )

        # Verify other fields are preserved
        assert logged_response["action"] == "GUARDRAIL_INTERVENED"
        assert (
            logged_response["assessments"][0]["sensitiveInformationPolicy"][
                "piiEntities"
            ][0]["type"]
            == "PHONE"
        )

        slg_list = request_data["metadata"]["standard_logging_guardrail_information"]
        assert (
            slg_list[0]["guardrail_response"]["assessments"][0][
                "sensitiveInformationPolicy"
            ]["piiEntities"][0]["match"]
            == "[REDACTED]"
        )

        print("Bedrock guardrail logging redaction test passed")


@pytest.mark.asyncio
async def test_bedrock_guardrail_original_response_not_modified():
    """Test that the original response is not modified by redaction, only the logged version"""

    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )

    # Mock the Bedrock API response with PII
    original_response_data = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Hello, my phone number is {PHONE}"}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "PHONE",
                            "match": "+1 412 555 1212",  # This should NOT be modified in original
                            "action": "ANONYMIZED",
                        }
                    ]
                }
            }
        ],
    }

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = original_response_data

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello, my phone number is +1 412 555 1212"},
        ],
    }

    # Mock AWS credentials to avoid credential loading issues in CI
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    # Mock AWS-related methods to ensure test runs without external dependencies
    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ) as mock_load_creds,
        patch.object(
            guardrail, "_prepare_request", return_value=MagicMock()
        ) as mock_prepare_request,
    ):

        mock_post.return_value = mock_bedrock_response

        # Call the method
        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=request_data.get("messages"),
            request_data=request_data,
        )

        # Verify that the original response data was not modified
        # (The json() method should return the original data)
        original_data = mock_bedrock_response.json()
        assert (
            original_data["assessments"][0]["sensitiveInformationPolicy"][
                "piiEntities"
            ][0]["match"]
            == "+1 412 555 1212"
        )

        # Verify that the returned BedrockGuardrailResponse contains original data
        assert (
            result["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0][
                "match"
            ]
            == "+1 412 555 1212"
        )

        print("Original response not modified test passed")


@pytest.mark.asyncio
async def test__redact_pii_matches_preserves_non_pii_entities():
    """Test that _redact_pii_matches only affects PII-related entities and preserves other assessment data"""

    response_with_mixed_data = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "EMAIL",
                            "match": "user@example.com",
                            "action": "ANONYMIZED",
                            "confidence": "HIGH",
                        }
                    ],
                    "regexes": [
                        {
                            "name": "custom_pattern",
                            "match": "some_pattern_match",
                            "action": "BLOCKED",
                        }
                    ],
                },
                "contentPolicy": {
                    "filters": [
                        {
                            "type": "VIOLENCE",
                            "confidence": "MEDIUM",
                            "action": "BLOCKED",
                        }
                    ]
                },
                "topicPolicy": {
                    "topics": [
                        {
                            "name": "Restricted Topic",
                            "type": "DENY",
                            "action": "BLOCKED",
                        }
                    ]
                },
            }
        ],
        "outputs": [{"text": "Content blocked"}],
    }

    # Call the redaction function
    redacted_response = _redact_pii_matches(response_with_mixed_data)

    # Verify that PII entity matches are redacted
    pii_entities = redacted_response["assessments"][0]["sensitiveInformationPolicy"][
        "piiEntities"
    ]
    assert pii_entities[0]["match"] == "[REDACTED]", "PII match should be redacted"
    assert pii_entities[0]["type"] == "EMAIL", "PII type should be preserved"
    assert pii_entities[0]["action"] == "ANONYMIZED", "PII action should be preserved"
    assert pii_entities[0]["confidence"] == "HIGH", "PII confidence should be preserved"

    # Verify that regex matches are also redacted (updated behavior)
    regexes = redacted_response["assessments"][0]["sensitiveInformationPolicy"][
        "regexes"
    ]
    assert regexes[0]["match"] == "[REDACTED]", "Regex match should be redacted"
    assert regexes[0]["name"] == "custom_pattern", "Regex name should be preserved"
    assert regexes[0]["action"] == "BLOCKED", "Regex action should be preserved"

    # Verify that other policies are completely unchanged
    content_policy = redacted_response["assessments"][0]["contentPolicy"]
    assert content_policy["filters"][0]["type"] == "VIOLENCE"
    assert content_policy["filters"][0]["confidence"] == "MEDIUM"

    topic_policy = redacted_response["assessments"][0]["topicPolicy"]
    assert topic_policy["topics"][0]["name"] == "Restricted Topic"

    # Verify top-level fields are unchanged
    assert redacted_response["action"] == "GUARDRAIL_INTERVENED"
    assert redacted_response["outputs"][0]["text"] == "Content blocked"

    print("Preserves non-PII entities test passed")


@pytest.mark.asyncio
async def test_pii_redaction_matches_debug_output_format():
    """Test that demonstrates the exact behavior shown in your debug output"""

    # This matches the structure from your debug output
    original_response = {
        "action": "GUARDRAIL_INTERVENED",
        "actionReason": "Guardrail blocked.",
        "assessments": [
            {
                "invocationMetrics": {
                    "guardrailCoverage": {
                        "textCharacters": {"guarded": 84, "total": 84}
                    },
                    "guardrailProcessingLatency": 322,
                    "usage": {
                        "contentPolicyImageUnits": 0,
                        "contentPolicyUnits": 0,
                        "contextualGroundingPolicyUnits": 0,
                        "sensitiveInformationPolicyFreeUnits": 0,
                        "sensitiveInformationPolicyUnits": 1,
                        "topicPolicyUnits": 0,
                        "wordPolicyUnits": 0,
                    },
                },
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "action": "BLOCKED",
                            "detected": True,
                            "match": "John Smith",
                            "type": "NAME",
                        },
                        {
                            "action": "BLOCKED",
                            "detected": True,
                            "match": "324-12-3212",
                            "type": "US_SOCIAL_SECURITY_NUMBER",
                        },
                        {
                            "action": "BLOCKED",
                            "detected": True,
                            "match": "607-456-7890",
                            "type": "PHONE",
                        },
                    ]
                },
            }
        ],
        "blockedResponse": "Input blocked by PII policy",
        "guardrailCoverage": {"textCharacters": {"guarded": 84, "total": 84}},
        "output": [{"text": "Input blocked by PII policy"}],
        "outputs": [{"text": "Input blocked by PII policy"}],
        "usage": {
            "contentPolicyImageUnits": 0,
            "contentPolicyUnits": 0,
            "contextualGroundingPolicyUnits": 0,
            "sensitiveInformationPolicyFreeUnits": 0,
            "sensitiveInformationPolicyUnits": 1,
            "topicPolicyUnits": 0,
            "wordPolicyUnits": 0,
        },
    }

    # Apply redaction
    redacted_response = _redact_pii_matches(original_response)

    # Verify the redacted response matches your expected debug output
    pii_entities = redacted_response["assessments"][0]["sensitiveInformationPolicy"][
        "piiEntities"
    ]

    # All PII matches should be redacted
    assert pii_entities[0]["match"] == "[REDACTED]", "NAME should be redacted"
    assert pii_entities[1]["match"] == "[REDACTED]", "SSN should be redacted"
    assert pii_entities[2]["match"] == "[REDACTED]", "PHONE should be redacted"

    # But all other fields should be preserved
    assert pii_entities[0]["type"] == "NAME"
    assert pii_entities[1]["type"] == "US_SOCIAL_SECURITY_NUMBER"
    assert pii_entities[2]["type"] == "PHONE"
    assert pii_entities[0]["action"] == "BLOCKED"
    assert pii_entities[0]["detected"] == True

    # Verify that the original response is unchanged
    original_pii_entities = original_response["assessments"][0][
        "sensitiveInformationPolicy"
    ]["piiEntities"]
    assert (
        original_pii_entities[0]["match"] == "John Smith"
    ), "Original should be unchanged"
    assert (
        original_pii_entities[1]["match"] == "324-12-3212"
    ), "Original should be unchanged"
    assert (
        original_pii_entities[2]["match"] == "607-456-7890"
    ), "Original should be unchanged"

    # Verify all other metadata is preserved in redacted response
    assert redacted_response["action"] == "GUARDRAIL_INTERVENED"
    assert redacted_response["actionReason"] == "Guardrail blocked."
    assert redacted_response["blockedResponse"] == "Input blocked by PII policy"
    assert (
        redacted_response["assessments"][0]["invocationMetrics"][
            "guardrailProcessingLatency"
        ]
        == 322
    )

    print("PII redaction matches debug output format test passed")
    print(
        f"Original PII values preserved: {[e['match'] for e in original_pii_entities]}"
    )
    print(f"Redacted PII values: {[e['match'] for e in pii_entities]}")


@pytest.mark.asyncio
async def test__redact_pii_matches_with_regex_matches():
    """Test redaction of regex matches in sensitive information policy"""

    response_with_regex = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "regexes": [
                        {
                            "name": "SSN_PATTERN",
                            "match": "123-45-6789",
                            "action": "BLOCKED",
                        },
                        {
                            "name": "CREDIT_CARD_PATTERN",
                            "match": "4111-1111-1111-1111",
                            "action": "ANONYMIZED",
                        },
                    ]
                }
            }
        ],
        "outputs": [{"text": "Regex patterns detected"}],
    }

    # Call the redaction function
    redacted_response = _redact_pii_matches(response_with_regex)

    # Verify that regex matches are redacted
    regexes = redacted_response["assessments"][0]["sensitiveInformationPolicy"][
        "regexes"
    ]

    assert regexes[0]["match"] == "[REDACTED]", "SSN regex match should be redacted"
    assert (
        regexes[1]["match"] == "[REDACTED]"
    ), "Credit card regex match should be redacted"

    # Verify other fields are preserved
    assert regexes[0]["name"] == "SSN_PATTERN", "Regex name should be preserved"
    assert regexes[0]["action"] == "BLOCKED", "Regex action should be preserved"
    assert regexes[1]["name"] == "CREDIT_CARD_PATTERN", "Regex name should be preserved"
    assert regexes[1]["action"] == "ANONYMIZED", "Regex action should be preserved"

    # Verify original response is unchanged
    original_regexes = response_with_regex["assessments"][0][
        "sensitiveInformationPolicy"
    ]["regexes"]
    assert original_regexes[0]["match"] == "123-45-6789", "Original should be unchanged"
    assert (
        original_regexes[1]["match"] == "4111-1111-1111-1111"
    ), "Original should be unchanged"

    print("Regex matches redaction test passed")


@pytest.mark.asyncio
async def test__redact_pii_matches_with_custom_words():
    """Test redaction of custom word matches in word policy"""

    response_with_custom_words = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "wordPolicy": {
                    "customWords": [
                        {
                            "match": "confidential_data",
                            "action": "BLOCKED",
                        },
                        {
                            "match": "secret_information",
                            "action": "ANONYMIZED",
                        },
                    ]
                }
            }
        ],
        "outputs": [{"text": "Custom words detected"}],
    }

    # Call the redaction function
    redacted_response = _redact_pii_matches(response_with_custom_words)

    # Verify that custom word matches are redacted
    custom_words = redacted_response["assessments"][0]["wordPolicy"]["customWords"]

    assert (
        custom_words[0]["match"] == "[REDACTED]"
    ), "First custom word match should be redacted"
    assert (
        custom_words[1]["match"] == "[REDACTED]"
    ), "Second custom word match should be redacted"

    # Verify other fields are preserved
    assert (
        custom_words[0]["action"] == "BLOCKED"
    ), "Custom word action should be preserved"
    assert (
        custom_words[1]["action"] == "ANONYMIZED"
    ), "Custom word action should be preserved"

    # Verify original response is unchanged
    original_custom_words = response_with_custom_words["assessments"][0]["wordPolicy"][
        "customWords"
    ]
    assert (
        original_custom_words[0]["match"] == "confidential_data"
    ), "Original should be unchanged"
    assert (
        original_custom_words[1]["match"] == "secret_information"
    ), "Original should be unchanged"

    print("Custom words redaction test passed")


@pytest.mark.asyncio
async def test__redact_pii_matches_with_managed_words():
    """Test redaction of managed word matches in word policy"""

    response_with_managed_words = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "wordPolicy": {
                    "managedWordLists": [
                        {
                            "match": "inappropriate_word",
                            "action": "BLOCKED",
                            "type": "PROFANITY",
                        },
                        {
                            "match": "offensive_term",
                            "action": "ANONYMIZED",
                            "type": "HATE_SPEECH",
                        },
                    ]
                }
            }
        ],
        "outputs": [{"text": "Managed words detected"}],
    }

    # Call the redaction function
    redacted_response = _redact_pii_matches(response_with_managed_words)

    # Verify that managed word matches are redacted
    managed_words = redacted_response["assessments"][0]["wordPolicy"][
        "managedWordLists"
    ]

    assert (
        managed_words[0]["match"] == "[REDACTED]"
    ), "First managed word match should be redacted"
    assert (
        managed_words[1]["match"] == "[REDACTED]"
    ), "Second managed word match should be redacted"

    # Verify other fields are preserved
    assert (
        managed_words[0]["action"] == "BLOCKED"
    ), "Managed word action should be preserved"
    assert (
        managed_words[0]["type"] == "PROFANITY"
    ), "Managed word type should be preserved"
    assert (
        managed_words[1]["action"] == "ANONYMIZED"
    ), "Managed word action should be preserved"
    assert (
        managed_words[1]["type"] == "HATE_SPEECH"
    ), "Managed word type should be preserved"

    # Verify original response is unchanged
    original_managed_words = response_with_managed_words["assessments"][0][
        "wordPolicy"
    ]["managedWordLists"]
    assert (
        original_managed_words[0]["match"] == "inappropriate_word"
    ), "Original should be unchanged"
    assert (
        original_managed_words[1]["match"] == "offensive_term"
    ), "Original should be unchanged"

    print("Managed words redaction test passed")


@pytest.mark.asyncio
async def test__redact_pii_matches_comprehensive_coverage():
    """Test redaction across all supported policy types in a single response"""

    comprehensive_response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "EMAIL",
                            "match": "user@example.com",
                            "action": "ANONYMIZED",
                        }
                    ],
                    "regexes": [
                        {
                            "name": "PHONE_PATTERN",
                            "match": "555-123-4567",
                            "action": "BLOCKED",
                        }
                    ],
                },
                "wordPolicy": {
                    "customWords": [
                        {
                            "match": "confidential",
                            "action": "BLOCKED",
                        }
                    ],
                    "managedWordLists": [
                        {
                            "match": "inappropriate",
                            "action": "ANONYMIZED",
                            "type": "PROFANITY",
                        }
                    ],
                },
            }
        ],
        "outputs": [{"text": "Multiple policy violations detected"}],
    }

    # Call the redaction function
    redacted_response = _redact_pii_matches(comprehensive_response)

    # Verify all match fields are redacted
    assessment = redacted_response["assessments"][0]

    # PII entities
    pii_entities = assessment["sensitiveInformationPolicy"]["piiEntities"]
    assert (
        pii_entities[0]["match"] == "[REDACTED]"
    ), "PII entity match should be redacted"

    # Regex matches
    regexes = assessment["sensitiveInformationPolicy"]["regexes"]
    assert regexes[0]["match"] == "[REDACTED]", "Regex match should be redacted"

    # Custom words
    custom_words = assessment["wordPolicy"]["customWords"]
    assert (
        custom_words[0]["match"] == "[REDACTED]"
    ), "Custom word match should be redacted"

    # Managed words
    managed_words = assessment["wordPolicy"]["managedWordLists"]
    assert (
        managed_words[0]["match"] == "[REDACTED]"
    ), "Managed word match should be redacted"

    # Verify all other fields are preserved
    assert pii_entities[0]["type"] == "EMAIL"
    assert regexes[0]["name"] == "PHONE_PATTERN"
    assert managed_words[0]["type"] == "PROFANITY"

    # Verify original response is unchanged
    original_assessment = comprehensive_response["assessments"][0]
    assert (
        original_assessment["sensitiveInformationPolicy"]["piiEntities"][0]["match"]
        == "user@example.com"
    )
    assert (
        original_assessment["sensitiveInformationPolicy"]["regexes"][0]["match"]
        == "555-123-4567"
    )
    assert (
        original_assessment["wordPolicy"]["customWords"][0]["match"] == "confidential"
    )
    assert (
        original_assessment["wordPolicy"]["managedWordLists"][0]["match"]
        == "inappropriate"
    )

    print("Comprehensive coverage redaction test passed")


@pytest.mark.asyncio
async def test_bedrock_guardrail_respects_custom_runtime_endpoint(monkeypatch):
    """Test that BedrockGuardrail respects aws_bedrock_runtime_endpoint when set"""

    # Clear any existing environment variable to ensure clean test
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)

    # Create guardrail with custom runtime endpoint
    custom_endpoint = "https://custom-bedrock.example.com"
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        aws_bedrock_runtime_endpoint=custom_endpoint,
    )

    # Mock credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    # Test data
    data = {"source": "INPUT", "content": [{"text": {"text": "test content"}}]}
    optional_params = {}
    aws_region_name = "us-east-1"

    # Mock the _load_credentials method to avoid actual AWS credential loading
    with patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, aws_region_name)
    ):
        # Call _prepare_request which internally calls get_runtime_endpoint
        prepped_request = guardrail._prepare_request(
            credentials=mock_credentials,
            data=data,
            optional_params=optional_params,
            aws_region_name=aws_region_name,
        )

        # Verify that the custom endpoint is used in the URL
        expected_url = f"{custom_endpoint}/guardrail/{guardrail.guardrailIdentifier}/version/{guardrail.guardrailVersion}/apply"
        assert (
            prepped_request.url == expected_url
        ), f"Expected URL to contain custom endpoint. Got: {prepped_request.url}"

        print(f"Custom runtime endpoint test passed. URL: {prepped_request.url}")


@pytest.mark.asyncio
async def test_bedrock_guardrail_respects_env_runtime_endpoint(monkeypatch):
    """Test that BedrockGuardrail respects AWS_BEDROCK_RUNTIME_ENDPOINT environment variable"""

    custom_endpoint = "https://env-bedrock.example.com"

    # Set the environment variable
    monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", custom_endpoint)

    # Create guardrail without explicit aws_bedrock_runtime_endpoint
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )

    # Mock credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    # Test data
    data = {"source": "INPUT", "content": [{"text": {"text": "test content"}}]}
    optional_params = {}
    aws_region_name = "us-east-1"

    # Mock the _load_credentials method
    with patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, aws_region_name)
    ):
        # Call _prepare_request which internally calls get_runtime_endpoint
        prepped_request = guardrail._prepare_request(
            credentials=mock_credentials,
            data=data,
            optional_params=optional_params,
            aws_region_name=aws_region_name,
        )

        # Verify that the custom endpoint from environment is used in the URL
        expected_url = f"{custom_endpoint}/guardrail/{guardrail.guardrailIdentifier}/version/{guardrail.guardrailVersion}/apply"
        assert (
            prepped_request.url == expected_url
        ), f"Expected URL to contain env endpoint. Got: {prepped_request.url}"

        print(f"Environment runtime endpoint test passed. URL: {prepped_request.url}")


@pytest.mark.asyncio
async def test_bedrock_guardrail_uses_default_endpoint_when_no_custom_set(monkeypatch):
    """Test that BedrockGuardrail uses default endpoint when no custom endpoint is set"""

    # Ensure no environment variable is set
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)

    # Create guardrail without any custom endpoint
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )

    # Mock credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    # Test data
    data = {"source": "INPUT", "content": [{"text": {"text": "test content"}}]}
    optional_params = {}
    aws_region_name = "us-west-2"

    # Mock the _load_credentials method
    with patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, aws_region_name)
    ):
        # Call _prepare_request which internally calls get_runtime_endpoint
        prepped_request = guardrail._prepare_request(
            credentials=mock_credentials,
            data=data,
            optional_params=optional_params,
            aws_region_name=aws_region_name,
        )

        # Verify that the default endpoint is used
        expected_url = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com/guardrail/{guardrail.guardrailIdentifier}/version/{guardrail.guardrailVersion}/apply"
        assert (
            prepped_request.url == expected_url
        ), f"Expected default URL. Got: {prepped_request.url}"

        print(f"Default endpoint test passed. URL: {prepped_request.url}")


@pytest.mark.asyncio
async def test_bedrock_guardrail_parameter_takes_precedence_over_env(monkeypatch):
    """Test that aws_bedrock_runtime_endpoint parameter takes precedence over environment variable

    This test verifies the corrected behavior where the parameter should take precedence
    over the environment variable, consistent with the endpoint_url logic.
    """

    param_endpoint = "https://param-bedrock.example.com"
    env_endpoint = "https://env-bedrock.example.com"

    # Set environment variable
    monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", env_endpoint)

    # Create guardrail with explicit aws_bedrock_runtime_endpoint
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        aws_bedrock_runtime_endpoint=param_endpoint,
    )

    # Mock credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    # Test data
    data = {"source": "INPUT", "content": [{"text": {"text": "test content"}}]}
    optional_params = {}
    aws_region_name = "us-east-1"

    # Mock the _load_credentials method
    with patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, aws_region_name)
    ):
        # Call _prepare_request which internally calls get_runtime_endpoint
        prepped_request = guardrail._prepare_request(
            credentials=mock_credentials,
            data=data,
            optional_params=optional_params,
            aws_region_name=aws_region_name,
        )

        # Verify that the parameter takes precedence over environment variable
        expected_url = f"{param_endpoint}/guardrail/{guardrail.guardrailIdentifier}/version/{guardrail.guardrailVersion}/apply"
        assert (
            prepped_request.url == expected_url
        ), f"Expected parameter endpoint to take precedence. Got: {prepped_request.url}"

        print(f"Parameter precedence test passed. URL: {prepped_request.url}")


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_with_only_tool_calls_response():
    """Test that apply_guardrail handles response with tool_calls (no text content) without calling Bedrock API"""
    # Create a BedrockGuardrail instance
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )

    # Mock the make_bedrock_api_request method
    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api_request:
        # Test the apply_guardrail method with tool_calls in response
        inputs = {
            "texts": [],
            "tool_calls": [
                {
                    "id": "call_eFSCWFsyL7MclHYnzKrcQnMK",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"São Paulo"}',
                    },
                }
            ],
        }

        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="response",
            logging_obj=None,
        )

        # Verify the result - should succeed without errors
        assert guardrailed_inputs is not None
        assert "tool_calls" in guardrailed_inputs
        assert len(guardrailed_inputs["tool_calls"]) == 1
        assert (
            guardrailed_inputs["tool_calls"][0]["id"] == "call_eFSCWFsyL7MclHYnzKrcQnMK"
        )
        assert guardrailed_inputs["tool_calls"][0]["function"]["name"] == "get_weather"
        assert (
            guardrailed_inputs["tool_calls"][0]["function"]["arguments"]
            == '{"location":"São Paulo"}'
        )
        # Verify that the Bedrock API was NOT called since there's no text to process
        mock_api_request.assert_not_called()
        print("✅ apply_guardrail with tool_calls test passed - no API call made")


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_response_uses_OUTPUT_source():
    """input_type='response' must call Bedrock with source=OUTPUT and assistant content.

    Regression: apply_guardrail used to always use source=INPUT. Output-only Bedrock
    policies (e.g. PII on model output) then returned action=NONE for non-streaming
    completions that go through unified_guardrail -> process_output_response.
    """
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )
    bedrock_none = {"action": "NONE", "output": [], "outputs": []}

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        mock_api.return_value = bedrock_none

        await guardrail.apply_guardrail(
            inputs={"texts": ["first line", "second line"]},
            request_data={"model": "gpt-4o"},
            input_type="response",
        )

        mock_api.assert_called_once()
        kwargs = mock_api.call_args.kwargs
        assert kwargs["source"] == "OUTPUT"
        assert kwargs["request_data"] == {"model": "gpt-4o"}
        synthetic = kwargs["response"]
        assert isinstance(synthetic, ModelResponse)
        assert len(synthetic.choices) == 2
        assert synthetic.choices[0].message.content == "first line"
        assert synthetic.choices[0].message.role == "assistant"
        assert synthetic.choices[1].message.content == "second line"
        assert synthetic.choices[1].message.role == "assistant"


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_request_uses_INPUT_source():
    """input_type='request' must call Bedrock with source=INPUT and user messages."""
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )
    bedrock_none = {"action": "NONE", "output": [], "outputs": []}

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        mock_api.return_value = bedrock_none

        await guardrail.apply_guardrail(
            inputs={"texts": ["user prompt"]},
            request_data={},
            input_type="request",
        )

        mock_api.assert_called_once()
        kwargs = mock_api.call_args.kwargs
        assert kwargs["source"] == "INPUT"
        assert kwargs["messages"] is not None
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["messages"][0]["content"] == "user prompt"
        assert kwargs.get("response") is None


@pytest.mark.asyncio
async def test_bedrock_guardrail_blocked_content_with_masking_enabled():
    """Test that BLOCKED content raises exception even when masking is enabled

    This test verifies the bug fix where previously mask_request_content=True or
    mask_response_content=True would bypass all BLOCKED content checks. Now it
    properly distinguishes between BLOCKED (raise exception) and ANONYMIZED (apply masking).
    """

    # Create guardrail with masking enabled
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        mask_request_content=True,  # Masking enabled
        mask_response_content=True,  # Masking enabled
    )

    # Mock Bedrock response with BLOCKED content (hate speech)
    blocked_response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "contentPolicy": {
                    "filters": [
                        {
                            "type": "HATE",
                            "confidence": "HIGH",
                            "action": "BLOCKED",  # Should raise exception
                        }
                    ]
                },
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "NAME",
                            "match": "John Doe",
                            "action": "ANONYMIZED",  # Should be masked
                        }
                    ]
                },
            }
        ],
        "outputs": [{"text": "Content blocked due to policy violation"}],
    }

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = blocked_response

    # Mock credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Test message with PII and hate speech"},
        ],
    }

    # Mock AWS-related methods
    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = mock_bedrock_response

        # Should raise HTTPException for BLOCKED content
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=request_data.get("messages"),
                request_data=request_data,
            )

        # Verify exception details
        assert exc_info.value.status_code == 400
        assert "Violated guardrail policy" in str(exc_info.value.detail)

        print("✅ BLOCKED content with masking enabled raises exception correctly")


# ──────────────────────────────────────────────────────────────────────────────
# Null-safety tests for Bedrock guardrail responses
#
# The Bedrock ApplyGuardrail API can return explicit null/None for list fields
# such as "regexes", "piiEntities", "topics", "filters", "customWords", and
# "managedWordLists" when a particular policy category is present in the
# assessment but has no matches.
#
# Python's dict.get("key", []) returns None (NOT []) when the key exists with
# a None value.  The `or []` fallback ensures we always iterate over a list.
#
# Without the fix, iterating over None raises:
#   TypeError: 'NoneType' object is not iterable
# which surfaces to callers as:
#   openai.InternalServerError: Error code: 500
#   {'error': {'message': "Bedrock guardrail failed: 'NoneType' object is not iterable", ...}}
# ──────────────────────────────────────────────────────────────────────────────


class TestRedactPiiMatchesNullSafety:
    """Tests for _redact_pii_matches handling of null/None list fields from Bedrock API."""

    @pytest.mark.asyncio
    async def test_should_handle_null_regexes_in_sensitive_info_policy(self):
        """Bedrock can return regexes: null while piiEntities has data.

        Real-world scenario: guardrail detects PII (e.g. EMAIL) but has no
        custom regex patterns configured, so the API returns regexes: null.
        """
        response = {
            "action": "NONE",
            "actionReason": "No action.",
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [
                            {
                                "action": "NONE",
                                "detected": True,
                                "match": "joebloggs@gmail.com",
                                "type": "EMAIL",
                            }
                        ],
                        "regexes": None,  # Explicit null from Bedrock API
                    },
                }
            ],
        }

        # Should not raise TypeError: 'NoneType' object is not iterable
        redacted = _redact_pii_matches(response)

        # PII match should be redacted
        pii = redacted["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]
        assert pii[0]["match"] == "[REDACTED]"
        assert pii[0]["type"] == "EMAIL"

    @pytest.mark.asyncio
    async def test_should_handle_null_pii_entities_in_sensitive_info_policy(self):
        """Bedrock can return piiEntities: null while regexes has data."""
        response = {
            "action": "NONE",
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": None,  # null from Bedrock API
                        "regexes": [
                            {
                                "name": "CUSTOM_PATTERN",
                                "match": "secret-abc-123",
                                "action": "BLOCKED",
                            }
                        ],
                    },
                }
            ],
        }

        redacted = _redact_pii_matches(response)

        regexes = redacted["assessments"][0]["sensitiveInformationPolicy"]["regexes"]
        assert regexes[0]["match"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_should_handle_null_custom_words_and_managed_words(self):
        """Bedrock can return null for customWords and managedWordLists in wordPolicy."""
        response = {
            "action": "NONE",
            "assessments": [
                {
                    "wordPolicy": {
                        "customWords": None,  # null from Bedrock API
                        "managedWordLists": None,  # null from Bedrock API
                    },
                }
            ],
        }

        # Should not raise TypeError
        redacted = _redact_pii_matches(response)

        # Values should remain None (no crash)
        assert redacted["assessments"][0]["wordPolicy"]["customWords"] is None
        assert redacted["assessments"][0]["wordPolicy"]["managedWordLists"] is None

    @pytest.mark.asyncio
    async def test_should_handle_null_assessments_list(self):
        """Bedrock can return assessments: null."""
        response = {
            "action": "NONE",
            "assessments": None,  # null from Bedrock API
        }

        # Should not raise TypeError
        redacted = _redact_pii_matches(response)
        assert redacted["assessments"] is None

    @pytest.mark.asyncio
    async def test_should_handle_all_null_policy_sub_lists_together(self):
        """All sub-list fields are null at the same time — worst-case scenario."""
        response = {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": None,
                        "regexes": None,
                    },
                    "wordPolicy": {
                        "customWords": None,
                        "managedWordLists": None,
                    },
                    "topicPolicy": None,
                    "contentPolicy": None,
                    "contextualGroundingPolicy": None,
                }
            ],
        }

        # Should not raise any exception
        redacted = _redact_pii_matches(response)
        assert redacted is not None


class TestShouldRaiseGuardrailBlockedExceptionNullSafety:
    """Tests for _should_raise_guardrail_blocked_exception handling of null list fields."""

    def _create_guardrail(self) -> BedrockGuardrail:
        return BedrockGuardrail(
            guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
        )

    @pytest.mark.asyncio
    async def test_should_handle_all_null_policy_sub_lists(self):
        """All policy sub-lists are null — should not crash, should return False."""
        guardrail = self._create_guardrail()

        response = {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "topicPolicy": {
                        "topics": None,  # null from Bedrock API
                    },
                    "contentPolicy": {
                        "filters": None,  # null
                    },
                    "wordPolicy": {
                        "customWords": None,  # null
                        "managedWordLists": None,  # null
                    },
                    "sensitiveInformationPolicy": {
                        "piiEntities": None,  # null
                        "regexes": None,  # null
                    },
                    "contextualGroundingPolicy": {
                        "filters": None,  # null
                    },
                }
            ],
        }

        # No BLOCKED actions found (all lists null) → should return False
        result = guardrail._should_raise_guardrail_blocked_exception(response)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_detect_blocked_despite_other_null_lists(self):
        """A mix of null lists and a real BLOCKED action — should still detect it."""
        guardrail = self._create_guardrail()

        response = {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "topicPolicy": {
                        "topics": None,  # null — should not crash
                    },
                    "contentPolicy": {
                        "filters": [
                            {
                                "type": "HATE",
                                "confidence": "HIGH",
                                "action": "BLOCKED",
                            }
                        ],
                    },
                    "wordPolicy": {
                        "customWords": None,  # null
                        "managedWordLists": None,  # null
                    },
                    "sensitiveInformationPolicy": {
                        "piiEntities": None,  # null
                        "regexes": None,  # null
                    },
                    "contextualGroundingPolicy": None,  # entire policy is null
                }
            ],
        }

        # Should return True because contentPolicy has a BLOCKED filter
        result = guardrail._should_raise_guardrail_blocked_exception(response)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_handle_null_assessments_list(self):
        """assessments itself is null — should return False."""
        guardrail = self._create_guardrail()

        response = {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": None,  # null from Bedrock API
        }

        result = guardrail._should_raise_guardrail_blocked_exception(response)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_handle_null_topics_with_blocked_word_policy(self):
        """topics is null but wordPolicy has a BLOCKED customWord."""
        guardrail = self._create_guardrail()

        response = {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "topicPolicy": {
                        "topics": None,
                    },
                    "wordPolicy": {
                        "customWords": [{"match": "badword", "action": "BLOCKED"}],
                        "managedWordLists": None,
                    },
                }
            ],
        }

        result = guardrail._should_raise_guardrail_blocked_exception(response)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_handle_null_pii_with_blocked_regex(self):
        """piiEntities is null but regexes has a BLOCKED match."""
        guardrail = self._create_guardrail()

        response = {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": None,
                        "regexes": [
                            {"name": "SSN", "match": "123-45-6789", "action": "BLOCKED"}
                        ],
                    },
                }
            ],
        }

        result = guardrail._should_raise_guardrail_blocked_exception(response)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_handle_null_grounding_filters(self):
        """contextualGroundingPolicy.filters is null — should not crash."""
        guardrail = self._create_guardrail()

        response = {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "contextualGroundingPolicy": {
                        "filters": None,
                    },
                }
            ],
        }

        result = guardrail._should_raise_guardrail_blocked_exception(response)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_not_crash_when_action_is_not_intervened(self):
        """If action != GUARDRAIL_INTERVENED, null lists should never be reached."""
        guardrail = self._create_guardrail()

        response = {
            "action": "NONE",
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": None,
                        "regexes": None,
                    },
                }
            ],
        }

        result = guardrail._should_raise_guardrail_blocked_exception(response)
        assert result is False


class TestApplyGuardrailNullSafety:
    """Tests for apply_guardrail handling of null/None texts input."""

    @pytest.mark.asyncio
    async def test_should_handle_none_texts_in_inputs(self):
        """inputs[\"texts\"] is explicitly None — should not crash."""
        guardrail = BedrockGuardrail(
            guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
        )

        inputs = {"texts": None}  # Explicit None

        mock_credentials = MagicMock()

        with (
            patch.object(
                guardrail.async_handler, "post", new_callable=AsyncMock
            ) as mock_post,
            patch.object(
                guardrail,
                "_load_credentials",
                return_value=(mock_credentials, "us-east-1"),
            ),
            patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        ):
            # With empty texts (from None → []), no Bedrock API call should be made
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

            # Should return empty texts without crashing
            assert result.get("texts") == []
            # No Bedrock API call should be made for empty input
            mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_handle_missing_texts_key(self):
        """inputs has no \"texts\" key at all — should not crash."""
        guardrail = BedrockGuardrail(
            guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
        )

        inputs = {}  # No "texts" key

        mock_credentials = MagicMock()

        with (
            patch.object(
                guardrail.async_handler, "post", new_callable=AsyncMock
            ) as mock_post,
            patch.object(
                guardrail,
                "_load_credentials",
                return_value=(mock_credentials, "us-east-1"),
            ),
            patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        ):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

            assert result.get("texts") == []
            mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_bedrock_guardrail_blocked_vs_anonymized_actions():
    """Test that BLOCKED actions raise exceptions but ANONYMIZED actions do not"""
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )

    # Test 1: ANONYMIZED action should NOT raise exception
    anonymized_response = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Hello, my phone number is {PHONE}"}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "PHONE",
                            "match": "+1 412 555 1212",
                            "action": "ANONYMIZED",
                        }
                    ]
                }
            }
        ],
    }

    should_raise = guardrail._should_raise_guardrail_blocked_exception(
        anonymized_response
    )
    assert should_raise is False, "ANONYMIZED actions should not raise exceptions"

    # Test 2: BLOCKED action should raise exception
    blocked_response = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "I can't provide that information."}],
        "assessments": [
            {
                "topicPolicy": {
                    "topics": [
                        {"name": "Sensitive Topic", "type": "DENY", "action": "BLOCKED"}
                    ]
                }
            }
        ],
    }

    should_raise = guardrail._should_raise_guardrail_blocked_exception(blocked_response)
    assert should_raise is True, "BLOCKED actions should raise exceptions"

    # Test 3: Mixed actions - should raise if ANY action is BLOCKED
    mixed_response = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "I can't provide that information."}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "PHONE",
                            "match": "+1 412 555 1212",
                            "action": "ANONYMIZED",
                        }
                    ]
                },
                "topicPolicy": {
                    "topics": [
                        {"name": "Blocked Topic", "type": "DENY", "action": "BLOCKED"}
                    ]
                },
            }
        ],
    }

    should_raise = guardrail._should_raise_guardrail_blocked_exception(mixed_response)
    assert (
        should_raise is True
    ), "Mixed actions with any BLOCKED should raise exceptions"

    # Test 4: NONE action should not raise exception
    none_response = {
        "action": "NONE",
        "outputs": [],
        "assessments": [],
    }

    should_raise = guardrail._should_raise_guardrail_blocked_exception(none_response)
    assert should_raise is False, "NONE action should not raise exceptions"

    print("\u2705 BLOCKED vs ANONYMIZED actions test passed")


# ---------------------------------------------------------------------------
# Spend logs: guardrail_mode (pre/during/post) vs Bedrock INPUT/OUTPUT
# ---------------------------------------------------------------------------


def test_bedrock_guardrail_uses_native_during_call_hook():
    """during_call must use async_moderation_hook, not unified apply_guardrail(input=request)."""
    assert BedrockGuardrail.use_native_during_call_hook is True


@pytest.mark.asyncio
async def test_make_bedrock_api_request_logging_event_type_for_spend_logs():
    """
    Spend/UI use event_type from the proxy hook, not Bedrock's INPUT/OUTPUT alone.
    When logging_event_type is set, it must be forwarded to standard guardrail logging.
    When omitted, INPUT maps to pre_call (legacy).
    """
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "NONE",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {"type": "NAME", "match": "GG", "action": "BLOCKED"}
                    ]
                }
            }
        ],
    }

    request_data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch.object(
            guardrail,
            "add_standard_logging_guardrail_information_to_request_data",
        ) as mock_log,
    ):
        mock_post.return_value = mock_bedrock_response

        await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=request_data["messages"],
            request_data=request_data,
            logging_event_type=GuardrailEventHooks.during_call,
        )
        assert (
            mock_log.call_args.kwargs["event_type"] == GuardrailEventHooks.during_call
        )
        # Raw Bedrock JSON is forwarded; redaction runs once in
        # CustomGuardrail.add_standard_logging_guardrail_information_to_request_data.
        assert (
            mock_log.call_args.kwargs["guardrail_json_response"]["assessments"][0][
                "sensitiveInformationPolicy"
            ]["piiEntities"][0]["match"]
            == "GG"
        )

        mock_log.reset_mock()

        await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=request_data["messages"],
            request_data=request_data,
        )
        assert mock_log.call_args.kwargs["event_type"] == GuardrailEventHooks.pre_call


@pytest.mark.asyncio
async def test_make_bedrock_api_request_filters_dynamic_evaluation_overrides():
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {"action": "NONE", "assessments": []}

    prepared_request = MagicMock()
    prepared_request.url = "https://bedrock.test/apply"
    prepared_request.body = b"{}"
    prepared_request.headers = {}

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(
            guardrail, "_prepare_request", return_value=prepared_request
        ) as mock_prepare_request,
        patch.object(
            guardrail,
            "get_guardrail_dynamic_request_body_params",
            return_value={
                "content": [{"text": {"text": "benign replacement"}}],
                "source": "OUTPUT",
                "outputScope": "FULL",
            },
        ),
    ):
        mock_post.return_value = mock_bedrock_response

        await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=[{"role": "user", "content": "actual prompt"}],
            request_data={"model": "gpt-4o"},
        )

    prepared_data = mock_prepare_request.call_args.kwargs["data"]
    assert prepared_data["source"] == "INPUT"
    assert "actual prompt" in json.dumps(prepared_data["content"])
    assert "benign replacement" not in json.dumps(prepared_data["content"])
    assert prepared_data["outputScope"] == "FULL"


@pytest.mark.asyncio
async def test_during_call_hook_invokes_bedrock_async_moderation_hook():
    """
    Bedrock sets use_native_during_call_hook so ProxyLogging runs the real
    async_moderation_hook (unified apply_guardrail would log INPUT as pre_call).
    """
    cache = DualCache()
    proxy_logging = ProxyLogging(user_api_key_cache=cache)

    guardrail = BedrockGuardrail(
        guardrail_name="bedrock-during-test",
        guardrailIdentifier="gid",
        guardrailVersion="1",
        event_hook=GuardrailEventHooks.during_call,
        default_on=True,
    )
    mock_mod = AsyncMock(return_value=None)
    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []
    try:
        litellm.callbacks = [guardrail]
        with patch.object(guardrail, "async_moderation_hook", new=mock_mod):
            await proxy_logging.during_call_hook(
                data={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "test"}],
                },
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="test_key", user_id="test_user"
                ),
                call_type="completion",
            )
    finally:
        litellm.callbacks = original_callbacks

    mock_mod.assert_awaited_once()


# ---------------------------------------------------------------------------
# L3: _extract_blocked_assessments + _get_http_exception_for_blocked_guardrail
# Regression coverage for case 2026-04-10-internal-bedrock-guardrail-streaming-error.
# ---------------------------------------------------------------------------


def _make_guardrail() -> BedrockGuardrail:
    return BedrockGuardrail(
        guardrail_name="bedrock-pii-guard",
        guardrailIdentifier="amgllac6xf3r",
        guardrailVersion="1",
    )


def test_extract_blocked_assessments_pii_entity():
    """L3: PII entity match (BLOCKED) is surfaced with category, type, and match."""
    g = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {"type": "NAME", "action": "BLOCKED", "match": "Jack"},
                        {"type": "EMAIL", "action": "ANONYMIZED", "match": "x@y.z"},
                    ]
                }
            }
        ],
    }
    blocked = g._extract_blocked_assessments(response)
    assert len(blocked) == 1
    assert blocked[0]["policy"] == "sensitiveInformationPolicy"
    matches = blocked[0]["matches"]
    assert len(matches) == 1  # only the BLOCKED one is surfaced
    assert matches[0]["category"] == "piiEntities"
    assert matches[0]["type"] == "NAME"
    assert matches[0]["match"] == "Jack"


def test_extract_blocked_assessments_multiple_policies():
    """L3: multiple policies fired in one assessment must all be reported."""
    g = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "topicPolicy": {
                    "topics": [
                        {"name": "Investment", "type": "DENY", "action": "BLOCKED"}
                    ]
                },
                "contentPolicy": {
                    "filters": [
                        {
                            "type": "VIOLENCE",
                            "confidence": "HIGH",
                            "filterStrength": "HIGH",
                            "action": "BLOCKED",
                        }
                    ]
                },
                "wordPolicy": {
                    "customWords": [{"match": "forbidden", "action": "BLOCKED"}]
                },
            }
        ],
    }
    blocked = g._extract_blocked_assessments(response)
    policies = {entry["policy"] for entry in blocked}
    assert policies == {"topicPolicy", "contentPolicy", "wordPolicy"}


def test_extract_blocked_assessments_only_anonymized_returns_empty():
    """L3: if all matches are ANONYMIZED (not BLOCKED), the list is empty."""
    g = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {"type": "NAME", "action": "ANONYMIZED", "match": "Jack"}
                    ]
                }
            }
        ],
    }
    assert g._extract_blocked_assessments(response) == []


def test_extract_blocked_assessments_no_assessments():
    """L3: response with no assessments returns an empty list, not an error."""
    g = _make_guardrail()
    assert g._extract_blocked_assessments({"action": "NONE"}) == []
    assert g._extract_blocked_assessments({"assessments": None}) == []


def test_get_http_exception_includes_assessments_and_identifier():
    """L3: end-to-end — _get_http_exception_for_blocked_guardrail emits the new fields."""
    g = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Sorry, the model cannot answer this question."}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {"type": "NAME", "action": "BLOCKED", "match": "Jack"}
                    ]
                }
            }
        ],
    }
    exc = g._get_http_exception_for_blocked_guardrail(response)
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400
    assert exc.detail["error"] == "Violated guardrail policy"
    assert (
        exc.detail["bedrock_guardrail_response"]
        == "Sorry, the model cannot answer this question."
    )
    assert exc.detail["guardrailIdentifier"] == "amgllac6xf3r"
    assert exc.detail["guardrailVersion"] == "1"
    assert exc.detail["assessments"][0]["policy"] == "sensitiveInformationPolicy"
    assert exc.detail["assessments"][0]["matches"][0]["type"] == "NAME"
    assert exc.detail["assessments"][0]["matches"][0]["match"] == "[REDACTED]"


def test_extract_violation_category_names_mixed_policies():
    """Topic names, content-filter types, PII types, and managed-word types
    flatten into a single category-name list — using only the operator-
    defined `name`/`type` labels."""
    g = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "topicPolicy": {
                    "topics": [
                        {"name": "Fiduciary Advice", "action": "BLOCKED"},
                        {"name": "Tax Advice", "action": "BLOCKED"},
                    ]
                },
                "contentPolicy": {
                    "filters": [{"type": "VIOLENCE", "action": "BLOCKED"}]
                },
                "wordPolicy": {
                    "managedWordLists": [{"type": "PROFANITY", "action": "BLOCKED"}],
                },
                "sensitiveInformationPolicy": {
                    "piiEntities": [{"type": "EMAIL", "action": "BLOCKED"}]
                },
            }
        ],
    }
    names = g._extract_violation_category_names(response)
    assert "Fiduciary Advice" in names
    assert "Tax Advice" in names
    assert "VIOLENCE" in names
    assert "PROFANITY" in names
    assert "EMAIL" in names


def test_extract_violation_category_names_does_not_leak_user_input():
    """SECURITY: customWords.match is the raw user-submitted word that
    triggered the rule, and an unnamed regex match is the actual sensitive
    value (e.g. a credit-card number). Neither must appear in
    violation_categories — otherwise the content the guardrail blocked
    leaks straight into telemetry backends."""
    g = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "wordPolicy": {
                    "customWords": [
                        {"match": "secret-codeword-abc-123", "action": "BLOCKED"}
                    ],
                },
                "sensitiveInformationPolicy": {
                    "regexes": [{"match": "4111-1111-1111-1111", "action": "BLOCKED"}]
                },
            }
        ],
    }
    names = g._extract_violation_category_names(response)
    assert "secret-codeword-abc-123" not in names
    assert "4111-1111-1111-1111" not in names
    assert names == []


def test_extract_violation_category_names_named_regex_uses_name():
    """A regex with a `name` field surfaces that operator-defined label
    (safe to log), not the matched value."""
    g = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "regexes": [
                        {
                            "name": "credit-card-pattern",
                            "match": "4111-1111-1111-1111",
                            "action": "BLOCKED",
                        }
                    ]
                }
            }
        ],
    }
    names = g._extract_violation_category_names(response)
    assert names == ["credit-card-pattern"]


def test_extract_violation_category_names_skips_anonymized():
    """ANONYMIZED entries are not blocks — they must not contribute to the
    violation_categories list."""
    g = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [{"type": "NAME", "action": "ANONYMIZED"}]
                }
            }
        ],
    }
    assert g._extract_violation_category_names(response) == []


def test_extract_violation_category_names_no_assessments():
    """Empty / missing assessments → empty list, not an error."""
    g = _make_guardrail()
    assert g._extract_violation_category_names({"action": "NONE"}) == []
    assert g._extract_violation_category_names({"assessments": None}) == []


@pytest.mark.asyncio
async def test_make_bedrock_api_request_forwards_guardrail_action():
    """Bedrock's top-level ``action`` string must be propagated through
    ``tracing_detail`` so downstream loggers (OTEL, ...) can surface the
    raw provider verdict as a queryable attribute without re-parsing the
    redacted guardrail_response blob."""
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )
    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "topicPolicy": {
                    "topics": [{"name": "Fiduciary Advice", "action": "BLOCKED"}]
                }
            }
        ],
    }

    request_data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch.object(
            guardrail,
            "add_standard_logging_guardrail_information_to_request_data",
        ) as mock_log,
        patch.object(
            guardrail,
            "_get_http_exception_for_blocked_guardrail",
            return_value=Exception("blocked"),
        ),
    ):
        mock_post.return_value = mock_bedrock_response

        with pytest.raises(Exception):
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=request_data["messages"],
                request_data=request_data,
            )

        tracing_detail = mock_log.call_args.kwargs["tracing_detail"]
        assert tracing_detail is not None
        assert tracing_detail["guardrail_action"] == "GUARDRAIL_INTERVENED"


@pytest.mark.asyncio
async def test_make_bedrock_api_request_omits_guardrail_action_when_missing():
    """If the Bedrock response omits ``action`` (older / partial payloads),
    the field must be left off ``tracing_detail`` rather than written as
    ``None`` — downstream code expects strings or absence, not nulls."""
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )
    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {"assessments": []}

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch.object(
            guardrail,
            "add_standard_logging_guardrail_information_to_request_data",
        ) as mock_log,
    ):
        mock_post.return_value = mock_bedrock_response

        await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=[{"role": "user", "content": "hi"}],
            request_data={"model": "gpt-4o", "messages": []},
        )

        tracing_detail = mock_log.call_args.kwargs["tracing_detail"]
        # No violation categories and no action ⇒ tracing_detail stays None
        # (the hook collapses an empty dict before forwarding).
        if tracing_detail is not None:
            assert "guardrail_action" not in tracing_detail


def test_get_http_exception_no_blocked_assessments_omits_field():
    """L3: when no assessments are blocked, the `assessments` key is omitted entirely."""
    g = _make_guardrail()
    response = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "blocked"}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {"type": "NAME", "action": "ANONYMIZED", "match": "Jack"}
                    ]
                }
            }
        ],
    }
    exc = g._get_http_exception_for_blocked_guardrail(response)
    assert isinstance(exc, HTTPException)
    assert "assessments" not in exc.detail
    assert exc.detail["guardrailIdentifier"] == "amgllac6xf3r"


@pytest.mark.asyncio
async def test_streaming_post_call_only_runs_output_scan():
    """
    async_post_call_streaming_iterator_hook must pass request_data into OUTPUT
    make_bedrock_api_request so spend/standard_logging attaches to the real request
    (Greptile: previously OUTPUT used request_data=None / ephemeral {}).

    post_call only validates the response — no INPUT scan should run here.
    """
    request_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"stream_guardrail_logging": True},
    }
    guardrail = BedrockGuardrail(
        guardrail_name="bedrock-stream-reqdata",
        guardrailIdentifier="test-id",
        guardrailVersion="DRAFT",
        event_hook=GuardrailEventHooks.post_call,
        default_on=True,
    )
    mock_chunks = [
        litellm.ModelResponseStream(
            id="tid",
            choices=[
                litellm.types.utils.StreamingChoices(
                    delta=litellm.types.utils.Delta(content="Hi", role="assistant"),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        ),
        litellm.ModelResponseStream(
            id="tid",
            choices=[
                litellm.types.utils.StreamingChoices(
                    delta=litellm.types.utils.Delta(content="!", role="assistant"),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        ),
    ]

    async def mock_stream():
        for c in mock_chunks:
            yield c

    minimal = {"action": "NONE", "assessments": [], "outputs": []}
    with patch.object(
        guardrail, "make_bedrock_api_request", AsyncMock(return_value=minimal)
    ) as mock_make:
        out = []
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=mock_stream(),
            request_data=request_data,
        ):
            out.append(chunk)

    assert len(out) >= 1
    output_calls = [
        c for c in mock_make.call_args_list if c.kwargs.get("source") == "OUTPUT"
    ]
    assert len(output_calls) == 1
    assert output_calls[0].kwargs.get("request_data") is request_data
    assert (
        output_calls[0].kwargs.get("logging_event_type")
        == GuardrailEventHooks.post_call
    )
    input_calls = [
        c for c in mock_make.call_args_list if c.kwargs.get("source") == "INPUT"
    ]
    assert len(input_calls) == 0


@pytest.mark.asyncio
async def test_streaming_post_call_output_only_path_passes_request_data_to_make_bedrock():
    """When INPUT validation is skipped (pre/during already ran), OUTPUT still gets request_data."""
    request_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    }
    guardrail = BedrockGuardrail(
        guardrail_name="bedrock-stream-out-only",
        guardrailIdentifier="test-id",
        guardrailVersion="DRAFT",
        event_hook=GuardrailEventHooks.during_call,
        default_on=True,
    )
    mock_chunks = [
        litellm.ModelResponseStream(
            id="tid",
            choices=[
                litellm.types.utils.StreamingChoices(
                    delta=litellm.types.utils.Delta(content="x", role="assistant"),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        ),
    ]

    async def mock_stream():
        for c in mock_chunks:
            yield c

    minimal = {"action": "NONE", "assessments": [], "outputs": []}
    with patch.object(
        guardrail, "make_bedrock_api_request", AsyncMock(return_value=minimal)
    ) as mock_make:
        async for _ in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=mock_stream(),
            request_data=request_data,
        ):
            pass

    assert mock_make.call_count == 1
    c = mock_make.call_args
    assert c.kwargs.get("source") == "OUTPUT"
    assert c.kwargs.get("request_data") is request_data


# ---------------------------------------------------------------------------
# Regression: post_call only validates OUTPUT.
# Input scanning belongs to pre_call / during_call. Running an extra INPUT
# scan here used to produce a duplicate "post-call" trace entry and made no
# semantic sense for a "post-call" event.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_success_hook_only_runs_output_scan():
    """
    With only `post_call` configured, async_post_call_success_hook must call
    make_bedrock_api_request exactly once with source="OUTPUT". An INPUT call
    here would produce a duplicate post-call log entry.
    """
    guardrail = BedrockGuardrail(
        guardrail_name="bedrock-post-pii",
        guardrailIdentifier="gid",
        guardrailVersion="1",
        event_hook=GuardrailEventHooks.post_call,
        default_on=True,
    )

    request_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {},
    }
    response = ModelResponse(
        choices=[
            litellm.Choices(
                message=litellm.Message(role="assistant", content="hello"),
                index=0,
                finish_reason="stop",
            )
        ],
        model="gpt-4o-mini",
    )

    minimal = {"action": "NONE", "assessments": [], "outputs": []}
    with patch.object(
        guardrail, "make_bedrock_api_request", AsyncMock(return_value=minimal)
    ) as mock_make:
        await guardrail.async_post_call_success_hook(
            data=request_data,
            user_api_key_dict=UserAPIKeyAuth(),
            response=response,
        )

    sources = [c.kwargs.get("source") for c in mock_make.call_args_list]
    assert sources == ["OUTPUT"]
    assert (
        mock_make.call_args.kwargs.get("logging_event_type")
        == GuardrailEventHooks.post_call
    )


# ---------------------------------------------------------------------------
# Contextual grounding: request-side qualifiers
# ---------------------------------------------------------------------------
#
# Bedrock contextual grounding tags each ApplyGuardrail content block with a
# `qualifiers` array (grounding_source / query / guard_content). A caller marks
# message content blocks `{"type": "grounding_source", ...}` / `{"type": "query", ...}`;
# at post_call the hook assembles one source="OUTPUT" call carrying the source +
# query + the model response (as guard_content). A request without these tags
# produces the plain-text payload with no qualifiers.

_GROUNDING_SOURCE_TEXT = "Tokyo is the capital of Japan."
_GROUNDING_QUERY_TEXT = "What is the capital of Japan?"
_GROUNDING_RESPONSE_TEXT = "The capital of Japan is Tokyo."


def _grounding_guardrail() -> BedrockGuardrail:
    return BedrockGuardrail(
        guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT"
    )


def _grounding_messages() -> list:
    return [
        {
            "role": "system",
            "content": [{"type": "grounding_source", "text": _GROUNDING_SOURCE_TEXT}],
        },
        {
            "role": "user",
            "content": [{"type": "query", "text": _GROUNDING_QUERY_TEXT}],
        },
    ]


def _model_response(content: str) -> ModelResponse:
    from litellm.types.utils import Choices, Message, ModelResponse

    return ModelResponse(
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ]
    )


# Expected OUTPUT content blocks, keyed by their grounding qualifier, so the
# per-test assertions read as the block sequence they expect.
_GROUNDING_SOURCE_BLOCK = {
    "text": {"text": _GROUNDING_SOURCE_TEXT, "qualifiers": ["grounding_source"]}
}
_QUERY_BLOCK = {"text": {"text": _GROUNDING_QUERY_TEXT, "qualifiers": ["query"]}}
_GUARD_BLOCK = {
    "text": {"text": _GROUNDING_RESPONSE_TEXT, "qualifiers": ["guard_content"]}
}


def _input_request(messages: list) -> dict:
    """Arrange a guardrail and act: build the Bedrock INPUT payload."""
    return _grounding_guardrail().convert_to_bedrock_format(
        source="INPUT", messages=messages
    )


def _output_request(messages: list, response=None) -> dict:
    """Arrange a guardrail and act: build the Bedrock OUTPUT payload."""
    return _grounding_guardrail().convert_to_bedrock_format(
        source="OUTPUT", response=response, messages=messages
    )


def test_grounding_input_strips_grounding_and_query_qualifiers():
    """Grounding is OUTPUT-only: tagged source/query reach Bedrock as plain text on an
    INPUT scan, so a tag cannot change how input-safety policies scan content (no bypass).
    """
    expected_request = {
        "source": "INPUT",
        "content": [
            {"text": {"text": _GROUNDING_SOURCE_TEXT}},
            {"text": {"text": _GROUNDING_QUERY_TEXT}},
        ],
    }

    actual_request = _input_request(_grounding_messages())

    assert actual_request == expected_request


def test_grounding_input_leaves_existing_guarded_text_unqualified():
    """An existing guarded_text input block keeps its legacy unqualified payload."""
    expected_request = {"source": "INPUT", "content": [{"text": {"text": "policy"}}]}

    actual_request = _input_request(
        [{"role": "user", "content": [{"type": "guarded_text", "text": "policy"}]}]
    )

    assert actual_request == expected_request


def test_grounding_output_assembles_source_query_and_response():
    """OUTPUT emits grounding_source + query (from the request) then the response as
    guard_content, so Bedrock can grade the response against the source and query."""
    expected_request = {
        "source": "OUTPUT",
        "content": [_GROUNDING_SOURCE_BLOCK, _QUERY_BLOCK, _GUARD_BLOCK],
    }

    actual_request = _output_request(
        _grounding_messages(), _model_response(_GROUNDING_RESPONSE_TEXT)
    )

    assert actual_request == expected_request


def test_grounding_output_keeps_legacy_payload_without_tags():
    """Without grounding tags the OUTPUT payload is the legacy single response block."""
    expected_request = {
        "source": "OUTPUT",
        "content": [{"text": {"text": "Hi there."}}],
    }

    actual_request = _output_request(
        [{"role": "user", "content": "hello"}], _model_response("Hi there.")
    )

    assert actual_request == expected_request


def test_grounding_output_combines_multiple_sources():
    """Every grounding_source block is emitted; Bedrock combines them into one corpus."""
    uk_source_text = "London is the capital of UK."
    uk_source_block = {
        "text": {"text": uk_source_text, "qualifiers": ["grounding_source"]}
    }
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "grounding_source", "text": uk_source_text},
                {"type": "grounding_source", "text": _GROUNDING_SOURCE_TEXT},
            ],
        },
        {"role": "user", "content": [{"type": "query", "text": _GROUNDING_QUERY_TEXT}]},
    ]
    expected_request = {
        "source": "OUTPUT",
        "content": [
            uk_source_block,
            _GROUNDING_SOURCE_BLOCK,
            _QUERY_BLOCK,
            _GUARD_BLOCK,
        ],
    }

    actual_request = _output_request(
        messages, _model_response(_GROUNDING_RESPONSE_TEXT)
    )

    assert actual_request == expected_request


def test_grounding_output_keeps_grounding_for_non_model_response():
    """Harvested grounding blocks survive a non-ModelResponse output instead of being
    silently dropped (regression guard for the unconditional content assignment)."""
    expected_request = {
        "source": "OUTPUT",
        "content": [_GROUNDING_SOURCE_BLOCK, _QUERY_BLOCK],
    }

    actual_request = _output_request(_grounding_messages(), response=None)

    assert actual_request == expected_request


@pytest.mark.parametrize(
    "role, is_trusted",
    [
        ("system", True),
        ("developer", True),
        ("tool", False),
        ("function", False),
        ("user", False),
        ("assistant", False),
    ],
)
def test_grounding_source_trusted_only_from_app_roles(role, is_trusted):
    """grounding_source is honored only from app-authored roles (system/developer). A
    tag on a user, tool, function or assistant message is ignored, so neither a forwarded
    end user nor an externally-influenced tool result can supply fake evidence for the
    grounding check to grade the response against; query is always collected."""
    messages = [
        {
            "role": role,
            "content": [{"type": "grounding_source", "text": _GROUNDING_SOURCE_TEXT}],
        },
        {"role": "user", "content": [{"type": "query", "text": _GROUNDING_QUERY_TEXT}]},
    ]
    expected_content = [_QUERY_BLOCK, _GUARD_BLOCK]
    if is_trusted:
        expected_content = [_GROUNDING_SOURCE_BLOCK, *expected_content]

    actual_request = _output_request(
        messages, _model_response(_GROUNDING_RESPONSE_TEXT)
    )

    assert actual_request == {"source": "OUTPUT", "content": expected_content}


@pytest.mark.asyncio
async def test_grounding_output_blocked_raises_400():
    """A BLOCKED contextualGroundingPolicy filter raises HTTP 400."""
    guardrail = _grounding_guardrail()

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "contextualGroundingPolicy": {
                    "filters": [
                        {
                            "type": "GROUNDING",
                            "threshold": 0.7,
                            "score": 0.1,
                            "action": "BLOCKED",
                        }
                    ]
                }
            }
        ],
        "outputs": [{"text": "Response blocked: not grounded in the provided source."}],
    }

    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = mock_bedrock_response

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="OUTPUT",
                response=_model_response("The capital of Japan is Paris."),
                messages=_grounding_messages(),
                request_data={"messages": _grounding_messages()},
            )

    assert exc_info.value.status_code == 400


###############################################################################
# LIT-4186: disable_exception_on_block regression tests
#
# Before the fix, a Bedrock block with disable_exception_on_block=True raised
# GuardrailInterventionNormalStringError, which no proxy code handled: the
# unified pre_call path re-raised it, so the client saw HTTP 500 with the block
# message; the native during_call hook swallowed it and set data["mock_response"],
# which was dead code because route_request already unpacked kwargs.
#
# The fix converts blocks to ModifyResponseException at the raise site inside
# make_bedrock_api_request. That exception is already the industry-standard
# proxy contract (caught in proxy_server.py, anthropic_endpoints, etc.) and
# turns into a 200 response whose content is the block message.
###############################################################################


def _blocked_bedrock_httpx_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Sorry, the model cannot answer this question."}],
        "assessments": [
            {
                "topicPolicy": {
                    "topics": [{"name": "Denied", "type": "DENY", "action": "BLOCKED"}]
                }
            }
        ],
    }
    return response


@pytest.mark.asyncio
async def test_make_bedrock_api_request_block_raises_modify_response_when_flag_set():
    from litellm.exceptions import ModifyResponseException

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=True,
    )

    request_data = {"model": "bedrock-nova-micro"}
    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _blocked_bedrock_httpx_response()

        with pytest.raises(ModifyResponseException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=[{"role": "user", "content": "My name is John Doe"}],
                request_data=request_data,
            )

    assert exc_info.value.message == "Sorry, the model cannot answer this question."
    assert exc_info.value.model == "bedrock-nova-micro"
    assert exc_info.value.guardrail_name == "test-bedrock-guard"


@pytest.mark.asyncio
async def test_make_bedrock_api_request_block_raises_http_400_when_flag_unset():
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=False,
    )

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _blocked_bedrock_httpx_response()

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=[{"role": "user", "content": "hi"}],
                request_data={"model": "bedrock-nova-micro"},
            )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_async_pre_call_hook_propagates_modify_response_on_block():
    """pre_call: block with disable_exception_on_block=True must raise
    ModifyResponseException so the endpoint handler returns 200 with the block
    message. Before LIT-4186 the exception was swallowed and only data
    ["mock_response"] was mutated, which the unified pre_call path never read
    (surfaced as HTTP 500)."""
    from litellm.exceptions import ModifyResponseException

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=True,
    )

    request_data = {
        "model": "bedrock-nova-micro",
        "messages": [{"role": "user", "content": "My name is John Doe"}],
    }
    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _blocked_bedrock_httpx_response()

        with pytest.raises(ModifyResponseException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=request_data,
                call_type="acompletion",
            )

    assert exc_info.value.message == "Sorry, the model cannot answer this question."
    # No `mock_response` mutation: the old broken contract must be gone
    # (route_request unpacks kwargs before this hook runs, so `mock_response`
    # would never reach the LLM call anyway).
    assert "mock_response" not in request_data


@pytest.mark.asyncio
async def test_async_moderation_hook_propagates_modify_response_on_block():
    """during_call: block must raise ModifyResponseException from the moderation
    task so the surrounding asyncio.gather cancels the LLM call, instead of
    the old behavior of swallowing the block and letting the model call proceed
    (LIT-4186 symptom 2: silent bypass, model billed)."""
    from litellm.exceptions import ModifyResponseException

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=True,
    )

    request_data = {
        "model": "bedrock-nova-micro",
        "messages": [{"role": "user", "content": "My name is John Doe"}],
    }
    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _blocked_bedrock_httpx_response()

        with pytest.raises(ModifyResponseException) as exc_info:
            await guardrail.async_moderation_hook(
                data=request_data,
                user_api_key_dict=UserAPIKeyAuth(),
                call_type="acompletion",
            )

    assert exc_info.value.message == "Sorry, the model cannot answer this question."


@pytest.mark.asyncio
async def test_async_post_call_success_hook_attaches_original_response_on_block():
    """post_call: block must raise ModifyResponseException and attach the LLM
    response to `original_response` so the synthetic block reply reports the
    upstream call's real token usage instead of zero."""
    from litellm.exceptions import ModifyResponseException

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=True,
    )

    request_data = {
        "model": "bedrock-nova-micro",
        "messages": [{"role": "user", "content": "hi"}],
    }
    llm_response = _model_response("Hello John Doe! The capital of France is Paris.")
    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post,
        patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _blocked_bedrock_httpx_response()

        with pytest.raises(ModifyResponseException) as exc_info:
            await guardrail.async_post_call_success_hook(
                data=request_data,
                user_api_key_dict=UserAPIKeyAuth(),
                response=llm_response,
            )

    assert exc_info.value.original_response is llm_response


@pytest.mark.asyncio
async def test_apply_guardrail_propagates_modify_response_on_block():
    """apply_guardrail (unified path used by pre_call / /apply_guardrail
    endpoint) must let ModifyResponseException propagate as-is so the endpoint
    handler catches it and returns a 200."""
    from litellm.exceptions import ModifyResponseException

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=True,
    )

    with patch.object(
        guardrail, "make_bedrock_api_request", new_callable=AsyncMock
    ) as mock_api:
        mock_api.side_effect = ModifyResponseException(
            message="Sorry, the model cannot answer this question.",
            model="bedrock-nova-micro",
            request_data={},
            guardrail_name="test-bedrock-guard",
        )

        with pytest.raises(ModifyResponseException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["My name is John Doe"]},
                request_data={"model": "bedrock-nova-micro"},
                input_type="request",
            )

    assert exc_info.value.message == "Sorry, the model cannot answer this question."


@pytest.mark.asyncio
async def test_streaming_post_call_block_yields_synthetic_stream_not_raise():
    """LIT-4186 regression: with disable_exception_on_block=True, streaming
    post_call blocks must be delivered as a synthetic stream (finish_reason=
    content_filter, block message as content), NOT raised. Pre-fix the local
    handler already produced this shape; the LIT-4186 refactor briefly turned
    it into an SSE 500 by letting ModifyResponseException escape the streaming
    generator. This test locks in the correct streaming contract.
    """
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=True,
    )

    async def _stream():
        yield ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", content="Coffee is a popular"),
                )
            ]
        )
        yield ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content=" beverage."), finish_reason="stop")]
        )

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _blocked_bedrock_httpx_response()

        chunks = [
            c
            async for c in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_stream(),
                request_data={"model": "bedrock-nova-micro"},
            )
        ]

    assert chunks, "streaming block should yield synthetic chunks, not error out"
    assembled_content = "".join(
        (c.choices[0].delta.content or "")
        for c in chunks
        if getattr(c, "choices", None) and getattr(c.choices[0], "delta", None)
    )
    assert assembled_content == "Sorry, the model cannot answer this question."
    assert chunks[-1].choices[0].finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_streaming_post_call_block_preserves_upstream_usage():
    """LIT-4186: streaming block must report the usage the upstream LLM call
    actually consumed. Non-streaming blocks carry it via original_response +
    _blocked_response_usage in the endpoint handler; streaming has to copy it
    onto the synthetic ModelResponse directly since the exception can't escape
    the SSE generator. Without this, clients see accurate billing on
    non-streaming blocks and zero on streaming blocks -- silent revenue leak."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=True,
    )

    async def _stream_with_usage():
        # Terminal chunk carrying usage, as OpenAI-style streams do with
        # stream_options={"include_usage": True}. stream_chunk_builder
        # aggregates this into the assembled ModelResponse's .usage.
        yield ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(role="assistant", content="Coffee is delicious"))]
        )
        yield ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content=""), finish_reason="stop")],
            usage=Usage(prompt_tokens=42, completion_tokens=17, total_tokens=59),
        )

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _blocked_bedrock_httpx_response()

        chunks = [
            c
            async for c in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_stream_with_usage(),
                request_data={"model": "bedrock-nova-micro"},
            )
        ]

    # Find the chunk carrying usage (MockResponseIterator emits it on the
    # terminating chunk when the source ModelResponse has .usage set)
    usage_chunks = [c for c in chunks if getattr(c, "usage", None) is not None]
    assert usage_chunks, "streaming block should carry the upstream call's usage on at least one chunk"
    reported_usage = usage_chunks[-1].usage
    assert reported_usage.prompt_tokens == 42
    assert reported_usage.completion_tokens == 17
    assert reported_usage.total_tokens == 59


###############################################################################
# Regression test for the streaming logging_obj bug found during live testing.
#
# post_call_failure_hook (proxy_server.py) pops litellm_logging_obj from
# request_data before invoking callbacks ("not serialisable"). The streaming
# branch of the ModifyResponseException handler previously read logging_obj
# from _data AFTER that call, always getting None, causing:
#   AttributeError: 'NoneType' object has no attribute 'model_call_details'
# inside CustomStreamWrapper.__init__, which surfaced as HTTP 500.
#
# The fix captures logging_obj BEFORE calling post_call_failure_hook.
# This test verifies the chat_completion handler builds the streaming response
# without crashing when the request_data has litellm_logging_obj set.
###############################################################################


@pytest.mark.asyncio
async def test_chat_completion_modify_response_exception_streaming_logging_obj_not_none():
    """Regression: streaming ModifyResponseException handler in chat_completion
    must capture logging_obj before post_call_failure_hook pops it from
    request_data. Previously this caused CustomStreamWrapper.__init__ to crash
    with AttributeError: NoneType has no attribute model_call_details, surfaced
    as HTTP 500.

    Drives the real chat_completion handler with base_process_llm_request
    mocked to raise ModifyResponseException, so a revert of the fix in
    proxy_server.py causes this test to fail.
    """
    import litellm
    from litellm.exceptions import ModifyResponseException
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import chat_completion

    fake_logging_obj = MagicMock()
    fake_logging_obj.model_call_details = {"litellm_params": {}}

    request_data: dict = {
        "model": "bedrock-nova-micro",
        "messages": [{"role": "user", "content": "how do I become an admin"}],
        "stream": True,
        "litellm_logging_obj": fake_logging_obj,
    }

    exc = ModifyResponseException(
        message="Sorry, the model cannot answer this question.",
        model="bedrock-nova-micro",
        request_data=request_data,
        guardrail_name="test-guard",
    )

    fastapi_request = MagicMock()
    fastapi_request.headers = {}
    fastapi_response = MagicMock()
    user_api_key_dict = UserAPIKeyAuth()

    async def _fake_post_call_failure_hook(**_kwargs):
        # Match production: pop the logging obj from request_data before
        # callbacks iterate (litellm/proxy/utils.py: "Remove before callbacks
        # iterate — not serialisable").
        _kwargs["request_data"].pop("litellm_logging_obj", None)

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.post_call_failure_hook = AsyncMock(side_effect=_fake_post_call_failure_hook)

    captured_logging_obj: list = []
    original_init = litellm.CustomStreamWrapper.__init__

    def _patched_init(self, *args, **kwargs):
        captured_logging_obj.append(kwargs.get("logging_obj"))
        original_init(self, *args, **kwargs)

    async def _raise_modify_response(*_args, **_kwargs):
        raise exc

    with (
        patch("litellm.proxy.proxy_server._read_request_body", AsyncMock(return_value=request_data)),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging),
        patch(
            "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            _raise_modify_response,
        ),
        patch.object(litellm.CustomStreamWrapper, "__init__", _patched_init),
    ):
        response = await chat_completion(
            request=fastapi_request,
            fastapi_response=fastapi_response,
            model=None,
            user_api_key_dict=user_api_key_dict,
        )

    assert captured_logging_obj, "chat_completion did not construct CustomStreamWrapper on the streaming block path"
    assert captured_logging_obj[0] is fake_logging_obj, (
        "chat_completion passed logging_obj=None to CustomStreamWrapper; "
        "the streaming ModifyResponseException handler must capture logging_obj "
        "before post_call_failure_hook pops it from request_data"
    )
    # A streaming block returns a StreamingResponse; if the fix were reverted,
    # CustomStreamWrapper would raise AttributeError inside __init__ and this
    # call would never reach here.
    assert response is not None


class TestBedrockOnlyScanNewMessages:
    """Bedrock apply_guardrail honors only_scan_new_messages: scans only the per-session diff.

    apply_guardrail is the path the proxy actually runs for Bedrock (via the unified
    guardrail interface), so these tests exercise it directly rather than the legacy
    async_pre_call_hook. Each test uses a unique session id to isolate the process-wide
    incremental cache.
    """

    def _guardrail(self):
        return BedrockGuardrail(
            guardrail_name="bedrock-incremental",
            guardrailIdentifier="test-guardrail",
            guardrailVersion="DRAFT",
            default_on=True,
            only_scan_new_messages=True,
        )

    @pytest.mark.asyncio
    async def test_second_turn_scans_only_new_messages(self):
        guardrail = self._guardrail()
        session = {"litellm_session_id": "sess-bedrock-diff"}
        bedrock_none = {"action": "NONE", "output": [], "outputs": []}

        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = bedrock_none

            await guardrail.apply_guardrail(
                inputs={"texts": ["be helpful", "first question"]},
                request_data=session,
                input_type="request",
            )
            assert mock_api.call_count == 1
            first_scanned = mock_api.call_args.kwargs["messages"]
            assert [m["content"] for m in first_scanned] == ["be helpful", "first question"]

            mock_api.reset_mock()

            await guardrail.apply_guardrail(
                inputs={"texts": ["be helpful", "first question", "first answer", "second question"]},
                request_data=session,
                input_type="request",
            )
            assert mock_api.call_count == 1
            second_scanned = mock_api.call_args.kwargs["messages"]
            assert [m["content"] for m in second_scanned] == ["first answer", "second question"]

    @pytest.mark.asyncio
    async def test_identical_resend_skips_api_call(self):
        guardrail = self._guardrail()
        session = {"litellm_session_id": "sess-bedrock-resend"}

        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}

            await guardrail.apply_guardrail(
                inputs={"texts": ["only question"]}, request_data=session, input_type="request"
            )
            assert mock_api.call_count == 1

            mock_api.reset_mock()
            result = await guardrail.apply_guardrail(
                inputs={"texts": ["only question"]}, request_data=session, input_type="request"
            )
            mock_api.assert_not_called()
            assert result["texts"] == ["only question"]

    @pytest.mark.asyncio
    async def test_no_session_id_scans_full_context(self):
        guardrail = self._guardrail()

        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}

            await guardrail.apply_guardrail(
                inputs={"texts": ["q1", "a1", "q2"]},
                request_data={"metadata": {}},
                input_type="request",
            )
            assert mock_api.call_count == 1
            scanned = mock_api.call_args.kwargs["messages"]
            assert [m["content"] for m in scanned] == ["q1", "a1", "q2"]

    @pytest.mark.asyncio
    async def test_masking_guardrail_falls_back_and_does_not_persist(self):
        """A guardrail that anonymizes content must not be short-circuited.

        Regression: the incremental fast path used to ignore the guardrail response,
        so masked/anonymized output was dropped, the raw text reached the model, and
        the segment was marked scanned so it was never re-checked. Detecting masked
        output must force a full-context scan (which applies the masking) and must not
        persist session state, so an identical resend is scanned again.
        """
        guardrail = self._guardrail()
        session = {"litellm_session_id": "sess-bedrock-mask"}
        masked = {
            "action": "GUARDRAIL_INTERVENED",
            "output": [],
            "outputs": [{"text": "my ssn is [REDACTED]"}],
        }

        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = masked

            result = await guardrail.apply_guardrail(
                inputs={"texts": ["my ssn is 123-45-6789"]},
                request_data=session,
                input_type="request",
            )
            assert mock_api.call_count == 2
            assert result["texts"] == ["my ssn is [REDACTED]"]

            mock_api.reset_mock()
            await guardrail.apply_guardrail(
                inputs={"texts": ["my ssn is 123-45-6789"]},
                request_data=session,
                input_type="request",
            )
            assert mock_api.call_count >= 1
            first_scanned = mock_api.call_args_list[0].kwargs.get("messages")
            assert first_scanned is not None
            assert [m["content"] for m in first_scanned] == ["my ssn is 123-45-6789"]

    @pytest.mark.asyncio
    async def test_generic_agent_multi_turn_scans_only_new_each_turn(self):
        """A generic agent (not Claude Code) opts in by propagating a session id.

        Agent frameworks on the OpenAI SDK carry the session through the request
        body (metadata.session_id here), not the x-claude-code-session-id header.
        Across a growing multi-turn conversation every turn after the first must
        send Bedrock only the newly appended segments, never the whole context.
        """
        guardrail = self._guardrail()
        session = {"metadata": {"session_id": "agent-multi-turn"}}

        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}

            await guardrail.apply_guardrail(
                inputs={"texts": ["system prompt", "turn 1 question"]},
                request_data=session,
                input_type="request",
            )
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == [
                "system prompt",
                "turn 1 question",
            ]

            mock_api.reset_mock()
            await guardrail.apply_guardrail(
                inputs={"texts": ["system prompt", "turn 1 question", "turn 1 answer", "turn 2 question"]},
                request_data=session,
                input_type="request",
            )
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == [
                "turn 1 answer",
                "turn 2 question",
            ]

            mock_api.reset_mock()
            await guardrail.apply_guardrail(
                inputs={
                    "texts": [
                        "system prompt",
                        "turn 1 question",
                        "turn 1 answer",
                        "turn 2 question",
                        "turn 2 answer",
                        "turn 3 question",
                    ]
                },
                request_data=session,
                input_type="request",
            )
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == [
                "turn 2 answer",
                "turn 3 question",
            ]

    def test_incremental_scan_cache_prefers_proxy_shared_cache(self):
        guardrail = self._guardrail()
        shared = DualCache()
        proxy_logging = MagicMock()
        proxy_logging.internal_usage_cache.dual_cache = shared

        with patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging):
            assert guardrail._incremental_scan_cache() is shared

    def test_incremental_scan_cache_falls_back_when_proxy_logging_missing(self):
        from litellm.integrations.custom_guardrail import dc as fallback_cache

        guardrail = self._guardrail()
        with patch("litellm.proxy.proxy_server.proxy_logging_obj", None):
            assert guardrail._incremental_scan_cache() is fallback_cache

    def test_incremental_scan_cache_falls_back_when_proxy_not_importable(self):
        from litellm.integrations.custom_guardrail import dc as fallback_cache

        guardrail = self._guardrail()
        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": None}):
            assert guardrail._incremental_scan_cache() is fallback_cache

    @pytest.mark.asyncio
    async def test_blocked_turn_is_rescanned_on_retry(self):
        guardrail = self._guardrail()
        session = {"litellm_session_id": "sess-bedrock-blocked"}

        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = HTTPException(status_code=400, detail="blocked")
            with pytest.raises(HTTPException):
                await guardrail.apply_guardrail(
                    inputs={"texts": ["blocked prompt"]}, request_data=session, input_type="request"
                )

            mock_api.reset_mock()
            mock_api.side_effect = None
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await guardrail.apply_guardrail(
                inputs={"texts": ["blocked prompt"]}, request_data=session, input_type="request"
            )
            assert mock_api.call_count == 1
            scanned = mock_api.call_args.kwargs["messages"]
            assert [m["content"] for m in scanned] == ["blocked prompt"]


class TestBedrockIncrementalFlagInteractions:
    """Regression coverage for only_scan_new_messages combined with the other
    Bedrock guardrail flags, from the PR #33278 live validation. Live evidence:
    each of these was reproduced against a real Bedrock ApplyGuardrail first;
    the mocks here encode the wire payloads observed there.
    """

    def _guardrail(self, **overrides):
        params = dict(
            guardrail_name="bedrock-incremental-flags",
            guardrailIdentifier="test-guardrail",
            guardrailVersion="DRAFT",
            default_on=True,
            only_scan_new_messages=True,
        )
        params.update(overrides)
        return BedrockGuardrail(**params)

    @pytest.mark.asyncio
    async def test_edited_history_segment_rescans_only_that_segment(self):
        guardrail = self._guardrail()
        session = {"litellm_session_id": "sess-flags-edit"}
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await guardrail.apply_guardrail(
                inputs={"texts": ["q1", "a1", "q2"]}, request_data=session, input_type="request"
            )
            mock_api.reset_mock()
            await guardrail.apply_guardrail(
                inputs={"texts": ["q1 EDITED", "a1", "q2"]}, request_data=session, input_type="request"
            )
            assert mock_api.call_count == 1
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == ["q1 EDITED"]

    @pytest.mark.asyncio
    async def test_same_content_different_session_rescans_everything(self):
        guardrail = self._guardrail()
        texts = ["shared question", "shared answer"]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await guardrail.apply_guardrail(
                inputs={"texts": list(texts)}, request_data={"litellm_session_id": "sess-x1"}, input_type="request"
            )
            mock_api.reset_mock()
            await guardrail.apply_guardrail(
                inputs={"texts": list(texts)}, request_data={"litellm_session_id": "sess-x2"}, input_type="request"
            )
            assert mock_api.call_count == 1
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == texts

    @pytest.mark.asyncio
    async def test_litellm_masking_flag_disables_incremental_single_full_scan(self):
        """mask_request_content must fall back to exactly ONE full scan per turn
        and never persist hashes (verified live: 1 call/turn, no cache writes)."""
        guardrail = self._guardrail(mask_request_content=True)
        session = {"litellm_session_id": "sess-flags-mask"}
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await guardrail.apply_guardrail(
                inputs={"texts": ["q1"]}, request_data=session, input_type="request"
            )
            assert mock_api.call_count == 1
            mock_api.reset_mock()
            await guardrail.apply_guardrail(
                inputs={"texts": ["q1"]}, request_data=session, input_type="request"
            )
            assert mock_api.call_count == 1, "masking mode must re-scan every turn, exactly once"

    @pytest.mark.asyncio
    async def test_server_side_anonymize_falls_back_full_scan_and_never_persists(self):
        """A guardrail that rewrites content (Bedrock-side ANONYMIZE) must fall back
        to the full scan so masking applies, and record no session state. Live
        validation showed this costs 2 provider calls per turn; the count is
        asserted here as documentation of that intended-tradeoff behavior."""
        guardrail = self._guardrail()
        session = {"litellm_session_id": "sess-flags-anon"}
        masked = {"action": "NONE", "output": [{"text": "MASKED q1"}], "outputs": [{"text": "MASKED q1"}]}
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = masked
            result = await guardrail.apply_guardrail(
                inputs={"texts": ["q1"]}, request_data=session, input_type="request"
            )
            assert mock_api.call_count == 2, "incremental attempt + full-scan fallback"
            assert result["texts"] == ["MASKED q1"], "masked content must be applied"
            mock_api.reset_mock()
            await guardrail.apply_guardrail(
                inputs={"texts": ["q1"]}, request_data=session, input_type="request"
            )
            assert mock_api.call_count == 2, "no hashes persisted, so the double scan repeats"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR #33278 known gap: incremental path bypasses _select_messages_for_apply_guardrail, "
        "so experimental_use_latest_role_message_only is silently ignored. Intended semantics "
        "(pending DRI decision): incremental mode defers to the latest-role selection.",
        strict=False,
    )
    async def test_latest_role_only_is_respected_with_incremental(self):
        guardrail = self._guardrail(experimental_use_latest_role_message_only=True)
        session = {"litellm_session_id": "sess-flags-latestrole"}
        structured = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
        ]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await guardrail.apply_guardrail(
                inputs={"texts": ["sys", "q1"], "structured_messages": structured},
                request_data=session,
                input_type="request",
            )
            scanned = [m["content"] for m in mock_api.call_args.kwargs["messages"]]
            assert scanned == ["q1"], "latest-role selection must exclude the system prompt"
