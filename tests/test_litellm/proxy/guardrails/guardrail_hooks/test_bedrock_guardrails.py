"""
Unit tests for Bedrock Guardrails
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
    _redact_pii_matches,
)
from litellm.types.llms.anthropic import (
    AllAnthropicMessageValues,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockRole,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


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
        "assessments": "not_a_list",  # This should cause an exception
    }

    # Should not crash and return original response
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
    """Test that the Bedrock guardrail uses redacted response for logging"""

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
    with patch.object(
        guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post, patch(
        "litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.verbose_proxy_logger.debug"
    ) as mock_debug, patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
    ) as mock_load_creds, patch.object(
        guardrail, "_prepare_request", return_value=MagicMock()
    ) as mock_prepare_request:

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
    with patch.object(
        guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post, patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
    ) as mock_load_creds, patch.object(
        guardrail, "_prepare_request", return_value=MagicMock()
    ) as mock_prepare_request:

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
async def test_create_bedrock_input_content_request_with_anthropic_messages():
    """Test creating Bedrock input request from Anthropic message types"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", 
        guardrailVersion="DRAFT"
    )
    
    # Create Anthropic message structure
    anthropic_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello, my name is John and my SSN is 123-45-6789"
                }
            ]
        },
        {
            "role": "assistant", 
            "content": [
                {
                    "type": "text",
                    "text": "I understand you've provided personal information."
                }
            ]
        }
    ]
    
    # Test the conversion
    bedrock_request = guardrail._create_bedrock_input_content_request(anthropic_messages)
    
    # Verify the structure
    assert "content" in bedrock_request
    assert len(bedrock_request["content"]) == 2
    
    # Verify first message content
    assert bedrock_request["content"][0]["text"]["text"] == "Hello, my name is John and my SSN is 123-45-6789"
    
    # Verify second message content
    assert bedrock_request["content"][1]["text"]["text"] == "I understand you've provided personal information."
    
    print("Anthropic input messages conversion test passed")


@pytest.mark.asyncio
async def test_create_bedrock_output_content_request_with_anthropic_response():
    """Test creating Bedrock output request from AnthropicMessagesResponse"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", 
        guardrailVersion="DRAFT"
    )
    
    # Create AnthropicMessagesResponse structure
    anthropic_response: AnthropicMessagesResponse = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            AnthropicResponseContentBlockText(
                type="text",
                text="Hello! Your personal information has been processed."
            ),
            AnthropicResponseContentBlockRole(
                type="role",
                role="assistant", 
                content="Additional response content"
            )
        ],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20}
    }
    
    # Test the conversion
    bedrock_request = guardrail._create_bedrock_output_content_request(anthropic_response)
    
    # Verify the structure
    assert "content" in bedrock_request
    assert len(bedrock_request["content"]) == 2
    
    # Verify text content block conversion
    assert bedrock_request["content"][0]["text"]["text"] == "Hello! Your personal information has been processed."
    
    # Verify role content block conversion  
    assert bedrock_request["content"][1]["text"]["text"] == "Additional response content"
    
    print("Anthropic response conversion test passed")


@pytest.mark.asyncio
async def test_convert_to_bedrock_format_anthropic_input():
    """Test convert_to_bedrock_format with Anthropic messages as INPUT"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", 
        guardrailVersion="DRAFT"
    )
    
    anthropic_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "My email is user@example.com and phone is 555-123-4567"
                }
            ]
        }
    ]
    
    # Test conversion for INPUT source
    bedrock_request = guardrail.convert_to_bedrock_format(
        source="INPUT",
        messages=anthropic_messages
    )
    
    # Verify the structure
    assert "content" in bedrock_request
    assert len(bedrock_request["content"]) == 1
    assert bedrock_request["content"][0]["text"]["text"] == "My email is user@example.com and phone is 555-123-4567"
    
    print("Anthropic INPUT format conversion test passed")


@pytest.mark.asyncio
async def test_convert_to_bedrock_format_anthropic_output():
    """Test convert_to_bedrock_format with Anthropic response as OUTPUT"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", 
        guardrailVersion="DRAFT"
    )
    
    anthropic_response: AnthropicMessagesResponse = {
        "id": "msg_456", 
        "type": "message",
        "role": "assistant",
        "content": [
            AnthropicResponseContentBlockText(
                type="text",
                text="I'll help you with that sensitive information: credit card 4111-1111-1111-1111"
            )
        ],
        "model": "claude-3-sonnet-20240229", 
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 15, "output_tokens": 25}
    }
    
    # Test conversion for OUTPUT source
    bedrock_request = guardrail.convert_to_bedrock_format(
        source="OUTPUT",
        response=anthropic_response
    )
    
    # Verify the structure
    assert "content" in bedrock_request
    assert len(bedrock_request["content"]) == 1
    assert bedrock_request["content"][0]["text"]["text"] == "I'll help you with that sensitive information: credit card 4111-1111-1111-1111"
    
    print("Anthropic OUTPUT format conversion test passed")


@pytest.mark.asyncio
async def test_make_bedrock_api_request_with_anthropic_messages():
    """Test make_bedrock_api_request with Anthropic messages"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", 
        guardrailVersion="DRAFT"
    )
    
    anthropic_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "My SSN is 123-45-6789"
                }
            ]
        }
    ]
    
    # Mock Bedrock API response
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Input blocked due to PII"}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "US_SOCIAL_SECURITY_NUMBER",
                            "match": "123-45-6789", 
                            "action": "BLOCKED"
                        }
                    ]
                }
            }
        ]
    }
    
    # Mock credentials and AWS methods
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key" 
    mock_credentials.token = None
    
    with patch.object(
        guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post, patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
    ), patch.object(
        guardrail, "_prepare_request", return_value=MagicMock()
    ):
        mock_post.return_value = mock_bedrock_response
        
        # Call the API request method
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=anthropic_messages
            )        
        # Verify that HTTPException was raised with correct details
        assert exc_info.value.status_code == 400
        assert "Violated guardrail policy" in str(exc_info.value.detail)
        assert "Input blocked due to PII" in str(exc_info.value.detail)

    print("Anthropic API request test passed")


@pytest.mark.asyncio
async def test_make_bedrock_api_request_with_anthropic_response():
    """Test make_bedrock_api_request with AnthropicMessagesResponse"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT"
    )
    
    anthropic_response: AnthropicMessagesResponse = {
        "id": "msg_789",
        "type": "message", 
        "role": "assistant",
        "content": [
            AnthropicResponseContentBlockText(
                type="text",
                text="Your credit card number 4111-1111-1111-1111 has been processed"
            )
        ],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "end_turn", 
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 30}
    }
    
    # Mock Bedrock API response
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Output blocked due to credit card info"}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "CREDIT_DEBIT_CARD_NUMBER",
                            "match": "4111-1111-1111-1111",
                            "action": "BLOCKED"
                        }
                    ]
                }
            }
        ]
    }
    
    # Mock credentials and AWS methods
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None
    
    with patch.object(
        guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post, patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
    ), patch.object(
        guardrail, "_prepare_request", return_value=MagicMock()
    ):
        mock_post.return_value = mock_bedrock_response
        
        # Call the API request method        
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="OUTPUT",
                messages=anthropic_response
            )        
        # Verify that HTTPException was raised with correct details
        assert exc_info.value.status_code == 400
        assert "Violated guardrail policy" in str(exc_info.value.detail)
        assert "Output blocked due to credit card info" in str(exc_info.value.detail)


    print("Anthropic response API request test passed")


@pytest.mark.asyncio
async def test_apply_masking_to_anthropic_messages_response():
    """Test applying masking to AnthropicMessagesResponse"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail", 
        guardrailVersion="DRAFT"
    )
    
    # Create AnthropicMessagesResponse with sensitive content
    anthropic_response: AnthropicMessagesResponse = {
        "id": "msg_mask_test",
        "type": "message",
        "role": "assistant", 
        "content": [
            AnthropicResponseContentBlockText(
                type="text",
                text="Your SSN 123-45-6789 is sensitive"
            ),
            AnthropicResponseContentBlockRole(
                type="role",
                role="assistant",
                content="Credit card 4111-1111-1111-1111 detected"
            )
        ],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 25, "output_tokens": 35}
    }
    
    # Mock Bedrock guardrail response with masking
    mock_guardrail_response = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [
            {"text": "Your SSN [REDACTED] is sensitive"},
            {"text": "Credit card [REDACTED] detected"}
        ],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "US_SOCIAL_SECURITY_NUMBER",
                            "match": "123-45-6789",
                            "action": "BLOCKED"
                        },
                        {
                            "type": "CREDIT_DEBIT_CARD_NUMBER", 
                            "match": "4111-1111-1111-1111",
                            "action": "BLOCKED"
                        }
                    ]
                }
            }
        ]
    }
    
    # Apply masking
    guardrail._apply_masking_to_response(anthropic_response, mock_guardrail_response)
    
    # Verify that content has been masked
    content = anthropic_response["content"]
    assert len(content) == 2
    
    # Check first content block (AnthropicResponseContentBlockText)
    assert isinstance(content[0], AnthropicResponseContentBlockText)
    assert content[0].text == "Your SSN [REDACTED] is sensitive"
    
    # Check second content block (AnthropicResponseContentBlockRole)
    assert isinstance(content[1], AnthropicResponseContentBlockRole) 
    assert content[1].content == "Credit card [REDACTED] detected"
    
    # Verify other fields remain unchanged
    assert anthropic_response["id"] == "msg_mask_test"
    assert anthropic_response["role"] == "assistant"
    assert anthropic_response["model"] == "claude-3-sonnet-20240229"
    
    print("Anthropic response masking test passed")


@pytest.mark.asyncio
async def test_get_content_for_anthropic_message():
    """Test extracting content from Anthropic message format"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT"
    )
    
    # Test with Anthropic message containing multiple content blocks
    anthropic_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "First piece of content with SSN 123-45-6789"
            },
            {
                "type": "text", 
                "text": "Second piece with email user@example.com"
            }
        ]
    }
    
    # Extract content
    content_list = guardrail.get_content_for_message(anthropic_message)
    
    # Verify extraction
    assert content_list is not None
    assert len(content_list) == 2
    assert content_list[0] == "First piece of content with SSN 123-45-6789"
    assert content_list[1] == "Second piece with email user@example.com"
    
    print("Anthropic message content extraction test passed")


@pytest.mark.asyncio
async def test_apply_masking_to_anthropic_messages():
    """Test applying masking to Anthropic messages list"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT"
    )
    
    # Create Anthropic messages with sensitive content
    anthropic_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "My SSN is 123-45-6789"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": "Your credit card 4111-1111-1111-1111 is noted"
                }
            ]
        }
    ]
    
    # Define masked texts
    masked_texts = [
        "My SSN is [REDACTED]", 
        "Your credit card [REDACTED] is noted"
    ]
    
    # Apply masking
    masked_messages = guardrail._apply_masking_to_messages(anthropic_messages, masked_texts)
    
    # Verify masking was applied
    assert len(masked_messages) == 2
    
    # Check first message
    first_message_content = masked_messages[0]["content"][0]["text"]
    assert first_message_content == "My SSN is [REDACTED]"
    
    # Check second message  
    second_message_content = masked_messages[1]["content"][0]["text"]
    assert second_message_content == "Your credit card [REDACTED] is noted"
    
    # Verify roles remain unchanged
    assert masked_messages[0]["role"] == "user"
    assert masked_messages[1]["role"] == "assistant"
    
    print("Anthropic messages masking test passed")


@pytest.mark.asyncio
async def test_anthropic_integration_pre_call_hook():
    """Test pre_call_hook with Anthropic messages"""
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT"
    )
    
    # Mock data with Anthropic messages
    data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi, my phone number is 555-123-4567"
                    }
                ]
            }
        ]
    }
    
    # Mock Bedrock API response
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Hi, my phone number is [REDACTED]"}],
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [
                        {
                            "type": "PHONE",
                            "match": "555-123-4567",
                            "action": "BLOCKED"
                        }
                    ]
                }
            }
        ]
    }
    
    # Mock credentials and AWS methods
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None
    
    with patch.object(
        guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post, patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
    ), patch.object(
        guardrail, "_prepare_request", return_value=MagicMock()
    ):
        mock_post.return_value = mock_bedrock_response
        
        # Call pre_call_hook - should raise HTTPException when guardrail blocks
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=None, 
                data=data,
                call_type="completion"
            )
        
        # Verify that HTTPException was raised with correct details
        assert exc_info.value.status_code == 400
        assert "Violated guardrail policy" in str(exc_info.value.detail)
        
    print("Anthropic pre_call_hook integration test passed")
