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

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
    _redact_pii_matches,
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
            guardrailed_inputs["tool_calls"][0]["id"]
            == "call_eFSCWFsyL7MclHYnzKrcQnMK"
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
    with patch.object(
        guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post, patch.object(
        guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
    ), patch.object(
        guardrail, "_prepare_request", return_value=MagicMock()
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
                        "customWords": None,       # null from Bedrock API
                        "managedWordLists": None,   # null from Bedrock API
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
                        "topics": None,           # null from Bedrock API
                    },
                    "contentPolicy": {
                        "filters": None,          # null
                    },
                    "wordPolicy": {
                        "customWords": None,      # null
                        "managedWordLists": None,  # null
                    },
                    "sensitiveInformationPolicy": {
                        "piiEntities": None,      # null
                        "regexes": None,           # null
                    },
                    "contextualGroundingPolicy": {
                        "filters": None,          # null
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
                        "topics": None,            # null — should not crash
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
                        "customWords": None,       # null
                        "managedWordLists": None,   # null
                    },
                    "sensitiveInformationPolicy": {
                        "piiEntities": None,       # null
                        "regexes": None,            # null
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
                        "customWords": [
                            {"match": "badword", "action": "BLOCKED"}
                        ],
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

        with patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post, patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ), patch.object(
            guardrail, "_prepare_request", return_value=MagicMock()
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

        with patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
        ) as mock_post, patch.object(
            guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ), patch.object(
            guardrail, "_prepare_request", return_value=MagicMock()
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
    """L3: PII entity match (BLOCKED) is surfaced with category, type, and matched term."""
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
