"""
Unit tests for Bedrock Guardrails
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException


import litellm
from litellm.caching.caching import DualCache
from litellm.exceptions import ModifyResponseException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.constants import BEDROCK_APPLY_GUARDRAIL_CHUNK_BUDGET_CHARS
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockContentChunkResult,
    BedrockGuardrail,
    _redact_pii_matches,
)
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockContentItem,
    BedrockTextContent,
)
from litellm.types.utils import CallTypes, ModelResponse


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
    pii_entities = redacted_response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]

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
    assessment1_pii = redacted_response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]
    assessment2_pii = redacted_response["assessments"][1]["sensitiveInformationPolicy"]["piiEntities"]

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

    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch("litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.verbose_proxy_logger.debug") as mock_debug,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")) as mock_load_creds,
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()) as mock_prepare_request,
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

        assert bedrock_response_log_call is not None, "Should have logged Bedrock AI response"

        # Extract the logged response data
        logged_response = bedrock_response_log_call[0][1]  # Second argument to debug call

        # Verify that the logged response has redacted PII
        assert (
            logged_response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0]["match"] == "[REDACTED]"
        )

        # Verify other fields are preserved
        assert logged_response["action"] == "GUARDRAIL_INTERVENED"
        assert logged_response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0]["type"] == "PHONE"

        slg_list = request_data["metadata"]["standard_logging_guardrail_information"]
        assert (
            slg_list[0]["guardrail_response"]["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0]["match"]
            == "[REDACTED]"
        )

        print("Bedrock guardrail logging redaction test passed")


@pytest.mark.asyncio
async def test_bedrock_guardrail_original_response_not_modified():
    """Test that the original response is not modified by redaction, only the logged version"""

    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()

    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")) as mock_load_creds,
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()) as mock_prepare_request,
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
            original_data["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0]["match"]
            == "+1 412 555 1212"
        )

        # Verify that the returned BedrockGuardrailResponse contains original data
        assert result["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"][0]["match"] == "+1 412 555 1212"

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
    pii_entities = redacted_response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]
    assert pii_entities[0]["match"] == "[REDACTED]", "PII match should be redacted"
    assert pii_entities[0]["type"] == "EMAIL", "PII type should be preserved"
    assert pii_entities[0]["action"] == "ANONYMIZED", "PII action should be preserved"
    assert pii_entities[0]["confidence"] == "HIGH", "PII confidence should be preserved"

    # Verify that regex matches are also redacted (updated behavior)
    regexes = redacted_response["assessments"][0]["sensitiveInformationPolicy"]["regexes"]
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
                    "guardrailCoverage": {"textCharacters": {"guarded": 84, "total": 84}},
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
    pii_entities = redacted_response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]

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
    original_pii_entities = original_response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]
    assert original_pii_entities[0]["match"] == "John Smith", "Original should be unchanged"
    assert original_pii_entities[1]["match"] == "324-12-3212", "Original should be unchanged"
    assert original_pii_entities[2]["match"] == "607-456-7890", "Original should be unchanged"

    # Verify all other metadata is preserved in redacted response
    assert redacted_response["action"] == "GUARDRAIL_INTERVENED"
    assert redacted_response["actionReason"] == "Guardrail blocked."
    assert redacted_response["blockedResponse"] == "Input blocked by PII policy"
    assert redacted_response["assessments"][0]["invocationMetrics"]["guardrailProcessingLatency"] == 322

    print("PII redaction matches debug output format test passed")
    print(f"Original PII values preserved: {[e['match'] for e in original_pii_entities]}")
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
    regexes = redacted_response["assessments"][0]["sensitiveInformationPolicy"]["regexes"]

    assert regexes[0]["match"] == "[REDACTED]", "SSN regex match should be redacted"
    assert regexes[1]["match"] == "[REDACTED]", "Credit card regex match should be redacted"

    # Verify other fields are preserved
    assert regexes[0]["name"] == "SSN_PATTERN", "Regex name should be preserved"
    assert regexes[0]["action"] == "BLOCKED", "Regex action should be preserved"
    assert regexes[1]["name"] == "CREDIT_CARD_PATTERN", "Regex name should be preserved"
    assert regexes[1]["action"] == "ANONYMIZED", "Regex action should be preserved"

    # Verify original response is unchanged
    original_regexes = response_with_regex["assessments"][0]["sensitiveInformationPolicy"]["regexes"]
    assert original_regexes[0]["match"] == "123-45-6789", "Original should be unchanged"
    assert original_regexes[1]["match"] == "4111-1111-1111-1111", "Original should be unchanged"

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

    assert custom_words[0]["match"] == "[REDACTED]", "First custom word match should be redacted"
    assert custom_words[1]["match"] == "[REDACTED]", "Second custom word match should be redacted"

    # Verify other fields are preserved
    assert custom_words[0]["action"] == "BLOCKED", "Custom word action should be preserved"
    assert custom_words[1]["action"] == "ANONYMIZED", "Custom word action should be preserved"

    # Verify original response is unchanged
    original_custom_words = response_with_custom_words["assessments"][0]["wordPolicy"]["customWords"]
    assert original_custom_words[0]["match"] == "confidential_data", "Original should be unchanged"
    assert original_custom_words[1]["match"] == "secret_information", "Original should be unchanged"

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
    managed_words = redacted_response["assessments"][0]["wordPolicy"]["managedWordLists"]

    assert managed_words[0]["match"] == "[REDACTED]", "First managed word match should be redacted"
    assert managed_words[1]["match"] == "[REDACTED]", "Second managed word match should be redacted"

    # Verify other fields are preserved
    assert managed_words[0]["action"] == "BLOCKED", "Managed word action should be preserved"
    assert managed_words[0]["type"] == "PROFANITY", "Managed word type should be preserved"
    assert managed_words[1]["action"] == "ANONYMIZED", "Managed word action should be preserved"
    assert managed_words[1]["type"] == "HATE_SPEECH", "Managed word type should be preserved"

    # Verify original response is unchanged
    original_managed_words = response_with_managed_words["assessments"][0]["wordPolicy"]["managedWordLists"]
    assert original_managed_words[0]["match"] == "inappropriate_word", "Original should be unchanged"
    assert original_managed_words[1]["match"] == "offensive_term", "Original should be unchanged"

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
    assert pii_entities[0]["match"] == "[REDACTED]", "PII entity match should be redacted"

    # Regex matches
    regexes = assessment["sensitiveInformationPolicy"]["regexes"]
    assert regexes[0]["match"] == "[REDACTED]", "Regex match should be redacted"

    # Custom words
    custom_words = assessment["wordPolicy"]["customWords"]
    assert custom_words[0]["match"] == "[REDACTED]", "Custom word match should be redacted"

    # Managed words
    managed_words = assessment["wordPolicy"]["managedWordLists"]
    assert managed_words[0]["match"] == "[REDACTED]", "Managed word match should be redacted"

    # Verify all other fields are preserved
    assert pii_entities[0]["type"] == "EMAIL"
    assert regexes[0]["name"] == "PHONE_PATTERN"
    assert managed_words[0]["type"] == "PROFANITY"

    # Verify original response is unchanged
    original_assessment = comprehensive_response["assessments"][0]
    assert original_assessment["sensitiveInformationPolicy"]["piiEntities"][0]["match"] == "user@example.com"
    assert original_assessment["sensitiveInformationPolicy"]["regexes"][0]["match"] == "555-123-4567"
    assert original_assessment["wordPolicy"]["customWords"][0]["match"] == "confidential"
    assert original_assessment["wordPolicy"]["managedWordLists"][0]["match"] == "inappropriate"

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
    with patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, aws_region_name)):
        # Call _prepare_request which internally calls get_runtime_endpoint
        prepped_request = guardrail._prepare_request(
            credentials=mock_credentials,
            data=data,
            optional_params=optional_params,
            aws_region_name=aws_region_name,
        )

        # Verify that the custom endpoint is used in the URL
        expected_url = (
            f"{custom_endpoint}/guardrail/{guardrail.guardrailIdentifier}/version/{guardrail.guardrailVersion}/apply"
        )
        assert prepped_request.url == expected_url, (
            f"Expected URL to contain custom endpoint. Got: {prepped_request.url}"
        )

        print(f"Custom runtime endpoint test passed. URL: {prepped_request.url}")


@pytest.mark.asyncio
async def test_bedrock_guardrail_respects_env_runtime_endpoint(monkeypatch):
    """Test that BedrockGuardrail respects AWS_BEDROCK_RUNTIME_ENDPOINT environment variable"""

    custom_endpoint = "https://env-bedrock.example.com"

    # Set the environment variable
    monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", custom_endpoint)

    # Create guardrail without explicit aws_bedrock_runtime_endpoint
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

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
    with patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, aws_region_name)):
        # Call _prepare_request which internally calls get_runtime_endpoint
        prepped_request = guardrail._prepare_request(
            credentials=mock_credentials,
            data=data,
            optional_params=optional_params,
            aws_region_name=aws_region_name,
        )

        # Verify that the custom endpoint from environment is used in the URL
        expected_url = (
            f"{custom_endpoint}/guardrail/{guardrail.guardrailIdentifier}/version/{guardrail.guardrailVersion}/apply"
        )
        assert prepped_request.url == expected_url, f"Expected URL to contain env endpoint. Got: {prepped_request.url}"

        print(f"Environment runtime endpoint test passed. URL: {prepped_request.url}")


@pytest.mark.asyncio
async def test_bedrock_guardrail_uses_default_endpoint_when_no_custom_set(monkeypatch):
    """Test that BedrockGuardrail uses default endpoint when no custom endpoint is set"""

    # Ensure no environment variable is set
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)

    # Create guardrail without any custom endpoint
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

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
    with patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, aws_region_name)):
        # Call _prepare_request which internally calls get_runtime_endpoint
        prepped_request = guardrail._prepare_request(
            credentials=mock_credentials,
            data=data,
            optional_params=optional_params,
            aws_region_name=aws_region_name,
        )

        # Verify that the default endpoint is used
        expected_url = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com/guardrail/{guardrail.guardrailIdentifier}/version/{guardrail.guardrailVersion}/apply"
        assert prepped_request.url == expected_url, f"Expected default URL. Got: {prepped_request.url}"

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
    with patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, aws_region_name)):
        # Call _prepare_request which internally calls get_runtime_endpoint
        prepped_request = guardrail._prepare_request(
            credentials=mock_credentials,
            data=data,
            optional_params=optional_params,
            aws_region_name=aws_region_name,
        )

        # Verify that the parameter takes precedence over environment variable
        expected_url = (
            f"{param_endpoint}/guardrail/{guardrail.guardrailIdentifier}/version/{guardrail.guardrailVersion}/apply"
        )
        assert prepped_request.url == expected_url, (
            f"Expected parameter endpoint to take precedence. Got: {prepped_request.url}"
        )

        print(f"Parameter precedence test passed. URL: {prepped_request.url}")


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_with_only_tool_calls_response():
    """Test that apply_guardrail handles response with tool_calls (no text content) without calling Bedrock API"""
    # Create a BedrockGuardrail instance
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

    # Mock the make_bedrock_api_request method
    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api_request:
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
        assert guardrailed_inputs["tool_calls"][0]["id"] == "call_eFSCWFsyL7MclHYnzKrcQnMK"
        assert guardrailed_inputs["tool_calls"][0]["function"]["name"] == "get_weather"
        assert guardrailed_inputs["tool_calls"][0]["function"]["arguments"] == '{"location":"São Paulo"}'
        # Verify that the Bedrock API was NOT called since there's no text to process
        mock_api_request.assert_not_called()
        print("✅ apply_guardrail with tool_calls test passed - no API call made")


def _anthropic_tool_result_conversation(
    extra_blocks: tuple[dict[str, str], ...] = (),
) -> list[dict[str, object]]:
    """Anthropic /v1/messages history whose latest user turn is a tool_result follow-up."""
    return [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {"city": "Paris"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01A", "content": "18C and sunny"},
                *extra_blocks,
            ],
        },
    ]


@pytest.mark.asyncio
async def test_during_call_hook_skips_bedrock_call_for_tool_result_only_turn():
    """A tool_result-only latest user turn must not post an empty content list to Bedrock.

    Regression for `400: At least one GuardrailContentBlock must be provided` on
    /v1/messages: with experimental_use_latest_role_message_only the scanned turn is the
    Anthropic tool_result block, which carries no text, so ApplyGuardrail rejected the call.
    """
    guardrail = BedrockGuardrail(
        guardrail_name="bedrock-tool-result",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        event_hook=GuardrailEventHooks.during_call,
        default_on=True,
        experimental_use_latest_role_message_only=True,
    )
    data = {"model": "claude-sonnet-4-5", "messages": _anthropic_tool_result_conversation()}

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
        await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(),
            call_type=CallTypes.anthropic_messages.value,
        )

    mock_post.assert_not_called()
    assert data["messages"] == _anthropic_tool_result_conversation()


@pytest.mark.asyncio
async def test_during_call_hook_still_scans_tool_result_turn_carrying_text():
    """The skip must be limited to turns with nothing to scan, never to tool_result turns as such."""
    guardrail = BedrockGuardrail(
        guardrail_name="bedrock-tool-result-text",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        event_hook=GuardrailEventHooks.during_call,
        default_on=True,
        experimental_use_latest_role_message_only=True,
    )
    data = {
        "model": "claude-sonnet-4-5",
        "messages": _anthropic_tool_result_conversation(({"type": "text", "text": "now summarize that"},)),
    }
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"action": "NONE", "assessments": []}

    with (
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
    ):
        mock_post.return_value = mock_response
        await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(),
            call_type=CallTypes.anthropic_messages.value,
        )

    mock_post.assert_called_once()
    sent = mock_post.call_args.kwargs["data"].decode()
    assert "now summarize that" in sent
    # tool_result text is not extracted by this path (https://github.com/BerriAI/litellm/issues/33086)
    assert "18C and sunny" not in sent


@pytest.mark.asyncio
async def test_make_apply_guardrail_request_skips_output_scan_without_response_text():
    """A tool-calls-only assistant response yields no OUTPUT content, so it must not be posted."""
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")
    response = ModelResponse(
        choices=[
            litellm.Choices(
                index=0,
                message=litellm.Message(role="assistant", content=None, tool_calls=[]),
                finish_reason="tool_calls",
            )
        ]
    )

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
        bedrock_response = await guardrail.make_bedrock_api_request(source="OUTPUT", response=response)

    mock_post.assert_not_called()
    assert bedrock_response == {}


@pytest.mark.asyncio
async def test_make_apply_guardrail_request_skips_scan_without_credentials():
    """Skipping happens before credential resolution, so an empty scan costs no AWS work."""
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

    with (
        patch.object(guardrail, "_load_credentials", side_effect=AssertionError("credentials must not be loaded")),
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
    ):
        await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=[{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "out"}]}],
        )

    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_bedrock_apply_guardrail_response_uses_OUTPUT_source():
    """input_type='response' must call Bedrock with source=OUTPUT and assistant content.

    Regression: apply_guardrail used to always use source=INPUT. Output-only Bedrock
    policies (e.g. PII on model output) then returned action=NONE for non-streaming
    completions that go through unified_guardrail -> process_output_response.
    """
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")
    bedrock_none = {"action": "NONE", "output": [], "outputs": []}

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
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
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")
    bedrock_none = {"action": "NONE", "output": [], "outputs": []}

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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
        return BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

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
                        "regexes": [{"name": "SSN", "match": "123-45-6789", "action": "BLOCKED"}],
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
        guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

        inputs = {"texts": None}  # Explicit None

        mock_credentials = MagicMock()

        with (
            patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
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
        guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

        inputs = {}  # No "texts" key

        mock_credentials = MagicMock()

        with (
            patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
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
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

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

    should_raise = guardrail._should_raise_guardrail_blocked_exception(anonymized_response)
    assert should_raise is False, "ANONYMIZED actions should not raise exceptions"

    # Test 2: BLOCKED action should raise exception
    blocked_response = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "I can't provide that information."}],
        "assessments": [
            {"topicPolicy": {"topics": [{"name": "Sensitive Topic", "type": "DENY", "action": "BLOCKED"}]}}
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
                "topicPolicy": {"topics": [{"name": "Blocked Topic", "type": "DENY", "action": "BLOCKED"}]},
            }
        ],
    }

    should_raise = guardrail._should_raise_guardrail_blocked_exception(mixed_response)
    assert should_raise is True, "Mixed actions with any BLOCKED should raise exceptions"

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
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "NONE",
        "assessments": [
            {"sensitiveInformationPolicy": {"piiEntities": [{"type": "NAME", "match": "GG", "action": "BLOCKED"}]}}
        ],
    }

    request_data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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
        assert mock_log.call_args.kwargs["event_type"] == GuardrailEventHooks.during_call
        # Raw Bedrock JSON is forwarded; redaction runs once in
        # CustomGuardrail.add_standard_logging_guardrail_information_to_request_data.
        assert (
            mock_log.call_args.kwargs["guardrail_json_response"]["assessments"][0]["sensitiveInformationPolicy"][
                "piiEntities"
            ][0]["match"]
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
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")
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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=prepared_request) as mock_prepare_request,
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
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key", user_id="test_user"),
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
                "topicPolicy": {"topics": [{"name": "Investment", "type": "DENY", "action": "BLOCKED"}]},
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
                "wordPolicy": {"customWords": [{"match": "forbidden", "action": "BLOCKED"}]},
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
            {"sensitiveInformationPolicy": {"piiEntities": [{"type": "NAME", "action": "ANONYMIZED", "match": "Jack"}]}}
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
            {"sensitiveInformationPolicy": {"piiEntities": [{"type": "NAME", "action": "BLOCKED", "match": "Jack"}]}}
        ],
    }
    exc = g._get_http_exception_for_blocked_guardrail(response)
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400
    assert exc.detail["error"] == "Violated guardrail policy"
    assert exc.detail["bedrock_guardrail_response"] == "Sorry, the model cannot answer this question."
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
                "contentPolicy": {"filters": [{"type": "VIOLENCE", "action": "BLOCKED"}]},
                "wordPolicy": {
                    "managedWordLists": [{"type": "PROFANITY", "action": "BLOCKED"}],
                },
                "sensitiveInformationPolicy": {"piiEntities": [{"type": "EMAIL", "action": "BLOCKED"}]},
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
                    "customWords": [{"match": "secret-codeword-abc-123", "action": "BLOCKED"}],
                },
                "sensitiveInformationPolicy": {"regexes": [{"match": "4111-1111-1111-1111", "action": "BLOCKED"}]},
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
        "assessments": [{"sensitiveInformationPolicy": {"piiEntities": [{"type": "NAME", "action": "ANONYMIZED"}]}}],
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
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")
    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{"topicPolicy": {"topics": [{"name": "Fiduciary Advice", "action": "BLOCKED"}]}}],
    }

    request_data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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

        with pytest.raises(Exception, match="blocked"):
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
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")
    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {"assessments": []}

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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
            {"sensitiveInformationPolicy": {"piiEntities": [{"type": "NAME", "action": "ANONYMIZED", "match": "Jack"}]}}
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
    with patch.object(guardrail, "make_bedrock_api_request", AsyncMock(return_value=minimal)) as mock_make:
        out = []
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=mock_stream(),
            request_data=request_data,
        ):
            out.append(chunk)

    assert len(out) >= 1
    output_calls = [c for c in mock_make.call_args_list if c.kwargs.get("source") == "OUTPUT"]
    assert len(output_calls) == 1
    assert output_calls[0].kwargs.get("request_data") is request_data
    assert output_calls[0].kwargs.get("logging_event_type") == GuardrailEventHooks.post_call
    input_calls = [c for c in mock_make.call_args_list if c.kwargs.get("source") == "INPUT"]
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
    with patch.object(guardrail, "make_bedrock_api_request", AsyncMock(return_value=minimal)) as mock_make:
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
    with patch.object(guardrail, "make_bedrock_api_request", AsyncMock(return_value=minimal)) as mock_make:
        await guardrail.async_post_call_success_hook(
            data=request_data,
            user_api_key_dict=UserAPIKeyAuth(),
            response=response,
        )

    sources = [c.kwargs.get("source") for c in mock_make.call_args_list]
    assert sources == ["OUTPUT"]
    assert mock_make.call_args.kwargs.get("logging_event_type") == GuardrailEventHooks.post_call


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
    return BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")


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
_GROUNDING_SOURCE_BLOCK = {"text": {"text": _GROUNDING_SOURCE_TEXT, "qualifiers": ["grounding_source"]}}
_QUERY_BLOCK = {"text": {"text": _GROUNDING_QUERY_TEXT, "qualifiers": ["query"]}}
_GUARD_BLOCK = {"text": {"text": _GROUNDING_RESPONSE_TEXT, "qualifiers": ["guard_content"]}}


def _input_request(messages: list) -> dict:
    """Arrange a guardrail and act: build the Bedrock INPUT payload."""
    return _grounding_guardrail().convert_to_bedrock_format(source="INPUT", messages=messages)


def _output_request(messages: list, response=None) -> dict:
    """Arrange a guardrail and act: build the Bedrock OUTPUT payload."""
    return _grounding_guardrail().convert_to_bedrock_format(source="OUTPUT", response=response, messages=messages)


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

    actual_request = _input_request([{"role": "user", "content": [{"type": "guarded_text", "text": "policy"}]}])

    assert actual_request == expected_request


def test_grounding_output_assembles_source_query_and_response():
    """OUTPUT emits grounding_source + query (from the request) then the response as
    guard_content, so Bedrock can grade the response against the source and query."""
    expected_request = {
        "source": "OUTPUT",
        "content": [_GROUNDING_SOURCE_BLOCK, _QUERY_BLOCK, _GUARD_BLOCK],
    }

    actual_request = _output_request(_grounding_messages(), _model_response(_GROUNDING_RESPONSE_TEXT))

    assert actual_request == expected_request


def test_grounding_output_keeps_legacy_payload_without_tags():
    """Without grounding tags the OUTPUT payload is the legacy single response block."""
    expected_request = {
        "source": "OUTPUT",
        "content": [{"text": {"text": "Hi there."}}],
    }

    actual_request = _output_request([{"role": "user", "content": "hello"}], _model_response("Hi there."))

    assert actual_request == expected_request


def test_grounding_output_combines_multiple_sources():
    """Every grounding_source block is emitted; Bedrock combines them into one corpus."""
    uk_source_text = "London is the capital of UK."
    uk_source_block = {"text": {"text": uk_source_text, "qualifiers": ["grounding_source"]}}
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

    actual_request = _output_request(messages, _model_response(_GROUNDING_RESPONSE_TEXT))

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

    actual_request = _output_request(messages, _model_response(_GROUNDING_RESPONSE_TEXT))

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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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
        "assessments": [{"topicPolicy": {"topics": [{"name": "Denied", "type": "DENY", "action": "BLOCKED"}]}}],
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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
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

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
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


_ANTHROPIC_SSE_CHUNKS = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message",'
    b'"role":"assistant","model":"claude","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"my ssn is 123-45-6789"}}\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":9}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
)


async def _anthropic_sse_stream():
    for chunk in _ANTHROPIC_SSE_CHUNKS:
        yield chunk


async def _drain_streaming_hook(
    guardrail: BedrockGuardrail, request_data: dict[str, object] | None = None
) -> list[object]:
    return [
        chunk
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=_anthropic_sse_stream(),
            request_data=request_data
            if request_data is not None
            else {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "what is my ssn"}]},
        )
    ]


def _sse_guardrail(**kwargs: object) -> BedrockGuardrail:
    return BedrockGuardrail(
        guardrail_name="bedrock-sse",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        event_hook=GuardrailEventHooks.post_call,
        default_on=True,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_streaming_hook_scans_raw_anthropic_sse_instead_of_crashing():
    """A /v1/messages stream arrives as raw SSE frames and must be assembled, then scanned.

    Regression for `500 Error building chunks for logging/streaming usage calculation`:
    stream_chunk_builder subscripts each chunk, which raises TypeError on bytes.
    """
    guardrail = _sse_guardrail()

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = {"action": "NONE"}
        delivered = await _drain_streaming_hook(guardrail)

    mock_api.assert_called_once()
    kwargs = mock_api.call_args.kwargs
    assert kwargs["source"] == "OUTPUT"
    assert "my ssn is 123-45-6789" in str(kwargs["response"].choices[0].message.content)
    assert kwargs["messages"] == [{"role": "user", "content": "what is my ssn"}]
    assert tuple(delivered) == _ANTHROPIC_SSE_CHUNKS


@pytest.mark.asyncio
async def test_streaming_hook_emits_masked_text_for_raw_anthropic_sse():
    """Masking must reach the client on /v1/messages, with mask_response_content unset.

    The assembled path masks regardless of the flag, so forwarding the original frames here
    would ship exactly the text the guardrail redacted.
    """
    guardrail = _sse_guardrail()

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [{"text": "my ssn is {SSN}"}],
        }
        delivered = await _drain_streaming_hook(guardrail)

    body = b"".join(delivered)
    assert b"{SSN}" in body
    assert b"123-45-6789" not in body


@pytest.mark.asyncio
async def test_streaming_hook_block_stream_keeps_upstream_identity():
    """A blocked stream must carry the same id and model as the mask path, not the proxy alias."""
    guardrail = _sse_guardrail(disable_exception_on_block=True)

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.side_effect = ModifyResponseException(
            message="Sorry, the model cannot answer this question.",
            model="my-proxy-alias",
            request_data={},
        )
        delivered = await _drain_streaming_hook(guardrail)

    body = b"".join(delivered)
    # the shared block builder mints a new message id: the block is not the upstream message
    assert b'"id": "msg_' in body
    assert b'"model": "claude"' in body
    assert b"my-proxy-alias" not in body


@pytest.mark.asyncio
async def test_streaming_hook_reraises_guardrail_service_failures():
    """A Bedrock outage must keep its status, not be reported to the caller as a guardrail decision.

    A policy block is the only 400 detailing a Mapping.
    """
    guardrail = _sse_guardrail()

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.side_effect = HTTPException(
            status_code=500, detail="Bedrock guardrail throttle retries exhausted"
        )
        with pytest.raises(HTTPException) as exc:
            await _drain_streaming_hook(guardrail)

    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_streaming_hook_frames_a_service_failure_once_a_keepalive_ping_flushed_the_headers():
    """Past the ping the status line is already on the wire, so a raise reaches the client as nothing.

    The failure has to travel as a frame instead, carrying its real status in the message.
    """
    guardrail = _sse_guardrail()

    with (
        patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api,
        patch.object(litellm, "anthropic_sse_ping_interval_seconds", 0.0001),
    ):
        mock_api.side_effect = HTTPException(status_code=503, detail="Bedrock is unavailable")
        delivered = await _drain_streaming_hook(guardrail)

    body = b"".join(delivered).decode()
    frame = next(line for line in body.splitlines() if line.startswith("data: "))
    message = json.loads(frame[6:])["error"]["message"]
    assert message == "503: Bedrock is unavailable"


@pytest.mark.asyncio
async def test_streaming_hook_reraises_a_service_failure_that_details_a_mapping():
    """InvokeGuardrailChecks details a Mapping on its 500, so detail shape alone cannot mean "block"."""
    guardrail = _sse_guardrail()

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.side_effect = HTTPException(
            status_code=500,
            detail={"error": "Bedrock InvokeGuardrailChecks returned an unexpected response shape"},
        )
        with pytest.raises(HTTPException) as exc:
            await _drain_streaming_hook(guardrail)

    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_streaming_block_error_frame_message_is_a_string():
    """AnthropicErrorDetail.message is typed str, built by the proxy's own detail serializer."""
    guardrail = _sse_guardrail()

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.side_effect = HTTPException(
            status_code=400, detail={"error": "Violated guardrail policy", "guardrailIdentifier": "gid"}
        )
        delivered = await _drain_streaming_hook(guardrail)

    frame = next(line for line in b"".join(delivered).decode().splitlines() if line.startswith("data: "))
    message = json.loads(frame[6:])["error"]["message"]
    # AnthropicErrorDetail.message is typed str, and the proxy's own serializer produces the
    # readable message rather than a repr of the detail dict
    assert isinstance(message, str)
    assert message == "Violated guardrail policy"


@pytest.mark.asyncio
async def test_streaming_hook_fails_closed_when_raw_sse_cannot_be_assembled():
    """An unscannable stream must not be delivered: forwarding it silently disables the guardrail."""
    guardrail = _sse_guardrail()

    async def _unparseable_stream():
        yield b'data: {"type":"content_block_delta"}\n\n'

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        delivered = [
            chunk
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_unparseable_stream(),
                request_data={"model": "claude-sonnet-4-5"},
            )
        ]

    mock_api.assert_not_called()
    body = b"".join(delivered)
    # a raise cannot reach the client once a keepalive ping has flushed the headers
    assert b"event: error" in body
    assert b"could not be assembled" in body
    assert b"content_block_delta" not in body


@pytest.mark.asyncio
async def test_streaming_hook_fails_closed_when_assembler_raises_api_error():
    """stream_chunk_builder re-raises assembly failures as litellm.APIError; it must not escape.

    That exception message is the exact 500 this fix exists to remove.
    """
    guardrail = _sse_guardrail()

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers."
        "anthropic_passthrough_logging_handler.AnthropicPassthroughLoggingHandler."
        "_build_complete_streaming_response",
        side_effect=litellm.APIError(
            status_code=500,
            message="Error building chunks for logging/streaming usage calculation",
            llm_provider="",
            model="",
        ),
    ):
        delivered = await _drain_streaming_hook(guardrail)

    assert b"event: error" in b"".join(delivered)


@pytest.mark.asyncio
async def test_streaming_hook_preserves_message_id_and_model_when_re_emitting():
    """A rewritten stream must still look like the upstream Anthropic response."""
    guardrail = _sse_guardrail()

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [{"text": "my ssn is {SSN}"}],
        }
        delivered = await _drain_streaming_hook(guardrail)

    body = b"".join(delivered)
    assert b'"id": "msg_1"' in body
    assert b"unknown-model" not in body
    assert b'"model": "claude"' in body


@pytest.mark.asyncio
async def test_streaming_hook_blocks_raw_anthropic_sse_on_violation():
    """A block on the extracted text must stop the stream rather than deliver it."""
    guardrail = _sse_guardrail()

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.side_effect = HTTPException(status_code=400, detail={"error": "Violated guardrail policy"})
        delivered = await _drain_streaming_hook(guardrail)

    body = b"".join(delivered)
    # a keepalive ping may already have flushed the headers, so the block has to travel as a frame
    assert b"event: error" in body
    assert b"Violated guardrail policy" in body
    assert b"123-45-6789" not in body


@pytest.mark.asyncio
async def test_streaming_hook_yields_synthetic_block_stream_for_raw_anthropic_sse():
    """disable_exception_on_block must keep behaving as a stream, not an SSE 500 frame."""
    guardrail = _sse_guardrail(disable_exception_on_block=True)

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.side_effect = ModifyResponseException(
            message="Sorry, the model cannot answer this question.",
            model="claude",
            request_data={},
        )
        delivered = await _drain_streaming_hook(guardrail)

    body = b"".join(delivered)
    assert b"Sorry, the model cannot answer this question." in body
    assert b"123-45-6789" not in body
    # the upstream call was already paid for, so the block frame must still report its usage
    assert b'"input_tokens": 5' in body
    assert b'"output_tokens": 9' in body


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


def _too_large_validation_httpx_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 400
    response.json.return_value = {
        "message": "Input is too long. Content size exceeds the maximum input size in text units."
    }
    response.text = json.dumps(response.json.return_value)
    return response


def _other_validation_httpx_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 400
    response.json.return_value = {"message": "guardrailIdentifier is not valid"}
    response.text = json.dumps(response.json.return_value)
    return response


def _throttling_httpx_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 429
    response.json.return_value = {"message": "Rate exceeded"}
    response.text = json.dumps(response.json.return_value)
    response.headers = {}
    return response


def _too_large_throttling_httpx_response() -> MagicMock:
    """The shape AWS actually returns for an oversized ApplyGuardrail request when
    the guardrail has an active content-filter policy: a 429 ThrottlingException,
    not the documented 400 ValidationException. Message taken from a live call."""
    response = MagicMock()
    response.status_code = 429
    response.json.return_value = {
        "message": (
            "Input text size (3273 text units) exceeds the maximum allowed "
            "(1000 text units) for the content filter policy (Classic tier)."
        )
    }
    response.text = json.dumps(response.json.return_value)
    response.headers = {}
    return response


def _passing_bedrock_httpx_response(marker: str) -> MagicMock:
    """A successful ApplyGuardrail response tagged with `marker` so tests can
    verify which chunk produced which output/usage after merging."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "action": "NONE",
        "outputs": [{"text": marker}],
        "assessments": [],
        "usage": {"contentPolicyUnits": 1},
    }
    return response


def _blocking_bedrock_httpx_response(marker: str) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": marker}],
        "assessments": [{"topicPolicy": {"topics": [{"name": marker, "type": "DENY", "action": "BLOCKED"}]}}],
        "usage": {"contentPolicyUnits": 1},
    }
    return response


def _bedrock_guardrail_for_chunk_tests() -> "BedrockGuardrail":
    return BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=False,
    )


@pytest.mark.asyncio
async def test_apply_guardrail_chunks_on_too_large_validation_error():
    """A too-large 400 on the whole-content call must trigger a bisect-and-retry,
    and the two chunk responses must be merged (assessments concatenated, usage
    summed, outputs concatenated) rather than losing either half's result."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {"role": "user", "content": "first half of a very long message"},
        {"role": "user", "content": "second half of a very long message"},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _too_large_validation_httpx_response()
        if call_count == 2:
            return _passing_bedrock_httpx_response("chunk-1")
        return _blocking_bedrock_httpx_response("chunk-2")

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = _post_side_effect

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=messages,
                request_data={"model": "bedrock-nova-micro"},
            )

    assert call_count == 3
    detail = exc_info.value.detail
    assert exc_info.value.status_code == 400
    assert "chunk-2" in detail["bedrock_guardrail_response"]
    assert detail["assessments"][0]["matches"][0]["name"] == "chunk-2"


@pytest.mark.asyncio
async def test_apply_guardrail_merges_usage_and_outputs_across_chunks_when_both_pass():
    """When both chunks pass clean, the merged response must still carry both
    chunks' outputs/usage forward (needed for accurate logging/telemetry) and
    must not itself raise."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {"role": "user", "content": "chunk one text"},
        {"role": "user", "content": "chunk two text"},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _too_large_validation_httpx_response()
        if call_count == 2:
            return _passing_bedrock_httpx_response("chunk-1")
        return _passing_bedrock_httpx_response("chunk-2")

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = _post_side_effect

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert call_count == 3
    assert result.get("action") == "NONE"
    output_texts = [o.get("text") for o in result.get("outputs") or []]
    assert output_texts == ["chunk-1", "chunk-2"]
    assert result.get("usage", {}).get("contentPolicyUnits") == 2


@pytest.mark.asyncio
async def test_apply_guardrail_recurses_past_first_bisection_into_four_chunks():
    """A payload that is still too large after one bisection must keep splitting
    -- chunking is not capped at two pieces. Four messages where both the
    whole-content call AND both first-level halves are too large must recurse
    one level deeper into four chunks that all fit, not give up after the
    first split."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {"role": "user", "content": "message one"},
        {"role": "user", "content": "message two"},
        {"role": "user", "content": "message three"},
        {"role": "user", "content": "message four"},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    responses = [
        _too_large_validation_httpx_response(),  # whole content: [1,2,3,4]
        _too_large_validation_httpx_response(),  # first half: [1,2]
        _passing_bedrock_httpx_response("message one"),
        _passing_bedrock_httpx_response("message two"),
        _too_large_validation_httpx_response(),  # second half: [3,4]
        _passing_bedrock_httpx_response("message three"),
        _passing_bedrock_httpx_response("message four"),
    ]

    async def _post_side_effect(*_args, **_kwargs):
        return responses.pop(0)

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = _post_side_effect

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert mock_post.await_count == 7
    assert not responses
    assert result.get("action") == "NONE"
    output_texts = [o.get("text") for o in result.get("outputs") or []]
    assert output_texts == ["message one", "message two", "message three", "message four"]


@pytest.mark.asyncio
async def test_apply_guardrail_does_not_chunk_when_grounding_present():
    """Contextual-grounding requests are scored holistically against the whole
    source; chunking them would silently produce misleading grounding scores.
    A too-large error on a grounded request must propagate unchanged, not be
    bisected."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {
            "role": "system",
            "content": [{"type": "grounding_source", "text": "reference source text"}],
        },
        {"role": "user", "content": "what does the source say?"},
    ]
    model_response = ModelResponse()
    model_response.choices = [litellm.Choices(message=litellm.Message(content="a grounded answer", role="assistant"))]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _too_large_validation_httpx_response()

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="OUTPUT",
                messages=messages,
                response=model_response,
                request_data={"model": "bedrock-nova-micro"},
            )

    assert mock_post.await_count == 1
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_does_not_chunk_on_non_size_validation_error():
    """A 400 for an unrelated validation problem (e.g. a bad guardrail id) must
    not trigger chunking -- retrying a bad-config error split into pieces would
    just fail twice more and mask the real problem."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _other_validation_httpx_response()

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=messages,
                request_data={"model": "bedrock-nova-micro"},
            )

    assert mock_post.await_count == 1
    assert exc_info.value.status_code == 400
    assert "not valid" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_apply_guardrail_too_large_on_unsplittable_text_propagates_original_error():
    """A too-large error on content that has been bisected down to text too
    short to split further (< 2 characters) must propagate the original error
    rather than looping or crashing."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [{"role": "user", "content": "a"}]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _too_large_validation_httpx_response()

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=messages,
                request_data={"model": "bedrock-nova-micro"},
            )

    assert mock_post.await_count == 1
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_too_large_on_single_item_splits_by_text_and_succeeds():
    """A too-large error on content that is already down to a single content
    item must be bisected by that item's own text (not abandoned), so an
    oversized single message can still be scanned successfully in halves."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [{"role": "user", "content": "one giant single block of text"}]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _too_large_validation_httpx_response()
        return _passing_bedrock_httpx_response(f"half-{call_count}")

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = _post_side_effect

        response = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert mock_post.await_count == 3
    assert response.get("action") == "NONE"


def _raised_bedrock_error(status_code: int, message: str) -> httpx.HTTPStatusError:
    """A non-200 the way `AsyncHTTPHandler.post` actually surfaces it.

    That handler calls `response.raise_for_status()`, so in production a non-200 from
    Bedrock arrives as a raised `httpx.HTTPStatusError` carrying the response, never
    as a returned response object. Tests that return the response instead exercise a
    branch real traffic never reaches. A real `httpx.Response` is used rather than a
    MagicMock because the transport helper branches on
    `isinstance(err_response, httpx.Response)`."""
    response = httpx.Response(
        status_code=status_code,
        json={"message": message},
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/guardrail"),
    )
    return httpx.HTTPStatusError(message, request=response.request, response=response)


_TOO_LARGE_MESSAGE = "Input is too long. Content size exceeds the maximum input size in text units."


@pytest.mark.asyncio
async def test_apply_guardrail_chunking_logs_once_when_client_raises_for_status():
    """The too-large attempt recovered by chunking must still produce exactly one
    telemetry entry when the HTTP client raises for status, which is what really
    happens: `AsyncHTTPHandler.post` calls `raise_for_status()`.

    Regression for per-attempt `guardrail_failed_to_respond` entries leaking out of
    the transport helper on a request that ultimately succeeded, which made a
    recovered request look like several failures plus a success."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {"role": "user", "content": "chunk one text"},
        {"role": "user", "content": "chunk two text"},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _raised_bedrock_error(400, _TOO_LARGE_MESSAGE)
        return _passing_bedrock_httpx_response(f"chunk-{call_count}")

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch.object(
            guardrail,
            "add_standard_logging_guardrail_information_to_request_data",
        ) as mock_log,
    ):
        mock_post.side_effect = _post_side_effect

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert call_count == 3
    assert result.get("action") == "NONE"
    statuses = [call.kwargs.get("guardrail_status") for call in mock_log.call_args_list]
    assert statuses == ["success"]


@pytest.mark.asyncio
async def test_apply_guardrail_unrecoverable_failure_still_logs_once_when_client_raises():
    """Suppressing the transport helper's per-attempt logging must not swallow the only
    record of a genuine failure: an unsplittable too-large request still has to produce
    exactly one `guardrail_failed_to_respond` entry, not zero."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [{"role": "user", "content": "x"}]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    async def _post_side_effect(*_args, **_kwargs):
        raise _raised_bedrock_error(400, _TOO_LARGE_MESSAGE)

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch.object(
            guardrail,
            "add_standard_logging_guardrail_information_to_request_data",
        ) as mock_log,
    ):
        mock_post.side_effect = _post_side_effect

        with pytest.raises(HTTPException):
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=messages,
                request_data={"model": "bedrock-nova-micro"},
            )

    statuses = [call.kwargs.get("guardrail_status") for call in mock_log.call_args_list]
    assert statuses == ["guardrail_failed_to_respond"]


@pytest.mark.asyncio
async def test_apply_guardrail_single_item_split_twice_still_yields_one_output_per_item():
    """One oversized content item that needs two levels of text bisection ends up
    as four text fragments, and all four must still collapse back into exactly
    ONE output entry, because they all came from one original content item.

    Downstream masking (`_apply_masking_to_messages`) walks the merged outputs by
    a running index across the original, unchunked message list, so emitting more
    than one entry for a single message shifts every later message's masked text
    onto the wrong message and drops the surplus. Regression for fragment
    grouping assuming fragments only ever arrive as adjacent sibling *pairs*,
    which holds for one bisection level but not for two."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [{"role": "user", "content": "aaaa bbbb cccc dddd eeee ffff gggg hhhh"}]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    responses = [
        _too_large_validation_httpx_response(),  # whole single item
        _too_large_validation_httpx_response(),  # first half
        _passing_bedrock_httpx_response("q1"),
        _passing_bedrock_httpx_response("q2"),
        _too_large_validation_httpx_response(),  # second half
        _passing_bedrock_httpx_response("q3"),
        _passing_bedrock_httpx_response("q4"),
    ]

    async def _post_side_effect(*_args, **_kwargs):
        return responses.pop(0)

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = _post_side_effect

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert mock_post.await_count == 7
    assert not responses
    assert result.get("action") == "NONE"
    output_texts = [o.get("text") for o in result.get("outputs") or []]
    assert output_texts == ["q1q2q3q4"]


@pytest.mark.asyncio
async def test_apply_guardrail_chunk_retries_after_throttling_then_succeeds():
    """A chunk call throttled with a 429 must be retried with backoff and
    eventually succeed, rather than surfacing the 429 to the caller."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {"role": "user", "content": "chunk one text"},
        {"role": "user", "content": "chunk two text"},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _too_large_validation_httpx_response()
        if call_count == 2:
            return _throttling_httpx_response()
        if call_count == 3:
            return _passing_bedrock_httpx_response("chunk-1")
        return _passing_bedrock_httpx_response("chunk-2")

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch(
            "litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        mock_post.side_effect = _post_side_effect

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert call_count == 4
    mock_sleep.assert_awaited()
    output_texts = [o.get("text") for o in result.get("outputs") or []]
    assert output_texts == ["chunk-1", "chunk-2"]


@pytest.mark.asyncio
async def test_apply_guardrail_chunking_logs_exactly_once_as_success():
    """A too-large 400 that is recovered by chunking must not leave behind a
    'guardrail_failed_to_respond' telemetry entry for the initial oversized
    attempt: the whole logical request (1 too-large attempt + 2 chunk
    attempts here) must produce exactly one standard-logging entry, and it
    must reflect the eventual success, not the transient too-large failure."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {"role": "user", "content": "chunk one text"},
        {"role": "user", "content": "chunk two text"},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _too_large_validation_httpx_response()
        if call_count == 2:
            return _passing_bedrock_httpx_response("chunk-1")
        return _passing_bedrock_httpx_response("chunk-2")

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch.object(
            guardrail,
            "add_standard_logging_guardrail_information_to_request_data",
        ) as mock_log,
    ):
        mock_post.side_effect = _post_side_effect

        await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert call_count == 3
    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["guardrail_status"] == "success"


@pytest.mark.asyncio
async def test_apply_guardrail_unrecoverable_failure_logs_exactly_once_as_failed():
    """A too-large error that cannot be recovered (chunking disabled by
    contextual grounding) must still log exactly once, as a failure -- not be
    silently dropped by the chunking telemetry consolidation."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {
            "role": "system",
            "content": [{"type": "grounding_source", "text": "reference source text"}],
        },
        {"role": "user", "content": "what does the source say?"},
    ]
    model_response = ModelResponse()
    model_response.choices = [litellm.Choices(message=litellm.Message(content="a grounded answer", role="assistant"))]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch.object(
            guardrail,
            "add_standard_logging_guardrail_information_to_request_data",
        ) as mock_log,
    ):
        mock_post.return_value = _too_large_validation_httpx_response()

        with pytest.raises(HTTPException):
            await guardrail.make_bedrock_api_request(
                source="OUTPUT",
                messages=messages,
                response=model_response,
                request_data={"model": "bedrock-nova-micro"},
            )

    mock_post.assert_awaited_once()
    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["guardrail_status"] == "guardrail_failed_to_respond"


@pytest.mark.asyncio
async def test_apply_guardrail_chunk_merge_preserves_masking_position():
    """An earlier chunk that comes back clean (empty `outputs`) must not
    shift a later chunk's masked text onto the wrong message. Regression for:
    flattening outputs without positional metadata let a later chunk's PII
    redaction get applied to the first message while the actual PII-bearing
    message (in a later chunk) was forwarded unmasked."""
    guardrail = BedrockGuardrail(
        guardrail_name="test-bedrock-guard",
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        disable_exception_on_block=False,
        mask_request_content=True,
    )

    request_data = {
        "model": "bedrock-nova-micro",
        "messages": [
            {"role": "user", "content": "clean chunk with nothing to mask"},
            {"role": "user", "content": "chunk with PII: John Doe"},
        ],
    }

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    def _clean_httpx_response() -> MagicMock:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"action": "NONE", "assessments": []}
        return response

    def _masked_httpx_response() -> MagicMock:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "action": "GUARDRAIL_INTERVENED",
            "outputs": [{"text": "chunk with PII: [NAME]"}],
            "assessments": [
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [{"type": "NAME", "match": "John Doe", "action": "ANONYMIZED"}]
                    }
                }
            ],
        }
        return response

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _too_large_validation_httpx_response()
        if call_count == 2:
            return _clean_httpx_response()
        return _masked_httpx_response()

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = _post_side_effect

        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=request_data,
            call_type="acompletion",
        )

    assert call_count == 3
    updated_messages = request_data["messages"]
    assert updated_messages[0]["content"] == "clean chunk with nothing to mask"
    assert updated_messages[1]["content"] == "chunk with PII: [NAME]"


@pytest.mark.asyncio
async def test_apply_guardrail_accepted_content_costs_exactly_one_call():
    """Content AWS accepts must cost exactly one ApplyGuardrail call, however far over
    the chunk budget it is. Chunking is a recovery path, not something every request
    pays for. Regression for: bin-packing eagerly on every request, which split
    conversations AWS was happy to take whole and multiplied billed calls and guardrail
    latency on traffic that never had a size problem."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    item_text = "x" * (BEDROCK_APPLY_GUARDRAIL_CHUNK_BUDGET_CHARS // 2)
    messages = [{"role": "user", "content": item_text} for _ in range(3)]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return _passing_bedrock_httpx_response(f"batch-{call_count}")

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = _post_side_effect

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert call_count == 1
    assert result.get("action") == "NONE"


@pytest.mark.asyncio
async def test_apply_guardrail_small_content_makes_exactly_one_call():
    """Content that fits entirely within the budget in a single batch must
    make exactly one ApplyGuardrail call -- confirms bin-packing does not
    introduce an extra probe call for the common (small-request) case."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {"role": "user", "content": "short message one"},
        {"role": "user", "content": "short message two"},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.return_value = _passing_bedrock_httpx_response("single-batch")

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    mock_post.assert_awaited_once()
    assert result.get("action") == "NONE"


@pytest.mark.asyncio
async def test_apply_guardrail_batch_under_budget_still_rejected_falls_back_to_bisection():
    """A batch that fits the budget guess but is still rejected by AWS as too large
    (a lower real per-account/region/policy cap) must fall back to bisection for that
    batch only, and any other batch from the same request that AWS already accepted
    must not be re-sent.

    Three half-budget items pack into two batches once the whole-payload probe is
    rejected, so the sequence is probe, batch one (rejected), its two halves, batch
    two."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    item_text = "x" * (BEDROCK_APPLY_GUARDRAIL_CHUNK_BUDGET_CHARS // 2)
    messages = [
        {"role": "user", "content": item_text},
        {"role": "user", "content": item_text},
        {"role": "user", "content": item_text},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count in (1, 2):
            return _too_large_validation_httpx_response()
        return _passing_bedrock_httpx_response(f"chunk-{call_count}")

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = _post_side_effect

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert call_count == 5
    assert result.get("action") == "NONE"
    output_texts = [o.get("text") for o in result.get("outputs") or []]
    assert output_texts == ["chunk-3", "chunk-4", "chunk-5"]


def test_split_bedrock_content_single_item_splits_on_whitespace_not_mid_word():
    """A single content item whose raw character midpoint would fall inside a
    word must instead split at the nearest whitespace, so neither fragment
    ends or begins mid-token. Regression for the Veria AI review finding: a
    denied word/PII pattern straddling a raw character-midpoint cut could be
    truncated on both fragments and scan clean on each, then reassemble into
    the original unmasked text -- a detection bypass."""
    text = ("a" * 20) + " " + ("b" * 30)
    raw_midpoint = len(text) // 2
    assert text[raw_midpoint] == "b"
    content = [BedrockContentItem(text=BedrockTextContent(text=text))]

    split_content = BedrockGuardrail._split_bedrock_content(content)
    assert split_content is not None
    first_half, second_half = split_content

    first_text = first_half[0]["text"]["text"]
    second_text = second_half[0]["text"]["text"]

    assert first_text + second_text == text
    assert first_text == ("a" * 20) + " "
    assert second_text == "b" * 30


def test_split_bedrock_content_single_item_with_no_whitespace_falls_back_to_midpoint():
    """A single giant token with no whitespace anywhere has no safe split
    point, so the split must fall back to the raw character midpoint rather
    than failing or looping."""
    text = "a" * 40
    content = [BedrockContentItem(text=BedrockTextContent(text=text))]

    split_content = BedrockGuardrail._split_bedrock_content(content)
    assert split_content is not None
    first_half, second_half = split_content

    first_text = first_half[0]["text"]["text"]
    second_text = second_half[0]["text"]["text"]
    assert first_text + second_text == text
    assert len(first_text) == 20
    assert len(second_text) == 20


def test_bin_pack_bedrock_content_packs_minimal_batches_within_budget():
    """Many medium items should pack into the minimal number of in-order
    batches that each stay within budget, not one batch per item."""
    items = [BedrockContentItem(text=BedrockTextContent(text="x" * 30)) for _ in range(10)]

    batches = BedrockGuardrail._bin_pack_bedrock_content(items, budget=100)

    assert sum(len(batch) for batch in batches) == 10
    for batch in batches:
        combined_len = sum(len(item["text"]["text"]) for item in batch)
        assert combined_len <= 100
    assert len(batches) == 4


def test_bin_pack_bedrock_content_oversized_single_item_becomes_its_own_batch():
    """An item whose own text already exceeds the budget must not be
    pre-split here -- it becomes its own oversized batch, and only the
    reactive bisection fallback (on an AWS rejection) may split it later."""
    small_item = BedrockContentItem(text=BedrockTextContent(text="short"))
    oversized_item = BedrockContentItem(text=BedrockTextContent(text="x" * 200))
    items = [small_item, oversized_item, small_item]

    batches = BedrockGuardrail._bin_pack_bedrock_content(items, budget=100)

    assert batches == ((small_item,), (oversized_item,), (small_item,))


def test_bin_pack_bedrock_content_empty_content_makes_exactly_one_empty_batch():
    """Empty content must still pack into exactly one (empty) batch, matching
    pre-bin-packing behavior of sending the content list as-is in one call --
    bin-packing must not turn an empty request into zero ApplyGuardrail calls."""
    assert BedrockGuardrail._bin_pack_bedrock_content([], budget=100) == ((),)


@pytest.mark.asyncio
async def test_apply_guardrail_too_large_reported_as_429_bisects_without_burning_retries():
    """AWS reports an oversized ApplyGuardrail request as a 429 ThrottlingException
    (not the documented 400 ValidationException) when the guardrail has an active
    content-filter policy. That is not a transient throttle -- re-posting the same
    oversized content can never succeed -- so it must bisect immediately instead of
    consuming the exponential-backoff retry budget first.

    Regression for a bug found against a live guardrail: because the throttle retry
    only keyed off status 429, every oversized chunk burned all
    _BEDROCK_APPLY_GUARDRAIL_MAX_THROTTLE_RETRIES attempts (each a billed AWS call,
    each preceded by a backoff sleep) before bisection got a chance, at every level
    of the recursion."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    messages = [
        {"role": "user", "content": "chunk one text"},
        {"role": "user", "content": "chunk two text"},
    ]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    call_count = 0

    async def _post_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _too_large_throttling_httpx_response()
        return _passing_bedrock_httpx_response(f"half-{call_count}")

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch(
            "litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        mock_post.side_effect = _post_side_effect

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=messages,
            request_data={"model": "bedrock-nova-micro"},
        )

    assert call_count == 3
    mock_sleep.assert_not_awaited()
    assert result.get("action") == "NONE"
    output_texts = [o.get("text") for o in result.get("outputs") or []]
    assert output_texts == ["half-2", "half-3"]


def test_chunk_budget_defaults_to_apply_guardrail_per_second_quota():
    """The default budget must track ApplyGuardrail's default quota of 25 text units
    (about 1,000 characters each) per second. Packing to that size and posting
    sequentially is what stops chunking from trading a size error for a throttle, so
    this default is a deliberate match to AWS behaviour rather than an arbitrary
    number."""
    assert BEDROCK_APPLY_GUARDRAIL_CHUNK_BUDGET_CHARS == 25_000
    assert BedrockGuardrail(guardrailIdentifier="g", guardrailVersion="DRAFT").chunk_budget_chars == 25_000


@pytest.mark.asyncio
async def test_configured_chunk_budget_changes_how_content_is_packed():
    """An account with raised quotas can set a larger `chunk_budget_chars` and have it
    actually drive packing once AWS has rejected a payload, spending fewer
    ApplyGuardrail calls for the same content instead of being pinned to the
    conservative default.

    Four 20,000-character messages are 80,000 characters total, and every call here is
    preceded by the one whole-payload probe AWS rejects. At the 25,000 default only one
    message fits per batch, so it is the probe plus four; at 50,000 two fit per batch,
    so it is the probe plus two."""
    messages = [{"role": "user", "content": "x" * 20_000} for _ in range(4)]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    async def _calls_made_with_budget(budget: int) -> int:
        guardrail = BedrockGuardrail(
            guardrail_name="test-bedrock-guard",
            guardrailIdentifier="test-guardrail",
            guardrailVersion="DRAFT",
            disable_exception_on_block=False,
            chunk_budget_chars=budget,
        )
        with (
            patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
            patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
            patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        ):
            posted = 0

            async def _post_side_effect(*_args, **_kwargs):
                nonlocal posted
                posted += 1
                if posted == 1:
                    return _too_large_validation_httpx_response()
                return _passing_bedrock_httpx_response("ok")

            mock_post.side_effect = _post_side_effect
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=messages,
                request_data={"model": "bedrock-nova-micro"},
            )
            return mock_post.await_count

    assert await _calls_made_with_budget(BEDROCK_APPLY_GUARDRAIL_CHUNK_BUDGET_CHARS) == 5
    assert await _calls_made_with_budget(50_000) == 3


def test_split_index_never_produces_an_empty_fragment():
    """Both fragments must be non-empty for every splittable text, so bisection always
    makes progress.

    A text whose only qualifying whitespace is its final character is the dangerous
    shape: taking that boundary puts the split at len(text), leaving the first fragment
    identical to the input that was just rejected and the second empty. The recursion
    would then resubmit the unchanged fragment forever and exhaust the stack instead of
    scanning or surfacing Bedrock's error."""
    for text in ("ab ", "xxxx ", ("x" * 40) + " ", " ab", "a b", "ab", "  "):
        split_at = BedrockGuardrail._nearest_whitespace_split_index(text)
        assert 0 < split_at < len(text), f"degenerate split {split_at} for {text!r}"
        assert text[:split_at] and text[split_at:], f"empty fragment for {text!r}"
        assert text[:split_at] + text[split_at:] == text


@pytest.mark.asyncio
async def test_oversized_single_item_with_trailing_space_gives_up_instead_of_recursing():
    """An oversized single item whose only space is trailing must bottom out and
    surface Bedrock's error, not recurse forever.

    AWS is modelled the way it really behaves, rejecting every attempt, because the
    danger is a fragment identical to the input that was just rejected: AWS would
    reject it again, and each retry would split it into the same unchanged fragment.
    A split that always shrinks the text terminates and re-raises; one that can return
    the whole text raises RecursionError instead. The call-count bound is generous:
    halving 41 characters down to unsplittable is a handful of attempts, nowhere near
    a stack limit."""
    guardrail = _bedrock_guardrail_for_chunk_tests()
    messages = [{"role": "user", "content": ("x" * 40) + " "}]

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = lambda *_a, **_k: _too_large_validation_httpx_response()

        with pytest.raises(HTTPException) as excinfo:
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=messages,
                request_data={"model": "bedrock-nova-micro"},
            )

    assert excinfo.value.status_code == 400
    assert mock_post.await_count < 200


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
            await guardrail.apply_guardrail(inputs={"texts": ["q1"]}, request_data=session, input_type="request")
            assert mock_api.call_count == 1
            mock_api.reset_mock()
            await guardrail.apply_guardrail(inputs={"texts": ["q1"]}, request_data=session, input_type="request")
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
            await guardrail.apply_guardrail(inputs={"texts": ["q1"]}, request_data=session, input_type="request")
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, call_type, should_scan",
    [
        ("during_mcp_call", "call_mcp_tool", True),
        ("during_call", "completion", True),
        ("during_mcp_call", "completion", False),
        ("during_call", "call_mcp_tool", False),
    ],
)
async def test_moderation_hook_honors_the_mcp_event_type(mode, call_type, should_scan):
    """A guardrail configured mode: during_mcp_call must actually scan MCP tool calls.

    ProxyLogging.during_call_hook remaps call_mcp_tool to during_mcp_call before
    dispatching, so re-checking during_call here would reject the very requests the
    guardrail was configured for and let the tool call through unscanned.
    """
    guardrail = BedrockGuardrail(
        guardrail_name="bedrock-mcp",
        guardrailIdentifier="gid",
        guardrailVersion="1",
        event_hook=mode,
        default_on=True,
    )
    data = {
        "messages": [{"role": "user", "content": "scan me"}],
        "mcp_tool_name": "search",
        "mcp_arguments": {"query": "scan me"},
    }

    with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = MagicMock(action="NONE", output=[], outputs=[], assessments=[])
        await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="u"),
            call_type=call_type,
        )

    assert (mock_api.call_count == 1) is should_scan, (
        f"mode={mode} call_type={call_type}: expected scan={should_scan}, "
        f"bedrock api called {mock_api.call_count} times"
    )
    if should_scan:
        expected_event = (
            GuardrailEventHooks.during_mcp_call
            if call_type == CallTypes.call_mcp_tool.value
            else GuardrailEventHooks.during_call
        )
        assert mock_api.call_args.kwargs["logging_event_type"] == expected_event, (
            "the scan must be logged under the event it actually ran for, so guardrail logs, "
            "OTel spans, and Langfuse metadata do not misclassify MCP enforcement as an LLM call"
        )


class TestScanOnlyToolResultsWithLatestRoleFilter:
    @pytest.mark.asyncio
    async def test_warns_and_skips_when_scoped_payload_has_no_user_message(self):
        """scan_only_tool_results hands Bedrock a tool-role-only payload, but
        experimental_use_latest_role_message_only scans only the latest user
        message: the silent no-op must warn."""
        guardrail = BedrockGuardrail(
            guardrail_name="bedrock-latest-role-scoped",
            guardrailIdentifier="test-guardrail",
            guardrailVersion="DRAFT",
            default_on=True,
            experimental_use_latest_role_message_only=True,
        )
        guardrail.scan_only_tool_results = True
        inputs = {
            "texts": ["TOOL-RESULT"],
            "structured_messages": [{"role": "tool", "tool_call_id": "call_1", "content": "TOOL-RESULT"}],
        }

        with (
            patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api,
            patch(
                "litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.verbose_proxy_logger.warning"
            ) as mock_warning,
        ):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"litellm_call_id": "test-call-id"},
                input_type="request",
            )

        mock_api.assert_not_called()
        assert result["texts"] == ["TOOL-RESULT"]
        warning_text = " ".join(str(arg) for c in mock_warning.call_args_list for arg in c.args)
        assert "scan_only_tool_results" in warning_text


@pytest.mark.parametrize("separator", ["\n", "\t", "\r\n", "　"])
def test_split_bedrock_content_splits_on_any_whitespace_not_just_space(separator):
    """Regression: the midpoint split must land on any Unicode whitespace, not only an
    ASCII space.

    Matching only " " left the boundary unguarded for exactly the payloads that grow
    large enough to need splitting: JSON lines, source code, logs and transcripts are
    newline or tab delimited. A deny-listed word sitting at the midpoint of one was cut
    in half, scanned clean on both fragments, and reassembled intact, which is the
    single-token detection bypass the whitespace split exists to close."""
    text = separator.join(["aaaaaaa"] * 4) + separator + "BADWORDXYZ" + separator + separator.join(["bbbbbbb"] * 4)

    first, second = BedrockGuardrail._split_bedrock_content([BedrockContentItem(text=BedrockTextContent(text=text))])

    first_text = first[0]["text"]["text"]
    second_text = second[0]["text"]["text"]
    assert first_text + second_text == text, "split must stay lossless"
    assert "BADWORDXYZ" in first_text or "BADWORDXYZ" in second_text, "split severed the token"


def test_merge_bedrock_responses_preserves_fields_the_merge_has_no_opinion_on():
    """Regression: merging must not drop AWS response fields it does not itself merge.

    The merged response used to be rebuilt from an empty dict holding only action,
    outputs, assessments and usage, so actionReason, guardrailCoverage and anything AWS
    adds later vanished from the guardrail_json_response the Admin UI renders, on every
    ApplyGuardrail request rather than only chunked ones."""
    chunk = BedrockContentChunkResult(
        response={
            "action": "NONE",
            "actionReason": "No action.",
            "guardrailCoverage": {"textCharacters": {"guarded": 41, "total": 41}},
            "usage": {"contentPolicyUnits": 1},
        },
        content=[BedrockContentItem(text=BedrockTextContent(text="hello"))],
        fragment_group_size=1,
    )

    merged = BedrockGuardrail._merge_bedrock_guardrail_responses([chunk])

    assert merged["actionReason"] == "No action."
    assert merged["guardrailCoverage"] == {"textCharacters": {"guarded": 41, "total": 41}}


def test_merge_bedrock_usage_sums_counters_not_on_the_known_list():
    """Regression: usage counters were summed from a hardcoded list of six keys, so the
    ones AWS also returns (contentPolicyImageUnits, the automatedReasoning pair) were
    reported as absent no matter what the chunks actually used."""
    chunks = [
        BedrockContentChunkResult(
            response={"action": "NONE", "usage": {"contentPolicyImageUnits": units, "contentPolicyUnits": 1}},
            content=[BedrockContentItem(text=BedrockTextContent(text="x"))],
            fragment_group_size=1,
        )
        for units in (3, 4)
    ]

    usage = BedrockGuardrail._merge_bedrock_guardrail_responses(chunks)["usage"]

    assert usage["contentPolicyImageUnits"] == 7
    assert usage["contentPolicyUnits"] == 2


@pytest.mark.asyncio
async def test_apply_guardrail_exception_inside_200_logs_failure_and_proceeds():
    """Regression: AWS can report a failure inside an HTTP 200 body via Output.__type,
    and that must be logged as guardrail_failed_to_respond rather than success.

    Real AWS does this: an unrecognised operation path on bedrock-runtime answers
    HTTP 200 with {"Output": {"__type": "com.amazon.coral.service#UnknownOperationException"}}.
    Consolidating telemetry had replaced the derived status with a hardcoded "success",
    which reported a failed scan as a clean one. The request itself still proceeds, which
    is the behaviour of the code before chunking existed."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    exception_response = MagicMock()
    exception_response.status_code = 200
    exception_response.json.return_value = {
        "Output": {"__type": "com.amazon.coral.service#UnknownOperationException"},
        "Version": "1.0",
    }
    exception_response.text = json.dumps(exception_response.json.return_value)

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch.object(guardrail, "add_standard_logging_guardrail_information_to_request_data") as mock_log,
    ):
        mock_post.return_value = exception_response

        result = await guardrail.make_bedrock_api_request(
            source="INPUT",
            messages=[{"role": "user", "content": "hello"}],
            request_data={"model": "bedrock-nova-micro"},
        )

    assert result is not None, "the request proceeds, as it did before chunking existed"
    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["guardrail_status"] == "guardrail_failed_to_respond"


@pytest.mark.asyncio
async def test_apply_guardrail_failure_logs_a_dict_not_a_bare_string():
    """Regression: the consolidated failure logger must log guardrail_json_response as a
    dict, the shape the pre-chunking code and the InvokeGuardrailChecks path both use.

    Consolidating telemetry had changed it to a bare string on the ApplyGuardrail path
    only, which breaks any consumer that reads it as a mapping and leaves the two paths
    in this file inconsistent."""
    guardrail = _bedrock_guardrail_for_chunk_tests()

    mock_credentials = MagicMock()
    mock_credentials.access_key = "k"
    mock_credentials.secret_key = "s"
    mock_credentials.token = None

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
        patch.object(guardrail, "add_standard_logging_guardrail_information_to_request_data") as mock_log,
    ):
        mock_post.side_effect = _raised_bedrock_error(400, "guardrailIdentifier is not valid")

        with pytest.raises(HTTPException):
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=[{"role": "user", "content": "hello"}],
                request_data={"model": "bedrock-nova-micro"},
            )

    mock_log.assert_called_once()
    logged = mock_log.call_args.kwargs["guardrail_json_response"]
    assert isinstance(logged, dict), f"expected a dict, got {type(logged).__name__}"
    assert "error" in logged


def test_build_tracing_detail_surfaces_usage_counters_and_cost(monkeypatch):
    """LIT-5650/LIT-5651: AWS-billed usage must land as guardrail_usage priced into guardrail_cost."""
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "bedrock/guardrails": {
                "guardrail_cost_per_unit": {
                    "topicPolicyUnits": 0.00015,
                    "contentPolicyUnits": 0.00015,
                    "wordPolicyUnits": 0.0,
                }
            }
        },
    )
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

    detail = guardrail._build_tracing_detail(
        {
            "action": "GUARDRAIL_INTERVENED",
            "usage": {"topicPolicyUnits": 1, "contentPolicyUnits": 2, "wordPolicyUnits": 0, "oddball": "not-an-int"},
        },
        aws_region_name="us-east-1",
    )

    assert detail["guardrail_usage"] == {"topicPolicyUnits": 1, "contentPolicyUnits": 2, "wordPolicyUnits": 0}
    assert detail["guardrail_cost"] == pytest.approx(0.00045)


def test_build_tracing_detail_omits_guardrail_usage_when_bedrock_reports_none():
    guardrail = BedrockGuardrail(guardrailIdentifier="test-guardrail", guardrailVersion="DRAFT")

    for detail in (
        guardrail._build_tracing_detail({"action": "NONE"}, aws_region_name="us-east-1"),
        guardrail._build_tracing_detail({"action": "NONE", "usage": {}}, aws_region_name="us-east-1"),
    ):
        assert "guardrail_usage" not in detail
        assert "guardrail_cost" not in detail


@pytest.mark.asyncio
async def test_blocked_chunk_logs_usage_and_cost_of_prior_passed_chunks(monkeypatch):
    """LIT-5651 regression: a block on a later chunk must still bill the chunks AWS already processed."""
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "bedrock/guardrails": {
                "guardrail_cost_per_unit": {
                    "contentPolicyUnits": 0.00015,
                    "wordPolicyUnits": 0.0,
                }
            }
        },
    )
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        chunk_budget_chars=40,
    )

    too_large_response = MagicMock()
    too_large_response.status_code = 429
    too_large_response.json.return_value = {
        "message": "Input text size (60 text units) exceeds the maximum allowed (1 text units) for the content filter policy"
    }

    passed_chunk_response = MagicMock()
    passed_chunk_response.status_code = 200
    passed_chunk_response.json.return_value = {
        "action": "NONE",
        "outputs": [],
        "assessments": [],
        "usage": {"contentPolicyUnits": 2, "wordPolicyUnits": 1},
    }

    blocked_chunk_response = MagicMock()
    blocked_chunk_response.status_code = 200
    blocked_chunk_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{"contentPolicy": {"filters": [{"type": "HATE", "confidence": "HIGH", "action": "BLOCKED"}]}}],
        "outputs": [{"text": "Content blocked"}],
        "usage": {"contentPolicyUnits": 3},
    }

    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "a" * 30},
            {"role": "user", "content": "b" * 30},
        ],
    }

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = [too_large_response, passed_chunk_response, blocked_chunk_response]

        with pytest.raises(HTTPException):
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=request_data["messages"],
                request_data=request_data,
            )

    assert mock_post.call_count == 3
    logged_entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert len(logged_entries) == 1
    logged = logged_entries[0]
    assert logged["guardrail_usage"] == {"contentPolicyUnits": 5, "wordPolicyUnits": 1}
    assert logged["guardrail_cost"] == pytest.approx(0.00075)
    assert logged["guardrail_response"]["usage"] == {"contentPolicyUnits": 5, "wordPolicyUnits": 1}


@pytest.mark.asyncio
async def test_terminal_failure_logs_usage_and_cost_of_prior_passed_chunks(monkeypatch):
    """LIT-5651 regression: a terminal failure on a later chunk must still bill the chunks AWS already processed."""
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "bedrock/guardrails": {
                "guardrail_cost_per_unit": {
                    "contentPolicyUnits": 0.00015,
                    "wordPolicyUnits": 0.0,
                }
            }
        },
    )
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        chunk_budget_chars=40,
    )

    too_large_response = MagicMock()
    too_large_response.status_code = 429
    too_large_response.json.return_value = {
        "message": "Input text size (60 text units) exceeds the maximum allowed (1 text units) for the content filter policy"
    }

    passed_chunk_response = MagicMock()
    passed_chunk_response.status_code = 200
    passed_chunk_response.json.return_value = {
        "action": "NONE",
        "outputs": [],
        "assessments": [],
        "usage": {"contentPolicyUnits": 2, "wordPolicyUnits": 1},
    }

    failed_chunk_response = MagicMock()
    failed_chunk_response.status_code = 400
    failed_chunk_response.json.return_value = {"message": "ValidationException: guardrail is in a failed state"}

    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "a" * 30},
            {"role": "user", "content": "b" * 30},
        ],
    }

    with (
        patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post,
        patch.object(guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")),
        patch.object(guardrail, "_prepare_request", return_value=MagicMock()),
    ):
        mock_post.side_effect = [too_large_response, passed_chunk_response, failed_chunk_response]

        with pytest.raises(HTTPException):
            await guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=request_data["messages"],
                request_data=request_data,
            )

    assert mock_post.call_count == 3
    logged_entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert len(logged_entries) == 1
    logged = logged_entries[0]
    assert logged["guardrail_status"] == "guardrail_failed_to_respond"
    assert logged["guardrail_usage"] == {"contentPolicyUnits": 2, "wordPolicyUnits": 1}
    assert logged["guardrail_cost"] == pytest.approx(0.0003)
    assert logged["guardrail_response"]["usage"] == {"contentPolicyUnits": 2, "wordPolicyUnits": 1}
    assert "error" in logged["guardrail_response"]
