import os
import pytest
from litellm import mock_completion
from unittest.mock import patch

import litellm
from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    OPTIONAL_PresidioPIIMasking,
    PresidioPerRequestConfig,
)
from litellm.types.guardrails import PiiEntityType, PiiAction
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache
from litellm.exceptions import BlockedPiiEntityError


@pytest.mark.asyncio
async def test_presidio_with_blocked_entities():
    """Test for Presidio guardrail with blocked entities - requires actual Presidio API"""
    # Setup the guardrail with specific entities config - BLOCK for credit card
    litellm.turn_on_debug()
    pii_entities_config = {
        PiiEntityType.CREDIT_CARD: PiiAction.BLOCK,  # This entity should cause a block
        PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,  # This entity should be masked
    }

    presidio_guardrail = OPTIONAL_PresidioPIIMasking(
        pii_entities_config=pii_entities_config,
        presidio_analyzer_api_base=os.environ.get("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.environ.get("PRESIDIO_ANONYMIZER_API_BASE"),
    )

    # Test text with blocked PII type
    test_text = (
        "My credit card number is 4111-1111-1111-1111 and my email is test@example.com"
    )

    # Verify the analyze request configuration
    analyze_request = presidio_guardrail._get_presidio_analyze_request_payload(
        text=test_text, presidio_config=None, request_data={}
    )

    # Verify entities were passed correctly
    assert "entities" in analyze_request
    assert set(analyze_request["entities"]) == set(pii_entities_config.keys())

    # Test that BlockedPiiEntityError is raised when check_pii is called
    with pytest.raises(BlockedPiiEntityError) as excinfo:
        await presidio_guardrail.check_pii(
            text=test_text, output_parse_pii=True, presidio_config=None, request_data={}
        )

    # Verify the error contains the correct entity type
    assert excinfo.value.entity_type == PiiEntityType.CREDIT_CARD
    assert excinfo.value.guardrail_name == presidio_guardrail.guardrail_name


@pytest.mark.asyncio
async def test_presidio_pre_call_hook_with_blocked_entities():
    """Test for Presidio guardrail pre-call hook with blocked entities on a chat completion request"""
    # Setup the guardrail with specific entities config
    pii_entities_config = {
        PiiEntityType.CREDIT_CARD: PiiAction.BLOCK,  # This entity should cause a block
        PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,  # This entity should be masked
    }

    presidio_guardrail = OPTIONAL_PresidioPIIMasking(
        pii_entities_config=pii_entities_config,
        presidio_analyzer_api_base=os.environ.get("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.environ.get("PRESIDIO_ANONYMIZER_API_BASE"),
    )

    # Create a sample chat completion request with PII data
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "My credit card is 4111-1111-1111-1111 and my email is test@example.com.",
            },
        ],
        "model": "gpt-5-mini",
    }

    # Mock objects needed for the pre-call hook
    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
    cache = DualCache()

    # Call the pre-call hook and expect BlockedPiiEntityError
    with pytest.raises(BlockedPiiEntityError) as excinfo:
        await presidio_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="completion",
        )

    print(f"got error: {excinfo}")

    # Verify the error contains the correct entity type
    assert excinfo.value.entity_type == PiiEntityType.CREDIT_CARD
    assert excinfo.value.guardrail_name == presidio_guardrail.guardrail_name






# asyncio.run(test_output_parsing())


### UNIT TESTS FOR PRESIDIO PII MASKING ###

input_a_anonymizer_results = {
    "text": "hello world, my name is <PERSON>. My number is: <PHONE_NUMBER>",
    "items": [
        {
            "start": 48,
            "end": 62,
            "entity_type": "PHONE_NUMBER",
            "text": "<PHONE_NUMBER>",
            "operator": "replace",
        },
        {
            "start": 24,
            "end": 32,
            "entity_type": "PERSON",
            "text": "<PERSON>",
            "operator": "replace",
        },
    ],
}

input_b_anonymizer_results = {
    "text": "My name is <PERSON>, who are you? Say my name in your response",
    "items": [
        {
            "start": 11,
            "end": 19,
            "entity_type": "PERSON",
            "text": "<PERSON>",
            "operator": "replace",
        }
    ],
}


#   Test if PII masking works with input A


#   Test if PII masking works with input B (also test if the response != A's response)




@pytest.mark.asyncio
@patch.dict(
    os.environ,
    {
        "PRESIDIO_ANALYZER_API_BASE": "http://localhost:5002",
        "PRESIDIO_ANONYMIZER_API_BASE": "http://localhost:5001",
    },
)
async def test_presidio_pii_masking_logging_output_only_logged_response_guardrails_config():
    from typing import Dict, List, Optional

    import litellm
    from litellm.proxy.guardrails.init_guardrails import initialize_guardrails
    from litellm.types.guardrails import (
        GuardrailItemSpec,
        GuardrailEventHooks,
    )

    litellm.set_verbose = True
    # Environment variables are now patched via the decorator instead of setting them directly

    guardrails_config: List[Dict[str, GuardrailItemSpec]] = [
        {
            "pii_masking": {
                "callbacks": ["presidio"],
                "default_on": True,
                "logging_only": True,
            }
        }
    ]
    litellm_settings = {"guardrails": guardrails_config}

    assert len(litellm.guardrail_name_config_map) == 0
    initialize_guardrails(
        guardrails_config=guardrails_config,
        premium_user=True,
        config_file_path="",
        litellm_settings=litellm_settings,
    )

    assert len(litellm.guardrail_name_config_map) == 1

    pii_masking_obj: Optional[OPTIONAL_PresidioPIIMasking] = None
    for callback in litellm.callbacks:
        print(f"CALLBACK: {callback}")
        if isinstance(callback, OPTIONAL_PresidioPIIMasking):
            pii_masking_obj = callback

    assert pii_masking_obj is not None

    assert hasattr(pii_masking_obj, "logging_only")
    assert pii_masking_obj.event_hook == GuardrailEventHooks.logging_only

    assert pii_masking_obj.should_run_guardrail(
        data={}, event_type=GuardrailEventHooks.logging_only
    )
