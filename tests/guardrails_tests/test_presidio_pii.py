import sys
import os
import io, asyncio
import pytest
import time
from litellm import mock_completion
from unittest.mock import MagicMock, AsyncMock, patch
sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking, PresidioPerRequestConfig
from litellm.types.guardrails import PiiEntityType, PiiAction
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache
from litellm.exceptions import BlockedPiiEntityError
from litellm.types.utils import CallTypes as LitellmCallTypes




@pytest.mark.asyncio
async def test_presidio_with_entities_config():
    """Test for Presidio guardrail with entities config - requires actual Presidio API"""
    # Setup the guardrail with specific entities config
    litellm._turn_on_debug()
    pii_entities_config = {
        PiiEntityType.CREDIT_CARD: PiiAction.MASK,
        PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,
    }
    
    presidio_guardrail = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config=pii_entities_config,
        presidio_analyzer_api_base=os.environ.get("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.environ.get("PRESIDIO_ANONYMIZER_API_BASE")
    )
    
    # Test text with different PII types
    test_text = "My credit card number is 4111-1111-1111-1111, my email is test@example.com, and my phone is 555-123-4567"
    
    # Test the analyze request configuration
    analyze_request = presidio_guardrail._get_presidio_analyze_request_payload(
        text=test_text,
        presidio_config=None,
        request_data={}
    )
    
    # Verify entities were passed correctly
    assert "entities" in analyze_request
    assert set(analyze_request["entities"]) == set(pii_entities_config.keys())
    
    # Test the check_pii method - this will call the actual Presidio API
    redacted_text = await presidio_guardrail.check_pii(
        text=test_text,
        output_parse_pii=True,
        presidio_config=None,
        request_data={}
    )
    
    # Verify PII has been masked/replaced/redacted in the result
    assert "4111-1111-1111-1111" not in redacted_text
    assert "test@example.com" not in redacted_text

    # Since this entity is not in the config, it should not be masked
    assert "555-123-4567" in redacted_text
    
    # The specific replacements will vary based on Presidio's implementation
    print(f"Redacted text: {redacted_text}")


@pytest.mark.asyncio
async def test_presidio_apply_guardrail():
    """Test for Presidio guardrail apply guardrail - requires actual Presidio API"""
    litellm._turn_on_debug()
    presidio_guardrail = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config={},
        presidio_analyzer_api_base=os.environ.get("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.environ.get("PRESIDIO_ANONYMIZER_API_BASE")
    )

    test_text = "My credit card number is 4111-1111-1111-1111 and my email is test@example.com"
    response = await presidio_guardrail.apply_guardrail(
        inputs={"texts": [test_text]},
        request_data={},
        input_type="request",
    )
    print("response from apply guardrail for presidio: ", response)

    # Extract the modified text from the response
    modified_text = response["texts"][0] if response.get("texts") else ""

    # assert the default config masks the credit card and email
    assert "4111-1111-1111-1111" not in modified_text
    assert "test@example.com" not in modified_text

@pytest.mark.asyncio
async def test_presidio_with_blocked_entities():
    """Test for Presidio guardrail with blocked entities - requires actual Presidio API"""
    # Setup the guardrail with specific entities config - BLOCK for credit card
    litellm._turn_on_debug()
    pii_entities_config = {
        PiiEntityType.CREDIT_CARD: PiiAction.BLOCK,  # This entity should cause a block
        PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,  # This entity should be masked
    }
    
    presidio_guardrail = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config=pii_entities_config,
        presidio_analyzer_api_base=os.environ.get("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.environ.get("PRESIDIO_ANONYMIZER_API_BASE")
    )
    
    # Test text with blocked PII type
    test_text = "My credit card number is 4111-1111-1111-1111 and my email is test@example.com"
    
    # Verify the analyze request configuration
    analyze_request = presidio_guardrail._get_presidio_analyze_request_payload(
        text=test_text,
        presidio_config=None,
        request_data={}
    )
    
    # Verify entities were passed correctly
    assert "entities" in analyze_request
    assert set(analyze_request["entities"]) == set(pii_entities_config.keys())
    
    # Test that BlockedPiiEntityError is raised when check_pii is called
    with pytest.raises(BlockedPiiEntityError) as excinfo:
        await presidio_guardrail.check_pii(
            text=test_text,
            output_parse_pii=True,
            presidio_config=None,
            request_data={}
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
    
    presidio_guardrail = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config=pii_entities_config,
        presidio_analyzer_api_base=os.environ.get("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.environ.get("PRESIDIO_ANONYMIZER_API_BASE")
    )
    
    # Create a sample chat completion request with PII data
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My credit card is 4111-1111-1111-1111 and my email is test@example.com."}
        ],
        "model": "gpt-3.5-turbo"
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
            call_type="completion"
        )
    
    print(f"got error: {excinfo}")
    
    # Verify the error contains the correct entity type
    assert excinfo.value.entity_type == PiiEntityType.CREDIT_CARD
    assert excinfo.value.guardrail_name == presidio_guardrail.guardrail_name


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["completion", "acompletion"])
async def test_presidio_pre_call_hook_with_different_call_types(call_type):
    """Test for Presidio guardrail pre-call hook with both completion and acompletion call types"""
    # Setup the guardrail with specific entities config
    pii_entities_config = {
        PiiEntityType.CREDIT_CARD: PiiAction.MASK,
        PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,
    }
    
    presidio_guardrail = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config=pii_entities_config,
        presidio_analyzer_api_base=os.environ.get("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.environ.get("PRESIDIO_ANONYMIZER_API_BASE")
    )
    
    # Create a sample request with PII data
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My credit card is 4111-1111-1111-1111 and my email is test@example.com. My phone number is 555-123-4567"}
        ],
        "model": "gpt-3.5-turbo"
    }
    
    # Mock objects needed for the pre-call hook
    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
    cache = DualCache()
    
    # Call the pre-call hook with the specified call type
    modified_data = await presidio_guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type=call_type
    )
    
    # Verify the messages have been modified to mask PII
    assert modified_data["messages"][0]["content"] == "You are a helpful assistant."  # System prompt should be unchanged
    
    user_message = modified_data["messages"][1]["content"]
    assert "4111-1111-1111-1111" not in user_message
    assert "test@example.com" not in user_message

    # Since this entity is not in the config, it should not be masked
    assert "555-123-4567" in user_message
    
    print(f"Modified user message for call_type={call_type}: {user_message}")


@pytest.mark.parametrize(
    "base_url",
    [
        "presidio-analyzer-s3pa:10000",
        "https://presidio-analyzer-s3pa:10000",
        "http://presidio-analyzer-s3pa:10000",
    ],
)
def test_validate_environment_missing_http(base_url):
    pii_masking = _OPTIONAL_PresidioPIIMasking(mock_testing=True)

    # Use patch.dict to temporarily modify environment variables only for this test
    env_vars = {
        "PRESIDIO_ANALYZER_API_BASE": f"{base_url}/analyze",
        "PRESIDIO_ANONYMIZER_API_BASE": f"{base_url}/anonymize"
    }
    with patch.dict(os.environ, env_vars):
        pii_masking.validate_environment()

        expected_url = base_url
        if not (base_url.startswith("https://") or base_url.startswith("http://")):
            expected_url = "http://" + base_url

        assert (
            pii_masking.presidio_anonymizer_api_base == f"{expected_url}/anonymize/"
        ), "Got={}, Expected={}".format(
            pii_masking.presidio_anonymizer_api_base, f"{expected_url}/anonymize/"
        )
        assert pii_masking.presidio_analyzer_api_base == f"{expected_url}/analyze/"


@pytest.mark.asyncio
async def test_output_parsing():
    """
    - have presidio pii masking - mask an input message
    - make llm completion call
    - have presidio pii masking - output parse message
    - assert that no masked tokens are in the input message
    """
    litellm.set_verbose = True
    litellm.output_parse_pii = True
    pii_masking = _OPTIONAL_PresidioPIIMasking(mock_testing=True)

    initial_message = [
        {
            "role": "user",
            "content": "hello world, my name is Jane Doe. My number is: 034453334",
        }
    ]

    filtered_message = [
        {
            "role": "user",
            "content": "hello world, my name is <PERSON>. My number is: <PHONE_NUMBER>",
        }
    ]

    pii_masking.pii_tokens = {"<PERSON>": "Jane Doe", "<PHONE_NUMBER>": "034453334"}

    response = mock_completion(
        model="gpt-3.5-turbo",
        messages=filtered_message,
        mock_response="Hello <PERSON>! How can I assist you today?",
    )
    new_response = await pii_masking.async_post_call_success_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        data={
            "messages": [{"role": "system", "content": "You are an helpfull assistant"}]
        },
        response=response,
    )

    assert (
        new_response.choices[0].message.content
        == "Hello Jane Doe! How can I assist you today?"
    )


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
@pytest.mark.asyncio
async def test_presidio_pii_masking_input_a():
    """
    Tests to see if correct parts of sentence anonymized
    """
    pii_masking = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True, mock_redacted_text=input_a_anonymizer_results
    )

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    new_data = await pii_masking.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={
            "messages": [
                {
                    "role": "user",
                    "content": "hello world, my name is Jane Doe. My number is: 23r323r23r2wwkl",
                }
            ]
        },
        call_type="completion",
    )

    assert "<PERSON>" in new_data["messages"][0]["content"]
    assert "<PHONE_NUMBER>" in new_data["messages"][0]["content"]


#   Test if PII masking works with input B (also test if the response != A's response)
@pytest.mark.asyncio
async def test_presidio_pii_masking_input_b():
    """
    Tests to see if correct parts of sentence anonymized
    """
    pii_masking = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True, mock_redacted_text=input_b_anonymizer_results
    )

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    new_data = await pii_masking.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={
            "messages": [
                {
                    "role": "user",
                    "content": "My name is Jane Doe, who are you? Say my name in your response",
                }
            ]
        },
        call_type="completion",
    )

    assert "<PERSON>" in new_data["messages"][0]["content"]
    assert "<PHONE_NUMBER>" not in new_data["messages"][0]["content"]


@pytest.mark.asyncio
async def test_presidio_pii_masking_logging_output_only_no_pre_api_hook():
    from litellm.types.guardrails import GuardrailEventHooks

    pii_masking = _OPTIONAL_PresidioPIIMasking(
        logging_only=True,
        mock_testing=True,
        mock_redacted_text=input_b_anonymizer_results,
    )

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    test_messages = [
        {
            "role": "user",
            "content": "My name is Jane Doe, who are you? Say my name in your response",
        }
    ]

    assert (
        pii_masking.should_run_guardrail(
            data={"messages": test_messages},
            event_type=GuardrailEventHooks.pre_call,
        )
        is False
    )


@pytest.mark.asyncio
@patch.dict(os.environ, {
    "PRESIDIO_ANALYZER_API_BASE": "http://localhost:5002",
    "PRESIDIO_ANONYMIZER_API_BASE": "http://localhost:5001"
})
async def test_presidio_pii_masking_logging_output_only_logged_response_guardrails_config():
    from typing import Dict, List, Optional

    import litellm
    from litellm.proxy.guardrails.init_guardrails import initialize_guardrails
    from litellm.types.guardrails import (
        GuardrailItem,
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

    pii_masking_obj: Optional[_OPTIONAL_PresidioPIIMasking] = None
    for callback in litellm.callbacks:
        print(f"CALLBACK: {callback}")
        if isinstance(callback, _OPTIONAL_PresidioPIIMasking):
            pii_masking_obj = callback

    assert pii_masking_obj is not None

    assert hasattr(pii_masking_obj, "logging_only")
    assert pii_masking_obj.event_hook == GuardrailEventHooks.logging_only

    assert pii_masking_obj.should_run_guardrail(
        data={}, event_type=GuardrailEventHooks.logging_only
    )


@pytest.mark.asyncio
async def test_presidio_language_configuration():
    """Test that presidio_language parameter is properly set and used in analyze requests"""
    litellm._turn_on_debug()
    
    # Test with German language using mock testing to avoid API calls
    presidio_guardrail_de = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config={},
        presidio_language="de",
        mock_testing=True  # This bypasses the API validation
    )
    
    test_text = "Meine Telefonnummer ist +49 30 12345678"
    
    # Test the analyze request configuration
    analyze_request = presidio_guardrail_de._get_presidio_analyze_request_payload(
        text=test_text,
        presidio_config=None,
        request_data={}
    )
    
    # Verify the language is set to German
    assert analyze_request["language"] == "de"
    assert analyze_request["text"] == test_text
    
    # Test with Spanish language
    presidio_guardrail_es = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config={},
        presidio_language="es",
        mock_testing=True
    )
    
    test_text_es = "Mi número de teléfono es +34 912 345 678"
    
    analyze_request_es = presidio_guardrail_es._get_presidio_analyze_request_payload(
        text=test_text_es,
        presidio_config=None,
        request_data={}
    )
    
    # Verify the language is set to Spanish
    assert analyze_request_es["language"] == "es"
    assert analyze_request_es["text"] == test_text_es
    
    # Test default language (English) when not specified
    presidio_guardrail_default = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config={},
        mock_testing=True
    )
    
    test_text_en = "My phone number is +1 555-123-4567"
    
    analyze_request_default = presidio_guardrail_default._get_presidio_analyze_request_payload(
        text=test_text_en,
        presidio_config=None,
        request_data={}
    )
    
    # Verify the language defaults to English
    assert analyze_request_default["language"] == "en"
    assert analyze_request_default["text"] == test_text_en


@pytest.mark.asyncio
async def test_presidio_language_configuration_with_per_request_override():
    """Test that per-request language configuration overrides the default configured language"""
    litellm._turn_on_debug()
    
    # Set up guardrail with German as default language
    presidio_guardrail = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config={},
        presidio_language="de",
        mock_testing=True
    )
    
    test_text = "Test text with PII"
    
    # Test with per-request config overriding the default language
    presidio_config = PresidioPerRequestConfig(language="fr")
    
    analyze_request = presidio_guardrail._get_presidio_analyze_request_payload(
        text=test_text,
        presidio_config=presidio_config,
        request_data={}
    )
    
    # Verify the per-request language (French) overrides the default (German)
    assert analyze_request["language"] == "fr"
    assert analyze_request["text"] == test_text
    
    # Test without per-request config - should use default language
    analyze_request_default = presidio_guardrail._get_presidio_analyze_request_payload(
        text=test_text,
        presidio_config=None,
        request_data={}
    )
    
    # Verify the default language (German) is used
    assert analyze_request_default["language"] == "de"
    assert analyze_request_default["text"] == test_text
