import sys
import os
import io, asyncio
import pytest
sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.types.guardrails import PiiAction, PiiEntityType
from litellm.proxy.guardrails.guardrail_hooks.litellm_pii import LitellmPIIGuardrail


def test_litellm_pii_phone_number():
    """
    Test MASKING a phone number
    """
    litellm_pii_guardrail = LitellmPIIGuardrail(
        entities_config={
            PiiEntityType.PHONE_NUMBER: PiiAction.MASK
        }
    )
    results = litellm_pii_guardrail.analyze_and_handle_pii(
        text="My phone number is 212-555-5555"
    )
    print(results)

    assert results == "My phone number is <PHONE_NUMBER>"



def test_litellm_pii_mask_multiple_entities():
    """
    Test MASKING the following entities:
    - Email Address
    - Person
    - Location
    - URL
    - Credit Card
    """
    litellm_pii_guardrail = LitellmPIIGuardrail(
        entities_config={
            PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,
            PiiEntityType.PERSON: PiiAction.MASK,
            PiiEntityType.LOCATION: PiiAction.MASK,
            PiiEntityType.URL: PiiAction.MASK,
            PiiEntityType.CREDIT_CARD: PiiAction.MASK,
        }
    )
    results = litellm_pii_guardrail.analyze_and_handle_pii(
        text="My email address is test@test.com. My name is John Doe. My location is New York. My URL is https://www.test.com. My credit card is 1234-5678-9012-3456"
    )
    print(results)

    assert results == "My email address is <EMAIL_ADDRESS>. My name is <PERSON>. My location is <LOCATION>. My URL is <URL>. My credit card is <CREDIT_CARD>"



def test_litellm_pii_block_multiple_entities():
    """
    Test BLOCKING the following entities:
    - Email Address
    - Person
    - Location
    """
    litellm_pii_guardrail = LitellmPIIGuardrail(
        entities_config={
            PiiEntityType.EMAIL_ADDRESS: PiiAction.BLOCK,
            PiiEntityType.PERSON: PiiAction.BLOCK,
            PiiEntityType.LOCATION: PiiAction.BLOCK,
            PiiEntityType.URL: PiiAction.BLOCK,
            PiiEntityType.CREDIT_CARD: PiiAction.BLOCK,
        }
    )
    with pytest.raises(Exception) as e:
        results = litellm_pii_guardrail.analyze_and_handle_pii(
            text="My email address is test@test.com. My name is John Doe. My location is New York. My URL is https://www.test.com. My credit card is 1234-5678-9012-3456"
        )
        print(results)
    assert "PII detected" in str(e.value)