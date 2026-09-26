"""
Unit tests for ZTDS LiteLLM Guardrail
Validates 4 Core Protocol Invariants (IETF draft-sibiryakov-ztds-protocol-02)
https://datatracker.ietf.org/doc/draft-sibiryakov-ztds-protocol/
"""

import asyncio
import unittest
from litellm.proxy.guardrails.guardrail_hooks.ztds import ZTDSGuardrail


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockModelResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class TestZTDSLiteLLMGuardrail(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.guardrail = ZTDSGuardrail()

    def test_deterministic_surrogate_tokenization(self):
        """Invariant 2: Identical cleartext entities must receive identical tokens in session."""
        session_id = "test-session-1"
        text = "Contact alice@example.com or write to alice@example.com for secret sk-live12345678901234567890."
        sanitized, token_map = self.guardrail.sanitize_text(text, session_id)

        self.assertNotIn("alice@example.com", sanitized)
        self.assertNotIn("sk-live12345678901234567890", sanitized)
        self.assertIn("[EMAIL_TOKEN_1]", sanitized)
        self.assertIn("[API_SECRET_TOKEN_1]", sanitized)

        # Confirm identical surrogate reuse
        self.assertEqual(sanitized.count("[EMAIL_TOKEN_1]"), 2)

        # Restore test
        restored = self.guardrail.restore_text(sanitized, session_id)
        self.assertEqual(restored, text)

    def test_multi_entity_detection(self):
        """Invariant 1: Zero cleartext egress for emails, cards, phones, and API secrets."""
        session_id = "test-session-2"
        text = "Card 4111-2222-3333-4444 call +1-555-019-2834 server 192.168.1.100"
        sanitized, token_map = self.guardrail.sanitize_text(text, session_id)

        self.assertIn("[CREDIT_CARD_TOKEN_1]", sanitized)
        self.assertIn("[PHONE_TOKEN_1]", sanitized)
        self.assertIn("[IPV4_TOKEN_1]", sanitized)
        self.assertNotIn("4111-2222-3333-4444", sanitized)

    async def test_pre_and_post_call_lifecycle_with_zeroization(self):
        """Invariant 3: Ephemeral RAM Isolation and Theorem 2 RAM zeroization."""
        request_data = {
            "litellm_call_id": "call-101",
            "messages": [
                {"role": "user", "content": "Please verify user bob@enterprise.corp with IBAN DE89370400440532013000"}
            ]
        }

        # 1. Execute pre-call hook
        modified_data = await self.guardrail.async_pre_call_hook(
            user_api_key_dict={},
            cache={},
            data=request_data,
            call_type="chat_completion"
        )

        user_content = modified_data["messages"][0]["content"]
        self.assertNotIn("bob@enterprise.corp", user_content)
        self.assertIn("[EMAIL_TOKEN_1]", user_content)
        self.assertIn("[IBAN_TOKEN_1]", user_content)
        self.assertTrue(modified_data["metadata"]["ztds_sanitized"])

        # Check session table exists in volatile RAM before post-call
        self.assertIn("call-101", self.guardrail._session_maps)

        # 2. Simulate model response that mentions the token
        model_reply = "Verified account for [EMAIL_TOKEN_1] linked to [IBAN_TOKEN_1]."
        response_obj = MockModelResponse(model_reply)

        # 3. Execute post-call hook
        unmasked_response = await self.guardrail.async_post_call_success_hook(
            data=modified_data,
            user_api_key_dict={},
            response=response_obj
        )

        final_text = unmasked_response.choices[0].message.content
        self.assertIn("bob@enterprise.corp", final_text)
        self.assertIn("DE89370400440532013000", final_text)
        self.assertNotIn("[EMAIL_TOKEN_1]", final_text)

        # Invariant 3 / Theorem 2: Session tables MUST be completely zeroized from RAM
        self.assertNotIn("call-101", self.guardrail._session_maps)
        self.assertNotIn("call-101", self.guardrail._entity_maps)


if __name__ == "__main__":
    unittest.main()
