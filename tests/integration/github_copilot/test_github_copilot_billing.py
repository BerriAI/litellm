"""
GitHub Copilot Premium Request Billing Tests

Tests BILLING BEHAVIOR (not just response correctness) using real GitHub Copilot API.
Requires manual verification via GitHub dashboard for premium request counts.

Per PITFALLS.md: "Testing response correctness instead of billing correctness"
is a known anti-pattern. These tests verify actual premium consumption.

Requirements: INVST-01, INVST-02, INVST-03
Decisions: D-05 through D-11 from 01-CONTEXT.md
"""

import os
from typing import Dict, List

import pytest

from litellm import completion

# Skip all tests if API key not available (D-05)
pytestmark = pytest.mark.skipif(
    not os.environ.get("GITHUB_COPILOT_API_KEY"),
    reason="Requires GitHub Copilot API access for billing verification",
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get the current weather for a location. "
            "MUST be called when asked about weather."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. 'San Francisco, CA'",
                },
            },
            "required": ["location"],
        },
    },
}


class TestGitHubCopilotPremiumBilling:
    """
    Test actual premium request consumption with real GitHub Copilot API.

    IMPORTANT: These tests verify BILLING BEHAVIOR, not just response correctness.
    Manual verification via GitHub dashboard required (no programmatic billing API).

    How to use:
    1. Before running: note current premium request count at
       GitHub Settings -> Copilot -> Usage & billing
    2. Run tests with:
       GITHUB_COPILOT_API_KEY=<key> LITELLM_LOG=DEBUG pytest <this file> -v -s
    3. After running: note new premium request count and calculate delta
    4. Expected delta (after fix): 1 premium request per test, not 1 per turn
    """

    def test_baseline_single_turn_chat_api(self):
        """Baseline: Single user message consumes 1 premium request (D-10 pattern 1)"""
        print("\n" + "=" * 60)
        print("TEST: Baseline single-turn Chat API")
        print("=" * 60)

        response = completion(
            model="github_copilot/gpt-4",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert response is not None
        assert response.choices[0].message.content

        # Manual verification instructions (D-06)
        print("\nBaseline test complete.")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  1. Visit GitHub Settings -> Copilot -> Usage & billing")
        print("  2. Check premium request count")
        print("  3. Expected: +1 premium request from this test")
        print("=" * 60 + "\n")

    def test_multi_turn_conversation_chat_api(self):
        """Multi-turn conversation consumes 1 premium request (D-07, D-10 pattern 1)"""
        print("\n" + "=" * 60)
        print("TEST: Multi-turn conversation Chat API (FIXED: 1 premium request)")
        print("=" * 60)

        # Turn 1: Initial user message (should consume 1 premium request)
        messages: List[Dict] = [{"role": "user", "content": "What is 2+2?"}]
        response1 = completion(model="github_copilot/gpt-4", messages=messages)

        # Turn 2: Assistant response + follow-up (should NOT consume premium request)
        messages.append(
            {"role": "assistant", "content": response1.choices[0].message.content}
        )
        messages.append({"role": "user", "content": "And what is 3+3?"})
        response2 = completion(model="github_copilot/gpt-4", messages=messages)

        # Turn 3: Another follow-up (should NOT consume premium request)
        messages.append(
            {"role": "assistant", "content": response2.choices[0].message.content}
        )
        messages.append({"role": "user", "content": "And what is 4+4?"})
        response3 = completion(model="github_copilot/gpt-4", messages=messages)

        assert response3 is not None

        # Manual verification (D-06)
        print("\nMulti-turn test complete.")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  Turns executed: 3")
        print("  Expected premium requests: 1 (only from Turn 1)")
        print(
            "  Fixed behavior (X-Initiator fix): 1 premium "
            "request for all 3 turns"
        )
        print("  Visit: GitHub Settings -> Copilot -> Usage & billing")
        print("=" * 60 + "\n")

    def test_long_conversation_simulating_plan_mode(self):
        """10+ turn conversation simulating agent plan mode (D-07, D-10 pattern 4)"""
        print("\n" + "=" * 60)
        print("TEST: Long conversation (10+ turns) simulating plan mode")
        print("=" * 60)

        messages: List[Dict] = []

        # Initial user request (should consume 1 premium request)
        messages.append(
            {"role": "user", "content": "Help me plan a project with 5 tasks"}
        )
        response = completion(model="github_copilot/gpt-4", messages=messages)

        # Simulate 10 turns of agent planning (should NOT consume premium requests)
        for i in range(10):
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            messages.append({"role": "user", "content": f"Continue with task {i + 1}"})
            response = completion(model="github_copilot/gpt-4", messages=messages)

        assert response is not None

        # Manual verification (D-06)
        print("\nLong conversation test complete.")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  Turns executed: 11 (1 initial + 10 follow-ups)")
        print("  Expected premium requests: 1 (only from initial turn)")
        print(
            "  Fixed behavior (X-Initiator fix applied): "
            "1 premium request for all 11 turns"
        )
        print("  Visit: GitHub Settings -> Copilot -> Usage & billing")
        print("=" * 60 + "\n")

    def test_tool_calls_billing_chat_api(self):
        """
        TEST-04: Tool calls consume 1 premium request for full 2-turn exchange.

        Uses tool_choice='required' to force a tool call on Turn 1, making the test
        deterministic. Hard-asserts that Turn 1 returns a tool call — fails loudly
        instead of silently passing when the model answers directly.
        Expected: premium delta = 1 (Turn 1 only). Turn 2 (tool response) is agent-free.
        """
        print("\n" + "=" * 60)
        print("RECORD DASHBOARD VALUE BEFORE THIS TEST")
        print("TEST: test_tool_calls_billing_chat_api (TEST-04)")
        print("=" * 60)

        # Turn 1: User forces tool use via tool_choice="required"
        messages: List[Dict] = [
            {
                "role": "user",
                "content": "You MUST use the get_weather tool to answer: What is the current weather in San Francisco, CA?",
            }
        ]
        response1 = completion(
            model="github_copilot/gpt-4o",
            messages=messages,
            tools=[WEATHER_TOOL],
            tool_choice="required",
        )

        # Hard assertion — no silent pass if model ignores tool_choice
        assert (
            response1.choices[0].message.tool_calls is not None
            and len(response1.choices[0].message.tool_calls) > 0
        ), (
            "Turn 1 must return a tool call (tool_choice='required'). "
            "If GitHub Copilot API does not support tool_choice, update this test."
        )

        # Turn 2: Provide tool result (agent-initiated, no premium request)
        messages.append(response1.choices[0].message.model_dump())
        messages.append(
            {
                "role": "tool",
                "content": '{"temp": 65, "condition": "Cloudy"}',
                "tool_call_id": response1.choices[0].message.tool_calls[0].id,
            }
        )
        response2 = completion(model="github_copilot/gpt-4o", messages=messages)

        assert response2 is not None

        print(f"\nTest complete: 2 turns executed")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  Expected premium request delta: 1")
        print("  (Before count - after count should equal 1)")
        print("  Visit: https://github.com/settings/copilot — Usage & billing tab")
        print("=" * 60 + "\n")

    def test_baseline_single_turn_responses_api(self):
        """Baseline: Single user message should consume 1 premium request (Responses API, D-09)"""
        print("\n" + "=" * 60)
        print("TEST: Baseline single-turn Responses API (gpt-5.2)")
        print("=" * 60)

        response = completion(
            model="github_copilot/gpt-5.2",
            messages=[
                {"role": "user", "content": "Write a function to add two numbers"}
            ],
        )

        assert response is not None
        assert response.choices[0].message.content

        print("\nBaseline test complete (Responses API).")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  1. Visit GitHub Settings -> Copilot -> Usage & billing")
        print("  2. Check premium request count")
        print("  3. Expected: +1 premium request from this test")
        print("=" * 60 + "\n")

    def test_multi_turn_conversation_responses_api(self):
        """Multi-turn with Responses API should consume only 1 premium request (D-09, D-10 pattern 3)"""
        print("\n" + "=" * 60)
        print("TEST: Multi-turn conversation Responses API (FIXED: 1 premium request)")
        print("=" * 60)

        # Turn 1: Initial user message
        messages: List[Dict] = [
            {"role": "user", "content": "Write a function to add two numbers"}
        ]
        response1 = completion(model="github_copilot/gpt-5.2", messages=messages)

        # Turn 2: Follow-up building on previous response
        messages.append(
            {"role": "assistant", "content": response1.choices[0].message.content}
        )
        messages.append({"role": "user", "content": "Now add error handling"})
        response2 = completion(model="github_copilot/gpt-5.2", messages=messages)

        # Turn 3: Another follow-up
        messages.append(
            {"role": "assistant", "content": response2.choices[0].message.content}
        )
        messages.append({"role": "user", "content": "Add type hints"})
        response3 = completion(model="github_copilot/gpt-5.2", messages=messages)

        assert response3 is not None

        print("\nMulti-turn test complete (Responses API).")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  Turns executed: 3")
        print("  Expected premium requests: 1 (only from Turn 1)")
        print(
            "  Fixed behavior (X-Initiator fix applied): 1 premium request for all 3 turns"
        )
        print("  NOTE: Responses API may include encrypted_content in reasoning items")
        print("  Visit: GitHub Settings -> Copilot -> Usage & billing")
        print("=" * 60 + "\n")

    def test_responses_api_with_reasoning_items(self):
        """Verify encrypted_content field preservation in reasoning items (D-10 pattern 3, INVST-02)"""
        print("\n" + "=" * 60)
        print("TEST: Responses API reasoning items and encrypted_content")
        print("=" * 60)

        # Initial request
        messages: List[Dict] = [
            {"role": "user", "content": "Explain how quicksort works"}
        ]
        response1 = completion(model="github_copilot/gpt-5.2", messages=messages)

        # Check if response includes reasoning items (logged via diagnostic logging from Plan 01)
        # This test relies on LITELLM_LOG=DEBUG to show encrypted_content in logs
        print("  Turn 1 complete - check DEBUG logs for encrypted_content presence")

        # Follow-up to test field preservation across turns
        messages.append(
            {"role": "assistant", "content": response1.choices[0].message.content}
        )
        messages.append({"role": "user", "content": "Now show a code example"})
        response2 = completion(model="github_copilot/gpt-5.2", messages=messages)

        print("  Turn 2 complete - check DEBUG logs for encrypted_content preservation")

        assert response2 is not None

        print("\nReasoning items test complete.")
        print("VERIFICATION STEPS:")
        print("  1. Review DEBUG logs for 'Processing reasoning item' entries")
        print("  2. Verify encrypted_content field appears in Turn 1 response")
        print("  3. Verify encrypted_content preserved in Turn 2 request")
        print("  4. Check GitHub dashboard for premium request count")
        print("  Expected: 1 premium request total")
        print("  Fixed: X-Initiator=agent on Turn 2 suppresses premium charge")
        print("=" * 60 + "\n")

    def test_conversation_id_header_effect(self):
        """
        Resolves UNCONFIRMED status of COPILOT_CONVERSATION_ID_HEADER.

        Runs two paired 3-turn conversations back to back:
          - Conversation A: default behavior (no metadata["copilot_conversation_id"])
          - Conversation B: opt-in session billing (metadata={"copilot_conversation_id": "phase3-test-b"})

        Both should consume exactly 1 premium request each (2 total for both conversations).
        If x-conversation-id has no effect, both show delta=1 from X-Initiator alone.
        If x-conversation-id suppresses Turn 1 billing on Conversation B restart, delta differs.

        This test resolves open question from RESEARCH.md Section "Open Questions #1".
        """
        print("\n" + "=" * 60)
        print("RECORD DASHBOARD VALUE BEFORE THIS TEST")
        print("TEST: test_conversation_id_header_effect")
        print("=" * 60)

        # Conversation A: default behavior (no session key)
        messages_a: List[Dict] = [{"role": "user", "content": "What is 2+2?"}]
        response_a1 = completion(model="github_copilot/gpt-4o", messages=messages_a)
        assert response_a1 is not None
        messages_a.append(
            {"role": "assistant", "content": response_a1.choices[0].message.content}
        )

        messages_a.append({"role": "user", "content": "And 3+3?"})
        response_a2 = completion(model="github_copilot/gpt-4o", messages=messages_a)
        assert response_a2 is not None
        messages_a.append(
            {"role": "assistant", "content": response_a2.choices[0].message.content}
        )

        messages_a.append({"role": "user", "content": "And 4+4?"})
        response_a3 = completion(model="github_copilot/gpt-4o", messages=messages_a)
        assert response_a3 is not None

        print(
            "\nConversation A (no session key) complete. Expected premium delta so far: 1"
        )

        # Conversation B: opt-in session billing via metadata key
        messages_b: List[Dict] = [{"role": "user", "content": "What is 2+2?"}]
        response_b1 = completion(
            model="github_copilot/gpt-4o",
            messages=messages_b,
            metadata={"copilot_conversation_id": "phase3-x-conv-id-test"},
        )
        assert response_b1 is not None
        messages_b.append(
            {"role": "assistant", "content": response_b1.choices[0].message.content}
        )

        messages_b.append({"role": "user", "content": "And 3+3?"})
        response_b2 = completion(
            model="github_copilot/gpt-4o",
            messages=messages_b,
            metadata={"copilot_conversation_id": "phase3-x-conv-id-test"},
        )
        assert response_b2 is not None
        messages_b.append(
            {"role": "assistant", "content": response_b2.choices[0].message.content}
        )

        messages_b.append({"role": "user", "content": "And 4+4?"})
        response_b3 = completion(
            model="github_copilot/gpt-4o",
            messages=messages_b,
            metadata={"copilot_conversation_id": "phase3-x-conv-id-test"},
        )
        assert response_b3 is not None

        print("\nConversation B (with copilot_conversation_id session key) complete.")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  Total expected premium delta (both conversations A + B): 2")
        print(
            "  If GitHub dashboard shows delta = 2: x-conversation-id has no extra billing effect."
        )
        print(
            "  If GitHub dashboard shows delta < 2: x-conversation-id is actively suppressing premium charges."
        )
        print("  Visit: https://github.com/settings/copilot — Usage & billing tab")
        print(
            "  ACTION: Update COPILOT_CONVERSATION_ID_HEADER comment in common_utils.py:"
        )
        print(
            '    - If delta = 2: change UNCONFIRMED comment to "CONFIRMED NO-OP; X-Initiator sufficient"'
        )
        print(
            '    - If delta < 2: change to "CONFIRMED ACTIVE; reduces billing when session key provided"'
        )
        print("=" * 60 + "\n")

    def test_agentic_workflow_10plus_turns_chat_api(self):
        """
        TEST-01: 10+ turn agentic coding workflow consumes exactly 1 premium request.

        Simulates a realistic Plan Mode session: user requests a coding plan,
        agent works through planning steps turn by turn.
        Expected: premium delta = 1 (Turn 1 only). All subsequent turns are agent-free.
        """
        print("\n" + "=" * 60)
        print("RECORD DASHBOARD VALUE BEFORE THIS TEST")
        print("TEST: test_agentic_workflow_10plus_turns_chat_api (TEST-01)")
        print("=" * 60)

        # Turn 1 (premium): User-initiated planning request
        messages: List[Dict] = [
            {
                "role": "user",
                "content": (
                    "I need to build a Python REST API with FastAPI. "
                    "Create a detailed implementation plan with 10 specific steps, "
                    "one per message."
                ),
            }
        ]
        response = completion(model="github_copilot/gpt-4o", messages=messages)
        assert response is not None

        # Turns 2-11 (agent-initiated): Loop through 10 agentic steps
        for i in range(10):
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            messages.append(
                {"role": "user", "content": f"Execute step {i + 1} of the plan."}
            )
            response = completion(model="github_copilot/gpt-4o", messages=messages)
            assert response is not None

        print("\nTest complete: 11 turns executed (1 initial + 10 agentic steps)")
        print("MANUAL VERIFICATION REQUIRED:")
        print(
            "  Expected premium request delta: 1 (Turn 1 only — X-Initiator=agent on Turns 2-11)"
        )
        print("  (Before count - after count should equal 1)")
        print("  Visit: https://github.com/settings/copilot — Usage & billing tab")
        print("=" * 60 + "\n")

    def test_endpoint_parity_billing(self):
        """
        TEST-01 endpoint parity: Both Chat API and Responses API show correct billing.

        Runs one 3-turn Chat API conversation and one 3-turn Responses API conversation.
        Expected: premium delta = 2 (1 per conversation, first turn only).
        """
        print("\n" + "=" * 60)
        print("RECORD DASHBOARD VALUE BEFORE THIS TEST")
        print("TEST: test_endpoint_parity_billing (TEST-01 endpoint parity)")
        print("=" * 60)

        # Chat API: 3-turn conversation (model gpt-4o)
        chat_messages: List[Dict] = [
            {"role": "user", "content": "Write a Python hello world function."}
        ]
        chat_response1 = completion(
            model="github_copilot/gpt-4o", messages=chat_messages
        )
        assert chat_response1 is not None
        chat_messages.append(
            {"role": "assistant", "content": chat_response1.choices[0].message.content}
        )

        chat_messages.append({"role": "user", "content": "Add a docstring."})
        chat_response2 = completion(
            model="github_copilot/gpt-4o", messages=chat_messages
        )
        assert chat_response2 is not None
        chat_messages.append(
            {"role": "assistant", "content": chat_response2.choices[0].message.content}
        )

        chat_messages.append({"role": "user", "content": "Add type hints."})
        chat_response3 = completion(
            model="github_copilot/gpt-4o", messages=chat_messages
        )
        assert chat_response3 is not None

        print("\nChat API (3 turns) complete. Expected premium delta so far: 1")

        # Responses API: 3-turn conversation (model gpt-5.2)
        resp_messages: List[Dict] = [
            {"role": "user", "content": "Write a Python hello world function."}
        ]
        resp_response1 = completion(
            model="github_copilot/gpt-5.2", messages=resp_messages
        )
        assert resp_response1 is not None
        resp_messages.append(
            {"role": "assistant", "content": resp_response1.choices[0].message.content}
        )

        resp_messages.append({"role": "user", "content": "Add a docstring."})
        resp_response2 = completion(
            model="github_copilot/gpt-5.2", messages=resp_messages
        )
        assert resp_response2 is not None
        resp_messages.append(
            {"role": "assistant", "content": resp_response2.choices[0].message.content}
        )

        resp_messages.append({"role": "user", "content": "Add type hints."})
        resp_response3 = completion(
            model="github_copilot/gpt-5.2", messages=resp_messages
        )
        assert resp_response3 is not None

        print("\nEndpoint parity test complete.")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  Chat API (3 turns): expected delta = 1")
        print("  Responses API (3 turns): expected delta = 1")
        print("  Total expected premium request delta: 2")
        print("  If both show delta = 1 each: endpoint parity CONFIRMED.")
        print("  Visit: https://github.com/settings/copilot — Usage & billing tab")
        print("=" * 60 + "\n")
