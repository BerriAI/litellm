"""
Tests for the airline off-topic restriction policy template.

Verifies that off-topic messages are blocked and on-topic/conversational messages pass.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path

from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)

POLICY_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/policy_templates/airline_off_topic_restriction.yaml",
)


def _make_guardrail():
    """Create a ContentFilterGuardrail with the airline off-topic restriction loaded."""
    return ContentFilterGuardrail(
        guardrail_name="test-airline-off-topic",
        categories=[
            {
                "category": "airline_off_topic_restriction",
                "category_file": POLICY_TEMPLATE_PATH,
                "enabled": True,
                "action": "BLOCK",
            }
        ],
    )


class TestAirlineOffTopicRestriction:
    """Test the airline off-topic restriction policy template."""

    def test_on_topic_flight_booking(self):
        """Airline booking questions should pass."""
        guardrail = _make_guardrail()
        # Should not raise
        result = guardrail._filter_single_text("I want to book a flight to Dubai")
        assert result == "I want to book a flight to Dubai"

    def test_on_topic_baggage(self):
        """Baggage questions should pass."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("What is the baggage allowance for economy?")
        assert result == "What is the baggage allowance for economy?"

    def test_on_topic_checkin(self):
        """Check-in questions should pass."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("How do I check in online?")
        assert "check in" in result

    def test_on_topic_delay(self):
        """Flight delay questions should pass."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("My flight is delayed, what are my options?")
        assert "delayed" in result

    def test_conversational_hello(self):
        """Greetings should pass."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("Hello")
        assert result == "Hello"

    def test_conversational_thanks(self):
        """Thank you should pass."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("Thank you for your help")
        assert "Thank you" in result

    def test_conversational_help(self):
        """Help requests should pass."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("Can you help me?")
        assert "help" in result

    def test_conversational_yes_no(self):
        """Simple yes/no should pass."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("Yes")
        assert result == "Yes"

    def test_off_topic_news_always_block(self):
        """News questions should be blocked via always_block_keywords."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("What's in the news today?")
        assert exc_info.value.status_code == 403

    def test_off_topic_joke_always_block(self):
        """Joke requests should be blocked via always_block_keywords."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("Tell me a joke")
        assert exc_info.value.status_code == 403

    def test_off_topic_coding_always_block(self):
        """Coding requests should be blocked via always_block_keywords."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("Write me code in python")
        assert exc_info.value.status_code == 403

    def test_off_topic_ai_gateway_always_block(self):
        """AI gateway questions should be blocked via always_block_keywords."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("What is an AI gateway?")
        assert exc_info.value.status_code == 403

    def test_off_topic_capital_always_block(self):
        """General knowledge questions should be blocked via always_block_keywords."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("What is the capital of France?")
        assert exc_info.value.status_code == 403

    def test_off_topic_sports_conditional(self):
        """Sports questions should be blocked via conditional matching."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("Who won the football game today?")
        assert exc_info.value.status_code == 403

    def test_off_topic_recipe_always_block(self):
        """Recipe questions should be blocked via always_block_keywords."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("Give me a recipe for pasta")
        assert exc_info.value.status_code == 403

    def test_off_topic_movie_conditional(self):
        """Movie questions with a block word should be blocked."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("What is the top movie to watch on Netflix?")
        assert exc_info.value.status_code == 403

    def test_on_topic_recommend_seat(self):
        """Airline recommendation questions should pass (not false-positive)."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("Can you recommend the best seat?")
        assert "recommend" in result.lower()

    def test_on_topic_explain_booking(self):
        """Explain questions about airline topics should pass."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("Can you explain my booking details?")
        assert "explain" in result.lower()

    def test_off_topic_stock_conditional(self):
        """Stock market questions should be blocked via conditional matching."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("What is the stock price of Apple today?")
        assert exc_info.value.status_code == 403

    def test_off_topic_homework_always_block(self):
        """Homework requests should be blocked via always_block_keywords."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("Help me with my homework")
        assert exc_info.value.status_code == 403

    def test_off_topic_relationship_always_block(self):
        """Relationship advice should be blocked via always_block_keywords."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("Can you give me relationship advice?")
        assert exc_info.value.status_code == 403

    def test_exception_inflight_entertainment(self):
        """In-flight entertainment questions should pass (exception)."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text(
            "What movies are available on the in-flight entertainment?"
        )
        assert "in-flight entertainment" in result.lower()

    def test_exception_flight_price(self):
        """Flight price questions should pass (exception)."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("What is the flight price to London?")
        assert "flight price" in result.lower()

    def test_exception_travel_news(self):
        """Travel news should pass (exception)."""
        guardrail = _make_guardrail()
        result = guardrail._filter_single_text("Any travel news I should know about?")
        assert "travel news" in result.lower()

    def test_off_topic_president_always_block(self):
        """Political questions should be blocked."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("Who is the president of the United States?")
        assert exc_info.value.status_code == 403

    def test_off_topic_blockchain_always_block(self):
        """Blockchain questions should be blocked."""
        guardrail = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            guardrail._filter_single_text("What is blockchain technology?")
        assert exc_info.value.status_code == 403
