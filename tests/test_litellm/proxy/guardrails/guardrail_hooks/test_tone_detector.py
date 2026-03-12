"""
Tests for the Tone Detection feature of ContentFilterGuardrail.

Covers:
  - True positives: inappropriate tone is blocked
  - True negatives: professional tone passes through
  - False positive resistance: domain-safe terms and near-miss phrases pass
  - Safe-phrase override: user-defined safe phrases bypass blocking
  - Custom blocked phrases: user-added patterns are enforced
  - Empty / missing text handling
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../"))

from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_guardrail(
    blocked_phrases=None,
    safe_phrases=None,
) -> ContentFilterGuardrail:
    """Create a ContentFilterGuardrail with only tone detection enabled."""
    config = {}
    if blocked_phrases is not None:
        config["blocked_phrases"] = blocked_phrases
    if safe_phrases is not None:
        config["safe_phrases"] = safe_phrases
    return ContentFilterGuardrail(
        guardrail_name="test-tone",
        tone_detection_config=config,
    )


def _inputs(text: str) -> dict:
    return {"texts": [text]}


# ---------------------------------------------------------------------------
# TRUE POSITIVES — must be blocked
# ---------------------------------------------------------------------------

class TestTruePositives:
    """Each of these should raise HTTPException(400)."""

    # -- dismissive --
    @pytest.mark.parametrize(
        "text",
        [
            "That's not really my problem.",
            "That is not my problem.",
            "That's not my concern.",
            "I don't see what the big deal is.",
            "You're overthinking this.",
            "It's not that complicated, just read the FAQ.",
            "Just read the docs, it's all there.",
        ],
    )
    @pytest.mark.asyncio
    async def test_dismissive(self, text):
        g = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            await g.apply_guardrail(_inputs(text), {}, "response")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["category"] == "dismissive"

    # -- blaming --
    @pytest.mark.parametrize(
        "text",
        [
            "You should have read the terms before signing up.",
            "You should have checked the requirements first.",
            "That's your fault for not updating your settings.",
            "That is your mistake.",
            "If you had followed the instructions properly, this wouldn't have happened.",
            "If you had read the docs, you'd know.",
            "You clearly didn't set this up correctly.",
            "You clearly did not read the instructions before starting.",
            "This issue is on your end, not ours.",
            "This problem is on your end.",
        ],
    )
    @pytest.mark.asyncio
    async def test_blaming(self, text):
        g = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            await g.apply_guardrail(_inputs(text), {}, "response")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["category"] == "blaming"

    # -- refusal --
    @pytest.mark.parametrize(
        "text",
        [
            "I can't help you with that.",
            "There's nothing I can do about it.",
            "There is nothing we can do.",
            "You'll just have to figure it out yourself.",
            "You will have to figure it out.",
            "We don't do that. Try somewhere else.",
        ],
    )
    @pytest.mark.asyncio
    async def test_refusal(self, text):
        g = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            await g.apply_guardrail(_inputs(text), {}, "response")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["category"] == "refusal"

    # -- condescension --
    @pytest.mark.parametrize(
        "text",
        [
            "As I already explained, if you'd been paying attention...",
            "I'm not sure how to make this any simpler for you.",
            "Let me spell it out for you since you don't seem to get it.",
            "Since you don't get it, I'll try one more time.",
            "Oh, you want me to do your job for you too?",
        ],
    )
    @pytest.mark.asyncio
    async def test_condescension(self, text):
        g = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            await g.apply_guardrail(_inputs(text), {}, "response")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["category"] == "condescension"

    # -- impatience --
    @pytest.mark.parametrize(
        "text",
        [
            "I've already told you this three times.",
            "How many times do I have to explain this?",
            "Look, I don't have time to go over this again.",
            "I do not have time to go through this with you right now.",
            "Are you even listening to what I'm saying?",
            "Just do what I said already!",
        ],
    )
    @pytest.mark.asyncio
    async def test_impatience(self, text):
        g = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            await g.apply_guardrail(_inputs(text), {}, "response")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["category"] == "impatience"

    # -- unprofessional --
    @pytest.mark.parametrize(
        "text",
        [
            "lol yeah that's totally broken, my bad dude.",
            "Bruh, just restart the app and chill.",
            "Idk man, sounds like a you problem.",
            "LOL, that feature has been broken forever.",
            "SMH, that's the third time this week someone has asked about this.",
            "Whatever, just deal with it.",
            "This crap happens all the time, don't worry about it.",
        ],
    )
    @pytest.mark.asyncio
    async def test_unprofessional(self, text):
        g = _make_guardrail()
        with pytest.raises(HTTPException) as exc_info:
            await g.apply_guardrail(_inputs(text), {}, "response")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["category"] == "unprofessional"


# ---------------------------------------------------------------------------
# TRUE NEGATIVES — must pass through
# ---------------------------------------------------------------------------

class TestTrueNegatives:
    """Professional and helpful responses must NOT be blocked."""

    @pytest.mark.parametrize(
        "text",
        [
            # Professional tone
            "I understand your frustration. Let me look into this for you right away.",
            "I'm sorry to hear you're experiencing this issue. Here's what we can do to resolve it.",
            "Thank you for your patience. I've escalated this to our specialist team.",
            "Great question! You can find that setting under Account > Preferences.",
            "I appreciate you bringing this to our attention.",
            "That's a great point. While we don't currently support that feature, I can suggest a workaround.",
            "I completely understand your concern. Let me explain how this works.",
            "I want to make sure we get this right for you.",
            "You're all set! Is there anything else I can help you with today?",
            "I'm not able to process that request directly, but I can connect you with someone who can.",
            # Neutral / informational
            "Your order has been shipped and should arrive within 3-5 business days.",
            "The latest version includes several performance improvements.",
            "You can reset your password from the login page by clicking 'Forgot Password'.",
        ],
    )
    @pytest.mark.asyncio
    async def test_professional_passes(self, text):
        g = _make_guardrail()
        result = await g.apply_guardrail(_inputs(text), {}, "response")
        assert result["texts"] == [text]


# ---------------------------------------------------------------------------
# FALSE POSITIVE RESISTANCE — tricky near-miss phrases must pass
# ---------------------------------------------------------------------------

class TestFalsePositiveResistance:
    """Sentences that contain trigger-adjacent words but are NOT rude."""

    @pytest.mark.parametrize(
        "text,description",
        [
            (
                "You should have received a confirmation email within 5 minutes.",
                "informational 'should have received'",
            ),
            (
                "If the problem is on your end, here are some steps to troubleshoot.",
                "'the problem' not 'this problem/issue'",
            ),
            (
                "I can't help but notice you've been a loyal customer — thank you!",
                "'can't help but notice' is a compliment",
            ),
            (
                "Let me spell out the steps clearly so nothing is missed.",
                "'spell out the steps' is helpful, not condescending",
            ),
            (
                "I've already told our engineering team about this, and they're working on a fix.",
                "'told our team' is reassuring, not impatient",
            ),
            (
                "There's nothing I can do to speed up the shipment, but I can offer a discount on your next order.",
                "'nothing I can do to X, but Y' offers an alternative",
            ),
            (
                "Please check the FAQ for a list of supported file formats — it's very comprehensive.",
                "polite FAQ reference without 'just read the'",
            ),
            (
                "I understand that's your fault tolerance threshold — let me adjust it for you.",
                "'fault tolerance' is a technical term",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_near_miss_passes(self, text, description):
        g = _make_guardrail()
        result = await g.apply_guardrail(_inputs(text), {}, "response")
        assert result["texts"] == [text], f"False positive on: {description}"


# ---------------------------------------------------------------------------
# DOMAIN-SAFE TERMS — technical jargon must pass
# ---------------------------------------------------------------------------

class TestDomainSafeTerms:
    """Technical/domain jargon that sounds negative out of context must pass."""

    @pytest.mark.parametrize(
        "text",
        [
            "To kill the background process, open Task Manager and select 'End Task'.",
            "You can terminate your subscription at any time from the billing page.",
            "The aggressive caching strategy reduces load times by up to 40%.",
            "This will destroy the existing volume and create a new one.",
            "The dead letter queue captures messages that failed processing.",
            "Use the force push option only if you're sure no one else is working on that branch.",
            "The abort signal will cancel all in-flight requests when the user navigates away.",
            "Your trial has expired. You can reactivate your account by updating your payment method.",
            "The critical severity alert fires when CPU usage exceeds 95%.",
            "Run the nuke command to tear down the entire test environment.",
        ],
    )
    @pytest.mark.asyncio
    async def test_technical_jargon_passes(self, text):
        g = _make_guardrail()
        result = await g.apply_guardrail(_inputs(text), {}, "response")
        assert result["texts"] == [text]


# ---------------------------------------------------------------------------
# SAFE-PHRASE OVERRIDE
# ---------------------------------------------------------------------------

class TestSafePhraseOverride:
    """User-defined safe_phrases should exempt text from blocking."""

    @pytest.mark.asyncio
    async def test_safe_phrase_overrides_block(self):
        """A text that would normally be blocked is allowed if it matches a safe phrase."""
        g = _make_guardrail(safe_phrases=[r"help you with that"])
        # "I can't help you with that" normally triggers refusal
        result = await g.apply_guardrail(
            _inputs("I can't help you with that specific format, but here's an alternative."),
            {},
            "response",
        )
        assert result["texts"][0].startswith("I can't help you")

    @pytest.mark.asyncio
    async def test_safe_phrase_does_not_affect_other_violations(self):
        """A safe phrase for one pattern does not suppress unrelated violations."""
        g = _make_guardrail(safe_phrases=[r"help you with that"])
        with pytest.raises(HTTPException):
            await g.apply_guardrail(
                _inputs("You're overthinking this."),
                {},
                "response",
            )


# ---------------------------------------------------------------------------
# CUSTOM BLOCKED PHRASES
# ---------------------------------------------------------------------------

class TestCustomBlockedPhrases:
    """User-defined blocked_phrases extend the built-in patterns."""

    @pytest.mark.asyncio
    async def test_custom_blocked_phrase_fires(self):
        g = _make_guardrail(blocked_phrases=[r"\bper my last email\b"])
        with pytest.raises(HTTPException) as exc_info:
            await g.apply_guardrail(
                _inputs("Per my last email, the deadline was yesterday."),
                {},
                "response",
            )
        assert exc_info.value.detail["category"] == "custom_blocked"

    @pytest.mark.asyncio
    async def test_custom_blocked_phrase_case_insensitive(self):
        g = _make_guardrail(blocked_phrases=[r"\bPER MY LAST EMAIL\b"])
        with pytest.raises(HTTPException):
            await g.apply_guardrail(
                _inputs("per my last email, I mentioned this issue."),
                {},
                "response",
            )

    @pytest.mark.asyncio
    async def test_custom_blocked_does_not_affect_clean_text(self):
        g = _make_guardrail(blocked_phrases=[r"\bper my last email\b"])
        result = await g.apply_guardrail(
            _inputs("Thank you for reaching out! Here's how to fix that."),
            {},
            "response",
        )
        assert result["texts"][0].startswith("Thank you")


# ---------------------------------------------------------------------------
# EDGE CASES
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_empty_text_passes(self):
        g = _make_guardrail()
        result = await g.apply_guardrail({"texts": [""]}, {}, "response")
        assert result["texts"] == [""]

    @pytest.mark.asyncio
    async def test_no_texts_key(self):
        g = _make_guardrail()
        result = await g.apply_guardrail({}, {}, "response")
        assert result["texts"] == []

    @pytest.mark.asyncio
    async def test_multiple_texts_blocks_on_first_violation(self):
        g = _make_guardrail()
        with pytest.raises(HTTPException):
            await g.apply_guardrail(
                {"texts": [
                    "Thanks for reaching out!",
                    "That's not my problem.",
                ]},
                {},
                "response",
            )

    @pytest.mark.asyncio
    async def test_case_insensitive_detection(self):
        """Patterns should match regardless of case."""
        g = _make_guardrail()
        with pytest.raises(HTTPException):
            await g.apply_guardrail(
                _inputs("YOU'RE OVERTHINKING THIS."),
                {},
                "response",
            )


# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

class TestInit:

    def test_guardrail_name_set(self):
        g = _make_guardrail()
        assert g.guardrail_name == "test-tone"

    def test_tone_checker_enabled(self):
        g = _make_guardrail()
        assert g._tone_checker is not None

    def test_tone_checker_disabled_when_no_config(self):
        g = ContentFilterGuardrail(guardrail_name="test-no-tone")
        assert g._tone_checker is None

    def test_tone_checker_enabled_via_category(self):
        """Selecting tone_detection as a category should enable the ToneChecker."""
        g = ContentFilterGuardrail(
            guardrail_name="test-category-tone",
            categories=[{"category": "tone_detection", "enabled": True, "action": "BLOCK"}],
        )
        assert g._tone_checker is not None

    @pytest.mark.asyncio
    async def test_tone_detection_via_category_blocks(self):
        """Tone detection enabled via category selection should block bad tone."""
        g = ContentFilterGuardrail(
            guardrail_name="test-category-tone",
            categories=[{"category": "tone_detection", "enabled": True, "action": "BLOCK"}],
        )
        with pytest.raises(HTTPException):
            await g.apply_guardrail(
                _inputs("That's not my problem."),
                {},
                "response",
            )

    def test_tone_detection_category_disabled(self):
        """Disabling the tone_detection category should NOT enable ToneChecker."""
        g = ContentFilterGuardrail(
            guardrail_name="test-category-disabled",
            categories=[{"category": "tone_detection", "enabled": False, "action": "BLOCK"}],
        )
        assert g._tone_checker is None

    def test_invalid_regex_degrades_gracefully(self):
        """Invalid regex in blocked_phrases should degrade gracefully (tone checker disabled)."""
        g = _make_guardrail(blocked_phrases=[r"(unclosed"])
        # try/except in _init_tone_checker catches the ValueError; checker stays None
        assert g._tone_checker is None

    def test_pattern_too_long_degrades_gracefully(self):
        """Patterns exceeding the length limit should degrade gracefully."""
        g = _make_guardrail(blocked_phrases=["a" * 2000])
        assert g._tone_checker is None


# ---------------------------------------------------------------------------
# INPUT TYPE GUARD — tone detection only fires on responses
# ---------------------------------------------------------------------------

class TestInputTypeGuard:

    @pytest.mark.asyncio
    async def test_request_input_type_skips_tone_detection(self):
        """Tone detection should NOT fire when input_type is 'request'."""
        g = _make_guardrail()
        # This text would be blocked on 'response' but should pass on 'request'
        result = await g.apply_guardrail(
            _inputs("That's not my problem."),
            {},
            "request",
        )
        assert result["texts"] == ["That's not my problem."]

    @pytest.mark.asyncio
    async def test_response_input_type_fires_tone_detection(self):
        """Tone detection should fire when input_type is 'response'."""
        g = _make_guardrail()
        with pytest.raises(HTTPException):
            await g.apply_guardrail(
                _inputs("That's not my problem."),
                {},
                "response",
            )
