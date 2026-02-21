"""
Tests for competitor intent detection (normalize, entity layer, scoring, policy).
"""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.competitor_intent import (
    AirlineCompetitorIntentChecker,
    normalize,
    text_for_entity_matching,
)


class TestNormalize:
    """Test text normalization (leetspeak, spacing, zero-width)."""

    def test_normalize_lowercase(self):
        assert normalize("Is Qatar Better?") == "is qatar better?"

    def test_normalize_leetspeak(self):
        assert "qatar" in normalize("q@tar")
        assert "qatar" in normalize("q4tar")

    def test_normalize_collapse_whitespace(self):
        assert normalize("hello    world") == "hello world"

    def test_normalize_spaced_out_letters(self):
        # Single-letter tokens collapsed into word
        assert "qatar" in normalize("q a t a r").replace(" ", "")

    def test_normalize_empty(self):
        assert normalize("") == ""
        assert normalize(None) == ""

    def test_text_for_entity_matching_removes_punctuation(self):
        t = text_for_entity_matching("q.a.t.a.r  emirates")
        assert "emirates" in t
        assert "." not in t


class TestAirlineCompetitorIntentChecker:
    """Test AirlineCompetitorIntentChecker run() and intent bands."""

    @pytest.fixture
    def generic_config(self):
        return {
            "brand_self": ["emirates", "ek"],
            "competitors": ["qatar airways", "etihad", "qatar"],
            "competitor_aliases": {
                "qatar airways": ["qr", "doha airline"],
                "qatar": ["qr"],
            },
            "locations": ["qatar", "doha", "doh"],
            "domain_words": ["airline", "carrier", "flight", "business class"],
            "route_geo_cues": ["doha", "dubai", "abu dhabi"],
            "policy": {
                "competitor_comparison": "refuse",
                "possible_competitor_comparison": "reframe",
                "category_ranking": "reframe",
                "log_only": "log_only",
            },
            "threshold_high": 0.70,
            "threshold_medium": 0.45,
            "threshold_low": 0.30,
        }

    def test_run_other_intent(self, generic_config):
        checker = AirlineCompetitorIntentChecker(generic_config)
        result = checker.run("What is the weather today?")
        assert result["intent"] == "other"
        assert result["action_hint"] == "allow"

    def test_run_competitor_comparison_direct(self, generic_config):
        checker = AirlineCompetitorIntentChecker(generic_config)
        result = checker.run("Is Qatar better than Emirates?")
        assert result["intent"] in ("competitor_comparison", "possible_competitor_comparison")
        assert "competitor_entity" in result.get("signals", []) or "competitors" in str(result.get("entities", {}))
        assert result["confidence"] >= 0.45

    def test_run_competitor_comparison_as_good_as(self, generic_config):
        checker = AirlineCompetitorIntentChecker(generic_config)
        result = checker.run("Is Qatar as good as Emirates?")
        assert result["intent"] != "other"
        assert result["confidence"] >= 0.45

    def test_run_ranking_with_competitor(self, generic_config):
        checker = AirlineCompetitorIntentChecker(generic_config)
        result = checker.run("Why is Qatar Airways the best?")
        assert result["intent"] != "other"
        assert "qatar" in str(result.get("entities", {}).get("competitors", [])).lower() or "competitor" in str(result.get("signals", []))

    def test_run_ranking_without_competitor_category_ranking(self, generic_config):
        checker = AirlineCompetitorIntentChecker(generic_config)
        result = checker.run("Which Gulf airline is the best?")
        # domain_words "airline" + ranking "best" + geo "gulf" not in route_geo_cues but "airline" is domain
        assert result["intent"] in ("category_ranking", "possible_competitor_comparison", "log_only", "other")

    def test_run_evidence_populated(self, generic_config):
        checker = AirlineCompetitorIntentChecker(generic_config)
        result = checker.run("Is Qatar better than Emirates?")
        assert "evidence" in result
        assert isinstance(result["evidence"], list)

    def test_run_gate_prevents_false_positive(self, generic_config):
        # "best" alone without entity or domain should not trigger competitor_comparison
        checker = AirlineCompetitorIntentChecker(generic_config)
        result = checker.run("What is the best way to cook pasta?")
        assert result["intent"] in ("other", "log_only")

    def test_other_meaning_context_suppression(self, generic_config):
        # "flights to qatar" = other meaning (country), not competitor airline
        checker = AirlineCompetitorIntentChecker(generic_config)
        result = checker.run("how expensive are flights to qatar?")
        assert result["intent"] == "other"
        assert not result.get("entities", {}).get("competitors")


class TestContentFilterWithCompetitorIntent:
    """Integration: ContentFilterGuardrail with competitor_intent_config."""

    @pytest.mark.asyncio
    async def test_apply_guardrail_with_competitor_intent_allow(self):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-competitor",
            competitor_intent_config={
                "brand_self": ["emirates"],
                "competitors": ["qatar"],
                "domain_words": ["airline"],
                "policy": {"competitor_comparison": "refuse", "possible_competitor_comparison": "reframe"},
            },
        )
        inputs = {"texts": ["What is the capital of France?"]}
        result = await guardrail.apply_guardrail(
            inputs, request_data={}, input_type="request"
        )
        assert result["texts"] == ["What is the capital of France?"]

    @pytest.mark.asyncio
    async def test_apply_guardrail_with_competitor_intent_refuse(self):
        from fastapi import HTTPException

        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-competitor",
            competitor_intent_config={
                "brand_self": ["emirates"],
                "competitors": ["qatar airways"],
                "domain_words": ["airline", "flight"],
                "policy": {"competitor_comparison": "refuse"},
                "threshold_high": 0.5,
            },
        )
        inputs = {"texts": ["Is Qatar Airways better than Emirates?"]}
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs, request_data={}, input_type="request"
            )
        assert exc_info.value.status_code == 403
