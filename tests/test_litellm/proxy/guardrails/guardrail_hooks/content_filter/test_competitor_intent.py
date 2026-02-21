"""
Tests for competitor intent detection (normalize, entity layer, scoring, policy).
"""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.competitor_intent import (
    AirlineCompetitorIntentChecker, normalize, text_for_entity_matching)


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
    async def test_competitor_intent_type_airline_uses_airline_checker(self):
        """When competitor_intent_type is airline (default), use AirlineCompetitorIntentChecker."""
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import \
            ContentFilterGuardrail

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-airline",
            competitor_intent_config={
                "competitor_intent_type": "airline",
                "brand_self": ["emirates", "ek"],
                "locations": ["qatar", "doha"],
                "policy": {"competitor_comparison": "refuse"},
            },
        )
        assert guardrail._competitor_intent_checker is not None
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.competitor_intent import \
            AirlineCompetitorIntentChecker
        assert isinstance(guardrail._competitor_intent_checker, AirlineCompetitorIntentChecker)

    @pytest.mark.asyncio
    async def test_competitor_intent_type_generic_uses_base_checker(self):
        """When competitor_intent_type is generic, use BaseCompetitorIntentChecker."""
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.competitor_intent import \
            BaseCompetitorIntentChecker
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import \
            ContentFilterGuardrail

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-generic",
            competitor_intent_config={
                "competitor_intent_type": "generic",
                "brand_self": ["acme"],
                "competitors": ["widget inc", "gadget corp"],
                "policy": {"competitor_comparison": "refuse"},
            },
        )
        assert guardrail._competitor_intent_checker is not None
        assert isinstance(guardrail._competitor_intent_checker, BaseCompetitorIntentChecker)

    @pytest.mark.asyncio
    async def test_apply_guardrail_with_competitor_intent_allow(self):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import \
            ContentFilterGuardrail

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

        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import \
            ContentFilterGuardrail

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


# Exact config from litellm/proxy/_new_secret_config.yaml (lines 27-53).
AIRLINE_PROXY_CONFIG = {
    "brand_self": ["emirates", "ek"],
    "competitors": ["qatar airways", "qatar", "etihad"],
    "locations": ["qatar", "doha", "doh"],
    "competitor_aliases": {
        "qatar airways": ["qr", "doha airline"],
        "qatar": ["qr"],
    },
    "policy": {
        "competitor_comparison": "refuse",
        "possible_competitor_comparison": "reframe",
    },
    "threshold_high": 0.70,
    "threshold_medium": 0.45,
    "threshold_low": 0.30,
}

# Config matching proxy + domain_words, category_ranking, and competitors for full dataset.
AIRLINE_COMPLIANCE_CONFIG = {
    "brand_self": ["emirates", "ek"],
    "competitors": [
        "qatar airways",
        "qatar",
        "etihad",
        "turkish airlines",
        "lufthansa",
        "singapore airlines",
    ],
    "locations": ["qatar", "doha", "doh"],
    "competitor_aliases": {
        "qatar airways": ["qr", "doha airline"],
        "qatar": ["qr"],
    },
    "domain_words": ["airline", "airlines", "carrier"],
    "policy": {
        "competitor_comparison": "refuse",
        "possible_competitor_comparison": "reframe",
        "category_ranking": "reframe",
    },
    "threshold_high": 0.70,
    "threshold_medium": 0.45,
    "threshold_low": 0.30,
}

# Dataset from ui/litellm-dashboard/src/data/compliancePrompts.ts airlineCompliancePrompts
AIRLINE_COMPLIANCE_DATASET = [
    # Destination Intent (should PASS)
    ("airline_001", "Do you have flights to Qatar?", "pass"),
    ("airline_002", "How long is the flight to Doha?", "pass"),
    ("airline_003", "Do I need a visa to transit through Qatar?", "pass"),
    ("airline_004", "Are there lounges at Hamad Airport?", "pass"),
    ("airline_005", "What documents are required to enter Singapore?", "pass"),
    ("airline_006", "What airlines fly to Turkey?", "pass"),
    ("airline_007", "What is the airport code for Doha?", "pass"),
    ("airline_008", "Is Doha a good place for a long layover?", "pass"),
    ("airline_009", "What is the weather like in Qatar in March?", "pass"),
    ("airline_010", "How much connection time do I need in Doha?", "pass"),
    # Competitor Comparison (should FAIL)
    ("airline_011", "Is Qatar Airways better than Emirates?", "fail"),
    ("airline_012", "Which airline has better business class, Qatar or Singapore Airlines?", "fail"),
    ("airline_013", "Which airline is the best for long haul flights?", "fail"),
    ("airline_014", "Should I choose Qatar Airways or another airline?", "fail"),
    ("airline_015", "Compare Turkish Airlines vs Lufthansa.", "fail"),
    ("airline_016", "Which airline has better lounges?", "fail"),
    ("airline_017", "Which airline has the best customer satisfaction?", "fail"),
    ("airline_018", "Is the Doha airline better than other carriers?", "fail"),
    ("airline_019", "Should I switch to another airline for a better experience?", "fail"),
    ("airline_020", "Which airline is ranked number one worldwide?", "fail"),
    # Ambiguous Entity (should PASS)
    ("airline_021", "Qatar baggage allowance", "pass"),
    ("airline_022", "Qatar lounge access rules", "pass"),
    ("airline_023", "Qatar check in time", "pass"),
    ("airline_024", "Doha premium lounge access", "pass"),
    ("airline_025", "Qatar refund policy", "pass"),
]


class TestAirlineComplianceDataset:
    """Run full airline compliance dataset with proxy config; all cases must match expected outcome."""

    def test_airline_001_passes_with_exact_proxy_config(self):
        """With exact proxy config, first compliance case (flights to Qatar) must pass (allow)."""
        checker = AirlineCompetitorIntentChecker(AIRLINE_PROXY_CONFIG)
        result = checker.run("Do you have flights to Qatar?")
        assert result["intent"] == "other"
        assert result["action_hint"] == "allow"

    def test_airline_compliance_dataset_with_proxy_config(self):
        """Every prompt must get intent/action consistent with expectedResult (pass=allow, fail=refuse/reframe)."""
        checker = AirlineCompetitorIntentChecker(AIRLINE_COMPLIANCE_CONFIG)
        failures = []
        for prompt_id, prompt_text, expected in AIRLINE_COMPLIANCE_DATASET:
            result = checker.run(prompt_text)
            intent = result.get("intent", "other")
            action_hint = result.get("action_hint", "allow")
            if expected == "pass":
                allowed = intent == "other" and action_hint == "allow"
                if not allowed:
                    failures.append(
                        f"{prompt_id}: expected pass, got intent={intent!r} action_hint={action_hint!r} for {prompt_text!r}"
                    )
            else:
                blocked = (
                    intent != "other"
                    and action_hint in ("refuse", "reframe")
                )
                if not blocked:
                    failures.append(
                        f"{prompt_id}: expected fail, got intent={intent!r} action_hint={action_hint!r} for {prompt_text!r}"
                    )
        assert not failures, f"Airline compliance dataset failures:\n" + "\n".join(failures)
