"""
Tests for ComplianceChecker EU AI Act and GDPR compliance checks.
"""

from litellm.proxy.compliance_checks import ComplianceChecker
from litellm.types.proxy.compliance_endpoints import ComplianceCheckRequest


def _make_request(**kwargs) -> ComplianceCheckRequest:
    defaults = {"request_id": "req-1"}
    defaults.update(kwargs)
    return ComplianceCheckRequest(**defaults)


class TestEUAIActChecks:
    """Tests for EU AI Act compliance checks."""

    def test_all_pass_with_full_data(self):
        """All checks pass when full data + pre-call + post-call guardrails are present."""
        req = _make_request(
            user_id="user-1",
            model="gpt-4",
            timestamp="2024-01-01T00:00:00Z",
            guardrail_information=[
                {"guardrail_mode": "pre_call", "guardrail_status": "success"},
                {"guardrail_mode": "post_call", "guardrail_status": "success"},
            ],
        )
        checker = ComplianceChecker(req)
        results = checker.check_eu_ai_act()

        assert len(results) == 15
        for r in results:
            assert r.passed, f"{r.article} - {r.check_name} failed: {r.detail}"

    def test_no_guardrails_fails_most_checks(self):
        """Without guardrails, most checks fail."""
        req = _make_request()
        checker = ComplianceChecker(req)
        results = checker.check_eu_ai_act()

        assert len(results) == 15
        passed = [r for r in results if r.passed]
        assert len(passed) == 0

    def test_article_5_subcategories_present(self):
        """Verify all Article 5 subcategories are returned."""
        req = _make_request()
        checker = ComplianceChecker(req)
        results = checker.check_eu_ai_act()

        articles = [r.article for r in results]
        assert "Art. 5" in articles
        assert "Art. 5.1(a)" in articles
        assert "Art. 5.1(b)" in articles
        assert "Art. 5.1(c)" in articles
        assert "Art. 5.1(d)" in articles
        assert "Art. 5.1(f)" in articles
        assert "Art. 5.1(h)" in articles

    def test_additional_articles_present(self):
        """Verify articles beyond Art. 5 are returned."""
        req = _make_request()
        checker = ComplianceChecker(req)
        results = checker.check_eu_ai_act()

        articles = [r.article for r in results]
        assert "Art. 9" in articles
        assert "Art. 10" in articles
        assert "Art. 12" in articles
        assert "Art. 13" in articles
        assert "Art. 14" in articles
        assert "Art. 15" in articles
        assert "Art. 26" in articles
        assert "Art. 50" in articles

    def test_art_15_requires_both_pre_and_post_call(self):
        """Art. 15 accuracy/robustness requires both pre-call and post-call guardrails."""
        # Only pre-call -> should fail
        req = _make_request(
            user_id="u",
            model="m",
            timestamp="t",
            guardrail_information=[
                {"guardrail_mode": "pre_call", "guardrail_status": "success"},
            ],
        )
        checker = ComplianceChecker(req)
        results = checker.check_eu_ai_act()
        art_15 = [r for r in results if r.article == "Art. 15"][0]
        assert not art_15.passed

        # Both pre-call and post-call -> should pass
        req2 = _make_request(
            user_id="u",
            model="m",
            timestamp="t",
            guardrail_information=[
                {"guardrail_mode": "pre_call", "guardrail_status": "success"},
                {"guardrail_mode": "post_call", "guardrail_status": "success"},
            ],
        )
        checker2 = ComplianceChecker(req2)
        results2 = checker2.check_eu_ai_act()
        art_15_2 = [r for r in results2 if r.article == "Art. 15"][0]
        assert art_15_2.passed

    def test_art_26_deployer_obligations(self):
        """Art. 26 requires user, model, timestamp, guardrails, and pre-call."""
        req = _make_request(
            user_id="u",
            model="m",
            timestamp="t",
            guardrail_information=[
                {"guardrail_mode": "post_call", "guardrail_status": "success"},
            ],
        )
        checker = ComplianceChecker(req)
        results = checker.check_eu_ai_act()
        art_26 = [r for r in results if r.article == "Art. 26"][0]
        # Has everything except pre-call
        assert not art_26.passed
        assert "pre-call risk screening" in art_26.detail

    def test_art_50_requires_post_call(self):
        """Art. 50 requires post-call guardrails."""
        req = _make_request(
            guardrail_information=[
                {"guardrail_mode": "pre_call", "guardrail_status": "success"},
            ],
        )
        checker = ComplianceChecker(req)
        results = checker.check_eu_ai_act()
        art_50 = [r for r in results if r.article == "Art. 50"][0]
        assert not art_50.passed


class TestGDPRChecks:
    """Tests for GDPR compliance checks."""

    def test_gdpr_checks_count(self):
        req = _make_request()
        checker = ComplianceChecker(req)
        results = checker.check_gdpr()
        assert len(results) == 3

    def test_gdpr_all_pass(self):
        req = _make_request(
            user_id="u",
            model="m",
            timestamp="t",
            guardrail_information=[
                {"guardrail_mode": "pre_call", "guardrail_status": "success"},
            ],
        )
        checker = ComplianceChecker(req)
        results = checker.check_gdpr()
        for r in results:
            assert r.passed, f"{r.article} - {r.check_name} failed: {r.detail}"

    def test_sensitive_data_intervention(self):
        req = _make_request(
            guardrail_information=[
                {"guardrail_mode": "pre_call", "guardrail_status": "guardrail_intervened"},
            ],
        )
        checker = ComplianceChecker(req)
        results = checker.check_gdpr()
        art_5_1c = [r for r in results if r.article == "Art. 5(1)(c)"][0]
        assert art_5_1c.passed
        assert "intervened" in art_5_1c.detail
