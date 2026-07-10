"""
Unit tests for compliance check endpoints (EU AI Act and GDPR).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

from litellm.proxy.compliance_checks import ComplianceChecker
from litellm.types.proxy.compliance_endpoints import ComplianceCheckRequest

# ---------------------------------------------------------------------------
# EU AI Act — Non-compliant cases (Task #3)
# ---------------------------------------------------------------------------


class TestEuAiActNonCompliant:
    """Requests that should NOT be EU AI Act compliant."""

    def test_no_guardrails_applied(self):
        """Request with no guardrail information at all."""
        data = ComplianceCheckRequest(
            request_id="req-001",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=None,
        )
        checks = ComplianceChecker(data).check_eu_ai_act()
        results = {c.check_name: c.passed for c in checks}
        assert results["Guardrails applied"] is False
        assert results["Content screened before LLM"] is False
        assert results["Audit record complete"] is False

    def test_empty_guardrails_list(self):
        """Request with an empty guardrail list."""
        data = ComplianceCheckRequest(
            request_id="req-002",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[],
        )
        checks = ComplianceChecker(data).check_eu_ai_act()
        results = {c.check_name: c.passed for c in checks}
        assert results["Guardrails applied"] is False
        assert results["Content screened before LLM"] is False
        assert results["Audit record complete"] is False

    def test_no_prohibited_practices_screening(self):
        """Guardrails exist but only post-call (no pre-call screening)."""
        data = ComplianceCheckRequest(
            request_id="req-003",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "content_filter",
                    "guardrail_mode": "post_call",
                    "guardrail_status": "success",
                }
            ],
        )
        checks = ComplianceChecker(data).check_eu_ai_act()
        results = {c.check_name: c.passed for c in checks}
        assert results["Guardrails applied"] is True
        assert results["Content screened before LLM"] is False

    def test_incomplete_audit_missing_user_id(self):
        """Audit record missing user_id."""
        data = ComplianceCheckRequest(
            request_id="req-004",
            user_id=None,
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "prohibited_practices",
                    "guardrail_status": "success",
                }
            ],
        )
        checks = ComplianceChecker(data).check_eu_ai_act()
        results = {c.check_name: c.passed for c in checks}
        assert results["Audit record complete"] is False

    def test_incomplete_audit_missing_model(self):
        """Audit record missing model."""
        data = ComplianceCheckRequest(
            request_id="req-005",
            user_id="user-1",
            model=None,
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "prohibited_practices",
                    "guardrail_status": "success",
                }
            ],
        )
        checks = ComplianceChecker(data).check_eu_ai_act()
        results = {c.check_name: c.passed for c in checks}
        assert results["Audit record complete"] is False

    def test_incomplete_audit_missing_timestamp(self):
        """Audit record missing timestamp."""
        data = ComplianceCheckRequest(
            request_id="req-006",
            user_id="user-1",
            model="gpt-4",
            timestamp=None,
            guardrail_information=[
                {
                    "guardrail_name": "prohibited_practices",
                    "guardrail_status": "success",
                }
            ],
        )
        checks = ComplianceChecker(data).check_eu_ai_act()
        results = {c.check_name: c.passed for c in checks}
        assert results["Audit record complete"] is False

    def test_incomplete_audit_missing_guardrails(self):
        """Audit record has user/model/timestamp but no guardrails."""
        data = ComplianceCheckRequest(
            request_id="req-007",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[],
        )
        checks = ComplianceChecker(data).check_eu_ai_act()
        results = {c.check_name: c.passed for c in checks}
        assert results["Audit record complete"] is False


# ---------------------------------------------------------------------------
# GDPR — Non-compliant cases (Task #3)
# ---------------------------------------------------------------------------


class TestGdprNonCompliant:
    """Requests that should NOT be GDPR compliant."""

    def test_no_pii_detection(self):
        """Guardrails exist but only post-call (no pre-call data protection)."""
        data = ComplianceCheckRequest(
            request_id="req-101",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "content_filter",
                    "guardrail_mode": "post_call",
                    "guardrail_status": "success",
                }
            ],
        )
        checks = ComplianceChecker(data).check_gdpr()
        results = {c.check_name: c.passed for c in checks}
        assert results["Data protection applied"] is False
        assert results["Sensitive data protected"] is False

    def test_empty_guardrails(self):
        """Empty guardrail list — no PII scan."""
        data = ComplianceCheckRequest(
            request_id="req-102",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[],
        )
        checks = ComplianceChecker(data).check_gdpr()
        results = {c.check_name: c.passed for c in checks}
        assert results["Data protection applied"] is False
        assert results["Audit record complete"] is False

    def test_pii_sent_in_plaintext(self):
        """PII detection ran but status indicates PII was passed through."""
        data = ComplianceCheckRequest(
            request_id="req-103",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "pii_detection",
                    "guardrail_status": "pii_detected_not_blocked",
                }
            ],
        )
        checks = ComplianceChecker(data).check_gdpr()
        results = {c.check_name: c.passed for c in checks}
        assert results["Data protection applied"] is True
        assert results["Sensitive data protected"] is False

    def test_gdpr_audit_missing_user_id(self):
        """GDPR audit missing user_id."""
        data = ComplianceCheckRequest(
            request_id="req-104",
            user_id=None,
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "pii_detection",
                    "guardrail_status": "success",
                }
            ],
        )
        checks = ComplianceChecker(data).check_gdpr()
        results = {c.check_name: c.passed for c in checks}
        assert results["Audit record complete"] is False

    def test_gdpr_audit_missing_model(self):
        """GDPR audit missing model."""
        data = ComplianceCheckRequest(
            request_id="req-105",
            user_id="user-1",
            model=None,
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "pii_detection",
                    "guardrail_status": "success",
                }
            ],
        )
        checks = ComplianceChecker(data).check_gdpr()
        results = {c.check_name: c.passed for c in checks}
        assert results["Audit record complete"] is False

    def test_no_guardrails_at_all(self):
        """None guardrail_information."""
        data = ComplianceCheckRequest(
            request_id="req-106",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=None,
        )
        checks = ComplianceChecker(data).check_gdpr()
        results = {c.check_name: c.passed for c in checks}
        assert results["Data protection applied"] is False
        assert results["Audit record complete"] is False


# ---------------------------------------------------------------------------
# EU AI Act — Compliant cases (Task #4)
# ---------------------------------------------------------------------------


class TestEuAiActCompliant:
    """Requests that SHOULD be EU AI Act compliant."""

    def test_fully_compliant(self):
        """All checks pass: guardrails, prohibited_practices, full audit."""
        data = ComplianceCheckRequest(
            request_id="req-201",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "content_filter",
                    "guardrail_status": "success",
                },
                {
                    "guardrail_name": "prohibited_practices",
                    "guardrail_status": "success",
                },
            ],
        )
        checks = ComplianceChecker(data).check_eu_ai_act()
        results = {c.check_name: c.passed for c in checks}
        assert results["Guardrails applied"] is True
        assert results["Content screened before LLM"] is True
        assert results["Audit record complete"] is True
        assert all(c.passed for c in checks)

    def test_compliant_with_multiple_guardrails(self):
        """Multiple guardrails including prohibited_practices."""
        data = ComplianceCheckRequest(
            request_id="req-202",
            user_id="user-2",
            model="claude-3",
            timestamp="2026-02-17T12:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "pii_detection",
                    "guardrail_status": "success",
                },
                {
                    "guardrail_name": "prohibited_practices",
                    "guardrail_status": "success",
                },
                {
                    "guardrail_name": "content_filter",
                    "guardrail_status": "success",
                },
            ],
        )
        checks = ComplianceChecker(data).check_eu_ai_act()
        assert all(c.passed for c in checks)


# ---------------------------------------------------------------------------
# GDPR — Compliant cases (Task #4)
# ---------------------------------------------------------------------------


class TestGdprCompliant:
    """Requests that SHOULD be GDPR compliant."""

    def test_fully_compliant_pii_no_issues(self):
        """PII scan ran, found nothing (status=success), full audit."""
        data = ComplianceCheckRequest(
            request_id="req-301",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "pii_detection",
                    "guardrail_status": "success",
                }
            ],
        )
        checks = ComplianceChecker(data).check_gdpr()
        results = {c.check_name: c.passed for c in checks}
        assert results["Data protection applied"] is True
        assert results["Sensitive data protected"] is True
        assert results["Audit record complete"] is True
        assert all(c.passed for c in checks)

    def test_compliant_pii_masked(self):
        """PII detected and masked (guardrail_intervened) — still compliant."""
        data = ComplianceCheckRequest(
            request_id="req-302",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "pii_detection",
                    "guardrail_status": "guardrail_intervened",
                }
            ],
        )
        checks = ComplianceChecker(data).check_gdpr()
        results = {c.check_name: c.passed for c in checks}
        assert results["Data protection applied"] is True
        assert results["Sensitive data protected"] is True
        assert results["Audit record complete"] is True
        assert all(c.passed for c in checks)

    def test_compliant_with_other_guardrails(self):
        """PII detection plus other guardrails — still compliant."""
        data = ComplianceCheckRequest(
            request_id="req-303",
            user_id="user-2",
            model="claude-3",
            timestamp="2026-02-17T12:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "content_filter",
                    "guardrail_status": "success",
                },
                {
                    "guardrail_name": "pii_detection",
                    "guardrail_status": "success",
                },
                {
                    "guardrail_name": "prohibited_practices",
                    "guardrail_status": "success",
                },
            ],
        )
        checks = ComplianceChecker(data).check_gdpr()
        assert all(c.passed for c in checks)


# ---------------------------------------------------------------------------
# guardrail_mode matching — LitellmParams.mode is Union[str, List[str], Mode],
# so a spend log's guardrail_mode can be None / str / list / tuple / set / dict.
# A prior implementation compared `g_mode == mode`, which silently failed for
# the non-str shapes (a list never equals a string), so multi-mode guardrails
# were counted for no mode and every mode-based check reported NON-COMPLIANT.
# For the tag-based dict Mode only `default` is trusted: the spend log does not
# record which tags were active for the audited request, so counting tag
# overrides would over-report (a post_call tag override must not make a
# pre_call-default guardrail count as post_call).
# ---------------------------------------------------------------------------


class TestModeMatching:
    """Direct coverage of ComplianceChecker._mode_matches for every shape."""

    @pytest.mark.parametrize(
        "g_mode, mode, expected",
        [
            # None → defaults to pre_call only.
            (None, "pre_call", True),
            (None, "post_call", False),
            (None, "during_call", False),
            # str → exact match.
            ("pre_call", "pre_call", True),
            ("post_call", "pre_call", False),
            ("during_call", "during_call", True),
            # list → membership (the primary regression the fix addresses).
            (["pre_call", "post_call"], "pre_call", True),
            (["pre_call", "post_call"], "post_call", True),
            (["pre_call"], "post_call", False),
            ([], "pre_call", False),
            # tuple / set → membership.
            (("during_call",), "during_call", True),
            ({"pre_call", "post_call"}, "post_call", True),
            # dict → only `default` is trusted.
            ({"default": "pre_call"}, "pre_call", True),
            ({"default": "pre_call"}, "post_call", False),
            ({"default": ["pre_call", "post_call"]}, "post_call", True),
            # dict → per-tag overrides are IGNORED (not in the spend log).
            ({"default": "pre_call", "tags": {"eu": "post_call"}}, "pre_call", True),
            ({"default": "pre_call", "tags": {"eu": "post_call"}}, "post_call", False),
            ({"default": "pre_call", "tags": {"eu": ["during_call"]}}, "during_call", False),
            # dict → no usable default falls back to the pre_call convention.
            ({"tags": {"x": "post_call"}}, "pre_call", True),
            ({"tags": {"x": "post_call"}}, "post_call", False),
            ({}, "pre_call", True),
            ({}, "post_call", False),
            ({"default": 123}, "pre_call", True),  # junk default → pre_call convention
            # Unknown / unexpected top-level types never match.
            (5, "pre_call", False),
            (object(), "pre_call", False),
        ],
    )
    def test_mode_matches(self, g_mode, mode, expected):
        assert ComplianceChecker._mode_matches(g_mode, mode) is expected

    def test_list_mode_counted_for_each_listed_mode(self):
        """Regression: a guardrail configured with mode ["pre_call", "post_call"]
        is counted under BOTH modes. The old `g_mode == mode` compare matched
        neither (a list never equals a string), so every mode-based EU AI Act
        check reported NON-COMPLIANT."""
        data = ComplianceCheckRequest(
            request_id="req-mode-1",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "pii_masking",
                    "guardrail_mode": ["pre_call", "post_call"],
                    "guardrail_status": "success",
                }
            ],
        )
        checker = ComplianceChecker(data)
        assert len(checker._get_guardrails_by_mode("pre_call")) == 1
        assert len(checker._get_guardrails_by_mode("post_call")) == 1
        assert len(checker._get_guardrails_by_mode("during_call")) == 0
        results = {c.check_name: c.passed for c in checker.check_eu_ai_act()}
        assert results["Content screened before LLM"] is True

    def test_dict_tag_override_not_over_counted(self):
        """A tag override must NOT make the guardrail count for that mode: with
        default=pre_call and a post_call tag, it counts for pre_call only, so a
        pre-call compliance check cannot pass on a post-call tag override."""
        data = ComplianceCheckRequest(
            request_id="req-mode-2",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T12:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "pii_masking",
                    "guardrail_mode": {"default": "pre_call", "tags": {"eu": "post_call"}},
                    "guardrail_status": "success",
                }
            ],
        )
        checker = ComplianceChecker(data)
        assert len(checker._get_guardrails_by_mode("pre_call")) == 1
        assert len(checker._get_guardrails_by_mode("post_call")) == 0

    def test_none_mode_defaults_to_pre_call(self):
        """A guardrail logged without a mode counts as pre_call only."""
        data = ComplianceCheckRequest(
            request_id="req-mode-3",
            user_id="user-1",
            model="gpt-4",
            timestamp="2026-02-17T12:00:00Z",
            guardrail_information=[{"guardrail_name": "pii_masking", "guardrail_status": "success"}],
        )
        checker = ComplianceChecker(data)
        assert len(checker._get_guardrails_by_mode("pre_call")) == 1
        assert len(checker._get_guardrails_by_mode("post_call")) == 0
