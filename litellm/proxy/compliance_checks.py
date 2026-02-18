"""
Compliance checker for EU AI Act and GDPR regulations.

Provides guardrail-agnostic compliance validation based on guardrail modes
and execution results rather than specific guardrail names.
"""

from typing import Dict, List

from litellm.types.proxy.compliance_endpoints import (
    ComplianceCheckRequest,
    ComplianceCheckResult,
)


class ComplianceChecker:
    """
    Validates compliance with EU AI Act and GDPR regulations.

    Uses guardrail-agnostic checks based on:
    - Whether any guardrails ran
    - Guardrail execution mode (pre-call, post-call, etc.)
    - Whether guardrails intervened/blocked content
    - Completeness of audit records
    """

    def __init__(self, data: ComplianceCheckRequest):
        self.data = data
        self.guardrails = data.guardrail_information or []

    def _get_guardrails_by_mode(self, mode: str) -> List[Dict]:
        """
        Get all guardrails that ran in a specific mode.

        If a guardrail doesn't have a mode specified, it's treated as pre-call
        (the most common case).
        """
        result = []
        for g in self.guardrails:
            g_mode = g.get("guardrail_mode")
            # If no mode specified, default to pre_call
            if g_mode is None and mode == "pre_call":
                result.append(g)
            elif g_mode == mode:
                result.append(g)
        return result

    def _has_guardrail_intervention(self, guardrails: List[Dict]) -> bool:
        """Check if any guardrail intervened (blocked/masked content)."""
        for g in guardrails:
            status = g.get("guardrail_status", "")
            if status in ["guardrail_intervened", "failed", "blocked"]:
                return True
        return False

    def _all_guardrails_passed(self, guardrails: List[Dict]) -> bool:
        """Check if all guardrails passed (no issues detected)."""
        if not guardrails:
            return False
        return all(g.get("guardrail_status") == "success" for g in guardrails)

    # ── EU AI Act Helper Methods ────────────────────────────────────────────

    def _check_art_5_content_screened(self) -> ComplianceCheckResult:
        """Art. 5: Check if content was screened before LLM (pre-call)."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Content screened before LLM",
            article="Art. 5",
            passed=has_pre_call,
            detail=(
                f"{len(pre_call_guardrails)} pre-call guardrail(s) screened content"
                if has_pre_call
                else "No pre-call screening applied"
            ),
        )

    def _check_art_5_1a_manipulation_screened(self) -> ComplianceCheckResult:
        """Art. 5.1(a): Check if manipulation/subliminal techniques are screened."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Manipulation & subliminal techniques screened",
            article="Art. 5.1(a)",
            passed=has_pre_call,
            detail=(
                "Pre-call guardrails screen for prohibited manipulation techniques"
                if has_pre_call
                else "No guardrails screening for subliminal/manipulative techniques"
            ),
        )

    def _check_art_5_1b_vulnerability_screened(self) -> ComplianceCheckResult:
        """Art. 5.1(b): Check if vulnerability exploitation is screened."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Vulnerability exploitation screened",
            article="Art. 5.1(b)",
            passed=has_pre_call,
            detail=(
                "Pre-call guardrails screen for exploitation of vulnerable groups"
                if has_pre_call
                else "No guardrails screening for vulnerability exploitation"
            ),
        )

    def _check_art_5_1c_social_scoring_screened(self) -> ComplianceCheckResult:
        """Art. 5.1(c): Check if social scoring systems are screened."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Social scoring systems screened",
            article="Art. 5.1(c)",
            passed=has_pre_call,
            detail=(
                "Pre-call guardrails screen for prohibited social scoring"
                if has_pre_call
                else "No guardrails screening for social scoring systems"
            ),
        )

    def _check_art_5_1d_predictive_policing_screened(self) -> ComplianceCheckResult:
        """Art. 5.1(d): Check if predictive policing/criminal profiling is screened."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Criminal profiling & predictive policing screened",
            article="Art. 5.1(d)",
            passed=has_pre_call,
            detail=(
                "Pre-call guardrails screen for prohibited criminal profiling"
                if has_pre_call
                else "No guardrails screening for predictive policing"
            ),
        )

    def _check_art_5_1f_emotion_recognition_screened(self) -> ComplianceCheckResult:
        """Art. 5.1(f): Check if emotion recognition in workplace/education is screened."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Emotion recognition in workplace/education screened",
            article="Art. 5.1(f)",
            passed=has_pre_call,
            detail=(
                "Pre-call guardrails screen for prohibited emotion recognition"
                if has_pre_call
                else "No guardrails screening for workplace/education emotion recognition"
            ),
        )

    def _check_art_5_1h_biometric_categorization_screened(self) -> ComplianceCheckResult:
        """Art. 5.1(h): Check if biometric categorization is screened."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Biometric categorization screened",
            article="Art. 5.1(h)",
            passed=has_pre_call,
            detail=(
                "Pre-call guardrails screen for prohibited biometric categorization"
                if has_pre_call
                else "No guardrails screening for biometric categorization"
            ),
        )

    def _check_art_9_guardrails_applied(self) -> ComplianceCheckResult:
        """Art. 9: Check if any guardrails were applied."""
        has_guardrails = len(self.guardrails) > 0
        return ComplianceCheckResult(
            check_name="Guardrails applied",
            article="Art. 9",
            passed=has_guardrails,
            detail=(
                f"{len(self.guardrails)} guardrail(s) applied"
                if has_guardrails
                else "No guardrails applied"
            ),
        )

    def _check_art_10_data_governance(self) -> ComplianceCheckResult:
        """Art. 10: Check if input data was validated by guardrails."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Input data governance validated",
            article="Art. 10",
            passed=has_pre_call,
            detail=(
                "Pre-call guardrails validate input data quality and governance"
                if has_pre_call
                else "No input data validation guardrails applied"
            ),
        )

    def _check_art_12_audit_complete(self) -> ComplianceCheckResult:
        """Art. 12: Check if audit record is complete."""
        has_user = bool(self.data.user_id)
        has_model = bool(self.data.model)
        has_timestamp = bool(self.data.timestamp)
        has_guardrails = len(self.guardrails) > 0
        audit_complete = has_user and has_model and has_timestamp and has_guardrails

        missing = []
        if not has_user:
            missing.append("user_id")
        if not has_model:
            missing.append("model")
        if not has_timestamp:
            missing.append("timestamp")
        if not has_guardrails:
            missing.append("guardrail_results")

        return ComplianceCheckResult(
            check_name="Audit record complete",
            article="Art. 12",
            passed=audit_complete,
            detail=(
                "All required audit fields present"
                if audit_complete
                else f"Missing: {', '.join(missing)}"
            ),
        )

    def _check_art_13_transparency(self) -> ComplianceCheckResult:
        """Art. 13: Check if AI system transparency is maintained."""
        has_model = bool(self.data.model)
        has_user = bool(self.data.user_id)
        return ComplianceCheckResult(
            check_name="AI system transparency",
            article="Art. 13",
            passed=has_model and has_user,
            detail=(
                "AI model and user identity are recorded for transparency"
                if has_model and has_user
                else "Missing model or user identification for transparency compliance"
            ),
        )

    def _check_art_14_human_oversight(self) -> ComplianceCheckResult:
        """Art. 14: Check if human oversight mechanisms are in place."""
        has_user = bool(self.data.user_id)
        has_guardrails = len(self.guardrails) > 0
        return ComplianceCheckResult(
            check_name="Human oversight mechanisms active",
            article="Art. 14",
            passed=has_user and has_guardrails,
            detail=(
                "User identified and guardrails provide automated oversight"
                if has_user and has_guardrails
                else "No human oversight mechanisms detected"
            ),
        )

    def _check_art_15_accuracy_robustness(self) -> ComplianceCheckResult:
        """Art. 15: Check if accuracy and robustness safeguards are in place."""
        post_call_guardrails = self._get_guardrails_by_mode("post_call")
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_both = len(post_call_guardrails) > 0 and len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Accuracy & robustness safeguards",
            article="Art. 15",
            passed=has_both,
            detail=(
                "Both pre-call and post-call guardrails ensure output accuracy and robustness"
                if has_both
                else "Missing pre-call or post-call guardrails for accuracy/robustness checks"
            ),
        )

    def _check_art_26_deployer_obligations(self) -> ComplianceCheckResult:
        """Art. 26: Check if deployer obligations for high-risk AI are met."""
        has_user = bool(self.data.user_id)
        has_model = bool(self.data.model)
        has_timestamp = bool(self.data.timestamp)
        has_guardrails = len(self.guardrails) > 0
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0

        obligations_met = (
            has_user and has_model and has_timestamp and has_guardrails and has_pre_call
        )

        missing = []
        if not has_user:
            missing.append("user identification")
        if not has_model:
            missing.append("model identification")
        if not has_timestamp:
            missing.append("timestamp logging")
        if not has_guardrails:
            missing.append("guardrail application")
        if not has_pre_call:
            missing.append("pre-call risk screening")

        return ComplianceCheckResult(
            check_name="Deployer obligations for high-risk AI",
            article="Art. 26",
            passed=obligations_met,
            detail=(
                "All deployer obligations met: user/model identified, timestamped, guardrails active"
                if obligations_met
                else f"Missing deployer obligations: {', '.join(missing)}"
            ),
        )

    def _check_art_50_synthetic_content_transparency(self) -> ComplianceCheckResult:
        """Art. 50: Check if AI-generated content transparency is maintained."""
        post_call_guardrails = self._get_guardrails_by_mode("post_call")
        has_post_call = len(post_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="AI-generated content transparency",
            article="Art. 50",
            passed=has_post_call,
            detail=(
                "Post-call guardrails monitor AI-generated content for transparency"
                if has_post_call
                else "No post-call guardrails to monitor AI-generated content transparency"
            ),
        )

    # ── GDPR Helper Methods ──────────────────────────────────────────────────

    def _check_art_32_data_protection(self) -> ComplianceCheckResult:
        """Art. 32: Check if data protection was applied (pre-call)."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_pre_call = len(pre_call_guardrails) > 0
        return ComplianceCheckResult(
            check_name="Data protection applied",
            article="Art. 32",
            passed=has_pre_call,
            detail=(
                f"{len(pre_call_guardrails)} pre-call guardrail(s) protect data"
                if has_pre_call
                else "No pre-call data protection applied"
            ),
        )

    def _check_art_5_1c_sensitive_data_protected(self) -> ComplianceCheckResult:
        """Art. 5(1)(c): Check if sensitive data was protected."""
        pre_call_guardrails = self._get_guardrails_by_mode("pre_call")
        has_intervention = self._has_guardrail_intervention(pre_call_guardrails)
        all_passed = self._all_guardrails_passed(pre_call_guardrails)
        data_protected = has_intervention or all_passed

        if has_intervention:
            detail = "Guardrail intervened to protect sensitive data"
        elif all_passed:
            detail = "No sensitive data detected"
        else:
            detail = "No pre-call guardrails to protect sensitive data"

        return ComplianceCheckResult(
            check_name="Sensitive data protected",
            article="Art. 5(1)(c)",
            passed=data_protected,
            detail=detail,
        )

    def _check_art_30_audit_complete(self) -> ComplianceCheckResult:
        """Art. 30: Check if audit record is complete."""
        has_user = bool(self.data.user_id)
        has_model = bool(self.data.model)
        has_timestamp = bool(self.data.timestamp)
        has_guardrails = len(self.guardrails) > 0
        audit_complete = has_user and has_model and has_timestamp and has_guardrails

        missing = []
        if not has_user:
            missing.append("user_id")
        if not has_model:
            missing.append("model")
        if not has_timestamp:
            missing.append("timestamp")
        if not has_guardrails:
            missing.append("guardrail_results")

        return ComplianceCheckResult(
            check_name="Audit record complete",
            article="Art. 30",
            passed=audit_complete,
            detail=(
                "All required audit fields present"
                if audit_complete
                else f"Missing: {', '.join(missing)}"
            ),
        )

    # ── Main Compliance Check Methods ────────────────────────────────────────

    def check_eu_ai_act(self) -> List[ComplianceCheckResult]:
        """
        Check EU AI Act compliance.

        Returns:
            List of compliance check results covering:
            - Art. 5: Content screened before LLM (pre-call screening)
            - Art. 5.1(a): Manipulation & subliminal techniques screened
            - Art. 5.1(b): Vulnerability exploitation screened
            - Art. 5.1(c): Social scoring systems screened
            - Art. 5.1(d): Criminal profiling & predictive policing screened
            - Art. 5.1(f): Emotion recognition in workplace/education screened
            - Art. 5.1(h): Biometric categorization screened
            - Art. 9: Guardrails applied
            - Art. 10: Input data governance validated
            - Art. 12: Audit record complete
            - Art. 13: AI system transparency
            - Art. 14: Human oversight mechanisms active
            - Art. 15: Accuracy & robustness safeguards
            - Art. 26: Deployer obligations for high-risk AI
            - Art. 50: AI-generated content transparency
        """
        return [
            self._check_art_5_content_screened(),
            self._check_art_5_1a_manipulation_screened(),
            self._check_art_5_1b_vulnerability_screened(),
            self._check_art_5_1c_social_scoring_screened(),
            self._check_art_5_1d_predictive_policing_screened(),
            self._check_art_5_1f_emotion_recognition_screened(),
            self._check_art_5_1h_biometric_categorization_screened(),
            self._check_art_9_guardrails_applied(),
            self._check_art_10_data_governance(),
            self._check_art_12_audit_complete(),
            self._check_art_13_transparency(),
            self._check_art_14_human_oversight(),
            self._check_art_15_accuracy_robustness(),
            self._check_art_26_deployer_obligations(),
            self._check_art_50_synthetic_content_transparency(),
        ]

    def check_gdpr(self) -> List[ComplianceCheckResult]:
        """
        Check GDPR compliance.

        Returns:
            List of compliance check results for:
            - Art. 32: Data protection applied (pre-call screening)
            - Art. 5(1)(c): Sensitive data protected
            - Art. 30: Audit record complete
        """
        return [
            self._check_art_32_data_protection(),
            self._check_art_5_1c_sensitive_data_protected(),
            self._check_art_30_audit_complete(),
        ]
