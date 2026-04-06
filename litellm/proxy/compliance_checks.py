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
            List of compliance check results for:
            - Art. 9: Guardrails applied
            - Art. 5: Content screened before LLM (pre-call screening)
            - Art. 12: Audit record complete
        """
        return [
            self._check_art_9_guardrails_applied(),
            self._check_art_5_content_screened(),
            self._check_art_12_audit_complete(),
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
