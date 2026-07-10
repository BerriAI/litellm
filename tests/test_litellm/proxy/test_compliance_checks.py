from typing import List, Optional, Union

from litellm.proxy.compliance_checks import ComplianceChecker
from litellm.types.proxy.compliance_endpoints import ComplianceCheckRequest


def _make_checker(guardrail_mode: Optional[Union[str, List[str]]]) -> ComplianceChecker:
    return ComplianceChecker(
        ComplianceCheckRequest(
            request_id="req-1",
            user_id="user-1",
            model="gpt-5.2",
            timestamp="2026-07-05T00:00:00Z",
            guardrail_information=[
                {
                    "guardrail_name": "my-pii-guardrail",
                    "guardrail_mode": guardrail_mode,
                    "guardrail_status": "success",
                }
            ],
        )
    )


def test_list_mode_counts_as_pre_call():
    checker = _make_checker(["pre_call", "post_call"])
    assert len(checker._get_guardrails_by_mode("pre_call")) == 1
    assert all(check.passed for check in checker.check_gdpr())
    assert all(check.passed for check in checker.check_eu_ai_act())


def test_list_mode_without_requested_mode_is_excluded():
    checker = _make_checker(["post_call"])
    assert checker._get_guardrails_by_mode("pre_call") == []


def test_string_mode_matching_still_works():
    assert len(_make_checker("pre_call")._get_guardrails_by_mode("pre_call")) == 1
    assert _make_checker("post_call")._get_guardrails_by_mode("pre_call") == []


def test_missing_mode_defaults_to_pre_call():
    assert len(_make_checker(None)._get_guardrails_by_mode("pre_call")) == 1
    assert _make_checker(None)._get_guardrails_by_mode("post_call") == []
