"""Regression tests for the /guardrails/test exec timeout fix (#28259).

Module-level exec() in TestCustomCodeGuardrail runs inside the same
timeout-protected ThreadPoolExecutor scope as apply_fn, so a top-level
infinite loop in otherwise-valid Python returns an execution-timeout
error instead of hanging the handler thread indefinitely.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_endpoints import (
    TestCustomCodeGuardrailRequest,
    test_custom_code_guardrail as _test_custom_code_guardrail,
)

INFINITE_LOOP_CODE = """
while True:
    pass

def apply_guardrail(inputs, request_data, input_type):
    return {"action": "allow"}
"""

VALID_CODE = """
def apply_guardrail(inputs, request_data, input_type):
    return {"action": "allow"}
"""


def _admin_key() -> UserAPIKeyAuth:
    key = MagicMock(spec=UserAPIKeyAuth)
    key.user_role = LitellmUserRoles.PROXY_ADMIN
    return key


def _request(code: str) -> TestCustomCodeGuardrailRequest:
    return TestCustomCodeGuardrailRequest(
        custom_code=code,
        test_input={"texts": ["hello"]},
        input_type="texts",
        request_data={},
    )


@pytest.mark.asyncio
async def test_top_level_infinite_loop_times_out():
    """A module-level infinite loop must hit the execution timeout, not hang."""
    result = await _test_custom_code_guardrail(
        request=_request(INFINITE_LOOP_CODE),
        user_api_key_dict=_admin_key(),
    )
    assert result.success is False
    assert result.error_type == "execution"
    assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_valid_code_still_runs():
    """Well-formed code defines apply_guardrail and executes normally."""
    result = await _test_custom_code_guardrail(
        request=_request(VALID_CODE),
        user_api_key_dict=_admin_key(),
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_missing_apply_guardrail_reports_compilation():
    """Code without apply_guardrail gets the compilation error type."""
    result = await _test_custom_code_guardrail(
        request=_request("x = 1\n"),
        user_api_key_dict=_admin_key(),
    )
    assert result.success is False
    assert result.error_type == "compilation"
    assert "apply_guardrail" in result.error


@pytest.mark.asyncio
async def test_non_admin_forbidden():
    """Non-admin callers are rejected before any code executes."""
    key = MagicMock(spec=UserAPIKeyAuth)
    key.user_role = LitellmUserRoles.INTERNAL_USER
    with pytest.raises(HTTPException) as excinfo:
        await _test_custom_code_guardrail(
            request=_request(VALID_CODE),
            user_api_key_dict=key,
        )
    assert excinfo.value.status_code == 403
