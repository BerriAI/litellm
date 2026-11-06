from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.workflow_execution_endpoints import (
    TestRunRequest as PaidRunRequest, WorkflowInvocation, _execution_key,
)


@pytest.mark.parametrize("overrides", [
    {"is_session_token": True}, {"via_virtual_key": False}, {"team_id": "other"}, {"token": None},
])
def test_execution_requires_a_team_bound_application_key(overrides: dict[str, object]) -> None:
    owner: Final = UserAPIKeyAuth(token="owner", team_id="team").model_copy(update={"via_virtual_key": True})
    assert _execution_key(owner, "team") == "owner"
    with pytest.raises(HTTPException) as denied:
        _execution_key(owner.model_copy(update=overrides), "team")
    assert denied.value.status_code == 403


@pytest.mark.parametrize("extra", [{"stream": True}, {"tools": []}, {"temperature": 0}])
def test_workflow_invocation_rejects_unsupported_chat_parameters(extra: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkflowInvocation.model_validate({
            "model": "workflow/example", "messages": [{"role": "user", "content": "go"}], **extra,
        })


@pytest.mark.parametrize("consent", ["true", 1, None])
def test_paid_test_requires_explicit_boolean_billing_consent(consent: object) -> None:
    with pytest.raises(ValidationError):
        PaidRunRequest.model_validate({"messages": [{"role": "user", "content": "go"}], "authorize_billing": consent})
