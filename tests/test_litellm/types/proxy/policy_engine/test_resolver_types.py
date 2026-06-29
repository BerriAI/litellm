"""
Tests for pipeline field on policy CRUD types (resolver_types.py).
"""

import pytest
from pydantic import ValidationError

from litellm.types.proxy.policy_engine.resolver_types import (
    PolicyAttachmentCreateRequest,
    PolicyCreateRequest,
    PolicyDBResponse,
    PolicyUpdateRequest,
)
from litellm.types.proxy.policy_engine.policy_types import PolicyAttachment


def test_policy_create_request_with_pipeline():
    pipeline_data = {
        "mode": "pre_call",
        "steps": [
            {"guardrail": "g1", "on_fail": "next", "on_pass": "allow"},
            {"guardrail": "g2", "on_fail": "block", "on_pass": "allow"},
        ],
    }
    req = PolicyCreateRequest(
        policy_name="test-policy",
        guardrails_add=["g1", "g2"],
        pipeline=pipeline_data,
    )
    assert req.pipeline is not None
    assert req.pipeline["mode"] == "pre_call"
    assert len(req.pipeline["steps"]) == 2


def test_policy_create_request_without_pipeline():
    req = PolicyCreateRequest(
        policy_name="test-policy",
        guardrails_add=["g1"],
    )
    assert req.pipeline is None


def test_policy_update_request_with_pipeline():
    pipeline_data = {
        "mode": "pre_call",
        "steps": [
            {"guardrail": "g1", "on_fail": "block", "on_pass": "allow"},
        ],
    }
    req = PolicyUpdateRequest(pipeline=pipeline_data)
    assert req.pipeline is not None
    assert req.pipeline["steps"][0]["guardrail"] == "g1"


def test_policy_db_response_with_pipeline():
    pipeline_data = {
        "mode": "pre_call",
        "steps": [
            {"guardrail": "g1", "on_fail": "next", "on_pass": "allow"},
            {"guardrail": "g2", "on_fail": "block", "on_pass": "allow"},
        ],
    }
    resp = PolicyDBResponse(
        policy_id="test-id",
        policy_name="test-policy",
        guardrails_add=["g1", "g2"],
        pipeline=pipeline_data,
    )
    assert resp.pipeline is not None
    assert resp.pipeline["mode"] == "pre_call"
    dumped = resp.model_dump()
    assert dumped["pipeline"]["steps"][0]["guardrail"] == "g1"


def test_policy_db_response_without_pipeline():
    resp = PolicyDBResponse(
        policy_id="test-id",
        policy_name="test-policy",
    )
    assert resp.pipeline is None
    dumped = resp.model_dump()
    assert dumped["pipeline"] is None


def test_policy_create_request_roundtrip():
    pipeline_data = {
        "mode": "post_call",
        "steps": [
            {
                "guardrail": "g1",
                "on_fail": "modify_response",
                "on_pass": "next",
                "pass_data": True,
                "modify_response_message": "custom msg",
            },
        ],
    }
    req = PolicyCreateRequest(
        policy_name="roundtrip-test",
        guardrails_add=["g1"],
        pipeline=pipeline_data,
    )
    dumped = req.model_dump()
    restored = PolicyCreateRequest(**dumped)
    assert restored.pipeline == pipeline_data


def test_policy_attachment_create_request_rejects_empty_specific_scope():
    with pytest.raises(ValidationError, match="at least one non-empty selector"):
        PolicyAttachmentCreateRequest(policy_name="pii-policy")


def test_policy_attachment_create_request_rejects_empty_selector_list():
    with pytest.raises(ValidationError, match="at least one non-empty selector"):
        PolicyAttachmentCreateRequest(policy_name="pii-policy", teams=[])


def test_policy_attachment_create_request_allows_explicit_global_scope():
    request = PolicyAttachmentCreateRequest(
        policy_name="pii-policy",
        scope="*",
    )

    assert request.scope == "*"


@pytest.mark.parametrize("selector_field", ["teams", "keys", "models", "tags"])
def test_policy_attachment_create_request_allows_selector_scope(selector_field):
    selector_value = f"{selector_field}-a"
    request = PolicyAttachmentCreateRequest(
        policy_name="pii-policy",
        **{selector_field: [selector_value]},
    )

    assert getattr(request, selector_field) == [selector_value]


def test_policy_attachment_rejects_config_empty_specific_scope():
    with pytest.raises(ValidationError, match="at least one non-empty selector"):
        PolicyAttachment(policy="pii-policy")


def test_policy_attachment_rejects_config_empty_selector_list():
    with pytest.raises(ValidationError, match="at least one non-empty selector"):
        PolicyAttachment(policy="pii-policy", teams=[])


def test_policy_attachment_allows_config_explicit_global_scope():
    attachment = PolicyAttachment(policy="pii-policy", scope="*")

    assert attachment.scope == "*"


@pytest.mark.parametrize("selector_field", ["teams", "keys", "models", "tags"])
def test_policy_attachment_allows_config_selector_scope(selector_field):
    selector_value = f"{selector_field}-a"
    attachment = PolicyAttachment(
        policy="pii-policy",
        **{selector_field: [selector_value]},
    )

    assert getattr(attachment, selector_field) == [selector_value]
