import sys
import os
import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.guardrails.guardrail_hooks.model_armor.model_armor import ModelArmorGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache

@pytest.mark.asyncio
async def test_model_armor_pre_call_hook_inspect_and_deidentify():
    """
    Test Model Armor guardrail pre-call hook for both inspectResult and deidentifyResult handling.
    """
    guardrail = ModelArmorGuardrail(
        template_id="dummy-template",
        project_id="dummy-project",
        location="us-central1",
        credentials=None,
    )
    armor_response = {
        "sanitizationResult": {
            "filterResults": [
                {
                    "sdpFilterResult": {
                        "inspectResult": {
                            "executionState": "EXECUTION_SUCCESS",
                            "matchState": "NO_MATCH_FOUND",
                            "findings": []
                        },
                        "deidentifyResult": {
                            "executionState": "EXECUTION_SUCCESS",
                            "matchState": "MATCH_FOUND",
                            "data": {"text": "sanitized text here"}
                        }
                    }
                }
            ]
        }
    }
    with patch.object(guardrail, "make_model_armor_request", AsyncMock(return_value=armor_response)):
        user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
        cache = DualCache()
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My SSN is 123-45-6789."}
            ],
            "model": "gpt-3.5-turbo",
            "metadata": {}
        }
        guardrail.mask_request_content = True
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="completion"
            )
        assert exc_info.value.status_code == 400
        assert "Content blocked by Model Armor" in str(exc_info.value.detail)

def test_model_armor_should_block_content():
    guardrail = ModelArmorGuardrail(
        template_id="dummy-template",
        project_id="dummy-project",
        location="us-central1",
        credentials=None,
    )
    # Block on inspectResult
    armor_response_inspect = {
        "sanitizationResult": {
            "filterResults": [
                {"sdpFilterResult": {"inspectResult": {"matchState": "MATCH_FOUND"}}}
            ]
        }
    }
    assert guardrail._should_block_content(armor_response_inspect)
    # Block on deidentifyResult
    armor_response_deidentify = {
        "sanitizationResult": {
            "filterResults": [
                {"sdpFilterResult": {"deidentifyResult": {"matchState": "MATCH_FOUND"}}}
            ]
        }
    }
    assert guardrail._should_block_content(armor_response_deidentify)
    # No block if neither
    armor_response_none = {
        "sanitizationResult": {
            "filterResults": [
                {"sdpFilterResult": {"inspectResult": {"matchState": "NO_MATCH_FOUND"}, "deidentifyResult": {"matchState": "NO_MATCH_FOUND"}}}
            ]
        }
    }
    assert not guardrail._should_block_content(armor_response_none)
