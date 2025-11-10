"""
Regression test for removing unnecessary dict.copy() in completion hot paths.

Verifies that spreading deployment["litellm_params"] directly (without copy)
doesn't cause side effects that mutate the deployment in router.model_list.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router
from unittest.mock import AsyncMock, Mock, patch


@pytest.mark.asyncio
async def test_acompletion_deployment_not_mutated():
    """
    Test async completion doesn't mutate deployment when .copy() is removed.
    
    Optimization: Remove deployment["litellm_params"].copy() in _acompletion
    since data is only read and spread into input_kwargs dict.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "temperature": 0.7,
                },
            }
        ]
    )
    
    deployment_before = router.get_deployment_by_model_group_name("gpt-3.5")
    assert deployment_before is not None
    original_params = deployment_before.litellm_params.model_dump()
    
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        from litellm import ModelResponse
        
        mock_acompletion.return_value = ModelResponse(
            id="test",
            choices=[{"message": {"role": "assistant", "content": "test"}, "index": 0}],
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        
        try:
            await router.acompletion(
                model="gpt-3.5",
                messages=[{"role": "user", "content": "test"}],
            )
        except Exception:
            pass
    
    # Critical: Deployment params must be unchanged
    deployment_after = router.get_deployment_by_model_group_name("gpt-3.5")
    assert deployment_after is not None
    assert deployment_after.litellm_params.model_dump() == original_params


def test_completion_deployment_not_mutated():
    """
    Test sync completion doesn't mutate deployment when .copy() is removed.
    
    Optimization: Remove deployment["litellm_params"].copy() in _completion
    since data is only read and spread into input_kwargs dict.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "max_tokens": 100,
                },
            }
        ]
    )
    
    deployment_before = router.get_deployment_by_model_group_name("gpt-3.5")
    assert deployment_before is not None
    original_params = deployment_before.litellm_params.model_dump()
    
    with patch("litellm.completion", new_callable=Mock) as mock_completion:
        from litellm import ModelResponse
        
        mock_completion.return_value = ModelResponse(
            id="test",
            choices=[{"message": {"role": "assistant", "content": "test"}, "index": 0}],
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        
        try:
            router.completion(
                model="gpt-3.5",
                messages=[{"role": "user", "content": "test"}],
            )
        except Exception:
            pass
    
    # Critical: Deployment params must be unchanged
    deployment_after = router.get_deployment_by_model_group_name("gpt-3.5")
    assert deployment_after is not None
    assert deployment_after.litellm_params.model_dump() == original_params

