"""
Regression tests: caller-supplied credential/endpoint/project
kwargs must not override the deployment's server-pinned values in the Router's
``{**litellm_params, ..., **kwargs}`` merge.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import Router
from litellm.router import (
    _DEPLOYMENT_OWNED_CREDENTIAL_KWARGS,
    _strip_deployment_owned_credential_kwargs,
)


def test_strip_helper_removes_credential_kwargs_in_place():
    kwargs = {
        "temperature": 0.2,
        "vertex_project": "attacker-project",
        "aws_access_key_id": "AKIA-ATTACKER",
        "api_version": "attacker-version",
        "base_model": "azure/forged",
    }
    _strip_deployment_owned_credential_kwargs(kwargs)
    assert kwargs == {"temperature": 0.2}


def test_strip_helper_leaves_sanctioned_clientside_keys():
    # api_key / api_base / base_url go through the explicit opt-in path, not
    # this merge, so they must NOT be in the stripped set.
    for key in ("api_key", "api_base", "base_url"):
        assert key not in _DEPLOYMENT_OWNED_CREDENTIAL_KWARGS


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.0-flash",
                    "vertex_project": "admin-project",
                    "vertex_credentials": '{"private_key":"admin"}',
                    "api_key": "sk-admin",
                },
            }
        ]
    )


def test_sync_completion_ignores_caller_credential_kwargs():
    router = _router()
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    mock_completion = MagicMock(return_value=mock_response)

    with patch.object(litellm, "completion", mock_completion):
        router.completion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            vertex_project="attacker-project",
            vertex_credentials='{"private_key":"attacker"}',
        )

    assert mock_completion.call_count == 1
    _, kwargs = mock_completion.call_args
    assert kwargs["vertex_project"] == "admin-project"
    assert kwargs["vertex_credentials"] == '{"private_key":"admin"}'
    assert "attacker" not in str(kwargs["vertex_credentials"])


@pytest.mark.asyncio
async def test_async_completion_ignores_caller_credential_kwargs():
    router = _router()
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    mock_acompletion = AsyncMock(return_value=mock_response)

    with patch.object(litellm, "acompletion", mock_acompletion):
        await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            vertex_project="attacker-project",
            aws_access_key_id="AKIA-ATTACKER",
        )

    assert mock_acompletion.call_count == 1
    _, kwargs = mock_acompletion.call_args
    assert kwargs["vertex_project"] == "admin-project"
    assert kwargs.get("aws_access_key_id") is None


def test_router_strips_caller_credential_kwargs_before_merge():
    """The deployment-owned strip must run BEFORE the
    ``{**litellm_params, **kwargs}`` merge.

    The deployment pins vertex_project="deployment-value"; the caller supplies a
    colliding vertex_project="caller-value". Three possible orderings produce
    three distinct forwarded values:
      - strip BEFORE merge (correct): the deployment value survives.
      - strip AFTER merge: the field is popped entirely -> absent.
      - no strip: kwargs wins the merge -> caller value.
    Asserting the forwarded value equals the deployment value rejects both the
    after-merge mutation (absent) and the no-strip mutation (caller value).
    """
    router = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.0-flash",
                    "vertex_project": "deployment-value",
                },
            }
        ]
    )
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    mock_completion = MagicMock(return_value=mock_response)

    with patch.object(litellm, "completion", mock_completion):
        router._completion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            vertex_project="caller-value",
        )

    assert mock_completion.call_count == 1
    _, forwarded = mock_completion.call_args
    assert "vertex_project" in forwarded
    assert forwarded["vertex_project"] == "deployment-value"


@pytest.mark.asyncio
async def test_router_strips_caller_credential_kwargs_before_merge_async():
    """Async variant: the strip in _acompletion must precede the merge."""
    router = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.0-flash",
                    "vertex_project": "deployment-value",
                },
            }
        ]
    )
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    mock_acompletion = AsyncMock(return_value=mock_response)

    with patch.object(litellm, "acompletion", mock_acompletion):
        await router._acompletion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            vertex_project="caller-value",
        )

    assert mock_acompletion.call_count == 1
    _, forwarded = mock_acompletion.call_args
    assert "vertex_project" in forwarded
    assert forwarded["vertex_project"] == "deployment-value"
