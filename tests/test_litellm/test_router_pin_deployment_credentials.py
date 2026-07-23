"""Regression tests for litellm.pin_deployment_credentials (issue #32892).

When enabled, caller-supplied aws_* credential/identity overrides in the request
body must be ignored so the deployment-configured values are used.
"""

import pytest

import litellm
from litellm import Router

CONFIG_ROLE = "arn:aws:iam::111111111111:role/config-role"
CALLER_ROLE = "arn:aws:iam::222222222222:role/caller-role"


def _bedrock_deployment() -> dict:
    return {
        "model_name": "claude",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "aws_role_name": CONFIG_ROLE,
            "aws_session_name": "config-session",
            "aws_region_name": "us-east-1",
        },
    }


def _merged(deployment: dict, kwargs: dict) -> dict:
    """Mirror the router call-site merge: config litellm_params first, caller kwargs last."""
    return {**deployment["litellm_params"], **kwargs}


@pytest.fixture(autouse=True)
def _reset_flag():
    original = litellm.pin_deployment_credentials
    yield
    litellm.pin_deployment_credentials = original


def test_pinned_credentials_ignore_caller_override():
    litellm.pin_deployment_credentials = True
    deployment = _bedrock_deployment()
    kwargs = {
        "aws_role_name": CALLER_ROLE,
        "aws_access_key_id": "AKIA-caller",
        "aws_secret_access_key": "caller-secret",
    }

    Router._pin_deployment_aws_credentials(deployment=deployment, kwargs=kwargs)

    assert "aws_role_name" not in kwargs
    assert "aws_access_key_id" not in kwargs
    assert "aws_secret_access_key" not in kwargs

    merged = _merged(deployment, kwargs)
    assert merged["aws_role_name"] == CONFIG_ROLE
    # A credential param the caller tried to inject but config never set must not leak through.
    assert "aws_access_key_id" not in merged


def test_region_is_not_pinned():
    litellm.pin_deployment_credentials = True
    deployment = _bedrock_deployment()
    kwargs = {"aws_role_name": CALLER_ROLE, "aws_region_name": "us-west-2"}

    Router._pin_deployment_aws_credentials(deployment=deployment, kwargs=kwargs)

    assert "aws_region_name" in kwargs
    assert _merged(deployment, kwargs)["aws_region_name"] == "us-west-2"


def test_disabled_keeps_caller_override():
    litellm.pin_deployment_credentials = False
    deployment = _bedrock_deployment()
    kwargs = {"aws_role_name": CALLER_ROLE}

    Router._pin_deployment_aws_credentials(deployment=deployment, kwargs=kwargs)

    assert kwargs["aws_role_name"] == CALLER_ROLE
    assert _merged(deployment, kwargs)["aws_role_name"] == CALLER_ROLE


def test_missing_model_is_noop():
    litellm.pin_deployment_credentials = True
    deployment = {"model_name": "claude", "litellm_params": {"aws_role_name": CONFIG_ROLE}}
    kwargs = {"aws_role_name": CALLER_ROLE}

    Router._pin_deployment_aws_credentials(deployment=deployment, kwargs=kwargs)

    assert kwargs["aws_role_name"] == CALLER_ROLE


def test_sagemaker_provider_is_pinned():
    litellm.pin_deployment_credentials = True
    deployment = {
        "model_name": "sm",
        "litellm_params": {"model": "sagemaker_chat/my-endpoint", "aws_role_name": CONFIG_ROLE},
    }
    kwargs = {"aws_role_name": CALLER_ROLE}

    Router._pin_deployment_aws_credentials(deployment=deployment, kwargs=kwargs)

    assert "aws_role_name" not in kwargs


def test_non_aws_provider_untouched():
    litellm.pin_deployment_credentials = True
    deployment = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
    }
    kwargs = {"aws_role_name": CALLER_ROLE}

    Router._pin_deployment_aws_credentials(deployment=deployment, kwargs=kwargs)

    assert kwargs["aws_role_name"] == CALLER_ROLE


def test_end_to_end_via_update_kwargs_with_deployment():
    litellm.pin_deployment_credentials = True
    router = Router(model_list=[_bedrock_deployment()])
    deployment = (router.get_model_list(model_name="claude") or [])[0]
    kwargs = {
        "model": "claude",
        "messages": [{"role": "user", "content": "hi"}],
        "aws_role_name": CALLER_ROLE,
    }

    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    assert "aws_role_name" not in kwargs
    assert _merged(deployment, kwargs)["aws_role_name"] == CONFIG_ROLE
