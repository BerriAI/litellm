"""
Regression tests: caller-supplied credential/endpoint/project
kwargs must not override the deployment's server-pinned values in the Router's
``{**litellm_params, ..., **kwargs}`` merge.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import Router
from litellm._logging import verbose_router_logger
from litellm.router import (
    _DEPLOYMENT_OWNED_CREDENTIAL_KWARGS,
    _strip_deployment_owned_credential_kwargs,
)


@pytest.fixture(autouse=True)
def _proxy_running():
    """The strip is only active behind the proxy boundary. These tests exercise
    that security path, so enable the flag for their duration."""
    original = litellm.proxy_is_running
    litellm.proxy_is_running = True
    try:
        yield
    finally:
        litellm.proxy_is_running = original


def test_strip_helper_passthrough_when_proxy_not_running():
    """In direct SDK use (no proxy), deployment-owned kwargs must NOT be
    stripped, so existing per-call overrides like api_version on Azure
    keep working. The proxy is the security boundary; SDK callers build
    their own."""
    original = litellm.proxy_is_running
    litellm.proxy_is_running = False
    try:
        kwargs = {"api_version": "2024-02-01", "vertex_project": "p"}
        _strip_deployment_owned_credential_kwargs(kwargs)
        assert kwargs == {"api_version": "2024-02-01", "vertex_project": "p"}
    finally:
        litellm.proxy_is_running = original


def test_strip_helper_warns_once_on_dropped_nonempty_value():
    kwargs = {"temperature": 0.2, "vertex_project": "caller-project"}
    with patch.object(verbose_router_logger, "warning") as mock_warn:
        _strip_deployment_owned_credential_kwargs(kwargs)
    mock_warn.assert_called_once()
    logged_keys = mock_warn.call_args.args[1]
    assert "vertex_project" in logged_keys
    # The value is a credential surface and must never be logged.
    assert "caller-project" not in str(mock_warn.call_args)


def test_strip_helper_silent_when_nothing_dropped():
    with patch.object(verbose_router_logger, "warning") as mock_warn:
        _strip_deployment_owned_credential_kwargs({"temperature": 0.2})
    mock_warn.assert_not_called()


def test_strip_helper_silent_on_present_but_empty_values():
    with patch.object(verbose_router_logger, "warning") as mock_warn:
        _strip_deployment_owned_credential_kwargs(
            {"vertex_project": "", "aws_access_key_id": None, "region_name": {}}
        )
    mock_warn.assert_not_called()


def test_strip_helper_removes_credential_kwargs_in_place():
    kwargs = {
        "temperature": 0.2,
        "vertex_project": "caller-project",
        "aws_access_key_id": "AKIA-CALLER",
        "api_version": "caller-version",
        "base_model": "azure/caller-supplied",
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
            vertex_project="caller-project",
            vertex_credentials='{"private_key":"caller"}',
        )

    assert mock_completion.call_count == 1
    _, kwargs = mock_completion.call_args
    assert kwargs["vertex_project"] == "admin-project"
    assert kwargs["vertex_credentials"] == '{"private_key":"admin"}'
    assert "caller" not in str(kwargs["vertex_credentials"])


@pytest.mark.asyncio
async def test_async_completion_ignores_caller_credential_kwargs():
    router = _router()
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    mock_acompletion = AsyncMock(return_value=mock_response)

    with patch.object(litellm, "acompletion", mock_acompletion):
        await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            vertex_project="caller-project",
            aws_access_key_id="AKIA-CALLER",
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


def test_router_completion_warns_when_credential_kwarg_stripped():
    """The strip is security-correct but backwards-incompatible for SDK Router
    callers who previously passed per-call ``vertex_project`` / ``api_version``.
    A single warning when a non-empty value is dropped surfaces the change."""
    router = _router()
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])

    with (
        patch.object(litellm, "completion", MagicMock(return_value=mock_response)),
        patch.object(verbose_router_logger, "warning") as mock_warn,
    ):
        router.completion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            vertex_project="caller-project",
        )

    strip_warnings = [
        call
        for call in mock_warn.call_args_list
        if "deployment-owned kwargs" in str(call.args[0])
    ]
    assert len(strip_warnings) == 1
    assert "vertex_project" in strip_warnings[0].args[1]
    assert "caller-project" not in str(strip_warnings[0])


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


def _router_with_admin_key() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-deployment-secret",
                    "api_base": "https://admin.example/v1",
                },
            }
        ]
    )


def test_router_clears_deployment_api_key_on_base_override():
    """A caller who redirects api_base without supplying a key must not have the
    deployment's own api_key forwarded to the caller-controlled endpoint.

    Captures the kwargs actually forwarded to litellm.completion (the merged
    input_kwargs). Pre-fix the deployment api_key rode along to the new base;
    asserting it is gone rejects that.
    """
    router = _router_with_admin_key()
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    mock_completion = MagicMock(return_value=mock_response)

    with patch.object(litellm, "completion", mock_completion):
        router._completion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"model_group": "primary"},
            api_base="https://caller.example/v1",
        )

    assert mock_completion.call_count == 1
    _, forwarded = mock_completion.call_args
    assert forwarded.get("api_base") == "https://caller.example/v1"
    assert forwarded.get("api_key") != "sk-deployment-secret"


def test_router_forwards_caller_api_key_on_base_override():
    """When the caller supplies their own key alongside the base override, that
    key is used, not the deployment's."""
    router = _router_with_admin_key()
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    mock_completion = MagicMock(return_value=mock_response)

    with patch.object(litellm, "completion", mock_completion):
        router._completion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"model_group": "primary"},
            api_base="https://caller.example/v1",
            api_key="sk-caller-byok",
        )

    assert mock_completion.call_count == 1
    _, forwarded = mock_completion.call_args
    assert forwarded.get("api_key") == "sk-caller-byok"


@pytest.mark.asyncio
async def test_router_clears_deployment_api_key_on_base_override_async():
    """Async variant of the base-override key clearing."""
    router = _router_with_admin_key()
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])
    mock_acompletion = AsyncMock(return_value=mock_response)

    with patch.object(litellm, "acompletion", mock_acompletion):
        await router._acompletion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"model_group": "primary"},
            api_base="https://caller.example/v1",
        )

    assert mock_acompletion.call_count == 1
    _, forwarded = mock_acompletion.call_args
    assert forwarded.get("api_base") == "https://caller.example/v1"
    assert forwarded.get("api_key") != "sk-deployment-secret"
