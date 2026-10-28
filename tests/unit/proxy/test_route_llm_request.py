import asyncio
import contextlib
import gc
import threading
import weakref
import httpx
import json
import litellm
import pytest
import redis
import respx


from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping
from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4
from typing import Final
from unittest.mock import MagicMock

from fakeredis import TcpFakeServer
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.route_llm_request import (
    JSON_OBJECT_ADAPTER,
    ProxyModelNotFoundError,
    UserConfigRouters,
    route_request,
    user_config_routers,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router_strategy.adaptive_router.config import (
    ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY,
    ADAPTIVE_ROUTER_RESPONSE_HEADER,
)
from litellm.types.router import RouterErrors, RouterGeneralSettings, RouterRateLimitError


@pytest.mark.parametrize(
    "route_type, required_body_params",
    [
        ("atext_completion", {"prompt": "Hello"}),
        ("acompletion", {"messages": [{"role": "user", "content": "Hello"}]}),
        ("aembedding", {"input": "Hello"}),
        ("aimage_generation", {"prompt": "a cat"}),
        ("aspeech", {"input": "Hello"}),
        ("atranscription", {"file": b"audio"}),
        ("amoderation", {"input": "Hello"}),
        ("arerank", {"query": "Hello", "documents": ["hi"]}),
    ],
)
@pytest.mark.asyncio
async def test_route_request_dynamic_credentials(route_type, required_body_params):
    data = {
        "model": "openai/gpt-4o-mini-2024-07-18",
        "api_key": "my-bad-key",
        "api_base": "https://api.openai.com/v1 ",
        **required_body_params,
    }
    llm_router = MagicMock()
    # Ensure that the dynamic method exists on the llm_router mock.
    getattr(llm_router, route_type).return_value = "fake_response"

    response = await route_request(data, llm_router, None, route_type)
    # Optionally verify the response if needed:
    assert response == "fake_response"
    # Now assert that the dynamic method was called once with the expected kwargs.
    getattr(llm_router, route_type).assert_called_once_with(**data)


@pytest.mark.asyncio
async def test_route_request_proxy_admin_can_call_all_team_scoped_deployments_without_team_id():
    import litellm

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    router = litellm.Router(
        model_list=[
            {
                "model_name": "internal-team-azure-east",
                "litellm_params": {
                    "model": "azure/gpt-4o",
                    "api_key": "fake",
                    "api_base": "https://east.example.openai.azure.com",
                    "api_version": "2024-02-15-preview",
                    "mock_response": "east",
                },
                "model_info": {
                    "id": "team-azure-east",
                    "team_id": "team-a",
                    "team_public_model_name": "team-azure",
                },
            },
            {
                "model_name": "internal-team-azure-west",
                "litellm_params": {
                    "model": "azure/gpt-4o",
                    "api_key": "fake",
                    "api_base": "https://west.example.openai.azure.com",
                    "api_version": "2024-02-15-preview",
                    "mock_response": "west",
                },
                "model_info": {
                    "id": "team-azure-west",
                    "team_id": "team-a",
                    "team_public_model_name": "team-azure",
                },
            },
        ]
    )
    admin_auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    data = {
        "model": "team-azure",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"user_api_key_auth": admin_auth},
    }

    llm_call = await route_request(
        data=data,
        llm_router=router,
        user_model=None,
        route_type="acompletion",
        user_api_key_dict=admin_auth,
    )
    response = await llm_call
    deployments = await router.async_get_healthy_deployments(
        model="team-azure",
        request_kwargs=data,
    )

    assert response.choices[0].message.content in {"east", "west"}
    assert {deployment["model_info"]["id"] for deployment in deployments} == {
        "team-azure-east",
        "team-azure-west",
    }

    non_admin_auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    with pytest.raises(ProxyModelNotFoundError):
        await route_request(
            data={
                **data,
                "metadata": {"user_api_key_auth": non_admin_auth},
            },
            llm_router=router,
            user_model=None,
            route_type="acompletion",
            user_api_key_dict=non_admin_auth,
        )

    from litellm.types.router import Deployment

    router.add_deployment(
        Deployment(
            model_name="internal-team-only",
            litellm_params={
                "model": "azure/gpt-4o",
                "api_key": "fake",
                "api_base": "https://internal.example.openai.azure.com",
                "api_version": "2024-02-15-preview",
            },
            model_info={
                "id": "internal-team-only-id",
                "team_id": "team-a",
            },
        )
    )
    internal_deployments = await router.async_get_healthy_deployments(
        model="internal-team-only",
        request_kwargs={
            **data,
            "model": "internal-team-only",
        },
    )

    assert {deployment["model_info"]["id"] for deployment in internal_deployments} == {"internal-team-only-id"}

    router.add_deployment(
        Deployment(
            model_name="internal-other-team-azure",
            litellm_params={
                "model": "azure/gpt-4o",
                "api_key": "fake",
                "api_base": "https://other.example.openai.azure.com",
                "api_version": "2024-02-15-preview",
                "mock_response": "other",
            },
            model_info={
                "id": "other-team-azure",
                "team_id": "team-b",
                "team_public_model_name": "team-azure",
            },
        )
    )

    async def _route_and_await():
        ambiguous_call = await route_request(
            data=data,
            llm_router=router,
            user_model=None,
            route_type="acompletion",
            user_api_key_dict=admin_auth,
        )
        await ambiguous_call

    with pytest.raises(litellm.BadRequestError, match="multiple teams"):
        await _route_and_await()

    router.add_deployment(
        Deployment(
            model_name="team-azure",
            litellm_params={
                "model": "azure/gpt-4o",
                "api_key": "fake",
                "api_base": "https://legacy.example.openai.azure.com",
                "api_version": "2024-02-15-preview",
            },
            model_info={
                "id": "legacy-team-azure",
                "team_id": "team-a",
                "team_public_model_name": "team-azure",
            },
        )
    )
    router.add_deployment(
        Deployment(
            model_name="team-azure",
            litellm_params={
                "model": "azure/gpt-4o",
                "api_key": "fake",
                "api_base": "https://other-legacy.example.openai.azure.com",
                "api_version": "2024-02-15-preview",
            },
            model_info={
                "id": "other-legacy-team-azure",
                "team_id": "team-b",
                "team_public_model_name": "team-azure",
            },
        )
    )

    with pytest.raises(litellm.BadRequestError, match="multiple teams"):
        await router.async_get_healthy_deployments(
            model="team-azure",
            request_kwargs=data,
        )

    router.add_deployment(
        Deployment(
            model_name="team-azure",
            litellm_params={
                "model": "azure/gpt-4o",
                "api_key": "fake",
                "api_base": "https://global.example.openai.azure.com",
                "api_version": "2024-02-15-preview",
            },
            model_info={"id": "global-team-azure"},
        )
    )

    collision_deployments = await router.async_get_healthy_deployments(
        model="team-azure",
        request_kwargs=data,
    )

    assert {deployment["model_info"]["id"] for deployment in collision_deployments} == {"global-team-azure"}


@pytest.mark.asyncio
async def test_route_request_no_model_required():
    """Test route types that don't require model parameter"""
    test_cases = [
        "amoderation",
        "aget_responses",
        "adelete_responses",
        "avector_store_create",
        "avector_store_search",
    ]

    for route_type in test_cases:
        # Test data without model parameter
        data = {"input": "test input", "query": "test query", "api_key": "test-key"}

        llm_router = MagicMock()
        getattr(llm_router, route_type).return_value = "fake_response"

        response = await route_request(data, llm_router, None, route_type)

        # Verify response
        assert response == "fake_response"
        # Verify the method was called with correct parameters
        getattr(llm_router, route_type).assert_called_once_with(**data)

        # Reset mock for next iteration
        llm_router.reset_mock()


@pytest.mark.asyncio
async def test_route_request_no_model_required_with_router_settings():
    """Test route types that don't require model parameter with router settings"""
    test_cases = [
        "amoderation",
        "aget_responses",
        "adelete_responses",
        "avector_store_create",
        "avector_store_search",
    ]

    for route_type in test_cases:
        # Test data with model parameter (it will be ignored for these route types)
        data = {
            "input": "test input",
            "query": "test query",
            "model": "test-model",  # Include dummy model to avoid KeyError
        }

        llm_router = MagicMock()
        # Set up router settings
        llm_router.router_general_settings.pass_through_all_models = False
        llm_router.default_deployment = None
        llm_router.pattern_router.patterns = []
        llm_router.model_names = []  # Empty model names list
        llm_router.get_model_ids.return_value = []  # Empty model IDs
        llm_router.model_group_alias = None  # No model group alias

        # Mock the async route call
        getattr(llm_router, route_type).return_value = "fake_response"

        # Run the request
        response = await route_request(data, llm_router, None, route_type)

        # Assert the mocked method was called with expected input
        assert response == "fake_response"
        getattr(llm_router, route_type).assert_called_once_with(**data)

        # Reset the mock for the next route
        llm_router.reset_mock()


@pytest.mark.asyncio
async def test_route_request_vector_store_routes_model_none_no_api_key_in_body():
    """
    GET /vector_stores/{id} and related routes do not send api_key in the body.
    Router must still accept model=None (as set by common_processing_pre_call_logic).
    """
    cases: list[tuple[str, dict]] = [
        ("avector_store_retrieve", {"vector_store_id": "vs_123", "model": None}),
        ("avector_store_list", {"model": None}),
        (
            "avector_store_update",
            {"vector_store_id": "vs_123", "name": "n", "model": None},
        ),
        ("avector_store_delete", {"vector_store_id": "vs_123", "model": None}),
    ]

    for route_type, data in cases:
        llm_router = MagicMock()
        llm_router.router_general_settings.pass_through_all_models = False
        llm_router.default_deployment = None
        llm_router.pattern_router.patterns = []
        llm_router.model_names = []
        llm_router.has_model_id.return_value = False
        llm_router.deployment_names = []
        llm_router.model_group_alias = None

        getattr(llm_router, route_type).return_value = "fake_response"

        response = await route_request(dict(data), llm_router, None, route_type)

        assert response == "fake_response"
        mock_method = getattr(llm_router, route_type)
        mock_method.assert_called_once()
        actual_kwargs = mock_method.call_args.kwargs
        for key, value in data.items():
            assert actual_kwargs.get(key) == value, (
                f"{route_type}: expected {key}={value!r}, got {actual_kwargs.get(key)!r}"
            )
        llm_router.reset_mock()


@pytest.mark.asyncio
async def test_route_request_no_model_required_with_router_settings_and_no_router():
    """Test route types that don't require model parameter with router settings and no router"""
    from unittest.mock import patch

    import litellm
    from litellm.proxy.route_llm_request import route_request

    data = {
        "model": "my-model-id",
        "api_key": "my-api-key",
        "messages": [{"role": "user", "content": "what llm are you"}],
    }

    with patch.object(litellm, "acompletion", return_value="fake_response") as mock_completion:
        await route_request(data, None, "gpt-3.5-turbo", "acompletion")

        mock_completion.assert_called_once_with(**data)


@pytest.mark.asyncio
async def test_route_request_with_router_settings_override():
    """
    Test that route_request handles router_settings_override by merging settings into kwargs
    instead of creating a new Router (which is expensive and was the old behavior).
    """
    # Mock data with router_settings_override containing per-request settings
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "router_settings_override": {
            "fallbacks": [{"gpt-3.5-turbo": ["gpt-4"]}],
            "num_retries": 5,
            "timeout": 30,
            "model_group_retry_policy": {"gpt-3.5-turbo": {"RateLimitErrorRetries": 3}},
            "routing_strategy": "least-busy",
            "model_group_alias": {"alias": "real_model"},
        },
    }

    llm_router = MagicMock()
    llm_router.acompletion.return_value = "success"

    response = await route_request(data, llm_router, None, "acompletion")

    assert response == "success"
    # Verify the router method was called with merged settings
    call_kwargs = llm_router.acompletion.call_args[1]
    assert call_kwargs["fallbacks"] == [{"gpt-3.5-turbo": ["gpt-4"]}]
    assert call_kwargs["num_retries"] == 5
    assert call_kwargs["timeout"] == 30
    assert call_kwargs["model_group_retry_policy"] == {"gpt-3.5-turbo": {"RateLimitErrorRetries": 3}}
    assert call_kwargs["routing_strategy"] == "least-busy"
    # Verify unsupported settings were NOT merged
    assert "model_group_alias" not in call_kwargs
    # Verify router_settings_override was removed from data
    assert "router_settings_override" not in call_kwargs


@pytest.mark.asyncio
async def test_route_request_with_router_settings_override_no_router():
    """
    Test that router_settings_override works when no router is provided,
    falling back to litellm module directly.
    """
    import litellm

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "router_settings_override": {
            "fallbacks": [{"gpt-3.5-turbo": ["gpt-4"]}],
            "num_retries": 3,
        },
    }

    # Use MagicMock explicitly to avoid auto-AsyncMock behavior in Python 3.12+
    mock_completion = MagicMock(return_value="success")
    original_acompletion = litellm.acompletion
    litellm.acompletion = mock_completion

    try:
        response = await route_request(data, None, "gpt-3.5-turbo", "acompletion")

        assert response == "success"
        # Verify litellm.acompletion was called with merged settings
        call_kwargs = mock_completion.call_args[1]
        assert "fallbacks" not in call_kwargs
        assert call_kwargs["num_retries"] == 3
    finally:
        litellm.acompletion = original_acompletion


@pytest.mark.asyncio
async def test_route_request_with_router_settings_override_preserves_existing():
    """
    Test that router_settings_override does not override settings already in the request.
    Request-level settings take precedence over key/team settings.
    """
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "num_retries": 10,  # Request-level setting
        "router_settings_override": {
            "num_retries": 3,  # Key/team setting - should NOT override
            "timeout": 30,  # Key/team setting - should be applied
        },
    }

    llm_router = MagicMock()
    llm_router.acompletion.return_value = "success"

    response = await route_request(data, llm_router, None, "acompletion")

    assert response == "success"
    call_kwargs = llm_router.acompletion.call_args[1]
    # Request-level num_retries should take precedence
    assert call_kwargs["num_retries"] == 10
    # Key/team timeout should be applied since not in request
    assert call_kwargs["timeout"] == 30


def test_gated_mock_params_cover_mock_router_testing_params():
    """``GATED_MOCK_PARAM_NAMES`` is hardcoded to avoid a cyclic import
    against ``litellm.types.router``. This test guards against drift — if a
    new ``mock_testing_*`` field is added to ``MockRouterTestingParams`` the
    gate must be updated to keep covering it. The gate is a superset: it also
    covers params consumed outside that dataclass."""
    from dataclasses import fields

    from litellm.proxy.route_llm_request import GATED_MOCK_PARAM_NAMES
    from litellm.types.router import MockRouterTestingParams

    assert {f.name for f in fields(MockRouterTestingParams)} <= set(GATED_MOCK_PARAM_NAMES)
    assert {"mock_testing_rate_limit_error", "mock_timeout", "mock_delay"} <= set(GATED_MOCK_PARAM_NAMES)


def test_e2e_proxy_config_opts_in_to_the_mock_params_its_suite_sends():
    """The ``build_and_test`` CI job runs the top-level ``tests/test_*.py`` suite
    against a proxy mounted with ``proxy_server_config.yaml``. Several of those
    tests drive fallback, retry and timeout paths by asking the proxy to
    fabricate a failure, which the gate rejects with a 400 unless the config
    opts in. Without the opt-in the positive cases fail outright and the
    negative ones pass for the wrong reason, so the suite and the config it
    runs against have to move together."""
    from pathlib import Path

    import yaml

    from litellm.proxy.route_llm_request import (
        GATED_MOCK_PARAM_NAMES,
        MOCK_TESTING_CONFIG_KEY,
    )

    repo_root = Path(__file__).parents[3]
    suite_sources = tuple(
        (path, path.read_text(encoding="utf-8")) for path in sorted(repo_root.glob("tests/test_*.py"))
    )
    senders = frozenset(
        f"{path.name}:{param}"
        for path, source in suite_sources
        for param in GATED_MOCK_PARAM_NAMES
        if f"{param}=" in source or f'"{param}"' in source
    )
    config = yaml.safe_load((repo_root / "proxy_server_config.yaml").read_text(encoding="utf-8"))
    general_settings = config.get("general_settings") or {}

    assert not senders or general_settings.get(MOCK_TESTING_CONFIG_KEY) is True, (
        f"proxy_server_config.yaml must set general_settings.{MOCK_TESTING_CONFIG_KEY}: true — "
        f"the E2E suite sends gated mock testing params ({', '.join(sorted(senders))}) "
        "and the proxy rejects them with a 400 otherwise"
    )


@pytest.mark.parametrize(
    "mock_param",
    [
        "mock_testing_fallbacks",
        "mock_testing_context_fallbacks",
        "mock_testing_content_policy_fallbacks",
        "mock_testing_rate_limit_error",
        "mock_timeout",
        "mock_delay",
    ],
)
def test_mock_params_rejected_when_not_allowed(mock_param):
    """Every gated param must be rejected by name when the proxy has not
    opted in, and the error must point the caller at the config key."""
    from litellm.proxy.route_llm_request import (
        MOCK_TESTING_CONFIG_KEY,
        raise_if_mock_testing_params_disallowed,
    )

    data = {"model": "gpt-3.5-turbo", mock_param: True}

    with pytest.raises(HTTPException) as exc_info:
        raise_if_mock_testing_params_disallowed(data, allowed=False)

    assert exc_info.value.status_code == 400
    error_message = exc_info.value.detail["error"]
    assert mock_param in error_message
    assert MOCK_TESTING_CONFIG_KEY in error_message


@pytest.mark.parametrize(
    "mock_param",
    [
        "mock_testing_fallbacks",
        "mock_testing_context_fallbacks",
        "mock_testing_content_policy_fallbacks",
        "mock_testing_rate_limit_error",
        "mock_timeout",
        "mock_delay",
    ],
)
def test_mock_params_pass_through_when_allowed(mock_param):
    """With the opt-in set, gated params must survive untouched — a gate that
    rejects correctly but strips anyway would leave the feature unusable."""
    from litellm.proxy.route_llm_request import raise_if_mock_testing_params_disallowed

    data = {"model": "gpt-3.5-turbo", mock_param: True}

    raise_if_mock_testing_params_disallowed(data, allowed=True)

    assert data[mock_param] is True


def test_mock_param_gate_reports_every_param_present():
    """A request carrying several gated params must name all of them, so a
    caller fixing one is not surprised by the next."""
    from litellm.proxy.route_llm_request import raise_if_mock_testing_params_disallowed

    data = {
        "model": "gpt-3.5-turbo",
        "mock_testing_fallbacks": True,
        "mock_delay": 30,
    }

    with pytest.raises(HTTPException) as exc_info:
        raise_if_mock_testing_params_disallowed(data, allowed=False)

    error_message = exc_info.value.detail["error"]
    assert "mock_testing_fallbacks" in error_message
    assert "mock_delay" in error_message


def test_ordinary_request_is_not_rejected_by_the_mock_param_gate():
    """The gate must not fire on a request that carries no gated param."""
    from litellm.proxy.route_llm_request import raise_if_mock_testing_params_disallowed

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "mock_response": "hi",
    }

    raise_if_mock_testing_params_disallowed(data, allowed=False)


@pytest.mark.asyncio
async def test_route_request_rejects_mock_params_by_default(monkeypatch):
    """End-to-end through ``route_request``: with no opt-in configured the
    request is rejected before it ever reaches the router."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {}, raising=False)

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "mock_testing_fallbacks": True,
    }
    llm_router = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await route_request(data, llm_router, None, "acompletion")

    assert exc_info.value.status_code == 400
    llm_router.acompletion.assert_not_called()


@pytest.mark.asyncio
async def test_route_request_forwards_mock_params_when_opted_in(monkeypatch):
    """End-to-end through ``route_request``: with the opt-in set the param
    reaches the router, which is what makes a fallback drill possible."""
    import litellm.proxy.proxy_server as proxy_server

    from litellm.proxy.route_llm_request import MOCK_TESTING_CONFIG_KEY

    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {MOCK_TESTING_CONFIG_KEY: True},
        raising=False,
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "mock_testing_fallbacks": True,
    }
    llm_router = MagicMock()
    llm_router.acompletion.return_value = "ok"

    await route_request(data, llm_router, None, "acompletion")

    call_kwargs = llm_router.acompletion.call_args[1]
    assert call_kwargs["mock_testing_fallbacks"] is True


@pytest.mark.parametrize("route_type", ["agenerate_content", "agenerate_content_stream"])
@pytest.mark.asyncio
async def test_route_request_maps_generation_config_for_google_routes(route_type):
    """For Google generate_content routes, route_request must rename
    `generationConfig` (Google's wire format) to `config` (the kwarg the
    router method expects). Without this mapping the request reaches the
    LLM with the field under the wrong name and the config is dropped."""
    data = {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": "9:16", "imageSize": "4K"},
        },
    }
    llm_router = MagicMock()
    getattr(llm_router, route_type).return_value = "ok"

    await route_request(data, llm_router, None, route_type)

    call_kwargs = getattr(llm_router, route_type).call_args[1]
    assert "generationConfig" not in call_kwargs
    assert "config" in call_kwargs
    assert call_kwargs["config"]["responseModalities"] == ["TEXT", "IMAGE"]
    assert call_kwargs["config"]["imageConfig"]["aspectRatio"] == "9:16"
    assert call_kwargs["config"]["imageConfig"]["imageSize"] == "4K"


@pytest.mark.parametrize("route_type", ["agenerate_content", "agenerate_content_stream"])
@pytest.mark.asyncio
async def test_route_request_preserves_existing_config_for_google_routes(route_type):
    """If the caller already supplies `config`, route_request must not
    overwrite it with `generationConfig`."""
    data = {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "config": {"existing": True},
        "generationConfig": {"shouldNotWin": True},
    }
    llm_router = MagicMock()
    getattr(llm_router, route_type).return_value = "ok"

    await route_request(data, llm_router, None, route_type)

    call_kwargs = getattr(llm_router, route_type).call_args[1]
    assert call_kwargs["config"] == {"existing": True}


async def _invoke_realtime_route(
    data: dict,
    llm_router,
    route_type: str = "acreate_realtime_client_secret",
):
    llm_call = await route_request(data, llm_router, None, route_type)
    return await llm_call


@pytest.fixture
def openai_realtime_credential():
    import litellm
    from litellm.types.utils import CredentialItem

    litellm.credential_list = [
        CredentialItem(
            credential_name="openai-realtime-cred",
            credential_info={"custom_llm_provider": "openai"},
            credential_values={"api_key": "resolved-credential-key"},
        )
    ]
    yield
    litellm.credential_list = []


@pytest.mark.asyncio
async def test_route_request_realtime_wildcard_model_resolves_credentials(
    monkeypatch,
):
    """
    POST /realtime/client_secrets with a request model like openai/gpt-realtime
    must match an openai/* deployment and forward its api_key upstream.
    """
    import httpx
    import litellm
    from unittest.mock import AsyncMock, patch

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "wildcard-realtime-key",
                },
            }
        ]
    )
    with patch(
        "litellm.realtime_api.main.base_llm_http_handler.async_realtime_client_secret_handler",
        new_callable=AsyncMock,
    ) as mock_handler:
        mock_handler.return_value = httpx.Response(200, json={"value": "ephemeral"})
        await _invoke_realtime_route(
            {"model": "openai/gpt-realtime"},
            router,
        )

    assert mock_handler.call_args.kwargs["api_key"] == "wildcard-realtime-key"


@pytest.mark.asyncio
async def test_route_request_realtime_team_scoped_model_resolves_credentials(
    monkeypatch,
):
    """
    Team-scoped deployments (team_public_model_name) must be selected when
    user_api_key_team_id is present, same as /chat/completions.
    """
    import httpx
    import litellm
    from unittest.mock import AsyncMock, patch

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    router = litellm.Router(
        model_list=[
            {
                "model_name": "internal-realtime",
                "litellm_params": {
                    "model": "openai/gpt-realtime",
                    "api_key": "team-realtime-key",
                },
                "model_info": {
                    "team_id": "team-a",
                    "team_public_model_name": "team-realtime",
                },
            }
        ]
    )
    with patch(
        "litellm.realtime_api.main.base_llm_http_handler.async_realtime_client_secret_handler",
        new_callable=AsyncMock,
    ) as mock_handler:
        mock_handler.return_value = httpx.Response(200, json={"value": "ephemeral"})
        await _invoke_realtime_route(
            {
                "model": "team-realtime",
                "metadata": {"user_api_key_team_id": "team-a"},
            },
            router,
        )

    assert mock_handler.call_args.kwargs["api_key"] == "team-realtime-key"


@pytest.mark.asyncio
async def test_route_request_realtime_litellm_credential_name_resolves_api_key(
    openai_realtime_credential,
    monkeypatch,
):
    """
    litellm_credential_name on a wildcard deployment must resolve to the stored
    api_key when routing acreate_realtime_client_secret through the router.
    """
    import httpx
    import litellm
    from unittest.mock import AsyncMock, patch

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "litellm_credential_name": "openai-realtime-cred",
                },
            }
        ]
    )
    with patch(
        "litellm.realtime_api.main.base_llm_http_handler.async_realtime_client_secret_handler",
        new_callable=AsyncMock,
    ) as mock_handler:
        mock_handler.return_value = httpx.Response(200, json={"value": "ephemeral"})
        await _invoke_realtime_route({"model": "openai/gpt-realtime"}, router)

    assert mock_handler.call_args.kwargs["api_key"] == "resolved-credential-key"


@pytest.mark.asyncio
async def test_route_request_realtime_unresolvable_model_raises_not_found(
    monkeypatch,
):
    """
    An unknown model must not silently fall through to litellm with an empty
    OPENAI_API_KEY env var.
    """
    import litellm
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.route_llm_request import ProxyModelNotFoundError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    router = litellm.Router(
        model_list=[
            {
                "model_name": "other-model",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "other-key"},
            }
        ]
    )
    with patch(
        "litellm.realtime_api.main.base_llm_http_handler.async_realtime_client_secret_handler",
        new_callable=AsyncMock,
    ) as mock_handler:
        with pytest.raises(ProxyModelNotFoundError):
            await _invoke_realtime_route({"model": "nonexistent-realtime-model"}, router)

    mock_handler.assert_not_called()


@pytest.mark.asyncio
async def test_route_request_realtime_calls_resolves_api_base(monkeypatch):
    """
    /realtime/calls must resolve the deployment's api_base through the router so a
    non-default (self-hosted / proxied) OpenAI endpoint is honored, instead of
    defaulting to https://api.openai.com.
    """
    import httpx
    import litellm
    from unittest.mock import AsyncMock, patch

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-realtime",
                "litellm_params": {
                    "model": "openai/gpt-realtime",
                    "api_key": "calls-key",
                    "api_base": "https://custom-realtime.example.com/v1",
                },
            }
        ]
    )
    with patch(
        "litellm.realtime_api.main.base_llm_http_handler.async_realtime_calls_handler",
        new_callable=AsyncMock,
    ) as mock_handler:
        mock_handler.return_value = httpx.Response(200, content=b"v=0\r\n")
        await _invoke_realtime_route(
            {
                "model": "my-realtime",
                "openai_ephemeral_key": "ek_test",
                "sdp_body": b"v=0\r\n",
            },
            router,
            route_type="arealtime_calls",
        )

    assert mock_handler.call_args.kwargs["api_base"] == "https://custom-realtime.example.com/v1"


@pytest.mark.asyncio
async def test_route_request_realtime_transcription_session_resolves_credentials(monkeypatch):
    """
    /realtime/transcription_sessions must resolve credentials through the router
    (wildcard deployment) rather than falling back to an empty OPENAI_API_KEY.
    """
    import httpx
    import litellm
    from unittest.mock import AsyncMock, patch

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "transcription-key",
                },
            }
        ]
    )
    with patch(
        "litellm.realtime_api.main.base_llm_http_handler.async_realtime_transcription_session_handler",
        new_callable=AsyncMock,
    ) as mock_handler:
        mock_handler.return_value = httpx.Response(200, json={"client_secret": {"value": "ephemeral"}})
        await _invoke_realtime_route(
            {"model": "openai/gpt-realtime"},
            router,
            route_type="acreate_realtime_transcription_session",
        )

    assert mock_handler.call_args.kwargs["api_key"] == "transcription-key"


@pytest.mark.asyncio
async def test_route_request_merges_enable_tag_filtering_from_override():
    """Key/team router_settings carry enable_tag_filtering; the override
    whitelist must forward it to the router call or the team's tag-routing
    toggle saved in the UI is silently ignored at request time."""
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "router_settings_override": {
            "enable_tag_filtering": True,
        },
    }

    llm_router = MagicMock()
    llm_router.acompletion.return_value = "success"

    response = await route_request(data, llm_router, None, "acompletion")

    assert response == "success"
    call_kwargs = llm_router.acompletion.call_args[1]
    assert call_kwargs["enable_tag_filtering"] is True


@pytest.mark.asyncio
async def test_route_request_strips_client_supplied_enable_tag_filtering():
    """enable_tag_filtering influences deployment selection and is only
    trusted when it comes from key/team router_settings via
    router_settings_override. A caller putting it in the request body must
    not reach the router with it."""
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "enable_tag_filtering": True,
    }

    llm_router = MagicMock()
    llm_router.acompletion.return_value = "ok"

    await route_request(data, llm_router, None, "acompletion")

    call_kwargs = llm_router.acompletion.call_args[1]
    assert "enable_tag_filtering" not in call_kwargs
    assert "enable_tag_filtering" not in data


@pytest.mark.asyncio
async def test_route_request_override_enable_tag_filtering_beats_body_value():
    """A client-sent enable_tag_filtering must not shadow the key/team
    setting: the body copy is stripped first, so the override value is the
    one the router sees."""
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "enable_tag_filtering": False,
        "router_settings_override": {
            "enable_tag_filtering": True,
        },
    }

    llm_router = MagicMock()
    llm_router.acompletion.return_value = "ok"

    await route_request(data, llm_router, None, "acompletion")

    call_kwargs = llm_router.acompletion.call_args[1]
    assert call_kwargs["enable_tag_filtering"] is True


_PROXY_DEPLOYMENT_BASE: Final = "https://proxy-deployment.example/v1"
_HELLO_MESSAGES: Final = ({"role": "user", "content": "Hello"},)


def _chat_completion_response(model: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _proxy_router_with_two_model_groups() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": f"openai/{model}",
                    "api_key": "sk-proxy",
                    "api_base": _PROXY_DEPLOYMENT_BASE,
                },
            }
            for model in ("gpt-3.5-turbo", "gpt-4")
        ],
        num_retries=0,
    )


@pytest.mark.parametrize(
    "extra_body, upstream_base",
    [
        pytest.param({"api_key": "sk-from-pre-call-hook"}, _PROXY_DEPLOYMENT_BASE, id="clientside-api-key"),
        pytest.param(
            {"api_base": "https://clientside.example/v1"}, "https://clientside.example/v1", id="clientside-api-base"
        ),
        pytest.param({"model": "gpt-3.5-turbo,gpt-4"}, _PROXY_DEPLOYMENT_BASE, id="comma-separated-batch"),
        pytest.param({"drop_params": True}, _PROXY_DEPLOYMENT_BASE, id="drop-params"),
        pytest.param(
            {
                "user_config": {
                    "model_list": [
                        {
                            "model_name": "gpt-3.5-turbo",
                            "litellm_params": {
                                "model": "openai/gpt-3.5-turbo",
                                "api_key": "sk-user-config",
                                "api_base": "https://user-config.example/v1",
                            },
                        }
                    ],
                }
            },
            "https://user-config.example/v1",
            id="user-config",
        ),
    ],
)
@pytest.mark.asyncio
async def test_route_request_never_forwards_router_settings_override(
    monkeypatch: pytest.MonkeyPatch, extra_body: dict[str, object], upstream_base: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    data: Final = {
        "model": "gpt-3.5-turbo",
        "messages": list(_HELLO_MESSAGES),
        "router_settings_override": {"num_retries": 2, "timeout": None},
        **extra_body,
    }

    with respx.mock(assert_all_called=False) as respx_mock:
        primary: Final = respx_mock.post(f"{upstream_base}/chat/completions", json__model="gpt-3.5-turbo").mock(
            side_effect=[
                httpx.Response(500, json={"error": {"message": "boom"}}),
                _chat_completion_response("gpt-3.5-turbo"),
            ]
        )
        respx_mock.post(f"{upstream_base}/chat/completions", json__model="gpt-4").mock(
            return_value=_chat_completion_response("gpt-4")
        )
        response: Final = await (await route_request(data, _proxy_router_with_two_model_groups(), None, "acompletion"))

    primary_response: Final = response[0] if isinstance(response, list) else response
    sent_bodies: Final = [json.loads(call.request.content) for call in primary.calls]
    assert not [body for body in sent_bodies if "router_settings_override" in body], sent_bodies
    assert primary_response._hidden_params["additional_headers"]["x-litellm-max-retries"] == 2, (
        primary_response._hidden_params
    )


@pytest.mark.parametrize("extra_body", [{}, {"api_key": "sk-from-pre-call-hook"}], ids=["router", "clientside-api-key"])
@pytest.mark.asyncio
async def test_route_request_treats_null_router_settings_override_as_absent(
    monkeypatch: pytest.MonkeyPatch, extra_body: dict[str, str]
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    data: Final = {
        "model": "gpt-3.5-turbo",
        "messages": list(_HELLO_MESSAGES),
        "router_settings_override": None,
        **extra_body,
    }

    with respx.mock(assert_all_called=False) as respx_mock:
        upstream: Final = respx_mock.post(f"{_PROXY_DEPLOYMENT_BASE}/chat/completions").mock(
            return_value=_chat_completion_response("gpt-3.5-turbo")
        )
        response: Final = await (await route_request(data, _proxy_router_with_two_model_groups(), None, "acompletion"))

    assert response.choices[0].message.content == "hi", response
    assert "router_settings_override" not in json.loads(upstream.calls.last.request.content)


@pytest.mark.parametrize(
    "global_drop_params, extra_body",
    [
        pytest.param(False, {}, id="drop-params-off"),
        pytest.param(False, {"drop_params": False}, id="request-drop-params-false"),
    ],
)
@pytest.mark.parametrize("override", ["fallbacks", ["num_retries"], 3])
@pytest.mark.asyncio
async def test_route_request_rejects_non_object_router_settings_override(
    monkeypatch: pytest.MonkeyPatch, global_drop_params: bool, extra_body: dict[str, bool | str], override: object
) -> None:
    from litellm.proxy._types import ProxyException

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)
    data: Final = {
        "model": "gpt-3.5-turbo",
        "messages": list(_HELLO_MESSAGES),
        "router_settings_override": override,
        **extra_body,
    }

    with respx.mock(assert_all_called=False) as respx_mock:
        upstream: Final = respx_mock.post(f"{_PROXY_DEPLOYMENT_BASE}/chat/completions").mock(
            return_value=_chat_completion_response("gpt-3.5-turbo")
        )
        with pytest.raises(ProxyException) as exc_info:
            await route_request(data, _proxy_router_with_two_model_groups(), None, "acompletion")

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "router_settings_override"
    assert upstream.call_count == 0


@pytest.mark.parametrize(
    "global_drop_params, extra_body",
    [
        pytest.param(False, {"drop_params": True}, id="request-drop-params"),
        pytest.param(True, {}, id="global-drop-params"),
        pytest.param(True, {"drop_params": False}, id="global-drop-params-with-request-false"),
        pytest.param(False, {"drop_params": "true"}, id="request-drop-params-string"),
    ],
)
@pytest.mark.parametrize("override", ["fallbacks", ["num_retries"], 3])
@pytest.mark.asyncio
async def test_route_request_drops_non_object_router_settings_override_under_drop_params(
    monkeypatch: pytest.MonkeyPatch, global_drop_params: bool, extra_body: dict[str, bool | str], override: object
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)
    data: Final = {
        "model": "gpt-3.5-turbo",
        "messages": list(_HELLO_MESSAGES),
        "router_settings_override": override,
        **extra_body,
    }

    with respx.mock(assert_all_called=False) as respx_mock:
        upstream: Final = respx_mock.post(f"{_PROXY_DEPLOYMENT_BASE}/chat/completions").mock(
            return_value=_chat_completion_response("gpt-3.5-turbo")
        )
        response: Final = await (await route_request(data, _proxy_router_with_two_model_groups(), None, "acompletion"))

    assert response.choices[0].message.content == "hi", response
    assert upstream.call_count == 1
    assert "router_settings_override" not in json.loads(upstream.calls.last.request.content)


@pytest.mark.parametrize(
    "user_config_settings, expected_upstream_calls",
    [
        pytest.param({}, 2, id="key-setting-fills-unset-user-config-setting"),
        pytest.param({"num_retries": 0}, 1, id="user-config-setting-beats-key-setting"),
    ],
)
@pytest.mark.asyncio
async def test_route_request_user_config_router_settings_beat_key_settings(
    monkeypatch: pytest.MonkeyPatch, user_config_settings: dict[str, int], expected_upstream_calls: int
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    user_config_base: Final = "https://user-config.example/v1"
    data: Final = {
        "model": "gpt-3.5-turbo",
        "messages": list(_HELLO_MESSAGES),
        "router_settings_override": {"num_retries": 1},
        "user_config": {
            **user_config_settings,
            "model_list": [
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "openai/gpt-3.5-turbo",
                        "api_key": "sk-user-config",
                        "api_base": user_config_base,
                    },
                }
            ],
        },
    }

    with respx.mock(assert_all_called=False) as respx_mock:
        upstream: Final = respx_mock.post(f"{user_config_base}/chat/completions").mock(
            side_effect=[
                httpx.Response(500, json={"error": {"message": "boom"}}),
                _chat_completion_response("gpt-3.5-turbo"),
            ]
        )
        with contextlib.suppress(litellm.InternalServerError):
            await (await route_request(data, _proxy_router_with_two_model_groups(), None, "acompletion"))

    assert upstream.call_count == expected_upstream_calls


@pytest.mark.parametrize(
    "extra_body",
    [
        pytest.param({}, id="no-key-settings"),
        pytest.param({"router_settings_override": {"num_retries": 2}}, id="key-settings"),
    ],
)
@pytest.mark.asyncio
async def test_route_request_refuses_blocked_model_whether_or_not_key_has_router_settings(
    extra_body: dict[str, dict[str, int]],
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "openai/gpt-3.5-turbo", "api_key": "sk-proxy"},
                "model_info": {"blocked": True},
            }
        ],
    )
    data: Final = {"model": "gpt-3.5-turbo", "messages": list(_HELLO_MESSAGES), **extra_body}

    with pytest.raises(litellm.PermissionDeniedError):
        await route_request(data, router, None, "acompletion")


@pytest.mark.parametrize(
    "credential",
    [
        pytest.param({"api_key": "sk-from-pre-call-hook"}, id="clientside-api-key"),
        pytest.param({"api_base": _PROXY_DEPLOYMENT_BASE}, id="clientside-api-base"),
    ],
)
@pytest.mark.parametrize(
    "fallback_field, upstream_error",
    [
        pytest.param("fallbacks", httpx.Response(500, json={"error": {"message": "boom"}}), id="fallbacks"),
        pytest.param(
            "context_window_fallbacks",
            httpx.Response(
                400,
                json={
                    "error": {
                        "message": "This model's maximum context length is 16385 tokens",
                        "code": "context_length_exceeded",
                    }
                },
            ),
            id="context-window-fallbacks",
        ),
        pytest.param(
            "content_policy_fallbacks",
            httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Your request was rejected by the content filter",
                        "type": "invalid_request_error",
                        "code": "content_policy_violation",
                    }
                },
            ),
            id="content-policy-fallbacks",
        ),
    ],
)
@pytest.mark.asyncio
async def test_route_request_applies_override_fallbacks_when_request_carries_credentials(
    monkeypatch: pytest.MonkeyPatch,
    credential: dict[str, str],
    fallback_field: str,
    upstream_error: httpx.Response,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    data: Final = {
        "model": "gpt-3.5-turbo",
        "messages": list(_HELLO_MESSAGES),
        "router_settings_override": {fallback_field: [{"gpt-3.5-turbo": ["gpt-4"]}]},
        **credential,
    }

    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.post(f"{_PROXY_DEPLOYMENT_BASE}/chat/completions", json__model="gpt-3.5-turbo").mock(
            return_value=upstream_error
        )
        fallback: Final = respx_mock.post(f"{_PROXY_DEPLOYMENT_BASE}/chat/completions", json__model="gpt-4").mock(
            return_value=_chat_completion_response("gpt-4")
        )
        response: Final = await (await route_request(data, _proxy_router_with_two_model_groups(), None, "acompletion"))

    assert fallback.call_count == 1, [str(call.request.url) for call in respx_mock.calls]
    assert "router_settings_override" not in json.loads(fallback.calls.last.request.content)
    assert response.choices[0].message.content == "hi", response


@pytest.mark.parametrize(
    "router_only_setting",
    [
        pytest.param({"model_group_retry_policy": {"gpt-4": {"TimeoutErrorRetries": 1}}}, id="model-group-retry-policy"),
        pytest.param({"context_window_fallbacks": [{"gpt-4": ["gpt-3.5-turbo"]}]}, id="context-window-fallbacks"),
        pytest.param({"content_policy_fallbacks": [{"gpt-4": ["gpt-3.5-turbo"]}]}, id="content-policy-fallbacks"),
    ],
)
@pytest.mark.asyncio
async def test_route_request_keeps_router_only_override_settings_out_of_pass_through_provider_body(
    monkeypatch: pytest.MonkeyPatch, router_only_setting: dict[str, object]
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pass-through")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "sk-proxy", "api_base": _PROXY_DEPLOYMENT_BASE},
            }
        ],
        router_general_settings=RouterGeneralSettings(pass_through_all_models=True),
    )
    data: Final = {
        "model": "openai/gpt-pass-through",
        "messages": list(_HELLO_MESSAGES),
        "router_settings_override": router_only_setting,
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=_chat_completion_response("gpt-pass-through")
        )
        await (await route_request(data, router, None, "acompletion"))

    assert set(router_only_setting).isdisjoint(JSON_OBJECT_ADAPTER.validate_json(upstream.calls.last.request.content))


@pytest.mark.asyncio
async def test_route_request_keeps_key_fallbacks_out_of_pass_through_provider_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pass-through")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "sk-proxy", "api_base": _PROXY_DEPLOYMENT_BASE},
            }
        ],
        router_general_settings=RouterGeneralSettings(pass_through_all_models=True),
    )
    data: Final = {
        "model": "openai/gpt-pass-through",
        "messages": list(_HELLO_MESSAGES),
        "router_settings_override": {"fallbacks": [{"gpt-4": ["gpt-4"]}], "num_retries": 0},
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": {"message": "upstream down"}})
        )
        with pytest.raises(litellm.InternalServerError):
            await (await route_request(data, router, None, "acompletion"))

    provider_bodies: Final = tuple(JSON_OBJECT_ADAPTER.validate_json(call.request.content) for call in upstream.calls)
    assert provider_bodies and all("gpt-4" not in body for body in provider_bodies), provider_bodies


@pytest.mark.parametrize(
    "route_type, param, route",
    [
        ("acompletion", "messages", "/chat/completions"),
        ("aembedding", "input", "/embeddings"),
        ("aresponses", "input", "/responses"),
        ("acreate_batch", "input_file_id", "/batches"),
        ("aspeech", "input", "/audio/speech"),
        ("amoderation", "input", "/moderations"),
        ("aimage_generation", "prompt", "/image/generations"),
        ("asearch", "query", "/search"),
        ("atext_completion", "prompt", "/completions"),
        ("atranscription", "file", "/audio/transcriptions"),
        ("arerank", "query", "/rerank"),
        ("acompact_responses", "input", "/responses/compact"),
        ("anthropic_messages", "messages", "anthropic_messages"),
        ("agenerate_content", "contents", "agenerate_content"),
        ("aocr", "document", "/ocr"),
        ("avector_store_search", "query", "avector_store_search"),
        ("avector_store_file_create", "file_id", "avector_store_file_create"),
        ("avector_store_file_update", "attributes", "avector_store_file_update"),
        ("avideo_generation", "prompt", "/videos"),
        ("avideo_remix", "prompt", "/videos/{video_id}/remix"),
        ("avideo_edit", "prompt", "/videos/edits"),
        ("avideo_extension", "prompt", "/videos/extensions"),
        ("avideo_create_character", "name", "/videos/characters"),
        ("acreate_container", "name", "/containers"),
        ("aupload_container_file", "file", "/containers/{container_id}/files"),
        ("acreate_agent", "name", "/v1beta/agents"),
        ("acreate_eval", "data_source_config", "/evals"),
        ("acreate_run", "data_source", "/evals/{eval_id}/runs"),
    ],
)
def test_raise_if_required_body_param_missing_rejects_missing_param(route_type, param, route):
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(
            route_type=route_type,
            data={"model": "gpt-4o"},
            llm_router=None,
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.param == param
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.message == f"{route}: Missing required parameter: '{param}'."


@pytest.mark.parametrize(
    "data, param",
    [
        ({"endpoint": "/v1/chat/completions", "completion_window": "24h"}, "input_file_id"),
        ({"input_file_id": "file-abc", "completion_window": "24h"}, "endpoint"),
        ({"input_file_id": "file-abc", "endpoint": "/v1/chat/completions"}, "completion_window"),
        ({}, "input_file_id"),
    ],
)
def test_raise_if_required_body_param_missing_names_first_missing_batch_param(data, param):
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(route_type="acreate_batch", data=data, llm_router=None)

    assert exc_info.value.param == param


def test_raise_if_required_body_param_missing_rejects_null_for_merge_base_route() -> None:
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(
            route_type="aembedding",
            data={"model": "text-embedding-3-small", "input": None},
            llm_router=None,
        )

    assert exc_info.value.param == "input"


@pytest.mark.parametrize(
    ("route_type", "data"),
    (
        pytest.param(
            "anthropic_messages",
            {"model": "claude", "messages": [], "max_tokens": None},
            id="anthropic-max-tokens",
        ),
        pytest.param(
            "aimage_generation",
            {"model": "gpt-image-1", "prompt": None},
            id="image-prompt",
        ),
    ),
)
def test_required_present_body_param_accepts_explicit_null(route_type: str, data: dict[str, object]) -> None:
    from litellm.proxy.route_llm_request import raise_if_required_body_param_missing

    raise_if_required_body_param_missing(route_type=route_type, data=data, llm_router=None)


def test_required_present_body_param_uses_router_deployment_default() -> None:
    import litellm
    from litellm.proxy.route_llm_request import raise_if_required_body_param_missing

    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-default",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                    "max_tokens": 32,
                },
            }
        ]
    )

    raise_if_required_body_param_missing(
        route_type="anthropic_messages",
        data={"model": "claude-default", "messages": []},
        llm_router=router,
    )


def test_required_present_body_param_without_router_default_still_raises() -> None:
    import litellm
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-without-default",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                },
            }
        ]
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(
            route_type="anthropic_messages",
            data={"model": "claude-without-default", "messages": []},
            llm_router=router,
        )

    assert exc_info.value.param == "max_tokens"


@pytest.mark.parametrize(
    "route_type, data, param, default",
    [
        ("anthropic_messages", {"model": "claude-router-default", "messages": []}, "max_tokens", 16),
        ("arerank", {"model": "rerank-router-default", "query": "hi"}, "documents", ["router default document"]),
    ],
)
def test_required_present_body_param_uses_router_wide_default(
    route_type: str, data: dict[str, object], param: str, default: object
) -> None:
    import litellm
    from litellm.proxy.route_llm_request import raise_if_required_body_param_missing

    router = litellm.Router(
        model_list=[
            {
                "model_name": str(data["model"]),
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                },
            }
        ],
        default_litellm_params={param: default},
    )

    raise_if_required_body_param_missing(route_type=route_type, data=data, llm_router=router)


@pytest.mark.parametrize(
    "route_type, data, param",
    [
        ("anthropic_messages", {"model": "claude", "messages": []}, "max_tokens"),
        ("arerank", {"model": "rerank-model", "query": "hi"}, "documents"),
    ],
)
def test_required_present_body_param_with_none_router_wide_default_still_raises(
    route_type: str, data: dict[str, object], param: str
) -> None:
    import litellm
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": str(data["model"]),
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                },
            }
        ],
        default_litellm_params={param: None},
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(route_type=route_type, data=data, llm_router=router)

    assert exc_info.value.param == param


@pytest.mark.parametrize(
    "route_type, data, param",
    [
        ("asearch", {"model": "search-model"}, "query"),
        ("acreate_eval", {"model": "eval-model"}, "data_source_config"),
        ("acreate_run", {"model": "eval-model"}, "data_source"),
        ("acreate_agent", {"model": "agent-model"}, "name"),
        ("avector_store_search", {}, "query"),
    ],
)
def test_required_present_body_param_ignores_router_wide_default_when_dispatch_skips_router(
    route_type: str, data: dict[str, object], param: str
) -> None:
    import litellm
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": str(data.get("model") or "any-model"),
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                },
            }
        ],
        default_litellm_params={param: "supplied-by-router-default"},
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(route_type=route_type, data=data, llm_router=router)

    assert exc_info.value.param == param


def test_required_present_body_param_uses_user_config_router_defaults() -> None:
    from litellm.proxy.route_llm_request import raise_if_required_body_param_missing

    raise_if_required_body_param_missing(
        route_type="anthropic_messages",
        data={
            "model": "claude-user-config",
            "messages": [],
            "user_config": {"default_litellm_params": {"max_tokens": 16}},
        },
        llm_router=None,
    )


def test_required_present_body_param_ignores_global_router_defaults_for_user_config_requests() -> None:
    import litellm
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-user-config",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                },
            }
        ],
        default_litellm_params={"max_tokens": 16},
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(
            route_type="anthropic_messages",
            data={"model": "claude-user-config", "messages": [], "user_config": {}},
            llm_router=router,
        )

    assert exc_info.value.param == "max_tokens"


@pytest.mark.parametrize(
    "llm_router",
    [
        pytest.param(MagicMock(), id="magicmock-router"),
        pytest.param(SimpleNamespace(get_model_list=lambda **_kwargs: ()), id="router-without-defaults-attr"),
    ],
)
def test_required_present_body_param_ignores_non_mapping_router_defaults(llm_router: object) -> None:
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(
            route_type="arerank",
            data={"model": "rerank-model", "query": "hi"},
            llm_router=llm_router,  # pyright: ignore[reportArgumentType]  # deliberately duck-typed routers
        )

    assert exc_info.value.param == "documents"


@pytest.mark.parametrize(
    "route_type, data, param",
    [
        ("arerank", {"model": "rerank-model", "query": "hi"}, "documents"),
        ("anthropic_messages", {"model": "claude", "messages": []}, "max_tokens"),
        ("avideo_extension", {"model": "sora-2", "prompt": "longer"}, "seconds"),
        ("avideo_create_character", {"name": "hero"}, "video"),
        ("acreate_eval", {"data_source_config": {"type": "custom"}}, "testing_criteria"),
        ("acreate_interaction", {"input": "hi"}, "model"),
        ("acreate_interaction", {"model": None, "agent": None, "input": "hi"}, "model"),
        ("acreate_interaction", {"model": "gemini-3-pro-preview"}, "input"),
    ],
)
def test_raise_if_required_body_param_missing_names_each_missing_param(route_type, data, param):
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(route_type=route_type, data=data, llm_router=None)

    assert exc_info.value.code == "400"
    assert exc_info.value.param == param


@pytest.mark.parametrize(
    "route_type, data",
    [
        ("acompletion", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
        ("acompletion", {"model": "gpt-4o", "messages": []}),
        ("atext_completion", {"model": "gpt-4o", "prompt": "hi"}),
        ("aembedding", {"model": "text-embedding-3-small", "input": "hi"}),
        ("aresponses", {"model": "gpt-4o", "input": "hi"}),
        ("aresponses", {"model": "gpt-4o", "input": []}),
        ("arerank", {"model": "rerank-model", "query": "hi", "documents": ["hello"]}),
        ("aimage_edit", {"model": "gpt-image-1", "image": b"png", "prompt": "a hat"}),
        ("aimage_edit", {"model": "stability.stable-image-remove-background-v1:0", "image": b"png"}),
        ("aimage_edit", {"model": "stability.stable-style-transfer-v1:0", "init_image": b"png"}),
        ("anthropic_messages", {"model": "claude", "messages": [], "max_tokens": 16}),
        ("avideo_extension", {"model": "sora-2", "prompt": "longer", "seconds": "4"}),
        ("acreate_eval", {"data_source_config": {"type": "custom"}, "testing_criteria": []}),
        ("acreate_interaction", {"model": "gemini-3-pro-preview", "input": "hi"}),
        ("acreate_interaction", {"agent": "deep-research", "input": "hi"}),
        ("aimage_generation", {"model": "gpt-image-1", "prompt": "a cat"}),
        ("aspeech", {"model": "gpt-4o-mini-tts", "input": "hi", "voice": "alloy"}),
        ("amoderation", {"model": "omni-moderation-latest", "input": ""}),
        ("asearch", {"model": "perplexity-search", "query": "litellm"}),
        (
            "acreate_batch",
            {"input_file_id": "file-abc", "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        ),
    ],
)
def test_raise_if_required_body_param_missing_allows_valid_requests(route_type, data):
    from litellm.proxy.route_llm_request import raise_if_required_body_param_missing

    raise_if_required_body_param_missing(route_type=route_type, data=data, llm_router=None)


@pytest.mark.asyncio
async def test_route_request_rejects_chat_completion_without_messages():
    """A /chat/completions body without `messages` used to splat into
    Router.acompletion() and surface the resulting TypeError as a 500."""
    from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError

    llm_router = MagicMock()

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        await route_request({"model": "gpt-4o"}, llm_router, None, "acompletion")

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "messages"
    llm_router.acompletion.assert_not_called()


@pytest.mark.asyncio
async def test_route_request_rejects_responses_without_input():
    from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError

    llm_router = MagicMock()

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        await route_request({"model": "gpt-4o"}, llm_router, None, "aresponses")

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "input"
    llm_router.aresponses.assert_not_called()


class FakeProxyModelTable:
    def __init__(self, rows):
        self.rows = rows
        self.find_many_wheres = []

    async def find_many(self, where=None, **kwargs):
        self.find_many_wheres.append(where)
        return list(self.rows)


def _fake_prisma_client_with_models(rows):
    from types import SimpleNamespace

    table = FakeProxyModelTable(rows)
    return SimpleNamespace(db=SimpleNamespace(litellm_proxymodeltable=table)), table


def _db_model_row(model_name: str, mock_response: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        model_id=f"{model_name}-id",
        model_name=model_name,
        litellm_params={"model": "openai/gpt-4o", "api_key": "fake", "mock_response": mock_response},
        model_info={},
        blocked=False,
    )


@pytest.mark.asyncio
async def test_route_request_read_through_recovers_model_created_on_sibling_replica(monkeypatch):
    """Regression: a model written to the DB by another replica must be served on
    first request instead of 400ing until the periodic config reload."""
    import litellm
    import litellm.proxy.proxy_server as proxy_server

    model_name = "e2e-sibling-replica-model"
    router = litellm.Router(
        model_list=[
            {
                "model_name": "some-other-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
            }
        ]
    )
    fake_prisma, table = _fake_prisma_client_with_models([_db_model_row(model_name, "hello-from-db")])
    monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    llm_call = await route_request(
        data={"model": model_name, "messages": [{"role": "user", "content": "hi"}]},
        llm_router=router,
        user_model=None,
        route_type="acompletion",
    )
    response = await llm_call

    assert response.choices[0].message.content == "hello-from-db"
    assert len(table.find_many_wheres) == 1
    assert table.find_many_wheres[0] == {"model_name": model_name}


def _tagged_db_model_row(model_name: str, tag: str, order: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=f"{model_name}-{tag}-id",
        model_name=model_name,
        litellm_params={
            "model": "openai/gpt-4o",
            "api_key": "fake",
            "mock_response": f"{tag}-deployment",
            "tags": [tag],
            "order": order,
        },
        model_info={},
        blocked=False,
    )


@pytest.mark.asyncio
async def test_route_request_read_through_retry_keeps_key_tag_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm.proxy.proxy_server as proxy_server

    model_name: Final = "sibling-replica-model-for-key-with-tag-filtering"
    router: Final = litellm.Router(
        model_list=[{"model_name": "some-other-model", "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"}}]
    )
    fake_prisma, _ = _fake_prisma_client_with_models(
        [_tagged_db_model_row(model_name, "free", order=1), _tagged_db_model_row(model_name, "paid", order=2)]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    response: Final = await (
        await route_request(
            {
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"tags": ["paid"]},
                "router_settings_override": {"enable_tag_filtering": True},
            },
            router,
            None,
            "acompletion",
        )
    )

    assert response.choices[0].message.content == "paid-deployment"


@pytest.mark.asyncio
async def test_route_request_unknown_model_raises_and_hits_db_once_within_ttl(monkeypatch):
    import litellm
    import litellm.proxy.proxy_server as proxy_server

    model_name = "e2e-model-nobody-created"
    router = litellm.Router(
        model_list=[
            {
                "model_name": "some-other-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
            }
        ]
    )
    fake_prisma, table = _fake_prisma_client_with_models([])
    monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    data = {"model": model_name, "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(ProxyModelNotFoundError):
        await route_request(data=data, llm_router=router, user_model=None, route_type="acompletion")
    with pytest.raises(ProxyModelNotFoundError):
        await route_request(data=data, llm_router=router, user_model=None, route_type="acompletion")

    assert table.find_many_wheres == [{"model_name": model_name}, {"model_id": model_name}]


@pytest.mark.asyncio
async def test_route_request_read_through_disabled_without_store_model_in_db(monkeypatch):
    import litellm
    import litellm.proxy.proxy_server as proxy_server

    model_name = "e2e-config-only-proxy-model"
    router = litellm.Router(
        model_list=[
            {
                "model_name": "some-other-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
            }
        ]
    )
    fake_prisma, table = _fake_prisma_client_with_models([_db_model_row(model_name, "should-not-load")])
    monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
    monkeypatch.setattr(proxy_server, "store_model_in_db", False)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    with pytest.raises(ProxyModelNotFoundError):
        await route_request(
            data={"model": model_name, "messages": [{"role": "user", "content": "hi"}]},
            llm_router=router,
            user_model=None,
            route_type="acompletion",
        )

    assert table.find_many_wheres == []


@pytest.mark.asyncio
async def test_route_request_read_through_supplies_db_model_default_for_missing_param(monkeypatch):
    import litellm
    import litellm.proxy.proxy_server as proxy_server
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    model_name = "e2e-db-only-max-tokens-default"
    router = litellm.Router(
        model_list=[{"model_name": "some-other-model", "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"}}]
    )
    db_row = SimpleNamespace(
        model_id=f"{model_name}-id",
        model_name=model_name,
        litellm_params={"model": "anthropic/claude-sonnet-4-5", "api_key": "fake", "max_tokens": 64},
        model_info={},
        blocked=False,
    )
    fake_prisma, table = _fake_prisma_client_with_models([db_row])
    monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    data = {"model": model_name, "messages": [{"role": "user", "content": "hi"}]}

    with patch.object(router, "anthropic_messages", new=AsyncMock(return_value="db_default_used")) as spy:
        response = await (await route_request(data, router, None, "anthropic_messages"))

    assert response == "db_default_used"
    spy.assert_called_once()
    assert table.find_many_wheres[0] == {"model_name": model_name}


@pytest.mark.asyncio
async def test_route_request_missing_param_for_unknown_model_still_400s_after_read_through(monkeypatch):
    import litellm
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError

    model_name = "e2e-unknown-model-missing-max-tokens"
    router = litellm.Router(
        model_list=[{"model_name": "some-other-model", "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"}}]
    )
    fake_prisma, table = _fake_prisma_client_with_models([])
    monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        await route_request(
            {"model": model_name, "messages": [{"role": "user", "content": "hi"}]},
            router,
            None,
            "anthropic_messages",
        )

    assert (exc_info.value.code, exc_info.value.param) == ("400", "max_tokens")
    assert table.find_many_wheres[0] == {"model_name": model_name}


@pytest.mark.asyncio
async def test_route_request_routing_group_name_passes_model_gate():
    from unittest.mock import AsyncMock, patch

    from litellm import Router

    router = Router(
        model_list=[
            {"model_name": "member-a", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"}},
            {"model_name": "member-b", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"}},
        ],
        routing_groups=[
            {"group_name": "grouped-quality", "models": ["member-a", "member-b"], "routing_strategy": "simple-shuffle"}
        ],
    )
    data = {"model": "grouped-quality", "messages": [{"role": "user", "content": "hi"}]}

    with patch.object(router, "acompletion", new=AsyncMock(return_value="group_response")) as spy:
        response = await (await route_request(data, router, None, "acompletion"))

    assert response == "group_response"
    spy.assert_called_once_with(**data)


@pytest.mark.asyncio
async def test_route_request_a2a_agent_miss_does_not_consume_model_read_through(fresh_agent_read_through, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import litellm
    import litellm.proxy.proxy_server as proxy_server

    model_name = "a2a/agent-nobody-created"
    router = litellm.Router(
        model_list=[
            {
                "model_name": "some-other-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
            }
        ]
    )
    fake_prisma, model_table = _fake_prisma_client_with_models([])
    agents_find_unique = AsyncMock(return_value=None)
    fake_prisma.db.litellm_agentstable = SimpleNamespace(find_unique=agents_find_unique)
    monkeypatch.setattr(proxy_server, "prisma_client", fake_prisma)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    with pytest.raises(ProxyModelNotFoundError):
        await route_request(
            data={"model": model_name, "messages": [{"role": "user", "content": "hi"}]},
            llm_router=router,
            user_model=None,
            route_type="acompletion",
        )

    assert agents_find_unique.await_count == 2
    assert model_table.find_many_wheres == []


def test_proxy_model_not_found_error_keeps_the_raw_model_only_in_the_client_response():
    raw_model: Final = "opus-4.6 Please summarize my medical records\nPatient has diabetes"

    error: Final = ProxyModelNotFoundError(route="/chat/completions", model_name=raw_model)

    assert raw_model in error.detail["error"]
    assert raw_model not in error.spend_log_error_message
    assert error.spend_log_error_message.startswith("/chat/completions: Invalid model name passed in")


@pytest.mark.asyncio
async def test_route_request_without_model_on_model_routed_endpoint_is_a_400():
    import litellm
    from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError

    router = litellm.Router(
        model_list=[
            {"model_name": "rerank-model", "litellm_params": {"model": "cohere/rerank-v3.5", "api_key": "fake"}}
        ]
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        await route_request(
            data={"query": "hi", "documents": ["hello"]}, llm_router=router, user_model=None, route_type="arerank"
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "model"


@pytest.mark.asyncio
async def test_route_request_router_settings_override_skips_null_fields():
    """
    A key or team saved from the dashboard stores every unset router setting as null. Those nulls
    must not reach the router as explicit per-request values, or they switch the router-level
    fallbacks and retries off for that key.
    """
    data: Final = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "router_settings_override": {
            "fallbacks": None,
            "context_window_fallbacks": None,
            "num_retries": None,
            "model_group_retry_policy": None,
            "timeout": 600,
        },
    }

    llm_router: Final = MagicMock()
    llm_router.acompletion.return_value = "success"

    response: Final = await route_request(data, llm_router, None, "acompletion")

    assert response == "success"
    call_kwargs: Final = llm_router.acompletion.call_args[1]
    assert call_kwargs["timeout"] == 600
    assert "fallbacks" not in call_kwargs
    assert "context_window_fallbacks" not in call_kwargs
    assert "num_retries" not in call_kwargs
    assert "model_group_retry_policy" not in call_kwargs


def _router_with_defaults(default_litellm_params: dict[str, object]):
    import litellm

    return litellm.Router(
        model_list=[
            {
                "model_name": "served",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "test-key"},
                "model_info": {"id": "served-deployment"},
            }
        ],
        default_litellm_params=default_litellm_params,
    )


@pytest.mark.parametrize(
    "route_type, param",
    [
        ("aimage_generation", "prompt"),
        ("aspeech", "input"),
        ("atext_completion", "prompt"),
        ("atranscription", "file"),
    ],
)
def test_router_default_for_positional_param_still_raises(route_type: str, param: str) -> None:
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(
            route_type=route_type,
            data={"model": "served"},
            llm_router=_router_with_defaults({param: "supplied-by-router-default"}),
        )

    assert exc_info.value.param == param


@pytest.mark.parametrize(
    "route_type, param",
    [
        ("aimage_generation", "prompt"),
        ("aspeech", "input"),
    ],
)
def test_user_config_default_for_positional_param_still_raises(route_type: str, param: str) -> None:
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(
            route_type=route_type,
            data={"model": "img", "user_config": {"model_list": [], "default_litellm_params": {param: "x"}}},
            llm_router=None,
        )

    assert exc_info.value.param == param


def test_deployment_param_for_positional_param_still_raises() -> None:
    import litellm
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "img",
                "litellm_params": {"model": "openai/gpt-image-1", "api_key": "test-key", "prompt": "a cat"},
            }
        ]
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(route_type="aimage_generation", data={"model": "img"}, llm_router=router)

    assert exc_info.value.param == "prompt"


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("omni-moderation-2024-09-26", id="unknown-model"),
        pytest.param("openai/gpt-4o-mini", id="deployment-name"),
        pytest.param("served-deployment", id="deployment-id"),
    ],
)
def test_router_default_ignored_for_moderation_model_the_router_does_not_list(model: str) -> None:
    from litellm.proxy.route_llm_request import (
        ProxyMissingRequiredParamError,
        raise_if_required_body_param_missing,
    )

    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        raise_if_required_body_param_missing(
            route_type="amoderation",
            data={"model": model},
            llm_router=_router_with_defaults({"input": "supplied"}),
        )

    assert exc_info.value.param == "input"


@pytest.mark.asyncio
async def test_null_user_config_is_treated_as_absent() -> None:
    router: Final = _router_with_defaults({})

    response: Final = await (
        await route_request(
            {
                "model": "served",
                "messages": [{"role": "user", "content": "hi"}],
                "mock_response": "from the main router",
                "user_config": None,
            },
            router,
            None,
            "acompletion",
        )
    )

    assert response.choices[0].message.content == "from the main router"


def _registered_bedrock_kb_search(team_id: str) -> dict[str, object]:
    """The request a team key's /v1/vector_stores/{id}/search carries once the registry row is merged in."""
    return {
        "model": None,
        "vector_store_id": "KB12345678",
        "query": "litellm proxy",
        "custom_llm_provider": "bedrock",
        "aws_region_name": "us-east-1",
        "aws_access_key_id": "AKIAREGISTRYKEY00000",
        "aws_secret_access_key": "registry-secret",
        "metadata": {"user_api_key_team_id": team_id},
    }


def _team_a_router(*team_models: tuple[str, dict[str, object]]) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": f"internal-team-a-{public_name}",
                "litellm_params": litellm_params,
                "model_info": {
                    "id": f"team-a-{public_name}",
                    "team_id": "team-a",
                    "team_public_model_name": public_name,
                },
            }
            for public_name, litellm_params in team_models
        ]
    )


_LANGGRAPH_AGENT_PARAMS: Final[dict[str, object]] = {
    "model": "langgraph/repro-agent",
    "api_base": "http://127.0.0.1:2024",
}
_BEDROCK_CLAUDE_PARAMS: Final[dict[str, object]] = {
    "model": "bedrock/anthropic.claude-sonnet-4-6",
    "aws_region_name": "us-east-1",
}


@pytest.mark.parametrize(
    "team_models",
    [
        pytest.param(
            [("a-langgraph-agent", _LANGGRAPH_AGENT_PARAMS), ("a-bedrock-claude", _BEDROCK_CLAUDE_PARAMS)],
            id="langgraph-first",
        ),
        pytest.param(
            [("a-bedrock-claude", _BEDROCK_CLAUDE_PARAMS), ("a-langgraph-agent", _LANGGRAPH_AGENT_PARAMS)],
            id="bedrock-first",
        ),
        pytest.param([("a-langgraph-agent", _LANGGRAPH_AGENT_PARAMS)], id="no-bedrock-team-model"),
    ],
)
@pytest.mark.asyncio
async def test_team_key_search_of_registered_bedrock_store_reaches_bedrock_with_registry_credentials(
    monkeypatch: pytest.MonkeyPatch,
    team_models: list[tuple[str, dict[str, object]]],
) -> None:
    """A team key's model-less search of a registered Bedrock store was routed through the team's first
    deployment, so a langgraph team model turned it into "Vector store search is not supported for langgraph"."""

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(
            "https://bedrock-agent-runtime.us-east-1.amazonaws.com/knowledgebases/KB12345678/retrieve"
        ).mock(return_value=httpx.Response(200, json={"retrievalResults": []}))
        response: Final = await (
            await route_request(
                _registered_bedrock_kb_search("team-a"),
                _team_a_router(*team_models),
                None,
                "avector_store_search",
            )
        )

    request: Final = route.calls.last.request
    posted_url: Final = str(request.url)
    signed_auth: Final = request.headers["Authorization"]
    assert response["object"] == "vector_store.search_results.page", response
    assert route.call_count == 1
    assert posted_url == "https://bedrock-agent-runtime.us-east-1.amazonaws.com/knowledgebases/KB12345678/retrieve", (
        posted_url
    )
    assert "Credential=AKIAREGISTRYKEY00000/" in signed_auth, signed_auth


@pytest.fixture(autouse=True)
def discard_cached_user_config_routers() -> Iterator[None]:
    yield
    user_config_routers.clear()


@pytest.fixture
def fake_redis_port() -> Iterator[int]:
    server: Final = TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


async def _wait_out_a_rate_limit_minute_rollover() -> None:
    seconds_left: Final = 60 - datetime.now().second
    if seconds_left < 10:
        await asyncio.sleep(seconds_left)


def _proxy_logging_obj(model: str = "rate-limited") -> LiteLLMLoggingObj:
    logging_obj, _ = litellm.utils.function_setup(
        original_function="acompletion",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.now(),
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        litellm_call_id=str(uuid4()),
    )
    assert isinstance(logging_obj, LiteLLMLoggingObj)
    return logging_obj


def _mock_deployment(model_name: str, **deployment_limits: int) -> dict[str, object]:
    return {
        "model_name": model_name,
        **deployment_limits,
        "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-test", "mock_response": "ok"},
    }


async def _user_config_request_outcome(
    redis_port: int | None,
    routing_settings: Mapping[str, object],
    deployment_limits: Mapping[str, int],
    logging_obj: LiteLLMLoggingObj | None,
) -> str:
    data: Final = {
        "model": "rate-limited",
        "messages": [{"role": "user", "content": "hi"}],
        "num_retries": 0,
        **({"litellm_logging_obj": logging_obj} if logging_obj is not None else {}),
        "user_config": {
            **routing_settings,
            **({"redis_host": "127.0.0.1", "redis_port": redis_port} if redis_port is not None else {}),
            "model_list": [_mock_deployment("rate-limited", **deployment_limits)],
        },
    }
    try:
        await (await route_request(data, None, None, "acompletion"))
    except (litellm.RateLimitError, RouterRateLimitError):
        return "429"
    finally:
        await asyncio.sleep(0.3)
    return "200"


async def _tpm_for_three_requests() -> int:
    messages: Final = [{"role": "user", "content": "hi"}]
    probe: Final = await litellm.acompletion(model="openai/gpt-5.5", messages=messages, mock_response="ok")
    assert isinstance(probe, litellm.ModelResponse)
    probe_usage: Final = getattr(probe, "usage", None)
    assert isinstance(probe_usage, litellm.Usage)
    return 2 * probe_usage.total_tokens + litellm.token_counter(messages=messages)


async def _request_cost() -> float:
    probe: Final = await litellm.acompletion(
        model="openai/gpt-5.5", messages=[{"role": "user", "content": "hi"}], mock_response="ok"
    )
    return litellm.completion_cost(completion_response=probe)


async def _proxy_router_outcomes(
    redis_port: int | None,
    routing_settings: Mapping[str, object],
    deployment_limits: Mapping[str, int],
    requests: int,
) -> list[str]:
    router: Final = litellm.Router(
        **routing_settings,
        num_retries=0,
        model_list=[_mock_deployment("proxy-rate-limited", **deployment_limits)],
        **({"redis_host": "127.0.0.1", "redis_port": redis_port} if redis_port is not None else {}),
    )

    async def outcome() -> str:
        try:
            await router.acompletion(model="proxy-rate-limited", messages=[{"role": "user", "content": "hi"}])
        except (litellm.RateLimitError, RouterRateLimitError):
            return "429"
        finally:
            await asyncio.sleep(0.3)
        return "200"

    try:
        return [await outcome() for _ in range(requests)]
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_a_user_config_adaptive_request_answers_with_the_model_it_picked() -> None:
    data: Final = {
        "model": "smart-cheap-router",
        "messages": [{"role": "user", "content": "Write a one line python function that adds two numbers"}],
        "metadata": {},
        "user_config": {
            "model_list": [
                {
                    "model_name": "smart-cheap-router",
                    "litellm_params": {
                        "model": "auto_router/adaptive_router",
                        "adaptive_router_config": {"available_models": ["fast", "smart"]},
                    },
                },
                {
                    **_mock_deployment("fast"),
                    "model_info": {"adaptive_router_preferences": {"quality_tier": 2, "strengths": []}},
                },
                {
                    **_mock_deployment("smart"),
                    "model_info": {"adaptive_router_preferences": {"quality_tier": 3, "strengths": ["code_generation"]}},
                },
            ]
        },
    }

    await (await route_request(data, None, None, "acompletion"))
    headers: Final = await ProxyLogging(user_api_key_cache=DualCache()).post_call_response_headers_hook(
        data=data, user_api_key_dict=UserAPIKeyAuth(), response=None
    )

    assert headers == {ADAPTIVE_ROUTER_RESPONSE_HEADER: data["metadata"][ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY]}


def _global_router_callbacks() -> tuple[list[object], list[object]]:
    return list(litellm.callbacks), list(litellm.input_callback)


@pytest.mark.parametrize(
    "routing_settings, deployment_limits",
    [
        ({"routing_strategy": "usage-based-routing-v2"}, {"rpm": 2}),
        ({"routing_strategy": "usage-based-routing-v2"}, {"tpm": 30}),
        ({"routing_strategy": "usage-based-routing"}, {"rpm": 2}),
        ({"optional_pre_call_checks": ["enforce_model_rate_limits"]}, {"rpm": 2}),
    ],
)
@pytest.mark.parametrize("arrives_with_logging_obj", [True, False])
@pytest.mark.parametrize("with_redis", [True, False])
@pytest.mark.asyncio
async def test_user_config_router_enforces_its_limit_across_proxy_requests(
    fake_redis_port: int,
    routing_settings: Mapping[str, object],
    deployment_limits: Mapping[str, int],
    arrives_with_logging_obj: bool,
    with_redis: bool,
) -> None:
    await _wait_out_a_rate_limit_minute_rollover()
    callbacks_before: Final = _global_router_callbacks()
    outcomes: Final = [
        await _user_config_request_outcome(
            fake_redis_port if with_redis else None,
            routing_settings,
            deployment_limits,
            _proxy_logging_obj() if arrives_with_logging_obj else None,
        )
        for _ in range(6)
    ]

    assert "429" in outcomes, outcomes
    assert _global_router_callbacks() == callbacks_before
    assert outcomes == await _proxy_router_outcomes(
        fake_redis_port if with_redis else None, routing_settings, deployment_limits, len(outcomes)
    )


@pytest.mark.parametrize("arrives_with_logging_obj", [True, False])
@pytest.mark.asyncio
async def test_concurrent_user_config_requests_each_count_once_against_the_tpm_limit(
    fake_redis_port: int, arrives_with_logging_obj: bool
) -> None:
    tpm_for_three_requests: Final = await _tpm_for_three_requests()

    def request() -> Coroutine[None, None, str]:
        return _user_config_request_outcome(
            fake_redis_port,
            {"routing_strategy": "usage-based-routing-v2"},
            {"tpm": tpm_for_three_requests},
            _proxy_logging_obj() if arrives_with_logging_obj else None,
        )

    await _wait_out_a_rate_limit_minute_rollover()
    concurrent_outcomes: Final = await asyncio.gather(request(), request())
    later_outcomes: Final = [await request() for _ in range(2)]

    assert [*concurrent_outcomes, *later_outcomes] == ["200", "200", "200", "429"]


@pytest.mark.asyncio
async def test_overlapping_user_config_requests_share_one_rpm_limit(fake_redis_port: int) -> None:
    def request() -> Coroutine[None, None, str]:
        return _user_config_request_outcome(
            fake_redis_port, {"routing_strategy": "usage-based-routing-v2"}, {"rpm": 2}, _proxy_logging_obj()
        )

    await _wait_out_a_rate_limit_minute_rollover()
    outcomes: Final = await asyncio.gather(*(request() for _ in range(4)))

    assert outcomes.count("200") <= 2, outcomes
    assert "429" in outcomes, outcomes


@pytest.mark.parametrize(
    "routing_strategy, limited_by",
    [("usage-based-routing-v2", "rpm"), ("usage-based-routing-v2", "tpm"), ("usage-based-routing", "rpm")],
)
@pytest.mark.parametrize("proxy_router_shares_redis", [True, False])
@pytest.mark.asyncio
async def test_a_proxy_router_on_the_same_strategy_leaves_the_user_config_limit_unchanged(
    fake_redis_port: int, routing_strategy: str, limited_by: str, proxy_router_shares_redis: bool
) -> None:
    deployment_limits: Final = {"rpm": 3} if limited_by == "rpm" else {"tpm": await _tpm_for_three_requests()}

    def request() -> Coroutine[None, None, str]:
        return _user_config_request_outcome(
            fake_redis_port, {"routing_strategy": routing_strategy}, deployment_limits, _proxy_logging_obj()
        )

    async def outcomes() -> list[str]:
        first_instance: Final = [await request() for _ in range(2)]
        user_config_routers.clear()
        second_instance: Final = [await request() for _ in range(3)]
        return [*first_instance, *second_instance]

    await _wait_out_a_rate_limit_minute_rollover()
    alone: Final = await outcomes()
    user_config_routers.clear()
    await asyncio.sleep(0.3)
    redis.Redis(host="127.0.0.1", port=fake_redis_port).flushall()
    proxy_router: Final = litellm.Router(
        routing_strategy=routing_strategy,
        model_list=[_mock_deployment("proxy-model")],
        redis_host="127.0.0.1" if proxy_router_shares_redis else None,
        redis_port=fake_redis_port if proxy_router_shares_redis else None,
    )
    try:
        beside_proxy_router: Final = await outcomes()
    finally:
        proxy_router.discard()

    assert "429" in alone, alone
    assert beside_proxy_router == alone


@pytest.mark.parametrize("user_config_settings", [{}, {"optional_pre_call_checks": ["router_budget_limiting"]}])
@pytest.mark.asyncio
async def test_user_config_requests_stay_under_the_proxy_provider_budget(
    user_config_settings: Mapping[str, object],
) -> None:
    proxy_router: Final = litellm.Router(
        model_list=[_mock_deployment("proxy-model")],
        provider_budget_config={"openai": {"budget_limit": await _request_cost() / 2, "time_period": "1d"}},
    )
    try:
        await proxy_router.acompletion(model="proxy-model", messages=[{"role": "user", "content": "hi"}])
        await asyncio.sleep(0.3)
        with pytest.raises(ValueError, match=RouterErrors.no_deployments_with_provider_budget_routing.value):
            await _user_config_request_outcome(None, user_config_settings, {}, _proxy_logging_obj())
    finally:
        proxy_router.discard()


@pytest.mark.asyncio
async def test_a_user_config_provider_budget_limits_its_own_requests() -> None:
    budget_limit: Final = await _request_cost() / 2

    def request() -> Coroutine[None, None, str]:
        budget: Final = {"openai": {"budget_limit": budget_limit, "time_period": "1d"}}
        return _user_config_request_outcome(None, {"provider_budget_config": budget}, {}, _proxy_logging_obj())

    first: Final = await request()
    with pytest.raises(ValueError, match=RouterErrors.no_deployments_with_provider_budget_routing.value):
        await request()
    assert first == "200"


@pytest.mark.asyncio
async def test_a_request_in_flight_when_its_user_config_router_is_dropped_still_counts_against_its_budget(
    fake_redis_port: int,
) -> None:
    budget_limit: Final = 1.5 * await _request_cost()

    async def request() -> None:
        await (
            await route_request(
                {
                    "model": "budgeted",
                    "messages": [{"role": "user", "content": "hi"}],
                    "litellm_logging_obj": _proxy_logging_obj("budgeted"),
                    "user_config": {
                        "redis_host": "127.0.0.1",
                        "redis_port": fake_redis_port,
                        "provider_budget_config": {"openai": {"budget_limit": budget_limit, "time_period": "1d"}},
                        "model_list": [
                            {
                                "model_name": "budgeted",
                                "litellm_params": {
                                    "model": "openai/gpt-5.5",
                                    "api_key": "sk-test",
                                    "mock_response": "ok",
                                    "mock_delay": 0.5,
                                },
                            }
                        ],
                    },
                },
                None,
                None,
                "acompletion",
            )
        )

    await request()
    in_flight: Final = asyncio.create_task(request())
    await asyncio.sleep(0.2)
    user_config_routers.clear()
    await in_flight
    await asyncio.sleep(0.3)

    with pytest.raises(ValueError, match=RouterErrors.no_deployments_with_provider_budget_routing.value):
        await request()


async def _user_config_request(model: str, user_config: Mapping[str, object], arrives_with_logging_obj: bool) -> object:
    return await (
        await route_request(
            {
                "model": model,
                "messages": list(_HELLO_MESSAGES),
                "num_retries": 0,
                **({"litellm_logging_obj": _proxy_logging_obj(model)} if arrives_with_logging_obj else {}),
                "user_config": user_config,
            },
            None,
            None,
            "acompletion",
        )
    )


def _ordered_pair(model_name: str, first_base: str, second_base: str) -> list[dict[str, object]]:
    return [
        {
            "model_name": model_name,
            "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-test", "api_base": api_base, "order": order},
        }
        for order, api_base in ((1, first_base), (2, second_base))
    ]


@pytest.mark.parametrize("arrives_with_logging_obj", [True, False])
@pytest.mark.asyncio
async def test_a_rate_limited_user_config_deployment_cools_down_for_the_next_requests(
    monkeypatch: pytest.MonkeyPatch, arrives_with_logging_obj: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    rate_limited_base: Final = "https://rate-limited.example/v1"
    healthy_base: Final = "https://healthy.example/v1"
    user_config: Final = {"model_list": _ordered_pair("flaky", rate_limited_base, healthy_base)}

    with respx.mock(assert_all_called=False) as respx_mock:
        rate_limited: Final = respx_mock.post(f"{rate_limited_base}/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": {"message": "slow down"}})
        )
        respx_mock.post(f"{healthy_base}/chat/completions").mock(return_value=_chat_completion_response("gpt-5.5"))
        for _ in range(3):
            await _user_config_request("flaky", user_config, arrives_with_logging_obj)
            await asyncio.sleep(0.3)

    assert rate_limited.call_count == 1


async def _remaining_usage_headers_after_429_then_200s(
    send: Callable[[], Awaitable[object]], api_base: str
) -> list[tuple[object, object] | str]:
    async def remaining_usage() -> tuple[object, object] | str:
        try:
            response: Final = await send()
        except litellm.RateLimitError:
            return "429"
        finally:
            await asyncio.sleep(0.3)
        assert isinstance(response, litellm.ModelResponse)
        headers: Final = response._hidden_params["additional_headers"]
        return headers.get("x-ratelimit-remaining-requests"), headers.get("x-ratelimit-remaining-tokens")

    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.post(f"{api_base}/chat/completions").mock(
            side_effect=[
                httpx.Response(429, json={"error": {"message": "slow down"}}),
                _chat_completion_response("gpt-5.5"),
                _chat_completion_response("gpt-5.5"),
            ]
        )
        return [await remaining_usage() for _ in range(3)]


@pytest.mark.parametrize("arrives_with_logging_obj", [True, False])
@pytest.mark.asyncio
async def test_user_config_remaining_usage_headers_match_a_proxy_router_with_the_same_deployment(
    monkeypatch: pytest.MonkeyPatch, arrives_with_logging_obj: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    await _wait_out_a_rate_limit_minute_rollover()
    proxy_base: Final = "https://proxy-limited.example/v1"
    user_config_base: Final = "https://user-config-limited.example/v1"

    def deployment(model_name: str, api_base: str) -> dict[str, object]:
        return {
            "model_name": model_name,
            "litellm_params": {
                "model": "openai/gpt-5.5",
                "api_key": "sk-test",
                "api_base": api_base,
                "rpm": 10,
                "tpm": 1000,
            },
        }

    proxy_router: Final = litellm.Router(model_list=[deployment("proxy-limited", proxy_base)])
    try:
        proxy_router_headers: Final = await _remaining_usage_headers_after_429_then_200s(
            lambda: proxy_router.acompletion(model="proxy-limited", messages=list(_HELLO_MESSAGES), num_retries=0),
            proxy_base,
        )
    finally:
        proxy_router.discard()

    user_config: Final = {"model_list": [deployment("user-config-limited", user_config_base)]}
    user_config_headers: Final = await _remaining_usage_headers_after_429_then_200s(
        lambda: _user_config_request("user-config-limited", user_config, arrives_with_logging_obj), user_config_base
    )

    assert user_config_headers == proxy_router_headers


async def _first_deployment_calls_over_successes_then_failures(
    send: Callable[[], Awaitable[object]], first_base: str, second_base: str
) -> int:
    successes: Final = [_chat_completion_response("gpt-5.5") for _ in range(6)]
    failures: Final = [httpx.Response(500, json={"error": {"message": "boom"}}) for _ in range(6)]
    with respx.mock(assert_all_called=False) as respx_mock:
        first: Final = respx_mock.post(f"{first_base}/chat/completions").mock(side_effect=[*successes, *failures])
        respx_mock.post(f"{second_base}/chat/completions").mock(return_value=_chat_completion_response("gpt-5.5"))
        for _ in range(len(successes) + len(failures)):
            await send()
            await asyncio.sleep(0.3)
        return first.call_count


@pytest.mark.parametrize("arrives_with_logging_obj", [True, False])
@pytest.mark.asyncio
async def test_user_config_successes_keep_a_deployment_out_of_cooldown_like_a_proxy_router(
    monkeypatch: pytest.MonkeyPatch, arrives_with_logging_obj: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    await _wait_out_a_rate_limit_minute_rollover()
    proxy_router: Final = litellm.Router(
        model_list=_ordered_pair("proxy-pair", "https://proxy-first.example/v1", "https://proxy-second.example/v1")
    )
    try:
        proxy_router_calls: Final = await _first_deployment_calls_over_successes_then_failures(
            lambda: proxy_router.acompletion(model="proxy-pair", messages=list(_HELLO_MESSAGES), num_retries=0),
            "https://proxy-first.example/v1",
            "https://proxy-second.example/v1",
        )
    finally:
        proxy_router.discard()

    user_config: Final = {
        "model_list": _ordered_pair("user-pair", "https://user-first.example/v1", "https://user-second.example/v1")
    }

    user_config_calls: Final = await _first_deployment_calls_over_successes_then_failures(
        lambda: _user_config_request("user-pair", user_config, arrives_with_logging_obj),
        "https://user-first.example/v1",
        "https://user-second.example/v1",
    )

    assert user_config_calls == proxy_router_calls


@pytest.mark.asyncio
async def test_user_config_adaptive_router_learns_from_proxy_requests() -> None:
    user_config: Final = {
        "model_list": [
            _mock_deployment("smart"),
            _mock_deployment("fast"),
            {
                "model_name": "adaptive",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "adaptive_router_config": {"available_models": ["smart", "fast"]},
                },
            },
        ]
    }
    await (
        await route_request(
            {
                "model": "adaptive",
                "messages": [{"role": "user", "content": "write a sorting function"}],
                "metadata": {"session_id": "session-1"},
                "litellm_logging_obj": _proxy_logging_obj("adaptive"),
                "user_config": user_config,
            },
            None,
            None,
            "acompletion",
        )
    )
    await asyncio.sleep(0.3)

    router: Final = user_config_routers.get(user_config, lambda: pytest.fail("the user_config router was not reused"))
    snapshot: Final = await router.adaptive_routers["adaptive"][0].strategy.get_state_snapshot()
    assert snapshot["feedback_contexts_live"] == 1, snapshot


@pytest.mark.asyncio
async def test_user_config_routers_discard_the_least_recently_used_router_past_capacity() -> None:
    routers: Final = UserConfigRouters(max_size=2)

    def config(model_name: str) -> dict[str, object]:
        return {"routing_strategy": "usage-based-routing-v2", "model_list": [_mock_deployment(model_name)]}

    def build(model_name: str) -> litellm.Router:
        return litellm.Router(**config(model_name))

    def must_be_cached() -> litellm.Router:
        pytest.fail("a cached router was rebuilt")

    first_handler: Final = weakref.ref(routers.get(config("first"), lambda: build("first")).lowesttpm_logger_v2)
    second_handler: Final = weakref.ref(routers.get(config("second"), lambda: build("second")).lowesttpm_logger_v2)
    routers.get(config("first"), must_be_cached)
    routers.get(config("third"), lambda: build("third"))
    await asyncio.sleep(0.2)
    gc.collect()

    assert routers.get(config("first"), must_be_cached) is not None
    assert first_handler() is not None
    assert second_handler() is None
    routers.clear()
