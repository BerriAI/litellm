import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path


from unittest.mock import MagicMock

from litellm.proxy.route_llm_request import ProxyModelNotFoundError, route_request


@pytest.mark.parametrize(
    "route_type",
    [
        "atext_completion",
        "acompletion",
        "aembedding",
        "aimage_generation",
        "aspeech",
        "atranscription",
        "amoderation",
        "arerank",
    ],
)
@pytest.mark.asyncio
async def test_route_request_dynamic_credentials(route_type):
    data = {
        "model": "openai/gpt-4o-mini-2024-07-18",
        "api_key": "my-bad-key",
        "api_base": "https://api.openai.com/v1 ",
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

    with pytest.raises(litellm.BadRequestError, match="multiple teams"):
        ambiguous_call = await route_request(
            data=data,
            llm_router=router,
            user_model=None,
            route_type="acompletion",
            user_api_key_dict=admin_auth,
        )
        await ambiguous_call

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
        data = {"input": "test input", "api_key": "test-key"}

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
            # These settings should be ignored (not in per_request_settings list)
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
    # Verify unsupported settings were NOT merged
    assert "routing_strategy" not in call_kwargs
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
        response = await route_request(data, None, None, "acompletion")

        assert response == "success"
        # Verify litellm.acompletion was called with merged settings
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["fallbacks"] == [{"gpt-3.5-turbo": ["gpt-4"]}]
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


def test_mock_testing_kwarg_names_matches_dataclass():
    """``_MOCK_TESTING_KWARG_NAMES`` is hardcoded to avoid a cyclic import
    against ``litellm.types.router``. This test guards against drift —
    if a new ``mock_testing_*`` field is added to ``MockRouterTestingParams``
    the strip list must be updated to keep covering it."""
    from dataclasses import fields

    from litellm.proxy.route_llm_request import _MOCK_TESTING_KWARG_NAMES
    from litellm.types.router import MockRouterTestingParams

    assert set(_MOCK_TESTING_KWARG_NAMES) == {f.name for f in fields(MockRouterTestingParams)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mock_flag",
    [
        "mock_testing_fallbacks",
        "mock_testing_context_fallbacks",
        "mock_testing_content_policy_fallbacks",
    ],
)
async def test_route_request_strips_mock_testing_flags(mock_flag):
    """VERIA-44: router-internal testing flags must not survive a
    user-supplied request body. Without this strip, an attacker can
    combine ``mock_testing_fallbacks=true`` with an unauthorized fallback
    in ``router_settings_override`` to deterministically execute requests
    against restricted models."""
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        mock_flag: True,
    }
    llm_router = MagicMock()
    llm_router.acompletion.return_value = "ok"

    await route_request(data, llm_router, None, "acompletion")

    call_kwargs = llm_router.acompletion.call_args[1]
    assert mock_flag not in call_kwargs
    # The flag is also gone from the original data dict so any subsequent
    # processing (e.g. logging) doesn't see it either.
    assert mock_flag not in data


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
