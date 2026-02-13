import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock

from litellm.proxy.route_llm_request import _kwargs_for_llm, route_request


def test_kwargs_for_llm_strips_presidio_pii_tokens():
    """_kwargs_for_llm removes _presidio_pii_tokens so it is not sent to the LLM provider."""
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "_presidio_pii_tokens": {"guardrail_1": {"<PERSON>": "Jane"}},
    }
    result = _kwargs_for_llm(data)
    assert "model" in result
    assert "messages" in result
    assert "_presidio_pii_tokens" not in result
    assert result.get("_presidio_pii_tokens") is None


def test_kwargs_for_llm_preserves_other_keys():
    """_kwargs_for_llm leaves all other keys unchanged."""
    data = {"model": "gpt-4", "temperature": 0.7, "api_key": "sk-xxx"}
    result = _kwargs_for_llm(data)
    assert result == data


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

    with patch.object(
        litellm, "acompletion", return_value="fake_response"
    ) as mock_completion:
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


@pytest.mark.asyncio
async def test_route_request_user_config_filters_router_args_and_discards(monkeypatch):
    import litellm

    state = {"init_kwargs": None, "discard_called": False}

    class _FakeRouter:
        @staticmethod
        def get_valid_args():
            return ["model_list", "routing_strategy"]

        def __init__(self, **kwargs):
            state["init_kwargs"] = kwargs

        async def acompletion(self, **kwargs):
            return {"ok": True, "kwargs": kwargs}

        def discard(self):
            state["discard_called"] = True

    monkeypatch.setattr(litellm, "Router", _FakeRouter)

    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "_presidio_pii_tokens": {"guardrail": {"<PERSON>": "Jane"}},
        "user_config": {
            "model_list": [
                {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4"}}
            ],
            "routing_strategy": "least-busy",
            "invalid_key": "should_be_ignored",
        },
    }

    llm_call = await route_request(data, None, None, "acompletion")
    result = await llm_call

    assert result["ok"] is True
    assert "_presidio_pii_tokens" not in result["kwargs"]
    assert state["init_kwargs"] == {
        "model_list": [
            {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4"}}
        ],
        "routing_strategy": "least-busy",
    }
    assert state["discard_called"] is True


@pytest.mark.asyncio
async def test_route_request_user_config_batch_does_not_forward_model_kwarg(monkeypatch):
    import litellm

    state = {"models": None, "kwargs": None}

    class _FakeRouter:
        @staticmethod
        def get_valid_args():
            return []

        def __init__(self, **kwargs):
            pass

        async def abatch_completion(self, models, **kwargs):
            state["models"] = models
            state["kwargs"] = kwargs
            return "ok"

        def discard(self):
            pass

    monkeypatch.setattr(litellm, "Router", _FakeRouter)

    data = {
        "model": "gpt-4o-mini, claude-3-haiku",
        "messages": [{"role": "user", "content": "hi"}],
        "user_config": {},
    }

    llm_call = await route_request(data, None, None, "acompletion")
    result = await llm_call

    assert result == "ok"
    assert state["models"] == ["gpt-4o-mini", "claude-3-haiku"]
    assert "model" not in (state["kwargs"] or {})


@pytest.mark.asyncio
async def test_route_request_batch_with_router_does_not_forward_model_kwarg():
    data = {
        "model": "gpt-4o-mini, claude-3-haiku",
        "messages": [{"role": "user", "content": "hi"}],
    }
    llm_router = MagicMock()
    llm_router.model_names = []
    llm_router.abatch_completion.return_value = "ok"

    response = await route_request(data, llm_router, None, "acompletion")

    assert response == "ok"
    call_kwargs = llm_router.abatch_completion.call_args[1]
    assert call_kwargs["models"] == ["gpt-4o-mini", "claude-3-haiku"]
    assert "model" not in call_kwargs
