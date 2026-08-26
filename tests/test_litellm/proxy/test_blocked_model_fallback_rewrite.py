from types import SimpleNamespace
from unittest.mock import patch

import pytest

import litellm
from litellm.proxy.route_llm_request import route_request


def _stateful_rewrite_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"blocked": True},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "first fallback response",
                },
            },
            {
                "model_name": "safe-fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "safe fallback response",
                },
            },
            {
                "model_name": "restricted-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "restricted response",
                },
            },
        ],
        model_group_alias={"public-model": "primary-model"},
        fallbacks=[{"public-model": ["fallback-model", "safe-fallback-model"]}],
        num_retries=0,
    )


@pytest.mark.asyncio
async def test_blocked_primary_cannot_be_rewritten_before_trusted_fallback():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"blocked": True},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "server fallback response",
                },
            },
            {
                "model_name": "restricted-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "rewritten primary response",
                },
            },
        ],
        model_group_alias={"public-model": "primary-model"},
        fallbacks=[{"public-model": ["fallback-model"]}],
        num_retries=0,
    )

    async def rewrite_only_blocked_primary(*, model, messages, **kwargs):
        if model != "public-model":
            return None
        return SimpleNamespace(
            model="restricted-model",
            messages=messages,
            litellm_params=None,
        )

    router.async_pre_routing_hook = rewrite_only_blocked_primary

    with patch("litellm.proxy.route_llm_request.mock_testing_params_allowed", return_value=False):
        response = await route_request(
            data={
                "model": "public-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            llm_router=router,
            user_model=None,
            route_type="acompletion",
        )

    assert response.choices[0].message.content == "server fallback response"


@pytest.mark.asyncio
async def test_stateful_fallback_rewrite_cannot_escape_preflight_validation():
    router = _stateful_rewrite_router()
    fallback_hook_calls = 0

    async def stateful_fallback_rewrite(*, model, messages, **kwargs):
        nonlocal fallback_hook_calls
        if model != "fallback-model":
            return None
        fallback_hook_calls += 1
        if fallback_hook_calls == 1:
            return None
        return SimpleNamespace(
            model="restricted-model",
            messages=messages,
            litellm_params=None,
        )

    router.async_pre_routing_hook = stateful_fallback_rewrite

    with patch("litellm.proxy.route_llm_request.mock_testing_params_allowed", return_value=False):
        response = await route_request(
            data={
                "model": "public-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            llm_router=router,
            user_model=None,
            route_type="acompletion",
        )

    assert fallback_hook_calls == 2
    assert response.choices[0].message.content == "safe fallback response"


@pytest.mark.asyncio
async def test_stateful_fallback_concrete_deployment_cannot_escape_preflight_validation():
    router = _stateful_rewrite_router()
    restricted_deployment_id = next(
        deployment["model_info"]["id"]
        for deployment in router.model_list
        if deployment["model_name"] == "restricted-model"
    )
    fallback_hook_calls = 0

    async def stateful_fallback_rewrite(*, model, messages, **kwargs):
        nonlocal fallback_hook_calls
        if model != "fallback-model":
            return None
        fallback_hook_calls += 1
        if fallback_hook_calls == 1:
            return None
        return SimpleNamespace(
            model=restricted_deployment_id,
            messages=messages,
            litellm_params=None,
        )

    router.async_pre_routing_hook = stateful_fallback_rewrite

    with patch("litellm.proxy.route_llm_request.mock_testing_params_allowed", return_value=False):
        response = await route_request(
            data={
                "model": "public-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            llm_router=router,
            user_model=None,
            route_type="acompletion",
        )

    assert fallback_hook_calls == 2
    assert response.choices[0].message.content == "safe fallback response"


@pytest.mark.asyncio
async def test_failed_concrete_fallback_consumes_exclusions_before_next_trusted_target():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"blocked": True},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": Exception("first fallback failed"),
                },
            },
            {
                "model_name": "safe-fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "safe fallback response",
                },
            },
        ],
        model_group_alias={"public-model": "primary-model"},
        fallbacks=[{"public-model": ["fallback-model", "safe-fallback-model"]}],
        num_retries=0,
    )
    fallback_deployment_id = next(
        deployment["model_info"]["id"]
        for deployment in router.model_list
        if deployment["model_name"] == "fallback-model"
    )
    fallback_hook_calls = 0

    async def select_concrete_first_fallback(*, model, messages, **kwargs):
        nonlocal fallback_hook_calls
        if model != "fallback-model":
            return None
        fallback_hook_calls += 1
        return SimpleNamespace(
            model=fallback_deployment_id,
            messages=messages,
            litellm_params=None,
        )

    router.async_pre_routing_hook = select_concrete_first_fallback

    with patch("litellm.proxy.route_llm_request.mock_testing_params_allowed", return_value=False):
        response = await route_request(
            data={
                "model": "public-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            llm_router=router,
            user_model=None,
            route_type="acompletion",
        )

    assert fallback_hook_calls == 2
    assert response.choices[0].message.content == "safe fallback response"


@pytest.mark.asyncio
async def test_retry_cannot_escape_preflight_validated_concrete_deployment():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"blocked": True},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": Exception("validated fallback failed"),
                },
                "model_info": {"id": "validated-fallback-deployment"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "unvalidated retry response",
                },
                "model_info": {"id": "unvalidated-fallback-deployment"},
            },
            {
                "model_name": "safe-fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                    "mock_response": "safe fallback response",
                },
            },
        ],
        model_group_alias={"public-model": "primary-model"},
        fallbacks=[{"public-model": ["fallback-model", "safe-fallback-model"]}],
        num_retries=1,
    )
    hook_calls = 0

    async def pin_then_attempt_retry_escape(*, model, messages, **kwargs):
        nonlocal hook_calls
        if model not in {
            "fallback-model",
            "validated-fallback-deployment",
            "unvalidated-fallback-deployment",
        }:
            return None
        hook_calls += 1
        target = (
            "unvalidated-fallback-deployment"
            if hook_calls >= 3
            else "validated-fallback-deployment"
        )
        return SimpleNamespace(
            model=target,
            messages=messages,
            litellm_params=None,
        )

    router.async_pre_routing_hook = pin_then_attempt_retry_escape

    with patch("litellm.proxy.route_llm_request.mock_testing_params_allowed", return_value=False):
        response = await route_request(
            data={
                "model": "public-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            llm_router=router,
            user_model=None,
            route_type="acompletion",
        )

    assert hook_calls >= 3
    assert response.choices[0].message.content == "safe fallback response"
