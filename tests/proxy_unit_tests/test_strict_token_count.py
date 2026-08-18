"""
Tests for per-model strict token counting (`model_info.strict_token_count`).

Context: https://github.com/BerriAI/litellm/issues/37102

When a provider's token counting API cannot count a model, the proxy falls back
to a local tiktoken estimate. That estimate can be materially lower than the
real count, and `/v1/messages/count_tokens` returns it in a shape that is
indistinguishable from an exact count.

`litellm.disable_token_counter` already turns that fallback into an error, but
it is proxy-wide. `strict_token_count` in a model's `model_info` opts a single
model into the same behaviour.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
import litellm.proxy.proxy_server as proxy_server
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.proxy._types import ProxyException, TokenCountRequest
from litellm.proxy.proxy_server import token_counter
from litellm.router import Router

# the error Bedrock CountTokens returns for a model it cannot count
UNSUPPORTED_MODEL_ERROR = "This model doesn't support counting tokens."


def _router(model_info=None):
    deployment = {
        "model_name": "claude-opus-5",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-opus-5-20260101-v1:0",
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "fake",
            "aws_secret_access_key": "fake",
        },
    }
    if model_info is not None:
        deployment["model_info"] = model_info
    return Router(model_list=[deployment])


def _unsupported_count_tokens():
    """Patch Bedrock CountTokens to reject the model, as it does in the report."""
    return patch(
        "litellm.llms.bedrock.count_tokens.handler.BedrockCountTokensHandler.handle_count_tokens_request",
        new=AsyncMock(side_effect=BedrockError(status_code=400, message=UNSUPPORTED_MODEL_ERROR)),
    )


async def _count_tokens(router):
    original_router = getattr(proxy_server, "llm_router", None)
    setattr(proxy_server, "llm_router", router)
    try:
        return await token_counter(
            request=TokenCountRequest(
                model="claude-opus-5",
                messages=[{"role": "user", "content": "hello " * 400}],
            ),
            call_endpoint=True,
        )
    finally:
        setattr(proxy_server, "llm_router", original_router)


@pytest.mark.asyncio
async def test_strict_token_count_raises_instead_of_estimating():
    """A model marked strict must fail rather than return a local estimate."""
    router = _router(model_info={"strict_token_count": True})

    with _unsupported_count_tokens():
        with pytest.raises(ProxyException) as exc_info:
            await _count_tokens(router)

    assert exc_info.value.type == "token_counting_error"
    assert UNSUPPORTED_MODEL_ERROR in exc_info.value.message


@pytest.mark.asyncio
async def test_without_strict_token_count_falls_back_to_estimate():
    """Default behaviour is unchanged: fall back to the local estimate."""
    router = _router()

    with _unsupported_count_tokens():
        response = await _count_tokens(router)

    assert response.total_tokens > 0
    # the estimate came from a local tokenizer, not the Bedrock API
    assert response.tokenizer_type != "bedrock_api"


@pytest.mark.asyncio
async def test_strict_token_count_false_is_explicitly_permissive():
    """`strict_token_count: false` must behave like the flag being absent."""
    router = _router(model_info={"strict_token_count": False})

    with _unsupported_count_tokens():
        response = await _count_tokens(router)

    assert response.total_tokens > 0


@pytest.mark.asyncio
async def test_strict_token_count_does_not_affect_other_models():
    """
    The point of the flag: one strict model must not make every other model
    strict. `disable_token_counter` could not express this.
    """
    router = Router(
        model_list=[
            {
                "model_name": "strict-model",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-opus-5-20260101-v1:0",
                    "aws_region_name": "us-east-1",
                    "aws_access_key_id": "fake",
                    "aws_secret_access_key": "fake",
                },
                "model_info": {"strict_token_count": True},
            },
            {
                "model_name": "lenient-model",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-opus-5-20260101-v1:0",
                    "aws_region_name": "us-east-1",
                    "aws_access_key_id": "fake",
                    "aws_secret_access_key": "fake",
                },
            },
        ]
    )

    original_router = getattr(proxy_server, "llm_router", None)
    setattr(proxy_server, "llm_router", router)
    try:
        with _unsupported_count_tokens():
            # strict model refuses, with the provider's reason preserved.
            # Asserting the type matters: it distinguishes the provider-failure
            # path from the later generic "no provider result" checkpoint.
            with pytest.raises(ProxyException) as exc_info:
                await token_counter(
                    request=TokenCountRequest(
                        model="strict-model",
                        messages=[{"role": "user", "content": "hello " * 400}],
                    ),
                    call_endpoint=True,
                )
            assert exc_info.value.type == "token_counting_error"
            assert UNSUPPORTED_MODEL_ERROR in exc_info.value.message

            # the other model, same provider and same failure, still estimates
            response = await token_counter(
                request=TokenCountRequest(
                    model="lenient-model",
                    messages=[{"role": "user", "content": "hello " * 400}],
                ),
                call_endpoint=True,
            )
            assert response.total_tokens > 0
    finally:
        setattr(proxy_server, "llm_router", original_router)


@pytest.mark.asyncio
async def test_strict_read_from_the_selected_deployment():
    """Strictness on the *selected deployment* is honoured on its own.

    The router's configured list here does not carry the flag, so only the
    deployment returned by selection does. This pins the `model_info` branch
    independently of the router-config fallback.
    """
    router = _router()  # configured without strict_token_count

    original = Router.async_get_available_deployment

    async def _strict_deployment(self, *args, **kwargs):
        deployment = await original(self, *args, **kwargs)
        deployment = dict(deployment)
        deployment["model_info"] = {
            **(deployment.get("model_info") or {}),
            "strict_token_count": True,
        }
        return deployment

    with patch.object(Router, "async_get_available_deployment", new=_strict_deployment):
        with _unsupported_count_tokens():
            with pytest.raises(ProxyException) as exc_info:
                await _count_tokens(router)

    assert exc_info.value.type == "token_counting_error"
    assert UNSUPPORTED_MODEL_ERROR in exc_info.value.message


@pytest.mark.asyncio
async def test_strict_survives_deployment_selection_failure():
    """A strict model must not fall back to an estimate when routing fails.

    Deployment selection can fail for reasons unrelated to the policy - every
    deployment cooling down, rate limited, unhealthy. The selected deployment's
    `model_info` is unavailable then, so resolving strictness only from it would
    hand back exactly the estimate the flag exists to refuse.
    """
    router = _router(model_info={"strict_token_count": True})

    with patch.object(
        Router,
        "async_get_available_deployment",
        new=AsyncMock(side_effect=Exception("No deployments available - cooldown")),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await _count_tokens(router)

    assert exc_info.value.type == "token_counting_disabled"


@pytest.mark.asyncio
async def test_non_strict_model_still_estimates_when_selection_fails():
    """The failure path stays permissive for models that never opted in."""
    router = _router()

    with patch.object(
        Router,
        "async_get_available_deployment",
        new=AsyncMock(side_effect=Exception("No deployments available - cooldown")),
    ):
        response = await _count_tokens(router)

    assert response.total_tokens > 0


@pytest.mark.asyncio
async def test_disable_token_counter_still_applies_proxy_wide():
    """The existing proxy-wide flag must keep working for unmarked models."""
    router = _router()
    original = litellm.disable_token_counter
    litellm.disable_token_counter = True
    try:
        with _unsupported_count_tokens():
            with pytest.raises(ProxyException):
                await _count_tokens(router)
    finally:
        litellm.disable_token_counter = original
