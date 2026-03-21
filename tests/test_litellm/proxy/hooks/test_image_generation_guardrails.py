"""
Tests that guardrails (post_call_success_hook) fire for image generation requests.

The /images/generations endpoint in proxy/image_endpoints/endpoints.py calls
proxy_logging_obj.post_call_success_hook after a successful image generation.
These tests verify:
1. CustomGuardrail.async_post_call_success_hook is invoked for image generation.
2. A guardrail can inspect and transform the image response.
3. A guardrail that raises blocks the response (exception propagates).
"""

import os
import sys
from typing import Any, Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import ImageObject, ImageResponse


def _make_image_response(**kwargs) -> ImageResponse:
    """Helper to build a minimal ImageResponse for tests."""
    return ImageResponse(
        data=[ImageObject(url="https://example.com/img.png")],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Hook is invoked for image generation responses
# ---------------------------------------------------------------------------


class TrackingGuardrail(CustomGuardrail):
    """Guardrail that records whether it was called and with what args."""

    def __init__(self):
        super().__init__(
            guardrail_name="tracking_guardrail",
            default_on=True,
            event_hook=GuardrailEventHooks.post_call,
        )
        self.called = False
        self.received_data: Optional[dict] = None
        self.received_response: Optional[Any] = None

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        self.called = True
        self.received_data = data
        self.received_response = response
        return response


@pytest.mark.asyncio
async def test_post_call_success_hook_invoked_for_image_generation():
    """
    Verify that a default-on guardrail's async_post_call_success_hook is
    called when ProxyLogging.post_call_success_hook is invoked with an
    ImageResponse (the same path used by the /images/generations endpoint).
    """
    guardrail = TrackingGuardrail()
    image_response = _make_image_response()

    with patch("litellm.callbacks", [guardrail]):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        data = {"model": "dall-e-3", "prompt": "A sunset over mountains"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        result = await proxy_logging.post_call_success_hook(
            data=data,
            response=image_response,
            user_api_key_dict=user_api_key_dict,
        )

    assert guardrail.called is True, "Guardrail hook was not invoked for image generation"
    assert guardrail.received_data is not None
    assert guardrail.received_data["model"] == "dall-e-3"
    assert isinstance(guardrail.received_response, ImageResponse)
    # The response should be passed through unchanged
    assert result is image_response


# ---------------------------------------------------------------------------
# 2. Guardrail can transform image generation response
# ---------------------------------------------------------------------------


class TransformingGuardrail(CustomGuardrail):
    """Guardrail that replaces the image URL in the response."""

    def __init__(self):
        super().__init__(
            guardrail_name="transforming_guardrail",
            default_on=True,
            event_hook=GuardrailEventHooks.post_call,
        )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        # Return a modified image response (e.g., watermarked URL)
        return ImageResponse(
            data=[ImageObject(url="https://example.com/watermarked.png")],
        )


@pytest.mark.asyncio
async def test_guardrail_can_transform_image_response():
    """
    Verify that a guardrail can replace the ImageResponse returned to the client.
    """
    guardrail = TransformingGuardrail()
    original_response = _make_image_response()

    with patch("litellm.callbacks", [guardrail]):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        data = {"model": "dall-e-3", "prompt": "A sunset"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        result = await proxy_logging.post_call_success_hook(
            data=data,
            response=original_response,
            user_api_key_dict=user_api_key_dict,
        )

    assert result is not original_response
    assert isinstance(result, ImageResponse)
    assert result.data[0].url == "https://example.com/watermarked.png"


# ---------------------------------------------------------------------------
# 3. Guardrail that raises blocks the image response
# ---------------------------------------------------------------------------


class BlockingGuardrail(CustomGuardrail):
    """Guardrail that raises on unsafe image prompts."""

    def __init__(self):
        super().__init__(
            guardrail_name="blocking_guardrail",
            default_on=True,
            event_hook=GuardrailEventHooks.post_call,
        )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        raise ValueError("Image content blocked by guardrail")


@pytest.mark.asyncio
async def test_guardrail_exception_propagates_for_image_generation():
    """
    Verify that an exception raised in a guardrail's post_call_success_hook
    propagates up (the proxy endpoint wraps this in an error response).
    """
    guardrail = BlockingGuardrail()

    with patch("litellm.callbacks", [guardrail]):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        data = {"model": "dall-e-3", "prompt": "Something unsafe"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        with pytest.raises(ValueError, match="Image content blocked by guardrail"):
            await proxy_logging.post_call_success_hook(
                data=data,
                response=_make_image_response(),
                user_api_key_dict=user_api_key_dict,
            )


# ---------------------------------------------------------------------------
# 4. Non-guardrail CustomLogger also fires for image generation
# ---------------------------------------------------------------------------


class TrackingLogger(CustomLogger):
    """Plain CustomLogger (not a guardrail) that tracks invocations."""

    def __init__(self):
        self.called = False
        self.received_response = None

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        self.called = True
        self.received_response = response
        return response


@pytest.mark.asyncio
async def test_custom_logger_post_call_success_hook_fires_for_image_generation():
    """
    Verify that a plain CustomLogger (non-guardrail) callback also has its
    async_post_call_success_hook invoked for image generation responses.
    """
    logger = TrackingLogger()
    image_response = _make_image_response()

    with patch("litellm.callbacks", [logger]):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        data = {"model": "dall-e-3", "prompt": "A cat"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        result = await proxy_logging.post_call_success_hook(
            data=data,
            response=image_response,
            user_api_key_dict=user_api_key_dict,
        )

    assert logger.called is True
    assert isinstance(logger.received_response, ImageResponse)
    assert result is image_response


# ---------------------------------------------------------------------------
# 5. Guardrail with should_run_guardrail=False is skipped
# ---------------------------------------------------------------------------


class OptInGuardrail(CustomGuardrail):
    """Guardrail that is NOT default_on, so it only runs if explicitly requested."""

    def __init__(self):
        super().__init__(
            guardrail_name="opt_in_guardrail",
            default_on=False,
            event_hook=GuardrailEventHooks.post_call,
        )
        self.called = False

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        self.called = True
        return response


@pytest.mark.asyncio
async def test_non_default_guardrail_skipped_for_image_generation():
    """
    Verify that a guardrail with default_on=False is NOT invoked for image
    generation unless the request explicitly enables it.
    """
    guardrail = OptInGuardrail()

    with patch("litellm.callbacks", [guardrail]):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        # No guardrails key in data -> should_run_guardrail returns False
        data = {"model": "dall-e-3", "prompt": "A sunset"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        await proxy_logging.post_call_success_hook(
            data=data,
            response=_make_image_response(),
            user_api_key_dict=user_api_key_dict,
        )

    assert guardrail.called is False, "Opt-in guardrail should not fire without explicit request"
