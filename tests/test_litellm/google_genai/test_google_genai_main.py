#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content adapter functionality
"""

import datetime
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.google_genai.main import GenerateContentHelper
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig


@pytest.mark.asyncio
async def test_agenerate_content_stream():
    """
    Test that the agenerate_content_stream function works
    """
    from unittest.mock import AsyncMock, patch

    from litellm.google_genai.main import (
        agenerate_content_stream,
        base_llm_http_handler,
    )

    with patch.object(
        base_llm_http_handler, "generate_content_handler", new=AsyncMock()
    ) as mock_post:
        result = await agenerate_content_stream(
            model="gemini/gemini-2.0-flash-001",
            contents="Hello, world!",
            stream=True,
        )
        mock_post.assert_called_once()
        mock_post.call_args.kwargs["stream"] == True


def test_setup_generate_content_call_propagates_user_to_logging_obj():
    """
    Regression test for Google-native /v1beta/.../:generateContent route
    dropping the `user` field from the spend log.

    `setup_generate_content_call` must pass `user` from kwargs to
    `litellm_logging_obj.update_from_kwargs` so that `logging_obj.user` and
    `model_call_details["user"]` reflect what the client sent (via header or
    body) instead of defaulting to None / "".
    """
    # Use a real Logging instance (required by Pydantic model validation),
    # but stub out update_from_kwargs so we can assert what it was called with.
    real_logging_obj = Logging(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="agenerate_content",
        start_time=datetime.datetime.now(),
        litellm_call_id="call-id-123",
        function_id="",
    )
    mock_update = MagicMock()
    real_logging_obj.update_from_kwargs = mock_update  # type: ignore[method-assign]

    # Mock the provider config so we hit the update_from_kwargs path (not the
    # adapter early-return path). Use spec= on a real subclass so Pydantic's
    # is_instance_of check passes.
    mock_provider_config = MagicMock(spec=GoogleGenAIConfig())
    mock_provider_config.map_generate_content_optional_params.return_value = {}
    mock_provider_config.transform_generate_content_request.return_value = {}

    with (
        patch(
            "litellm.get_llm_provider",
            return_value=("gemini-2.5-pro", "gemini", None, None),
        ),
        patch(
            "litellm.utils.ProviderConfigManager.get_provider_google_genai_generate_content_config",
            return_value=mock_provider_config,
        ),
    ):
        GenerateContentHelper.setup_generate_content_call(
            model="gemini-2.5-pro",
            contents=[{"parts": [{"text": "hi"}], "role": "user"}],
            config={},
            custom_llm_provider="gemini",
            litellm_logging_obj=real_logging_obj,
            litellm_call_id="call-id-123",
            user="my-end-user-uuid-456",
            metadata={"tags": ["scan_id=abc"]},
        )

    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args.kwargs
    assert (
        call_kwargs.get("user") == "my-end-user-uuid-456"
    ), f"Expected user to be propagated, got: {call_kwargs.get('user')!r}"
