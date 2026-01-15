#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content adapter functionality
"""
import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import json
import os
import sys

import pytest

import litellm


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
