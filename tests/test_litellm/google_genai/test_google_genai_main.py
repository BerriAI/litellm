#!/usr/bin/env python3
"""Tests for Google GenAI main entrypoints."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))


@pytest.mark.asyncio
async def test_agenerate_content_stream():
    """
    Test that the agenerate_content_stream function works
    """
    from litellm.google_genai.main import (
        agenerate_content_stream,
        base_llm_http_handler,
    )

    with patch.object(
        base_llm_http_handler, "generate_content_handler", new=AsyncMock()
    ) as mock_post:
        await agenerate_content_stream(
            model="gemini/gemini-2.0-flash-001",
            contents="Hello, world!",
            stream=True,
        )
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["stream"] is True


def test_generate_content_stream_forwards_system_instruction():
    """Test that generate_content_stream forwards systemInstruction and toolConfig."""
    from litellm.google_genai.main import (
        base_llm_http_handler,
        generate_content_stream,
    )

    mock_response = MagicMock()
    tool_config = {"functionCallingConfig": {"mode": "ANY"}}

    with patch.object(
        base_llm_http_handler, "generate_content_handler", return_value=mock_response
    ) as mock_post:
        result = generate_content_stream(
            model="gemini/gemini-2.0-flash-001",
            contents="Hello, world!",
            stream=True,
            systemInstruction={"parts": [{"text": "You are helpful"}]},
            toolConfig=tool_config,
        )

        assert result is mock_response
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["stream"] is True
        assert mock_post.call_args.kwargs["tool_config"] == tool_config
        assert mock_post.call_args.kwargs["system_instruction"] == {
            "parts": [{"text": "You are helpful"}]
        }
