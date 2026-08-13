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
        assert mock_post.call_args.kwargs["stream"] is True


def _mock_gemini_post_response():
    """A minimal stand-in for a successful Gemini generateContent HTTP response."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "hi"}], "role": "model"},
                "finishReason": "STOP",
            }
        ]
    }
    return resp


NATIVE_TOP_LEVEL_FIELD_CASES = [
    (
        "safetySettings",
        [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
    ),
    ("toolConfig", {"functionCallingConfig": {"mode": "AUTO"}}),
    ("cachedContent", "cachedContents/abc123"),
    ("labels", {"team": "search"}),
]


@pytest.mark.parametrize("field_name, field_value", NATIVE_TOP_LEVEL_FIELD_CASES)
def test_native_top_level_field_forwarded_to_request_body(field_name, field_value):
    """
    Regression for https://github.com/BerriAI/litellm/issues/12671

    Google's native generateContent body carries fields like safetySettings at the
    top level (siblings of generationConfig). The proxy spreads them as loose kwargs
    into generate_content. They must reach Google's request body at the top level and
    must NOT be silently dropped nor nested under generationConfig.
    """
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            **{field_name: field_value},
        )

    assert mock_post.called, "expected the request to reach the HTTP client"
    body = mock_post.call_args.kwargs["json"]
    assert (
        body[field_name] == field_value
    ), f"{field_name} should be forwarded to Google at the top level"
    assert field_name not in body.get("generationConfig", {}), (
        f"{field_name} must be a top-level sibling of generationConfig, "
        "not nested inside it"
    )


@pytest.mark.asyncio
async def test_native_safety_settings_forwarded_async():
    """The async path (used by the proxy's :generateContent route) must also forward
    native top-level fields."""
    from unittest.mock import AsyncMock, patch

    from litellm.google_genai.main import agenerate_content
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]

    with patch.object(
        AsyncHTTPHandler,
        "post",
        new_callable=AsyncMock,
        return_value=_mock_gemini_post_response(),
    ) as mock_post:
        await agenerate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
        )

    assert mock_post.called
    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    assert "safetySettings" not in body.get("generationConfig", {})


def test_native_fields_coexist_with_generation_config():
    """Forwarding native top-level fields must not regress the already-working
    generationConfig path; both must land in their correct positions."""
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
            generationConfig={"temperature": 0, "responseMimeType": "application/json"},
        )

    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    generation_config = body["generationConfig"]
    assert generation_config["temperature"] == 0
    assert generation_config["responseMimeType"] == "application/json"


def test_explicit_extra_body_overrides_native_top_level_field():
    """An explicit extra_body value takes precedence over the same top-level field."""
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    native = [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]
    override = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}
    ]

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=native,
            extra_body={"safetySettings": override},
        )

    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == override


def test_native_fields_and_system_instruction_forwarded_on_sync_stream():
    """The sync streaming path (generate_content_stream) must forward native top-level
    fields AND systemInstruction. The PR changed the merge here and newly added the
    systemInstruction kwarg; without coverage a regression on either ships green."""
    from unittest.mock import patch

    from litellm.google_genai.main import generate_content_stream
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]
    system_instruction = {"parts": [{"text": "Be terse"}]}

    with patch.object(
        HTTPHandler, "post", return_value=_mock_gemini_post_response()
    ) as mock_post:
        generate_content_stream(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
            systemInstruction=system_instruction,
        )

    assert mock_post.called
    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    assert body["systemInstruction"] == system_instruction
    assert "safetySettings" not in body.get("generationConfig", {})


@pytest.mark.asyncio
async def test_native_fields_forwarded_on_async_stream():
    """The async streaming path (agenerate_content_stream) backs the proxy's
    :streamGenerateContent route and must forward native top-level fields too."""
    from unittest.mock import AsyncMock, patch

    from litellm.google_genai.main import agenerate_content_stream
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ]

    with patch.object(
        AsyncHTTPHandler,
        "post",
        new_callable=AsyncMock,
        return_value=_mock_gemini_post_response(),
    ) as mock_post:
        await agenerate_content_stream(
            model="gemini/gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
            custom_llm_provider="gemini",
            api_key="test-key",
            safetySettings=safety_settings,
        )

    assert mock_post.called
    body = mock_post.call_args.kwargs["json"]
    assert body["safetySettings"] == safety_settings
    assert "safetySettings" not in body.get("generationConfig", {})
