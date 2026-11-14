#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content adapter functionality
"""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig
from litellm.llms.vertex_ai.google_genai.transformation import VertexAIGoogleGenAIConfig
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.router import GenericLiteLLMParams


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
    ("continuationToken", "dG9rLTE="),
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


@pytest.mark.parametrize("base_model, uses_native_api", [("gemini-2.5-flash", True), (None, False)])
def test_vertex_generate_content_routes_by_deployment_base_model(base_model, uses_native_api):
    import datetime

    from litellm.google_genai.main import GenerateContentHelper
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = Logging(
        model="vertex_ai/openai/tuned-endpoint",
        messages=[],
        stream=False,
        call_type="agenerate_content",
        start_time=datetime.datetime.now(),
        litellm_call_id="route-test",
        function_id="route-test",
    )
    setup = GenerateContentHelper.setup_generate_content_call(
        model="vertex_ai/openai/tuned-endpoint",
        contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
        litellm_logging_obj=logging_obj,
        **({} if base_model is None else {"base_model": base_model}),
    )

    assert (setup.generate_content_provider_config is not None) is uses_native_api


_LABEL_FIELDS: Final = ("user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata")
_IDENTITY: Final = MappingProxyType({"user_api_key_alias": "chargeback-key", "user_api_key_team_alias": "ai-platform"})
_PROXY_METADATA: Final = MappingProxyType(
    {**_IDENTITY, "spend_logs_metadata": {"cost-center": "cc-1"}, "requester_metadata": {"team": "search"}}
)
_FORGED: Final = MappingProxyType({"user_api_key_alias": "forged", "app": "bot"})
_HANDLER_CALLS: Final = pytest.mark.parametrize(
    ("is_async", "stream"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["sync", "async", "sync-stream", "async-stream"],
)
_GEMINI_REPLY: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
}
_SENT_BODY: Final = TypeAdapter(Mapping[str, object])
_SENT_LABELS: Final = TypeAdapter(Mapping[str, str])


class _SignedInVertexGoogleGenAIConfig(VertexAIGoogleGenAIConfig):
    def _ensure_access_token(
        self,
        credentials: VERTEX_CREDENTIALS_TYPES | None,
        project_id: str | None,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    ) -> tuple[str, str]:
        return "token", "proj"

    async def _ensure_access_token_async(
        self,
        credentials: VERTEX_CREDENTIALS_TYPES | None,
        project_id: str | None,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    ) -> tuple[str, str]:
        return "token", "proj"


async def _generate_content_body(
    config: GoogleGenAIConfig,
    litellm_params: GenericLiteLLMParams,
    extra_body: Mapping[str, object],
    *,
    is_async: bool,
    stream: bool,
) -> Mapping[str, object]:
    import datetime
    from unittest.mock import MagicMock

    import httpx

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    reply: Final = MagicMock(side_effect=lambda _request: httpx.Response(200, json=_GEMINI_REPLY))
    transport: Final = httpx.MockTransport(reply)
    client: Final = (
        AsyncHTTPHandler(transport=transport) if is_async else HTTPHandler(client=httpx.Client(transport=transport))
    )
    call: Final = BaseLLMHTTPHandler().generate_content_handler(
        model="gemini-3.8-flash",
        contents=[{"role": "user", "parts": [{"text": "Say hi"}]}],
        generate_content_provider_config=config,
        generate_content_config_dict={},
        tools=None,
        custom_llm_provider=config.custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=Logging(
            model="gemini-3.8-flash",
            messages=[],
            stream=stream,
            call_type="agenerate_content" if is_async else "generate_content",
            start_time=datetime.datetime(2026, 1, 1),
            litellm_call_id="labels-test",
            function_id="labels-test",
        ),
        extra_body=dict(extra_body),
        _is_async=is_async,
        client=client,
        stream=stream,
    )
    if is_async:
        await call

    sent: Final[httpx.Request] = reply.call_args.args[0]
    return _SENT_BODY.validate_json(sent.content)


async def _vertex_generate_content_body(
    extra_body: Mapping[str, object], *, is_async: bool, stream: bool
) -> Mapping[str, object]:
    return await _generate_content_body(
        _SignedInVertexGoogleGenAIConfig(),
        GenericLiteLLMParams(metadata=dict(_PROXY_METADATA), vertex_location="us-central1"),
        extra_body,
        is_async=is_async,
        stream=stream,
    )


@pytest.mark.asyncio
@_HANDLER_CALLS
async def test_vertex_generate_content_sends_identity_labels_a_caller_cannot_forge(
    is_async: bool, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", _LABEL_FIELDS)

    body: Final = await _vertex_generate_content_body({"labels": dict(_FORGED)}, is_async=is_async, stream=stream)

    labels: Final = _SENT_LABELS.validate_python(body["labels"])
    assert list(labels.items()) == [*_IDENTITY.items(), ("app", "bot"), ("cost-center", "cc-1")]


@pytest.mark.asyncio
@_HANDLER_CALLS
async def test_vertex_generate_content_without_caller_labels_still_carries_identity(
    is_async: bool, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", _LABEL_FIELDS)

    body: Final = await _vertex_generate_content_body({}, is_async=is_async, stream=stream)

    labels: Final = _SENT_LABELS.validate_python(body["labels"])
    assert list(labels.items()) == [*_IDENTITY.items(), ("team", "search"), ("cost-center", "cc-1")]


@pytest.mark.asyncio
@_HANDLER_CALLS
@pytest.mark.parametrize(
    "extra_body, expected_labels",
    [({}, "absent"), ({"labels": dict(_FORGED)}, _FORGED), ({"labels": "not-a-mapping"}, "not-a-mapping")],
    ids=["no-labels", "labels", "non-mapping-labels"],
)
async def test_vertex_generate_content_with_the_setting_off_sends_only_the_callers_labels(
    is_async: bool,
    stream: bool,
    extra_body: Mapping[str, object],
    expected_labels: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", None)

    body: Final = await _vertex_generate_content_body(extra_body, is_async=is_async, stream=stream)

    assert body.get("labels", "absent") == expected_labels


@pytest.mark.asyncio
@_HANDLER_CALLS
async def test_ai_studio_generate_content_keeps_the_callers_labels_with_the_setting_on(
    is_async: bool, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", _LABEL_FIELDS)

    body: Final = await _generate_content_body(
        GoogleGenAIConfig(),
        GenericLiteLLMParams(metadata=dict(_PROXY_METADATA), api_key="test-key"),
        {"labels": dict(_FORGED)},
        is_async=is_async,
        stream=stream,
    )

    assert body["labels"] == _FORGED
