"""Regression tests for skills HTTP error surfacing (#31587).

The skills handlers (create/list/get/delete, sync + async) used to pass the
raw upstream response straight into ``transform_*_skill_response`` without
calling ``raise_for_status()``. When Anthropic returned an error response
(e.g. 401 on an invalid key, 400 on a malformed request), the body shape
``{"type": "error", "error": ...}`` was force-parsed as a success payload
and surfaced as a Pydantic ValidationError with HTTP 500, instead of as a
proper upstream error.

These tests pin the contract that an error-status response raises
``httpx.HTTPStatusError`` *before* the response transform runs.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.skills.transformation import AnthropicSkillsConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.llms.anthropic_skills import ListSkillsResponse
from litellm.types.router import GenericLiteLLMParams


class _RecordingConfig(AnthropicSkillsConfig):
    """Wrap the real Anthropic config so abstract methods stay satisfied, but
    record whether ``transform_*_skill_response`` actually ran.

    A raise_for_status() failure must skip the transform entirely; if the
    transform runs, the bug from #31587 is back.
    """

    def __init__(self) -> None:
        self.transform_called = False

    def transform_list_skills_response(self, raw_response, logging_obj):  # type: ignore[override]
        self.transform_called = True
        return ListSkillsResponse(data=[], has_more=False, next_page=None)

    def transform_create_skill_response(self, raw_response, logging_obj):  # type: ignore[override]
        self.transform_called = True
        raise AssertionError("transform_create_skill_response must not run on error response")

    def transform_get_skill_response(self, raw_response, logging_obj):  # type: ignore[override]
        self.transform_called = True
        raise AssertionError("transform_get_skill_response must not run on error response")

    def transform_delete_skill_response(self, raw_response, logging_obj):  # type: ignore[override]
        self.transform_called = True
        raise AssertionError("transform_delete_skill_response must not run on error response")


def _error_response(status_code: int, method: str) -> httpx.Response:
    """Anthropic-shaped error body — matches the issue's reproduction."""
    return httpx.Response(
        status_code=status_code,
        json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad key"}},
        request=httpx.Request(method, "https://api.anthropic.com/v1/skills"),
    )


def _ok_response(method: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={"data": [], "has_more": False, "next_page": None},
        request=httpx.Request(method, "https://api.anthropic.com/v1/skills"),
    )


def _make_sync_client(response: httpx.Response):
    client = MagicMock()
    client.get = MagicMock(return_value=response)
    client.post = MagicMock(return_value=response)
    client.delete = MagicMock(return_value=response)
    return client


def _make_async_client(response: httpx.Response):
    async def _co(*a, **kw):
        return response

    client = MagicMock()
    client.get = MagicMock(return_value=_co())
    client.post = MagicMock(return_value=_co())
    client.delete = MagicMock(return_value=_co())
    return client


def _common_kwargs():
    return dict(
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=MagicMock(),
    )


_HANDLER_MODULE = "litellm.llms.custom_httpx.llm_http_handler"


# ---------------------------------------------------------------------------
# Sync handlers
# ---------------------------------------------------------------------------


def test_list_skills_handler_raises_on_error_response():
    config = _RecordingConfig()
    response = _error_response(401, "GET")
    handler = BaseLLMHTTPHandler()
    with patch(f"{_HANDLER_MODULE}._get_httpx_client", return_value=_make_sync_client(response)):
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            handler.list_skills_handler(
                url="https://api.anthropic.com/v1/skills",
                query_params={},
                skills_api_provider_config=config,
                **_common_kwargs(),
            )

    assert exc_info.value.response.status_code == 401
    assert not config.transform_called, "transform must be skipped when raise_for_status fires"


def test_create_skills_handler_raises_on_error_response():
    config = _RecordingConfig()
    response = _error_response(400, "POST")
    handler = BaseLLMHTTPHandler()
    with patch(f"{_HANDLER_MODULE}._get_httpx_client", return_value=_make_sync_client(response)):
        with pytest.raises(httpx.HTTPStatusError):
            handler.create_skill_handler(
                url="https://api.anthropic.com/v1/skills",
                request_body={"display_title": "x"},
                skills_api_provider_config=config,
                **_common_kwargs(),
            )

    assert not config.transform_called


def test_get_skills_handler_raises_on_error_response():
    config = _RecordingConfig()
    response = _error_response(404, "GET")
    handler = BaseLLMHTTPHandler()
    with patch(f"{_HANDLER_MODULE}._get_httpx_client", return_value=_make_sync_client(response)):
        with pytest.raises(httpx.HTTPStatusError):
            handler.get_skill_handler(
                url="https://api.anthropic.com/v1/skills/skill_abc",
                skills_api_provider_config=config,
                **_common_kwargs(),
            )

    assert not config.transform_called


def test_delete_skills_handler_raises_on_error_response():
    config = _RecordingConfig()
    response = _error_response(403, "DELETE")
    handler = BaseLLMHTTPHandler()
    with patch(f"{_HANDLER_MODULE}._get_httpx_client", return_value=_make_sync_client(response)):
        with pytest.raises(httpx.HTTPStatusError):
            handler.delete_skill_handler(
                url="https://api.anthropic.com/v1/skills/skill_abc",
                skills_api_provider_config=config,
                **_common_kwargs(),
            )

    assert not config.transform_called


# ---------------------------------------------------------------------------
# Async handlers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_list_skills_handler_raises_on_error_response():
    config = _RecordingConfig()
    response = _error_response(401, "GET")
    handler = BaseLLMHTTPHandler()
    with patch(f"{_HANDLER_MODULE}.get_async_httpx_client", return_value=_make_async_client(response)):
        with pytest.raises(httpx.HTTPStatusError):
            await handler.async_list_skills_handler(
                url="https://api.anthropic.com/v1/skills",
                query_params={},
                skills_api_provider_config=config,
                **_common_kwargs(),
            )

    assert not config.transform_called


@pytest.mark.asyncio
async def test_async_create_skills_handler_raises_on_error_response():
    config = _RecordingConfig()
    response = _error_response(400, "POST")
    handler = BaseLLMHTTPHandler()
    with patch(f"{_HANDLER_MODULE}.get_async_httpx_client", return_value=_make_async_client(response)):
        with pytest.raises(httpx.HTTPStatusError):
            await handler.async_create_skill_handler(
                url="https://api.anthropic.com/v1/skills",
                request_body={"display_title": "x"},
                skills_api_provider_config=config,
                **_common_kwargs(),
            )

    assert not config.transform_called


@pytest.mark.asyncio
async def test_async_get_skills_handler_raises_on_error_response():
    config = _RecordingConfig()
    response = _error_response(404, "GET")
    handler = BaseLLMHTTPHandler()
    with patch(f"{_HANDLER_MODULE}.get_async_httpx_client", return_value=_make_async_client(response)):
        with pytest.raises(httpx.HTTPStatusError):
            await handler.async_get_skill_handler(
                url="https://api.anthropic.com/v1/skills/skill_abc",
                skills_api_provider_config=config,
                **_common_kwargs(),
            )

    assert not config.transform_called


@pytest.mark.asyncio
async def test_async_delete_skills_handler_raises_on_error_response():
    config = _RecordingConfig()
    response = _error_response(403, "DELETE")
    handler = BaseLLMHTTPHandler()
    with patch(f"{_HANDLER_MODULE}.get_async_httpx_client", return_value=_make_async_client(response)):
        with pytest.raises(httpx.HTTPStatusError):
            await handler.async_delete_skill_handler(
                url="https://api.anthropic.com/v1/skills/skill_abc",
                skills_api_provider_config=config,
                **_common_kwargs(),
            )

    assert not config.transform_called


# ---------------------------------------------------------------------------
# Sanity: success response still flows through to transform
# ---------------------------------------------------------------------------


def test_list_skills_handler_runs_transform_on_success():
    config = _RecordingConfig()
    response = _ok_response("GET")
    handler = BaseLLMHTTPHandler()
    with patch(f"{_HANDLER_MODULE}._get_httpx_client", return_value=_make_sync_client(response)):
        result = handler.list_skills_handler(
            url="https://api.anthropic.com/v1/skills",
            query_params={},
            skills_api_provider_config=config,
            **_common_kwargs(),
        )

    assert config.transform_called
    assert isinstance(result, ListSkillsResponse)
