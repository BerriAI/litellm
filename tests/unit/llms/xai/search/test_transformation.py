import json
from typing import Final

import openai
import pytest
import respx

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.xai.search.transformation import XAISearchConfig

RESPONSES_URL: Final = "https://api.x.ai/v1/responses"
ANSWER: Final = "Grok update from X."
URLS: Final = ("https://x.com/xai/status/123", "https://x.com/xai/status/456")


def _response(ticks: int = 123456789) -> dict[str, object]:
    return {
        "status": "completed",
        "output": [
            {"type": "custom_tool_call", "name": "x_keyword_search"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": ANSWER,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": URLS[0],
                                "title": "Update",
                                "start_index": 0,
                                "end_index": 0,
                            },
                            {
                                "type": "url_citation",
                                "url": URLS[1],
                                "start_index": 0,
                                "end_index": 0,
                            },
                            {"type": "url_citation", "url": ""},
                            {"type": "file_citation", "file_id": "ignored"},
                        ],
                    }
                ],
            },
        ],
        "usage": {
            "input_tokens": 500,
            "output_tokens": 100,
            "cost_in_usd_ticks": ticks,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_search_returns_citations_and_preserves_the_full_bill_after_limiting_results(
    respx_mock: respx.MockRouter, use_async: bool
) -> None:
    route = respx_mock.post(RESPONSES_URL).respond(json=_response())
    params = {
        "query": "Grok update",
        "search_provider": "xai",
        "api_key": "test-xai-key",
        "max_results": 1,
    }
    result = await litellm.asearch(**params) if use_async else litellm.search(**params)

    assert isinstance(result, SearchResponse)
    assert result.model_dump() == {
        "object": "search",
        "results": [
            {
                "title": "Update",
                "url": URLS[0],
                "snippet": ANSWER,
                "date": None,
                "last_updated": None,
            }
        ],
    }
    assert result.hidden_params["response_cost"] == pytest.approx(123456789 / 10_000_000_000)
    assert json.loads(route.calls.last.request.content) == {
        "model": "grok-4.7",
        "input": "Grok update",
        "tools": [{"type": "x_search"}],
        "tool_choice": "required",
        "store": False,
    }


@pytest.mark.asyncio
async def test_configured_search_model_and_filters_reach_xai_through_the_router(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(RESPONSES_URL).respond(json=_response(0))
    router = litellm.Router(
        search_tools=[
            {
                "search_tool_name": "social-search",
                "litellm_params": {
                    "search_provider": "xai",
                    "api_key": "test-xai-key",
                    "search_model": "grok-test",
                    "allowed_x_handles": ["xai"],
                    "from_date": "2026-01-01",
                    "to_date": "2026-02-01",
                    "enable_image_understanding": True,
                    "enable_video_understanding": False,
                },
            }
        ]
    )
    result = await router.asearch(search_tool_name="social-search", query=["first", "second"])

    assert [r.url for r in result.results] == list(URLS)
    assert [r.snippet for r in result.results] == [ANSWER, ANSWER]
    assert result.results[1].title == URLS[1]
    assert result.hidden_params["response_cost"] == 0.0
    assert json.loads(route.calls.last.request.content) == {
        "model": "grok-test",
        "input": "first\nsecond",
        "tools": [
            {
                "type": "x_search",
                "allowed_x_handles": ["xai"],
                "from_date": "2026-01-01",
                "to_date": "2026-02-01",
                "enable_image_understanding": True,
                "enable_video_understanding": False,
            }
        ],
        "tool_choice": "required",
        "store": False,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"output": [], "status": "failed", "error": {"message": "search failed"}},
        {"output": [], "status": "incomplete"},
        {"output": [], "status": "completed", "usage": {}},
        {"output": [], "status": "completed", "usage": {"cost_in_usd_ticks": -1}},
        {"output": [], "status": "completed", "usage": {"cost_in_usd_ticks": True}},
        {"output": [], "status": "completed", "usage": {"cost_in_usd_ticks": "123"}},
        {"unexpected": "body"},
    ],
)
def test_search_does_not_report_failed_or_unbillable_upstream_responses_as_success(
    respx_mock: respx.MockRouter, payload: dict[str, object]
) -> None:
    respx_mock.post(RESPONSES_URL).respond(json=payload)
    with pytest.raises(openai.APIStatusError) as error:
        litellm.search(query="test", search_provider="xai", api_key="test-xai-key")
    assert error.value.status_code == 502


@pytest.mark.parametrize("status", [401, 429, 500])
def test_upstream_http_error_keeps_its_status(respx_mock: respx.MockRouter, status: int) -> None:
    respx_mock.post(RESPONSES_URL).respond(status, json={"error": "upstream error"})
    with pytest.raises(openai.APIStatusError) as error:
        litellm.search(query="test", search_provider="xai", api_key="test-xai-key")
    assert error.value.status_code == status


def test_search_with_no_citations_still_records_the_bill(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(RESPONSES_URL).respond(json={"output": [], "usage": {"cost_in_usd_ticks": 99}})
    result = litellm.search(query="test", search_provider="xai", api_key="test-xai-key")
    assert isinstance(result, SearchResponse)
    assert result.results == []
    assert result.hidden_params["response_cost"] == pytest.approx(99 / 10_000_000_000)


def test_environment_credentials_cannot_be_sent_to_a_caller_controlled_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "server-test-key")
    monkeypatch.delenv("XAI_API_BASE", raising=False)
    with pytest.raises(ValueError, match="Refusing to send the server-configured"):
        XAISearchConfig().validate_environment({}, api_base="https://untrusted.example/v1")


def test_operator_base_and_explicit_model_are_used_without_mutating_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAI_API_BASE", "https://gateway.example/v1/")
    monkeypatch.setenv("XAI_API_KEY", "server-test-key")
    config = XAISearchConfig()
    headers = {"X-Custom": "preserved"}
    resolved = config.validate_environment(headers)
    assert resolved == {
        **headers,
        "Authorization": "Bearer server-test-key",
        "Content-Type": "application/json",
    }
    assert headers == {"X-Custom": "preserved"}
    assert config.get_complete_url(None, {}) == "https://gateway.example/v1/responses"
    assert (
        config.get_complete_url("https://gateway.example/v1/responses/", {}) == "https://gateway.example/v1/responses"
    )
    assert config.transform_search_request("test", {"search_model": "explicit-model"})["model"] == "explicit-model"


@pytest.mark.asyncio
async def test_websearch_interception_returns_xai_citations_and_cost(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
    from litellm.proxy import proxy_server

    respx_mock.post(RESPONSES_URL).respond(json=_response())
    router = litellm.Router(
        search_tools=[
            {
                "search_tool_name": "social-search",
                "litellm_params": {"search_provider": "xai", "api_key": "test-xai-key"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    logger = WebSearchInterceptionLogger(search_tool_name="social-search")
    text, result = await logger._execute_search("Grok update")

    assert result is not None
    assert [r.url for r in result.results] == list(URLS)
    assert result.hidden_params["response_cost"] == pytest.approx(123456789 / 10_000_000_000)
    assert all(url in text for url in URLS)
    assert ANSWER in text
