"""
Tests for Parallel AI Search API integration (v1 endpoint).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


import litellm

MOCK_V1_RESPONSE = {
    "search_id": "search_abc123",
    "session_id": "session_xyz",
    "results": [
        {
            "url": "https://example.com/1",
            "title": "Test Result 1",
            "publish_date": "2026-01-15",
            "excerpts": ["First excerpt.", "Second excerpt."],
        },
        {
            "url": "https://example.com/2",
            "title": None,
            "publish_date": None,
            "excerpts": ["Only excerpt."],
        },
    ],
    "usage": [{"name": "search_advanced", "count": 1}],
}


def _mock_response(payload=None):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = payload if payload is not None else MOCK_V1_RESPONSE
    return mock_response


@pytest.fixture
def httpx_transport(monkeypatch):
    monkeypatch.setattr(  # test-quality-ok: respx needs HTTPX enabled to fake the provider HTTP boundary.
        litellm,
        "disable_aiohttp_transport",
        True,
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture
def bundled_cost_map(monkeypatch):
    """Price lookups against the bundled cost map.

    litellm caches model-info lookups, so swapping ``model_cost`` only takes
    effect once those caches are invalidated -- on the way in and back out.
    """
    from litellm.utils import _invalidate_model_cost_lowercase_map

    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    _invalidate_model_cost_lowercase_map()
    yield
    monkeypatch.undo()
    _invalidate_model_cost_lowercase_map()


class TestParallelAISearch:
    @pytest.fixture(autouse=True)
    def _set_api_key(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "test-api-key")
        monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)

    @pytest.mark.asyncio
    async def test_v1_endpoint_and_headers(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="latest developments in AI",
                search_provider="parallel_ai",
            )

            call_args = mock_post.call_args
            assert call_args.kwargs["url"] == "https://api.parallel.ai/v1/search"

            headers = call_args.kwargs.get("headers", {})
            assert headers["x-api-key"] == "test-api-key"
            assert headers["Content-Type"] == "application/json"
            assert "parallel-beta" not in headers

    @pytest.mark.asyncio
    async def test_string_query_maps_to_search_queries_and_objective(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="latest developments in AI",
                search_provider="parallel_ai",
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["search_queries"] == ["latest developments in AI"]
            assert json_data["objective"] == "latest developments in AI"

    @pytest.mark.asyncio
    async def test_list_query_maps_to_search_queries(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query=["AI developments", "machine learning trends"],
                search_provider="parallel_ai",
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["search_queries"] == [
                "AI developments",
                "machine learning trends",
            ]
            assert "objective" not in json_data

    @pytest.mark.asyncio
    async def test_mode_param_passthrough(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                mode="turbo",
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["mode"] == "turbo"

    @pytest.mark.asyncio
    async def test_default_mode_is_basic(self):
        """v1 defaults to 'advanced' server-side; litellm must send 'basic' to keep v1beta's default tier and cost tracking accurate."""
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["mode"] == "basic"

    @pytest.mark.parametrize("processor,expected_mode", [("base", "basic"), ("pro", "advanced")])
    @pytest.mark.asyncio
    async def test_legacy_processor_maps_to_mode(self, processor, expected_mode):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                processor=processor,
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["mode"] == expected_mode
            assert "processor" not in json_data

    @pytest.mark.asyncio
    async def test_explicit_mode_wins_over_processor(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                mode="turbo",
                processor="pro",
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["mode"] == "turbo"
            assert "processor" not in json_data

    @pytest.mark.asyncio
    async def test_top_level_v1_params_pass_through(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                session_id="session_123",
                max_chars_total=4000,
                max_tokens_per_page=1024,
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["session_id"] == "session_123"
            assert json_data["max_chars_total"] == 4000
            assert "max_tokens_per_page" not in json_data

    @pytest.mark.asyncio
    async def test_optional_params_nest_under_advanced_settings(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                max_results=5,
                country="US",
                search_domain_filter=["arxiv.org", "nature.com"],
                exclude_domains=["reddit.com"],
                max_chars_per_result=1500,
            )

            json_data = mock_post.call_args.kwargs.get("json")
            advanced_settings = json_data["advanced_settings"]
            assert advanced_settings["max_results"] == 5
            assert advanced_settings["location"] == "US"
            assert advanced_settings["source_policy"]["include_domains"] == [
                "arxiv.org",
                "nature.com",
            ]
            assert advanced_settings["source_policy"]["exclude_domains"] == ["reddit.com"]
            assert advanced_settings["excerpt_settings"]["max_chars_per_result"] == 1500

            assert "max_results" not in json_data
            assert "source_policy" not in json_data
            assert "search_domain_filter" not in json_data
            assert "exclude_domains" not in json_data
            assert "max_chars_per_result" not in json_data
            assert "country" not in json_data

    @pytest.mark.asyncio
    async def test_explicit_advanced_settings_take_precedence(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                max_results=5,
                advanced_settings={"max_results": 7},
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["advanced_settings"]["max_results"] == 7

    @pytest.mark.asyncio
    async def test_response_transformation(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            response = await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
            )

            assert response.object == "search"
            assert len(response.results) == 2

            first = response.results[0]
            assert first.title == "Test Result 1"
            assert first.url == "https://example.com/1"
            assert first.snippet == "First excerpt. ... Second excerpt."
            assert first.date == "2026-01-15"

            second = response.results[1]
            assert second.title == ""
            assert second.snippet == "Only excerpt."
            assert second.date is None

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://proxy.internal.example.com",
            "https://proxy.internal.example.com/",
            "https://proxy.internal.example.com/v1",
            "https://proxy.internal.example.com/v1/",
            "https://proxy.internal.example.com/v1/search",
        ],
    )
    @pytest.mark.asyncio
    async def test_custom_api_base_appends_v1_search(self, api_base, monkeypatch):
        # Operator points at an internal base via the env override (a trusted
        # host), so the server key is still used and the URL is normalized.
        monkeypatch.setenv("PARALLEL_AI_API_BASE", api_base)
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
            )

            call_args = mock_post.call_args
            assert call_args.kwargs["url"] == "https://proxy.internal.example.com/v1/search"

    @pytest.mark.asyncio
    async def test_caller_api_base_without_key_is_refused(self, monkeypatch):
        # A caller-supplied api_base (untrusted host) while relying on the
        # server key must be refused without any outbound request.
        monkeypatch.setenv("PARALLEL_API_KEY", "server-secret")
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            with pytest.raises(Exception, match="Refusing to send"):
                await litellm.asearch(
                    query="AI developments",
                    search_provider="parallel_ai",
                    api_base="https://attacker.example.com",
                )
            mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)

        with pytest.raises(Exception, match="PARALLEL_API_KEY"):
            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
            )

    @pytest.mark.asyncio
    async def test_flat_source_and_fetch_params_nest_under_advanced_settings(self, respx_mock, httpx_transport):
        route = respx_mock.post("https://api.parallel.ai/v1/search").respond(json=MOCK_V1_RESPONSE)

        await litellm.asearch(
            query="AI developments",
            search_provider="parallel_ai",
            objective="find peer-reviewed AI research",
            include_domains=["arxiv.org"],
            after_date="2026-01-01",
            location="gb",
            fetch_policy={"max_age_seconds": 600, "disable_cache_fallback": True},
            client_model="claude-fable-5",
        )

        json_data = json.loads(route.calls[0].request.content)
        assert json_data["objective"] == "find peer-reviewed AI research"
        assert json_data["client_model"] == "claude-fable-5"

        advanced_settings = json_data["advanced_settings"]
        assert advanced_settings["location"] == "gb"
        assert advanced_settings["fetch_policy"] == {
            "max_age_seconds": 600,
            "disable_cache_fallback": True,
        }
        assert advanced_settings["source_policy"]["include_domains"] == ["arxiv.org"]
        assert advanced_settings["source_policy"]["after_date"] == "2026-01-01"

        assert "include_domains" not in json_data
        assert "after_date" not in json_data
        assert "location" not in json_data
        assert "fetch_policy" not in json_data

    @pytest.mark.asyncio
    async def test_response_preserves_raw_parallel_fields(self, respx_mock, httpx_transport):
        respx_mock.post("https://api.parallel.ai/v1/search").respond(json=MOCK_V1_RESPONSE)

        response = await litellm.asearch(
            query="AI developments",
            search_provider="parallel_ai",
        )

        dumped = response.model_dump()
        assert dumped["search_id"] == "search_abc123"
        assert dumped["session_id"] == "session_xyz"
        assert dumped["parallel_usage"] == [{"name": "search_advanced", "count": 1}]

        first = response.results[0].model_dump()
        assert first["excerpts"] == ["First excerpt.", "Second excerpt."]

    @pytest.mark.asyncio
    async def test_response_normalizes_null_result_fields(self, respx_mock, httpx_transport):
        response_payload = {
            **MOCK_V1_RESPONSE,
            "results": [{"url": None, "title": None, "publish_date": None, "excerpts": None}],
        }
        respx_mock.post("https://api.parallel.ai/v1/search").respond(json=response_payload)

        response = await litellm.asearch(
            query="AI developments",
            search_provider="parallel_ai",
        )

        assert len(response.results) == 1
        result = response.results[0]
        assert result.url == ""
        assert result.title == ""
        assert result.snippet == ""
        assert result.date is None
        assert result.model_dump()["excerpts"] == ()

    @pytest.mark.parametrize(
        "mode,usage,max_results,expected_cost",
        [
            ("turbo", [{"name": "sku_search", "count": 1}], None, 0.001),
            ("fast", [{"name": "sku_search", "count": 1}], None, 0.001),
            ("basic", [{"name": "sku_search", "count": 1}], None, 0.005),
            ("advanced", [{"name": "sku_search", "count": 1}], None, 0.005),
            (
                "basic",
                [
                    {"name": "sku_search", "count": 1},
                    {"name": "sku_search_additional_results", "count": 2},
                ],
                20,
                0.007,
            ),
            ("basic", None, 20, 0.015),
        ],
    )
    @pytest.mark.asyncio
    async def test_search_cost_uses_mode_and_provider_usage(
        self, mode, usage, max_results, expected_cost, bundled_cost_map, respx_mock, httpx_transport
    ):
        response_payload = {**MOCK_V1_RESPONSE, "usage": usage}
        respx_mock.post("https://api.parallel.ai/v1/search").respond(json=response_payload)

        response = await litellm.asearch(
            query="AI developments",
            search_provider="parallel_ai",
            mode=mode,
            max_results=max_results,
        )

        assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)

    @pytest.mark.asyncio
    async def test_search_cost_treats_keyword_queries_as_one_request(
        self, bundled_cost_map, respx_mock, httpx_transport
    ):
        response_payload = {
            **MOCK_V1_RESPONSE,
            "usage": [{"name": "sku_search", "count": 1}],
        }
        respx_mock.post("https://api.parallel.ai/v1/search").respond(json=response_payload)

        response = await litellm.asearch(
            query=["AI developments", "machine learning trends"],
            search_provider="parallel_ai",
            mode="basic",
        )

        assert response._hidden_params["response_cost"] == pytest.approx(0.005)

    @pytest.mark.asyncio
    async def test_caller_cannot_supply_provider_usage(self, bundled_cost_map, respx_mock, httpx_transport):
        """`_parallel_ai_usage` prices the request, so a caller must not be able to set it.

        The provider reports no usage here, which is the case where a caller-supplied
        value would otherwise survive into the cost calculation.
        """
        response_payload = {k: v for k, v in MOCK_V1_RESPONSE.items() if k != "usage"}
        route = respx_mock.post("https://api.parallel.ai/v1/search").respond(json=response_payload)

        response = await litellm.asearch(
            query="AI developments",
            search_provider="parallel_ai",
            mode="basic",
            _parallel_ai_usage=[{"name": "sku_search", "count": 0}],
        )

        assert response._hidden_params["response_cost"] == pytest.approx(0.005)
        assert "_parallel_ai_usage" not in json.loads(route.calls[0].request.content)
