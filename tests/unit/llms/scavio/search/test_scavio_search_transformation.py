import datetime
import json
from collections.abc import Mapping, Sequence
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.scavio.search.transformation import ScavioSearchConfig
from litellm.search.cost_calculator import search_provider_cost_per_query
from litellm.types.utils import SearchProviders
from litellm.utils import ProviderConfigManager

SCAVIO_URL: Final = "https://api.scavio.dev/api/v2/google"

LIVE_RESPONSE: Final[Mapping[str, object]] = {
    "search_parameters": {"q": "open source llm gateway", "hl": "en", "gl": "us", "device": "desktop", "start": 0},
    "search_information": {"query_displayed": "open source llm gateway", "total_results": 2},
    "organic_results": [
        {
            "position": 1,
            "title": "LiteLLM - GitHub",
            "link": "https://github.com/BerriAI/litellm",
            "displayed_link": "https://github.com > BerriAI > litellm",
            "snippet": "Python SDK, Proxy Server (AI Gateway) to call 100+ LLM APIs in OpenAI format.",
            "snippet_highlighted_words": ["AI Gateway"],
            "source": "github.com",
        },
        {
            "position": 2,
            "title": "Is there an open-source LLM gateway with runtime routing?",
            "link": "https://www.reddit.com/r/SelfHostedAI/comments/example",
            "source": "reddit.com",
        },
    ],
    "related_searches": [{"query": "llm gateway comparison"}],
    "cached": False,
    "response_time": 3.1,
    "credits_used": 1,
    "credits_remaining": 99,
}


def _config() -> ScavioSearchConfig:
    return ScavioSearchConfig()


def _logging_obj() -> Logging:
    return Logging(
        model="scavio/search",
        messages=[],
        stream=False,
        call_type="search",
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="test-call",
        function_id="test-function",
    )


def _resp(payload: Mapping[str, object] | str, status_code: int = 200) -> httpx.Response:
    body: Final = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(status_code, content=body.encode(), request=httpx.Request("POST", SCAVIO_URL))


def _result(title: str = "Test Title", link: str = "https://example.com") -> Mapping[str, object]:
    return {
        "position": 1,
        "title": title,
        "link": link,
        "displayed_link": f"{link} > docs",
        "snippet": "Test snippet",
        "source": "example.com",
    }


class _ScavioStub:
    def __init__(self, status_code: int, body: Mapping[str, object]) -> None:
        self.status_code: Final = status_code
        self.body: Final = body
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(self.status_code, content=json.dumps(self.body).encode())


def test_ui_friendly_name() -> None:
    assert _config().ui_friendly_name() == "Scavio"


def test_search_provider_resolves_to_scavio_config() -> None:
    config: Final = ProviderConfigManager.get_provider_search_config(provider=SearchProviders("scavio"))
    assert isinstance(config, ScavioSearchConfig)


def test_validate_environment_with_explicit_key() -> None:
    headers: Final = _config().validate_environment({}, api_key="explicit-key")
    assert headers["Authorization"] == "Bearer explicit-key"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_reads_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCAVIO_API_KEY", "env-key")
    assert _config().validate_environment({})["Authorization"] == "Bearer env-key"


def test_validate_environment_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCAVIO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SCAVIO_API_KEY"):
        _config().validate_environment({})


def test_validate_environment_does_not_mutate_and_is_idempotent() -> None:
    config: Final = _config()
    caller_headers: Final = {"X-Custom": "keep-me"}

    once: Final = config.validate_environment(caller_headers, api_key="k")
    twice: Final = config.validate_environment(once, api_key="k")

    assert caller_headers == {"X-Custom": "keep-me"}
    assert once == twice
    assert once["X-Custom"] == "keep-me"


def test_get_complete_url_default_base() -> None:
    assert _config().get_complete_url(None, {}) == SCAVIO_URL


@pytest.mark.parametrize(
    "api_base",
    [
        "https://proxy.local",
        "https://proxy.local/",
        "https://proxy.local/api/v2/google",
        "https://proxy.local/api/v2/google/",
    ],
)
def test_get_complete_url_appends_path_exactly_once(api_base: str) -> None:
    assert _config().get_complete_url(api_base, {}) == "https://proxy.local/api/v2/google"


def test_transform_search_request_joins_list_query() -> None:
    assert _config().transform_search_request(["foo", "bar"], {})["query"] == "foo bar"


def test_transform_search_request_maps_country_to_lowercase_gl() -> None:
    assert _config().transform_search_request("q", {"country": "GB"})["gl"] == "gb"


def test_transform_search_request_explicit_gl_wins_over_country() -> None:
    assert _config().transform_search_request("q", {"country": "GB", "gl": "de"})["gl"] == "de"


@pytest.mark.parametrize("unified_param", ["max_results", "max_tokens_per_page", "country", "search_domain_filter"])
def test_transform_search_request_does_not_leak_unified_params(unified_param: str) -> None:
    data: Final = _config().transform_search_request(
        "q",
        {"max_results": 3, "max_tokens_per_page": 1024, "country": "us", "search_domain_filter": ["a.com"]},
    )
    assert unified_param not in data


def test_transform_search_request_skips_ai_overview_resolution_by_default() -> None:
    assert _config().transform_search_request("q", {})["resolve_ai_overview"] is False


def test_transform_search_request_caller_can_enable_ai_overview_resolution() -> None:
    assert _config().transform_search_request("q", {"resolve_ai_overview": True})["resolve_ai_overview"] is True


@pytest.mark.parametrize(
    "domains, expected",
    [
        (["arxiv.org"], "q site:arxiv.org"),
        (["arxiv.org", "nature.com"], "q (site:arxiv.org OR site:nature.com)"),
        (["arxiv.org", "nature.com", "science.org"], "q (site:arxiv.org OR site:nature.com OR site:science.org)"),
        (["-spam.com"], "q -site:spam.com"),
        (["arxiv.org", "-spam.com", "-junk.net"], "q site:arxiv.org -site:spam.com -site:junk.net"),
        (["-spam.com", "arxiv.org", "nature.com"], "q (site:arxiv.org OR site:nature.com) -site:spam.com"),
        ([], "q"),
        (["", "-"], "q"),
        ("arxiv.org", "q"),
    ],
)
def test_transform_search_request_folds_domain_filter_into_query(domains: Sequence[str], expected: str) -> None:
    assert _config().transform_search_request("q", {"search_domain_filter": domains})["query"] == expected


def test_transform_search_request_without_domain_filter_leaves_query_unchanged() -> None:
    assert _config().transform_search_request("latest AI developments", {})["query"] == "latest AI developments"


def test_transform_search_request_never_wraps_user_query_in_parentheses() -> None:
    data: Final = _config().transform_search_request(
        ["latest", "AI developments"], {"search_domain_filter": ["github.com", "gitlab.com"]}
    )
    assert data["query"] == "latest AI developments (site:github.com OR site:gitlab.com)"


def test_transform_search_request_forwards_scavio_params() -> None:
    data: Final = _config().transform_search_request(
        "q", {"hl": "fr", "location": "Paris,France", "device": "mobile", "start": 10, "time_period": "last_week"}
    )
    assert (data["hl"], data["location"], data["device"], data["start"], data["time_period"]) == (
        "fr",
        "Paris,France",
        "mobile",
        10,
        "last_week",
    )


def test_transform_search_response_maps_organic_results_in_order() -> None:
    resp: Final = _config().transform_search_response(
        _resp({"organic_results": [_result(title=t, link=f"https://{t}.com") for t in ("first", "second", "third")]}),
        logging_obj=_logging_obj(),
    )
    assert [(r.title, r.url) for r in resp.results] == [
        ("first", "https://first.com"),
        ("second", "https://second.com"),
        ("third", "https://third.com"),
    ]
    assert resp.results[0].snippet == "Test snippet"
    assert resp.results[0].date is None


def test_transform_search_response_result_without_snippet_does_not_fail_the_call() -> None:
    no_snippet: Final = {"position": 4, "title": "Forum thread", "link": "https://forum.example"}
    resp: Final = _config().transform_search_response(
        _resp({"organic_results": [no_snippet, _result()]}), logging_obj=_logging_obj()
    )
    assert [r.snippet for r in resp.results] == ["", "Test snippet"]


@pytest.mark.parametrize("max_results, expected_count", [(2, 2), (10, 5), (0, 5), (True, 5), (None, 5)])
def test_transform_search_response_caps_to_max_results(max_results: int | bool | None, expected_count: int) -> None:
    resp: Final = _config().transform_search_response(
        _resp({"organic_results": [_result(title=str(i)) for i in range(5)]}),
        logging_obj=_logging_obj(),
        optional_params={"max_results": max_results},
    )
    assert [r.title for r in resp.results] == [str(i) for i in range(expected_count)]


def test_transform_search_response_zero_hits() -> None:
    payload: Final[Mapping[str, object]] = {
        "search_information": {"total_results": 0},
        "organic_results": [],
        "credits_used": 1,
    }
    assert _config().transform_search_response(_resp(payload), logging_obj=_logging_obj()).results == []


@pytest.mark.parametrize(
    "body",
    [
        "<html>502 Bad Gateway</html>",
        '{"organic_results": ["garbage"]}',
        '{"organic_results": {"unexpected": "shape"}}',
        '{"organic_results": null}',
        "{}",
    ],
)
def test_transform_search_response_malformed_body_raises_instead_of_reporting_empty(body: str) -> None:
    with pytest.raises(Exception, match="Scavio Search"):
        _config().transform_search_response(_resp(body, status_code=502), logging_obj=_logging_obj())


def test_get_error_class_unwraps_scavio_error_envelope() -> None:
    error: Final = _config().get_error_class(error_message='{"error":"Invalid API key"}', status_code=401, headers={})
    assert getattr(error, "status_code", None) == 401
    assert str(error) == "Scavio Search: Invalid API key. See https://scavio.dev/docs/search-api for details."


@pytest.mark.parametrize("body", ["<html>502 Bad Gateway</html>", '{"error": null}'])
def test_get_error_class_falls_back_to_the_raw_body(body: str) -> None:
    assert f"Scavio Search: {body}." in str(_config().get_error_class(body, status_code=502, headers={}))


def test_search_handler_sends_mapped_request_and_parses_response() -> None:
    stub: Final = _ScavioStub(200, LIVE_RESPONSE)
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(stub)))

    response: Final = BaseLLMHTTPHandler().search(
        query="open source llm gateway",
        optional_params={"max_results": 1, "country": "US", "search_domain_filter": ["github.com"]},
        timeout=10.0,
        logging_obj=_logging_obj(),
        api_key="test-api-key",
        api_base=None,
        custom_llm_provider="scavio",
        client=client,
        provider_config=_config(),
    )

    assert isinstance(response, SearchResponse)
    assert [str(r.url) for r in stub.requests] == [SCAVIO_URL]
    assert stub.requests[0].headers["Authorization"] == "Bearer test-api-key"
    assert json.loads(stub.requests[0].content) == {
        "resolve_ai_overview": False,
        "gl": "us",
        "query": "open source llm gateway site:github.com",
    }
    assert [(r.title, r.url, r.snippet) for r in response.results] == [
        (
            "LiteLLM - GitHub",
            "https://github.com/BerriAI/litellm",
            "Python SDK, Proxy Server (AI Gateway) to call 100+ LLM APIs in OpenAI format.",
        )
    ]


def test_search_handler_surfaces_scavio_error_message() -> None:
    stub: Final = _ScavioStub(401, {"error": "Invalid API key"})
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(stub)))

    with pytest.raises(Exception, match="Scavio Search: Invalid API key"):
        BaseLLMHTTPHandler().search(
            query="q",
            optional_params={},
            timeout=10.0,
            logging_obj=_logging_obj(),
            api_key="wrong-key",
            api_base=None,
            custom_llm_provider="scavio",
            client=client,
            provider_config=_config(),
        )
    assert len(stub.requests) == 1


@pytest.mark.asyncio
async def test_async_search_handler_sends_mapped_request_and_parses_response() -> None:
    stub: Final = _ScavioStub(200, LIVE_RESPONSE)
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(stub))

    response: Final = await BaseLLMHTTPHandler().async_search(
        query=["open source", "llm gateway"],
        optional_params={"max_results": 1},
        timeout=10.0,
        logging_obj=_logging_obj(),
        api_key="test-api-key",
        api_base=None,
        custom_llm_provider="scavio",
        client=client,
        provider_config=_config(),
    )

    assert json.loads(stub.requests[0].content)["query"] == "open source llm gateway"
    assert [r.title for r in response.results] == ["LiteLLM - GitHub"]


def test_search_cost_comes_from_the_cost_map() -> None:
    expected: Final[float] = litellm.model_cost["scavio/search"]["input_cost_per_query"]
    assert expected > 0
    assert search_provider_cost_per_query(model="scavio/search", custom_llm_provider="scavio") == (expected, 0.0)
