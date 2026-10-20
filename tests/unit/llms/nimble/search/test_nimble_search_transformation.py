import json, litellm
from unittest.mock import AsyncMock, Mock, patch

import pytest

from litellm.llms.nimble.search.transformation import NimbleSearchConfig
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


def _config() -> NimbleSearchConfig:
    return NimbleSearchConfig()


def _resp(payload, status_code: int = 200):
    r = Mock()
    r.status_code = status_code
    r.headers = {}
    r.content = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
    return r


def _result(**overrides):
    base = {
        "title": "Test Title",
        "description": "Test description",
        "url": "https://example.com",
        "content": "Test content",
        "metadata": {"position": 1, "entity_type": "organic"},
        "additional_data": None,
    }
    return {**base, **overrides}


def test_ui_friendly_name():
    assert _config().ui_friendly_name() == "Nimble"


def test_validate_environment_with_explicit_key():
    headers = _config().validate_environment({}, api_key="explicit-key")
    assert headers["Authorization"] == "Bearer explicit-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Client-Source"] == "litellm"


def test_validate_environment_reads_env_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NIMBLE_API_KEY", "env-key")
    assert _config().validate_environment({})["Authorization"] == "Bearer env-key"


def test_validate_environment_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NIMBLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NIMBLE_API_KEY"):
        _config().validate_environment({})


def test_validate_environment_does_not_mutate_and_is_idempotent():
    """The http handler re-runs validate_environment after search/main.py already did."""
    config = _config()
    caller_headers = {"X-Custom": "keep-me"}

    once = config.validate_environment(caller_headers, api_key="k")
    twice = config.validate_environment(once, api_key="k")

    assert caller_headers == {"X-Custom": "keep-me"}
    assert once == twice
    assert once["X-Custom"] == "keep-me"


def test_get_complete_url_default_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NIMBLE_API_BASE", raising=False)
    assert _config().get_complete_url(None, {}) == "https://sdk.nimbleway.com/v2/search"


def test_get_complete_url_reads_env_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NIMBLE_API_BASE", "https://env-base.local/v2")
    assert _config().get_complete_url(None, {}) == "https://env-base.local/v2/search"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://self-hosted.local/v2",
        "https://self-hosted.local/v2/",
        "https://self-hosted.local/v2/search",
        "https://self-hosted.local/v2/search/",
    ],
)
def test_get_complete_url_appends_search_exactly_once(api_base: str):
    assert _config().get_complete_url(api_base, {}) == "https://self-hosted.local/v2/search"


def test_transform_search_request_joins_list_query():
    assert _config().transform_search_request(["foo", "bar"], {})["query"] == "foo bar"


def test_transform_search_request_max_results_is_not_clamped():
    """Nimble validates 1-100 itself; a clearer error beats silently rewriting the request."""
    assert _config().transform_search_request("q", {"max_results": 500})["max_results"] == 500


def test_transform_search_request_uppercases_country():
    assert _config().transform_search_request("q", {"country": "us"})["country"] == "US"


def test_transform_search_request_drops_max_tokens_per_page():
    assert "max_tokens_per_page" not in _config().transform_search_request("q", {"max_tokens_per_page": 1024})


def test_transform_search_request_splits_domain_filter():
    data = _config().transform_search_request("q", {"search_domain_filter": ["arxiv.org", "-spam.com", "nature.com"]})
    assert data["include_domains"] == ("arxiv.org", "nature.com")
    assert data["exclude_domains"] == ("spam.com",)


def test_transform_search_request_omits_empty_domain_lists():
    data = _config().transform_search_request("q", {"search_domain_filter": ["arxiv.org"]})
    assert data["include_domains"] == ("arxiv.org",)
    assert "exclude_domains" not in data


def test_transform_search_request_ignores_non_list_domain_filter():
    assert "include_domains" not in _config().transform_search_request("q", {"search_domain_filter": "arxiv.org"})


@pytest.mark.parametrize("native_key", ["include_domains", "exclude_domains"])
def test_transform_search_request_native_domains_win(native_key: str):
    """An explicit provider-native value must not be silently clobbered by the unified param."""
    data = _config().transform_search_request(
        "q",
        {"search_domain_filter": ["derived.com", "-derived-ex.com"], native_key: ["native.com"]},
    )
    assert data[native_key] == ["native.com"]


def test_transform_search_response_prefers_content():
    resp = _config().transform_search_response(_resp({"results": [_result()]}), logging_obj=Mock())
    assert resp.results[0].snippet == "Test content"


def test_transform_search_response_falls_back_to_description():
    resp = _config().transform_search_response(_resp({"results": [_result(content="")]}), logging_obj=Mock())
    assert resp.results[0].snippet == "Test description"


def test_transform_search_response_reads_publish_date():
    resp = _config().transform_search_response(
        _resp({"results": [_result(additional_data={"publish_date": "2026-08-01"})]}),
        logging_obj=Mock(),
    )
    assert resp.results[0].date == "2026-08-01"


@pytest.mark.parametrize("additional_data", [{}, "not-a-dict"])
def test_transform_search_response_date_is_none_without_usable_publish_date(additional_data):
    resp = _config().transform_search_response(
        _resp({"results": [_result(additional_data=additional_data)]}), logging_obj=Mock()
    )
    assert resp.results[0].date is None


def test_transform_search_response_keeps_additional_data():
    """News results often carry only a relative `publish_date_raw`, which is not a date;
    it must still reach the caller rather than being dropped on the floor."""
    resp = _config().transform_search_response(
        _resp({"results": [_result(additional_data={"publish_date_raw": "1 day ago"})]}),
        logging_obj=Mock(),
    )
    assert resp.results[0].date is None
    assert resp.results[0].additional_data == {"publish_date_raw": "1 day ago"}


def test_transform_search_response_omits_additional_data_when_absent():
    resp = _config().transform_search_response(_resp({"results": [_result()]}), logging_obj=Mock())
    assert not hasattr(resp.results[0], "additional_data")


def test_transform_search_response_preserves_order():
    resp = _config().transform_search_response(
        _resp({"results": [_result(title=t) for t in ("first", "second", "third")]}),
        logging_obj=Mock(),
    )
    assert [r.title for r in resp.results] == ["first", "second", "third"]


def test_transform_search_response_degraded_result_does_not_fail_the_call():
    resp = _config().transform_search_response(
        _resp({"results": [{"url": "https://example.com"}, _result()]}), logging_obj=Mock()
    )
    assert len(resp.results) == 2
    assert resp.results[0].title == ""
    assert resp.results[0].snippet == ""
    assert resp.results[1].title == "Test Title"


def test_transform_search_response_zero_hits():
    """A search with no hits really does come back as `"results": []`."""
    payload = {"request_id": "abc", "total_results": 0, "results": []}
    assert _config().transform_search_response(_resp(payload), logging_obj=Mock()).results == []


@pytest.mark.parametrize(
    "body",
    [
        "<html>502 Bad Gateway</html>",  # non-JSON body
        '{"results": ["garbage"]}',  # right key, wrong element shape
        '{"results": {"unexpected": "shape"}}',
        '{"results": null}',  # must not degrade to a successful empty search
        "{}",  # ditto for an absent key
    ],
)
def test_transform_search_response_malformed_body_raises_instead_of_reporting_empty(body: str):
    """A body LiteLLM cannot parse must not be reported as a successful zero-result search."""
    with pytest.raises(Exception, match="Nimble Search"):
        _config().transform_search_response(_resp(body, status_code=502), logging_obj=Mock())


def test_get_error_class_attributes_the_provider():
    error = _config().get_error_class(error_message="quota exceeded", status_code=429, headers={})
    assert error.status_code == 429
    assert "Nimble Search: quota exceeded" in str(error)
    assert "docs.nimbleway.com" in str(error)


def test_get_error_class_unwraps_nimble_detail_envelope():
    """Verbatim body from a live 422; the raw JSON envelope should not reach the user."""
    error = _config().get_error_class(
        error_message='{"detail":"search_depth=\'fast\' is only supported with focus=\'general\'."}',
        status_code=422,
        headers={},
    )
    assert (
        str(error) == "Nimble Search: search_depth='fast' is only supported with focus='general'. "
        "See https://docs.nimbleway.com/api-reference/search/search for details."
    )


def test_get_error_class_unwraps_nimble_message_envelope():
    """Verbatim body from a live collection failure, which uses a different envelope."""
    error = _config().get_error_class(
        error_message='{"success":"false","task_id":"4f74af04","message":"can\'t download the query response"}',
        status_code=500,
        headers={},
    )
    assert (
        str(error) == "Nimble Search: can't download the query response. "
        "See https://docs.nimbleway.com/api-reference/search/search for details."
    )


@pytest.mark.parametrize("body", ["<html>502 Bad Gateway</html>", '{"detail": null}'])
def test_get_error_class_falls_back_to_the_raw_body(body: str):
    assert f"Nimble Search: {body}." in str(_config().get_error_class(body, status_code=500, headers={}))


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

MOCK_NIMBLE_RESPONSE = {
    "request_id": "0f8b3a1c-1d2e-4f5a-9b0c-6d7e8f9a0b1c",
    "total_results": 2,
    "results": [
        {
            "title": "Nimble Web API",
            "description": "Short SERP description",
            "url": "https://nimbleway.com/",
            "content": "Full markdown content for the first result",
            "metadata": {"position": 1, "entity_type": "organic", "country": "US", "locale": "en"},
            "additional_data": {"publish_date": "2026-07-15"},
        },
        {
            "title": "Nimble Docs",
            "description": "Only a description here",
            "url": "https://docs.nimbleway.com/",
            "content": "",
            "metadata": {"position": 2, "entity_type": "organic"},
            "additional_data": None,
        },
    ],
    "serp_data": None,
}

def _mock_response():
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.content = json.dumps(MOCK_NIMBLE_RESPONSE).encode()
    return response

@pytest.mark.usefixtures("_vcr_outcome_gate")
class TestNimbleSearchTransformation:
    """
    Full-stack tests through `litellm.search` / `litellm.asearch` with the HTTP layer mocked.
    Transformation details are unit-tested in tests/unit/llms/nimble/search/.
    """

    @pytest.fixture(autouse=True)
    def _server_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "test-api-key")
        monkeypatch.delenv("NIMBLE_API_BASE", raising=False)

    def test_nimble_search_request_and_response(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ) as mock_post:
            response = litellm.search(
                query="nimble web scraping",
                search_provider="nimble",
                max_results=2,
                country="us",
                search_domain_filter=["nimbleway.com", "-spam.example"],
            )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["url"] == "https://sdk.nimbleway.com/v2/search"
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"
        assert call_kwargs["headers"]["X-Client-Source"] == "litellm"

        request_body = call_kwargs["json"]
        assert request_body["query"] == "nimble web scraping"
        assert request_body["max_results"] == 2
        assert request_body["country"] == "US"
        assert request_body["include_domains"] == ("nimbleway.com",)
        assert request_body["exclude_domains"] == ("spam.example",)

        assert response.object == "search"
        assert len(response.results) == 2
        assert response.results[0].title == "Nimble Web API"
        assert response.results[0].url == "https://nimbleway.com/"
        assert response.results[0].snippet == "Full markdown content for the first result"
        assert response.results[0].date == "2026-07-15"
        # Second result has no `content`, so the SERP description is the snippet.
        assert response.results[1].snippet == "Only a description here"
        assert response.results[1].date is None

    def test_provider_specific_params_survive_to_the_wire(self):
        """Nimble-native params must not be eaten by `filter_out_litellm_params`."""
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ) as mock_post:
            litellm.search(
                query="test query",
                search_provider="nimble",
                focus="news",
                search_depth="deep",
                time_range="week",
                locale="fr",
                output_format="plain_text",
                max_subagents=5,
            )

        request_body = mock_post.call_args.kwargs["json"]
        assert request_body["focus"] == "news"
        assert request_body["search_depth"] == "deep"
        assert request_body["time_range"] == "week"
        assert request_body["locale"] == "fr"
        assert request_body["output_format"] == "plain_text"
        assert request_body["max_subagents"] == 5

    @pytest.mark.asyncio
    async def test_nimble_asearch(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=AsyncMock(return_value=_mock_response()),
        ) as mock_post:
            response = await litellm.asearch(
                query="latest ai developments",
                search_provider="nimble",
                focus="news",
            )

        assert mock_post.call_args.kwargs["json"]["focus"] == "news"
        assert len(response.results) == 2

    def test_nimble_search_tracks_cost(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ):
            response = litellm.search(query="pricing check", search_provider="nimble")

        assert response._hidden_params["response_cost"] == pytest.approx(0.005)
