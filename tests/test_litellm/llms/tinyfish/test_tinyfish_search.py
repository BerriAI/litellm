"""
Tests for TinyFish Search API integration.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.tinyfish.search.transformation import (
    TinyfishSearchConfig,
    _append_domain_filters,
    _default_missing_result_fields,
)

MOCK_TINYFISH_RESPONSE = {
    "query": "web automation tools",
    "results": [
        {
            "position": 1,
            "site_name": "tinyfish.ai",
            "title": "TinyFish - AI Web Automation",
            "snippet": "Automate any website with natural language.",
            "url": "https://tinyfish.ai",
        },
        {
            "position": 2,
            "site_name": "github.com",
            "title": "Top Web Automation Tools",
            "snippet": "A curated list of browser automation frameworks.",
            "url": "https://github.com/example/web-automation",
        },
    ],
    "total_results": 2,
    "page": 0,
}


def _make_mock_response(
    json_data: dict | None = None,
    status_code: int = 200,
    request_url: str | None = None,
    text: str | None = None,
    headers: dict | None = None,
) -> MagicMock:
    import json as _json

    mock = MagicMock()
    mock.status_code = status_code
    mock.headers = headers or {}
    if json_data is not None:
        mock.json.return_value = json_data
        mock.text = text if text is not None else _json.dumps(json_data)
    else:
        # Force .json() to raise as httpx.Response does for non-JSON bodies.
        mock.json.side_effect = _json.JSONDecodeError("Expecting value", text or "", 0)
        mock.text = text or ""
    if request_url:
        mock.request = MagicMock()
        mock.request.url = httpx.URL(request_url)
    else:
        mock.request = None
    return mock


class TestTinyfishSearchConfig:
    def test_ui_friendly_name(self):
        assert TinyfishSearchConfig.ui_friendly_name() == "TinyFish"

    def test_get_http_method(self):
        assert TinyfishSearchConfig().get_http_method() == "GET"

    def test_validate_environment_with_explicit_key(self):
        config = TinyfishSearchConfig()
        headers = config.validate_environment(headers={}, api_key="sk-tinyfish-test")
        assert headers["X-API-Key"] == "sk-tinyfish-test"
        assert headers["Accept"] == "application/json"

    def test_validate_environment_from_env(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_API_KEY", "sk-from-env")
        config = TinyfishSearchConfig()
        headers = config.validate_environment(headers={})
        assert headers["X-API-Key"] == "sk-from-env"

    def test_validate_environment_missing_key(self, monkeypatch):
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        config = TinyfishSearchConfig()
        with pytest.raises(ValueError, match="TINYFISH_API_KEY"):
            config.validate_environment(headers={})

    def test_validate_environment_uses_api_base_kwarg(self):
        config = TinyfishSearchConfig()
        headers = config.validate_environment(
            headers={},
            api_key="sk-test",
            api_base="https://custom.tinyfish.ai",
        )
        assert headers["X-API-Key"] == "sk-test"


class TestTransformSearchRequest:
    def test_basic_query(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="hello world", optional_params={}
        )
        assert result == {"_tinyfish_params": {"query": "hello world"}}

    def test_list_query_joined(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query=["hello", "world"], optional_params={}
        )
        assert result["_tinyfish_params"]["query"] == "hello world"

    def test_country_maps_to_location(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"country": "US"}
        )
        assert result["_tinyfish_params"]["location"] == "US"

    def test_max_results_not_sent_on_wire(self):
        # TinyFish doesn't honor max_results server-side; we apply it client-side
        # in transform_search_response. The querystring should be free of it.
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"max_results": 5}
        )
        assert "max_results" not in result["_tinyfish_params"]

    def test_max_results_clamped_upper_stored_on_self(self):
        config = TinyfishSearchConfig()
        config.transform_search_request(
            query="test", optional_params={"max_results": 100}
        )
        assert config._caller_max_results == 10  # TinyFish's natural cap

    def test_max_results_clamped_lower_stored_on_self(self):
        config = TinyfishSearchConfig()
        config.transform_search_request(
            query="test", optional_params={"max_results": 0}
        )
        assert config._caller_max_results == 1

    def test_max_results_normal_stored_on_self(self):
        config = TinyfishSearchConfig()
        config.transform_search_request(
            query="test", optional_params={"max_results": 5}
        )
        assert config._caller_max_results == 5

    def test_max_results_non_numeric_string_warns_and_skips(self, caplog):
        # `int("abc")` would raise ValueError; guard makes the failure visible
        # via warning and treats the value as if max_results wasn't set.
        config = TinyfishSearchConfig()
        with caplog.at_level("WARNING"):
            result = config.transform_search_request(
                query="test", optional_params={"max_results": "abc"}
            )
        assert config._caller_max_results is None
        assert "max_results" not in result["_tinyfish_params"]
        messages = [r.getMessage() for r in caplog.records]
        assert any("max_results" in m and "abc" in m for m in messages)

    def test_max_results_infinity_float_warns_and_skips(self, caplog):
        # `int(float('inf'))` raises OverflowError, not ValueError/TypeError.
        # Guard must catch it so a caller passing math.inf gets the same
        # warn-and-ignore behavior as other malformed values.
        config = TinyfishSearchConfig()
        with caplog.at_level("WARNING"):
            result = config.transform_search_request(
                query="test", optional_params={"max_results": float("inf")}
            )
        assert config._caller_max_results is None
        assert "max_results" not in result["_tinyfish_params"]
        messages = [r.getMessage() for r in caplog.records]
        assert any("max_results" in m for m in messages)

    def test_domain_filter_appends_site_operators(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="python tutorials",
            optional_params={"search_domain_filter": ["arxiv.org", "github.com"]},
        )
        query_value = result["_tinyfish_params"]["query"]
        assert "site:arxiv.org" in query_value
        assert "site:github.com" in query_value
        assert "(python tutorials) (site:arxiv.org OR site:github.com)" == query_value

    def test_domain_filter_empty_list_ignored(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"search_domain_filter": []}
        )
        assert result["_tinyfish_params"]["query"] == "test"

    def test_domain_filter_non_list_ignored(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"search_domain_filter": "not-a-list"}
        )
        assert result["_tinyfish_params"]["query"] == "test"

    def test_unknown_params_passed_through(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"language": "en", "page": 2}
        )
        params = result["_tinyfish_params"]
        assert params["language"] == "en"
        assert params["page"] == 2

    def test_perplexity_params_not_passed_through(self):
        config = TinyfishSearchConfig()
        supported = config.get_supported_perplexity_optional_params()
        if supported:
            param = next(p for p in supported if p != "max_results" and p != "country")
            result = config.transform_search_request(
                query="test", optional_params={param: "value"}
            )
            assert param not in result["_tinyfish_params"]

    def test_arbitrary_param_passed_through(self):
        # `fetch` is a TinyFish-specific param (JSON-encoded tf-fetch config).
        # The passthrough loop should forward it verbatim without LiteLLM needing
        # to know about it.
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"fetch": "{}"}
        )
        assert result["_tinyfish_params"]["fetch"] == "{}"

    def test_dict_param_auto_json_encoded(self):
        # Callers naturally pass dict-shaped params; we serialize so the
        # downstream urlencode step (which only accepts str|int|bool) doesn't reject.
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test",
            optional_params={"fetch": {"format": "html", "fetch_path": "fast"}},
        )
        assert (
            result["_tinyfish_params"]["fetch"]
            == '{"format":"html","fetch_path":"fast"}'
        )

    def test_bool_param_serialized_as_lowercase(self):
        # urlencode renders Python bool as capitalized "True"/"False"; ux-labs
        # rejects those (e.g. include_thumbnail must be literal "true"/"false").
        # Normalize before passing through.
        config = TinyfishSearchConfig()
        true_result = config.transform_search_request(
            query="test", optional_params={"include_thumbnail": True}
        )
        false_result = config.transform_search_request(
            query="test", optional_params={"include_thumbnail": False}
        )
        assert true_result["_tinyfish_params"]["include_thumbnail"] == "true"
        assert false_result["_tinyfish_params"]["include_thumbnail"] == "false"

    def test_pre_stringified_param_passed_unchanged(self):
        # If the caller already JSON-encoded, don't re-encode.
        config = TinyfishSearchConfig()
        already = '{"format":"html"}'
        result = config.transform_search_request(
            query="test", optional_params={"fetch": already}
        )
        assert result["_tinyfish_params"]["fetch"] == already


class TestGetCompleteUrl:
    def test_default_api_base(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(api_base=None, optional_params={})
        assert url == "https://api.search.tinyfish.ai"

    def test_custom_api_base(self):
        config = TinyfishSearchConfig()
        url = config.get_complete_url(
            api_base="https://custom.api.tinyfish.ai", optional_params={}
        )
        assert url == "https://custom.api.tinyfish.ai"

    def test_env_api_base(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value="https://env.tinyfish.ai",
        ):
            url = config.get_complete_url(api_base=None, optional_params={})
        assert url == "https://env.tinyfish.ai"

    def test_with_tinyfish_params(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(
                api_base=None,
                optional_params={},
                data={"_tinyfish_params": {"query": "hello", "max_results": 5}},
            )
        assert "query=hello" in url
        assert "max_results=5" in url
        assert url.startswith("https://api.search.tinyfish.ai?")

    def test_without_tinyfish_params_key(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(
                api_base=None, optional_params={}, data={"other": "value"}
            )
        assert url == "https://api.search.tinyfish.ai"

    def test_data_none(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(api_base=None, optional_params={}, data=None)
        assert url == "https://api.search.tinyfish.ai"


class TestTransformSearchResponse:
    def test_basic_response(self):
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert result.object == "search"
        assert len(result.results) == 2
        assert result.results[0].title == "TinyFish - AI Web Automation"
        assert result.results[0].url == "https://tinyfish.ai"
        assert (
            result.results[0].snippet == "Automate any website with natural language."
        )

    def test_empty_results(self):
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response({"results": []})
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert result.object == "search"
        assert len(result.results) == 0

    def test_max_results_truncates_from_self_state(self):
        config = TinyfishSearchConfig()
        # Simulate transform_search_request having set the threaded value.
        config._caller_max_results = 3
        many_results = {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "snippet": f"Snippet {i}",
                }
                for i in range(10)
            ]
        }
        mock_response = _make_mock_response(many_results)
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert len(result.results) == 3
        assert result.results[0].title == "Result 0"
        assert result.results[2].title == "Result 2"

    def test_max_results_default_is_tinyfish_cap(self):
        # No caller value → fall back to TinyFish's natural ceiling (10).
        config = TinyfishSearchConfig()
        many_results = {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "snippet": f"Snippet {i}",
                }
                for i in range(15)
            ]
        }
        mock_response = _make_mock_response(many_results)
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert len(result.results) == 10

    def test_missing_required_fields_default_to_empty_string(self):
        # title/url/snippet are required by LiteLLM's SearchResult schema.
        # We default missing/null values to "" so a degraded TinyFish result
        # flows through instead of failing the whole call.
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response(
            {"results": [{}, {"title": None, "url": None, "snippet": None}]}
        )
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert len(result.results) == 2
        for r in result.results:
            assert r.title == ""
            assert r.url == ""
            assert r.snippet == ""

    def test_extra_per_result_fields_surface_as_attributes(self):
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        first = result.results[0]
        assert getattr(first, "position", None) == 1
        assert getattr(first, "site_name", None) == "tinyfish.ai"

    def test_fetch_field_rides_through_to_search_result(self):
        # Mirrors browser-search's per-result `fetch` nested object (see
        # api/src/parser.rs SearchResult.fetch). Confirms `fetch=...` requests
        # surface their content to LiteLLM callers without provider changes.
        config = TinyfishSearchConfig()
        fetched = {
            "results": [
                {
                    "title": "TinyFish",
                    "url": "https://tinyfish.ai",
                    "snippet": "Web automation.",
                    "fetch": {
                        "url": "https://tinyfish.ai",
                        "title": "TinyFish",
                        "text": "Body text",
                        "cached": False,
                    },
                }
            ]
        }
        mock_response = _make_mock_response(fetched)
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        first = result.results[0]
        fetch_field = getattr(first, "fetch", None)
        assert isinstance(fetch_field, dict)
        assert fetch_field["text"] == "Body text"

    def test_no_request_uses_default_max_results(self):
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert len(result.results) == 2

    def test_parameter_warnings_reader_emits_log_lines(self, caplog):
        # When TinyFish responds with a top-level `parameter_warnings` array
        # (post-rollout of that contract), each entry is re-fired as a
        # verbose_logger.warning so callers see what was ignored.
        config = TinyfishSearchConfig()
        body = {
            "results": [
                {"title": "x", "url": "https://x", "snippet": "x"},
            ],
            "parameter_warnings": [
                {
                    "type": "unsupported",
                    "parameter": "max_tokens_per_page",
                    "message": "Parameter not supported by TinyFish Search.",
                    "docs_url": "https://docs.tinyfish.ai/search-api",
                },
            ],
        }
        mock_response = _make_mock_response(body)
        with caplog.at_level("WARNING"):
            config.transform_search_response(
                raw_response=mock_response, logging_obj=None
            )
        messages = [r.getMessage() for r in caplog.records]
        assert any("max_tokens_per_page" in m for m in messages)
        # The type is included in the message so agents can branch on it.
        assert any("unsupported" in m for m in messages)

    def test_parameter_warnings_absent_no_log(self, caplog):
        # Absence of the field is silent — most responses won't carry it.
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)
        with caplog.at_level("WARNING"):
            config.transform_search_response(
                raw_response=mock_response, logging_obj=None
            )
        assert not any(
            "TinyFish Search ignored" in r.getMessage() for r in caplog.records
        )

    def test_parameter_warnings_malformed_shapes_never_throw(self):
        # Every shape that doesn't match {parameter: str, message: str} should
        # silently no-op. None of these should raise an exception.
        config = TinyfishSearchConfig()

        good_results = [{"title": "x", "url": "https://x", "snippet": "x"}]

        malformed_field_values = [
            "not a list",  # string
            42,  # int
            {"parameter": "x", "message": "y"},  # dict instead of list
            True,  # bool
        ]
        for bad_value in malformed_field_values:
            body = {"results": good_results, "parameter_warnings": bad_value}
            config.transform_search_response(
                raw_response=_make_mock_response(body), logging_obj=None
            )  # must not raise

        malformed_entries = [
            "string in list",  # non-dict
            42,  # int
            {},  # missing all
            {"type": "unsupported", "parameter": "x"},  # missing message
            {"type": "unsupported", "message": "y"},  # missing parameter
            {"parameter": "x", "message": "y"},  # missing type
            {
                "type": "unsupported",
                "parameter": None,
                "message": "y",
            },  # null parameter
            {"type": "unsupported", "parameter": "x", "message": ""},  # empty message
            {
                "type": "unsupported",
                "parameter": 42,
                "message": "y",
            },  # non-string parameter
            {"type": 1, "parameter": "x", "message": "y"},  # non-string type
        ]
        body = {"results": good_results, "parameter_warnings": malformed_entries}
        config.transform_search_response(
            raw_response=_make_mock_response(body), logging_obj=None
        )  # must not raise

    def test_parameter_warnings_malformed_entries_emit_nothing(self, caplog):
        config = TinyfishSearchConfig()
        body = {
            "results": [{"title": "x", "url": "https://x", "snippet": "x"}],
            "parameter_warnings": [
                {"type": "unsupported", "parameter": "x"},  # missing message — skipped
                {
                    "type": "unsupported",
                    "parameter": "valid_one",
                    "message": "actual msg",
                },  # ok — emitted
                {"parameter": "x", "message": "y"},  # missing type — skipped
            ],
        }
        with caplog.at_level("WARNING"):
            config.transform_search_response(
                raw_response=_make_mock_response(body), logging_obj=None
            )
        messages = [r.getMessage() for r in caplog.records]
        assert sum("parameter_warning" in m for m in messages) == 1
        assert any("valid_one" in m for m in messages)


class TestErrorHandling:
    def test_4xx_response_raises_with_attribution_and_unwrapped_message(self):
        # Reproduces ux-labs' error envelope shape for an INVALID_INPUT response.
        config = TinyfishSearchConfig()
        body = {
            "error": {
                "code": "INVALID_INPUT",
                "message": "query is required",
                "details": [{"field": "query"}],
            }
        }
        mock_response = _make_mock_response(body, status_code=400)
        with pytest.raises(Exception) as exc_info:
            config.transform_search_response(
                raw_response=mock_response, logging_obj=None
            )
        msg = str(exc_info.value)
        assert "TinyFish Search:" in msg
        assert "query is required" in msg
        assert "docs.tinyfish.ai/search-api" in msg
        assert getattr(exc_info.value, "status_code", None) == 400

    def test_429_preserves_status_code_and_headers(self):
        config = TinyfishSearchConfig()
        body = {"error": {"code": "RATE_LIMIT_EXCEEDED", "message": "60 rpm"}}
        mock_response = _make_mock_response(
            body, status_code=429, headers={"Retry-After": "60"}
        )
        with pytest.raises(Exception) as exc_info:
            config.transform_search_response(
                raw_response=mock_response, logging_obj=None
            )
        assert getattr(exc_info.value, "status_code", None) == 429
        headers = getattr(exc_info.value, "headers", {}) or {}
        assert headers.get("Retry-After") == "60"

    def test_5xx_with_non_ux_labs_body_falls_back_to_raw_text(self):
        # Cloudflare-style JSON or any other envelope: unwrap fails, fall back to raw.
        config = TinyfishSearchConfig()
        body = {"errors": [{"code": "10000", "message": "Internal"}]}
        mock_response = _make_mock_response(body, status_code=502)
        with pytest.raises(Exception) as exc_info:
            config.transform_search_response(
                raw_response=mock_response, logging_obj=None
            )
        msg = str(exc_info.value)
        assert "TinyFish Search:" in msg
        # The raw JSON body string should appear in the message verbatim.
        assert "10000" in msg

    def test_non_json_4xx_body_uses_raw_text(self):
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response(
            json_data=None, status_code=502, text="<html>Bad Gateway</html>"
        )
        with pytest.raises(Exception) as exc_info:
            config.transform_search_response(
                raw_response=mock_response, logging_obj=None
            )
        msg = str(exc_info.value)
        assert "TinyFish Search:" in msg
        assert "Bad Gateway" in msg

    def test_non_json_200_body_routes_through_get_error_class(self):
        # 200 but the body isn't JSON (degraded backend, CDN-injected page, etc.)
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response(
            json_data=None, status_code=200, text="not json"
        )
        with pytest.raises(Exception) as exc_info:
            config.transform_search_response(
                raw_response=mock_response, logging_obj=None
            )
        msg = str(exc_info.value)
        assert "TinyFish Search:" in msg
        assert "Expected JSON response" in msg

    def test_wrap_error_returns_attributed_baselm_exception_directly(self):
        # Direct unit test of the private _wrap_error helper used by
        # transform_search_response. Network failures don't go through this;
        # they hit BaseSearchConfig.get_error_class via LiteLLM core.
        config = TinyfishSearchConfig()
        body = '{"error": {"code": "UNAUTHORIZED", "message": "bad key"}}'
        exc = config._wrap_error(
            error_message=body, status_code=401, headers={"x": "y"}
        )
        msg = str(exc)
        assert "TinyFish Search:" in msg
        assert "bad key" in msg
        assert exc.status_code == 401

    def test_schema_mismatch_wraps_with_attribution(self):
        # When TinyFish returns a 200 with a body shape that doesn't match
        # LiteLLM's SearchResponse contract (e.g. missing top-level `results`),
        # raise with TinyFish attribution + docs link so the caller knows to
        # check TinyFish's schema, not their own input.
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response({"query": "x"})  # no `results` key
        with pytest.raises(Exception) as exc_info:
            config.transform_search_response(
                raw_response=mock_response, logging_obj=None
            )
        msg = str(exc_info.value)
        assert "TinyFish Search:" in msg
        assert "schema" in msg.lower()
        assert "docs.tinyfish.ai/search-api" in msg


class TestAppendDomainFilters:
    def test_single_domain(self):
        result = _append_domain_filters("test", ["example.com"])
        assert result == "(test) (site:example.com)"

    def test_multiple_domains(self):
        result = _append_domain_filters("query", ["a.com", "b.com", "c.com"])
        assert result == "(query) (site:a.com OR site:b.com OR site:c.com)"


class TestDefaultMissingResultFields:
    def test_non_dict_raw_json_is_noop(self):
        # raw_json could be a string/list/None if TinyFish ever returns a
        # non-envelope shape; the helper just returns without mutating.
        for payload in ("not a dict", ["list"], None, 42):
            _default_missing_result_fields(payload)  # must not raise

    def test_non_dict_results_item_skipped(self):
        # If `results` contains a non-dict entry (string, int, etc.), the helper
        # skips it; SearchResponse.model_validate will reject it later.
        raw_json = {"results": ["string item", 42, {"title": "ok"}]}
        _default_missing_result_fields(raw_json)
        # Only the dict item gets defaulted; the others are unchanged.
        assert raw_json["results"][0] == "string item"
        assert raw_json["results"][1] == 42
        assert raw_json["results"][2] == {"title": "ok", "url": "", "snippet": ""}
