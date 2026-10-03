"""
Tests for XAI Search API transformation (the x_search Live Search tool).

Tests the XAISearchConfig class that reverse-engineers a SearchResponse out of an
xAI Responses API turn, since x_search only exists as a tool inside that turn.

Source: litellm/llms/xai/search/transformation.py
"""

import json
from unittest.mock import Mock

import pytest

from litellm.llms.xai.search import transformation
from litellm.llms.xai.search.transformation import XAISearchConfig
from litellm.types.utils import SearchProviders
from litellm.utils import ProviderConfigManager

XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in ("XAI_API_KEY", "XAI_API_BASE"):
        monkeypatch.delenv(var, raising=False)


def _config() -> XAISearchConfig:
    return XAISearchConfig()


def _resp(payload, status_code: int = 200) -> Mock:
    r = Mock()
    r.status_code = status_code
    r.headers = {}
    r.content = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
    return r


def _citation(url: str, title: str) -> dict:
    return {"type": "url_citation", "url": url, "title": title, "start_index": 0, "end_index": 0}


def _message_response(text: str, annotations: list, usage: dict | None = None) -> dict:
    payload = {
        "output": [
            {"type": "x_search_call", "status": "completed"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": annotations}],
            },
        ]
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


REAL_SHAPED_FIXTURE = _message_response(
    text=(
        "SpaceX's latest Starship flight reached orbit and completed a controlled reentry, "
        "according to the official SpaceX account and independent orbital trackers."
    ),
    annotations=[
        _citation("https://x.com/SpaceX/status/1234567890", "SpaceX on X"),
        _citation("https://x.com/spacetrackorg/status/2234567890", "Space Track on X"),
        _citation("https://x.com/SpaceX/status/1234567890", "SpaceX on X"),
    ],
    usage={
        "input_tokens": 512,
        "output_tokens": 128,
        "output_tokens_details": {"reasoning_tokens": 64},
        "cost_in_usd_ticks": 45000000,
    },
)


class TestXAISearchConfigRegistration:
    def test_provider_registration(self):
        config = ProviderConfigManager.get_provider_search_config(provider=SearchProviders.XAI)
        assert isinstance(config, XAISearchConfig)

    def test_ui_friendly_name(self):
        assert _config().ui_friendly_name() == "xAI Live Search (x_search)"


class TestXAISearchConfigValidateEnvironment:
    def test_uses_caller_api_key(self):
        headers = _config().validate_environment({}, api_key="caller-key")
        assert headers["Authorization"] == "Bearer caller-key"
        assert headers["Content-Type"] == "application/json"

    def test_reads_env_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XAI_API_KEY", "env-key")
        headers = _config().validate_environment({})
        assert headers["Authorization"] == "Bearer env-key"

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="XAI_API_KEY"):
            _config().validate_environment({})

    def test_does_not_mutate_caller_headers(self):
        caller_headers = {"X-Custom": "keep-me"}
        result = _config().validate_environment(caller_headers, api_key="k")
        assert caller_headers == {"X-Custom": "keep-me"}
        assert result["X-Custom"] == "keep-me"

    def test_refuses_env_key_for_untrusted_api_base(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XAI_API_KEY", "env-key")
        with pytest.raises(ValueError, match="Refusing to send the server-configured"):
            _config().validate_environment({}, api_base="https://attacker.example.com")

    def test_allows_env_key_for_default_trusted_api_base(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XAI_API_KEY", "env-key")
        headers = _config().validate_environment({}, api_base="https://api.x.ai/v1")
        assert headers["Authorization"] == "Bearer env-key"

    def test_allows_env_key_for_api_base_matching_xai_api_base_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XAI_API_KEY", "env-key")
        monkeypatch.setenv("XAI_API_BASE", "https://custom.x.ai/v1")
        headers = _config().validate_environment({}, api_base="https://custom.x.ai/v1")
        assert headers["Authorization"] == "Bearer env-key"

    def test_caller_api_key_bypasses_trust_check(self):
        headers = _config().validate_environment(
            {}, api_key="caller-key", api_base="https://attacker.example.com"
        )
        assert headers["Authorization"] == "Bearer caller-key"


class TestXAISearchConfigGetCompleteUrl:
    def test_default(self):
        assert _config().get_complete_url(None, {}) == XAI_RESPONSES_URL

    def test_reads_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XAI_API_BASE", "https://custom.x.ai/v1")
        assert _config().get_complete_url(None, {}) == "https://custom.x.ai/v1/responses"

    def test_explicit_api_base_overrides_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XAI_API_BASE", "https://ignored.x.ai/v1")
        assert _config().get_complete_url("https://explicit.x.ai/v1", {}) == "https://explicit.x.ai/v1/responses"

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://api.x.ai/v1",
            "https://api.x.ai/v1/",
            "https://api.x.ai/v1/responses",
            "https://api.x.ai/v1/responses/",
        ],
    )
    def test_appends_responses_path_exactly_once(self, api_base: str):
        assert _config().get_complete_url(api_base, {}) == XAI_RESPONSES_URL


class TestXAISearchConfigTransformRequest:
    def test_default_model(self):
        body = _config().transform_search_request("latest AI developments", {})
        assert body["model"] == "grok-4-fast"
        assert body["input"] == "latest AI developments"
        assert body["tools"] == [{"type": "x_search"}]

    def test_custom_model(self):
        body = _config().transform_search_request("q", {"model": "grok-4"})
        assert body["model"] == "grok-4"

    def test_joins_list_query(self):
        assert _config().transform_search_request(["foo", "bar"], {})["input"] == "foo bar"

    def test_x_search_filters_included(self):
        body = _config().transform_search_request(
            "q",
            {
                "allowed_x_handles": ["spacex", "nasa"],
                "excluded_x_handles": ["spam"],
                "from_date": "2026-01-01",
                "to_date": "2026-02-01",
                "enable_image_understanding": True,
                "enable_video_understanding": False,
            },
        )
        assert body["tools"] == [
            {
                "type": "x_search",
                "allowed_x_handles": ["spacex", "nasa"],
                "excluded_x_handles": ["spam"],
                "from_date": "2026-01-01",
                "to_date": "2026-02-01",
                "enable_image_understanding": True,
                "enable_video_understanding": False,
            }
        ]

    def test_omits_absent_filters(self):
        body = _config().transform_search_request("q", {})
        assert body["tools"] == [{"type": "x_search"}]

    def test_ignores_unrelated_optional_params(self):
        body = _config().transform_search_request("q", {"max_results": 5, "search_domain_filter": ["x.com"]})
        assert body["tools"] == [{"type": "x_search"}]

    @pytest.mark.parametrize(
        "key,bad_value",
        [
            ("allowed_x_handles", "not-a-list"),
            ("excluded_x_handles", 123),
            ("from_date", 20260101),
            ("to_date", True),
            ("enable_image_understanding", "yes"),
            ("enable_video_understanding", 1),
        ],
    )
    def test_ignores_wrong_typed_filters(self, key: str, bad_value: object):
        body = _config().transform_search_request("q", {key: bad_value})
        assert body["tools"] == [{"type": "x_search"}]

    @pytest.mark.parametrize(
        "key,bad_list",
        [
            ("allowed_x_handles", ["spacex", None]),
            ("allowed_x_handles", ["spacex", 123]),
            ("excluded_x_handles", [True, "spam"]),
        ],
    )
    def test_ignores_list_filters_with_non_string_elements(self, key: str, bad_list: list):
        body = _config().transform_search_request("q", {key: bad_list})
        assert body["tools"] == [{"type": "x_search"}]


class TestXAISearchConfigTransformResponse:
    def test_real_shaped_fixture_dedupes_and_preserves_order(self):
        resp = _config().transform_search_response(_resp(REAL_SHAPED_FIXTURE), logging_obj=Mock())
        assert resp.object == "search"
        assert [r.url for r in resp.results] == [
            "https://x.com/SpaceX/status/1234567890",
            "https://x.com/spacetrackorg/status/2234567890",
        ]
        assert resp.results[0].title == "SpaceX on X"
        assert resp.results[1].title == "Space Track on X"

    def test_shares_full_text_as_snippet_across_citations(self):
        resp = _config().transform_search_response(_resp(REAL_SHAPED_FIXTURE), logging_obj=Mock())
        expected_snippet = REAL_SHAPED_FIXTURE["output"][1]["content"][0]["text"]
        assert all(r.snippet == expected_snippet for r in resp.results)

    def test_multiple_message_items_join_text_with_a_separator(self):
        payload = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "first claim",
                            "annotations": [_citation("https://example.com/a", "A")],
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "second claim",
                            "annotations": [_citation("https://example.com/b", "B")],
                        }
                    ],
                },
            ]
        }
        resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
        assert all(r.snippet == "first claim\n\nsecond claim" for r in resp.results)

    def test_ignores_non_citation_annotations(self):
        payload = _message_response("text", [{"type": "file_citation", "url": "https://example.com"}])
        assert _config().transform_search_response(_resp(payload), logging_obj=Mock()).results == []

    def test_ignores_citation_without_url(self):
        payload = _message_response("text", [{"type": "url_citation", "title": "no url"}])
        assert _config().transform_search_response(_resp(payload), logging_obj=Mock()).results == []

    def test_no_message_output_returns_empty_results(self):
        payload = {"output": [{"type": "x_search_call", "status": "completed"}]}
        assert _config().transform_search_response(_resp(payload), logging_obj=Mock()).results == []

    @pytest.mark.parametrize(
        "body",
        [
            "<html>502 Bad Gateway</html>",
            '{"output": "garbage"}',
            '{"output": null}',
            "{}",
        ],
    )
    def test_malformed_body_raises_instead_of_reporting_empty(self, body: str):
        with pytest.raises(Exception, match="xAI x_search"):
            _config().transform_search_response(_resp(body, status_code=502), logging_obj=Mock())

    def test_failed_status_raises_with_error_message(self):
        payload = {"output": [], "status": "failed", "error": {"message": "content was filtered"}}
        with pytest.raises(Exception, match="content was filtered") as excinfo:
            _config().transform_search_response(_resp(payload), logging_obj=Mock())
        assert excinfo.value.status_code == 502

    def test_incomplete_with_no_results_raises_with_reason(self):
        payload = {"output": [], "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        with pytest.raises(Exception, match="incomplete: max_output_tokens"):
            _config().transform_search_response(_resp(payload), logging_obj=Mock())

    def test_incomplete_with_partial_results_returns_them(self):
        payload = _message_response("claim", [_citation("https://example.com", "Example")])
        payload["status"] = "incomplete"
        resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
        assert [r.url for r in resp.results] == ["https://example.com"]

    def test_caps_results_to_max_results(self):
        annotations = [_citation(f"https://example.com/{i}", f"T{i}") for i in range(5)]
        resp = _config().transform_search_response(
            _resp(_message_response("claim", annotations)), logging_obj=Mock(), optional_params={"max_results": 2}
        )
        assert [r.url for r in resp.results] == ["https://example.com/0", "https://example.com/1"]

    def test_without_max_results_returns_all_citations(self):
        annotations = [_citation(f"https://example.com/{i}", f"T{i}") for i in range(4)]
        resp = _config().transform_search_response(
            _resp(_message_response("claim", annotations)), logging_obj=Mock()
        )
        assert len(resp.results) == 4

    @pytest.mark.parametrize("max_results", [True, False, 0, -1, "5"])
    def test_ignores_invalid_max_results_and_returns_all_citations(self, max_results: object):
        annotations = [_citation(f"https://example.com/{i}", f"T{i}") for i in range(3)]
        resp = _config().transform_search_response(
            _resp(_message_response("claim", annotations)),
            logging_obj=Mock(),
            optional_params={"max_results": max_results},
        )
        assert len(resp.results) == 3


class TestXAISearchConfigCost:
    def test_uses_xai_reported_ticks_when_present(self):
        resp = _config().transform_search_response(_resp(REAL_SHAPED_FIXTURE), logging_obj=Mock())
        assert resp._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.0045

    def test_no_usage_leaves_no_cost_attached(self):
        payload = _message_response("claim", [_citation("https://example.com", "Example")])
        resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
        assert "additional_headers" not in resp._hidden_params

    def test_falls_back_to_token_cost_when_ticks_absent(self, monkeypatch: pytest.MonkeyPatch):
        def fake_get_model_info(model: str, custom_llm_provider: str):
            if model == "xai/grok-4-fast-non-reasoning":
                return {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
            raise Exception("not mapped")

        monkeypatch.setattr(transformation, "get_model_info", fake_get_model_info)
        payload = _message_response(
            "claim",
            [_citation("https://example.com", "Example")],
            usage={"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 0}},
        )
        resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
        expected = 100 * 1e-6 + 50 * 2e-6 + transformation._PER_CITATION_SURCHARGE_USD * 1
        assert resp._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == pytest.approx(
            expected
        )

    def test_uses_reasoning_suffix_when_reasoning_tokens_present(self, monkeypatch: pytest.MonkeyPatch):
        seen_models = []

        def fake_get_model_info(model: str, custom_llm_provider: str):
            seen_models.append(model)
            if model == "xai/grok-4-fast-reasoning":
                return {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
            raise Exception("not mapped")

        monkeypatch.setattr(transformation, "get_model_info", fake_get_model_info)
        payload = _message_response(
            "claim",
            [_citation("https://example.com", "Example")],
            usage={"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 10}},
        )
        resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
        assert "additional_headers" in resp._hidden_params
        assert seen_models == ["xai/grok-4-fast", "xai/grok-4-fast-reasoning"]

    def test_tries_bare_model_before_suffix(self, monkeypatch: pytest.MonkeyPatch):
        def fake_get_model_info(model: str, custom_llm_provider: str):
            if model == "xai/grok-4-fast":
                return {"input_cost_per_token": 3e-6, "output_cost_per_token": 4e-6}
            raise Exception("not mapped")

        monkeypatch.setattr(transformation, "get_model_info", fake_get_model_info)
        payload = _message_response(
            "claim",
            [_citation("https://example.com", "Example")],
            usage={"input_tokens": 10, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 0}},
        )
        resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
        expected = 10 * 3e-6 + 5 * 4e-6 + transformation._PER_CITATION_SURCHARGE_USD * 1
        assert resp._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == pytest.approx(
            expected
        )

    def test_unmapped_model_leaves_no_cost_attached(self, monkeypatch: pytest.MonkeyPatch):
        def fake_get_model_info(model: str, custom_llm_provider: str):
            raise Exception("not mapped")

        monkeypatch.setattr(transformation, "get_model_info", fake_get_model_info)
        payload = _message_response(
            "claim",
            [_citation("https://example.com", "Example")],
            usage={"input_tokens": 10, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 0}},
        )
        resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
        assert "additional_headers" not in resp._hidden_params

    def test_surcharge_uses_distinct_citation_count_not_raw_annotation_count(self, monkeypatch: pytest.MonkeyPatch):
        def fake_get_model_info(model: str, custom_llm_provider: str):
            if model == "xai/grok-4-fast-non-reasoning":
                return {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
            raise Exception("not mapped")

        monkeypatch.setattr(transformation, "get_model_info", fake_get_model_info)
        payload = _message_response(
            "claim",
            [
                _citation("https://example.com/a", "A"),
                _citation("https://example.com/a", "A"),
                _citation("https://example.com/b", "B"),
            ],
            usage={"input_tokens": 0, "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}},
        )
        resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
        assert resp._hidden_params["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ] == pytest.approx(transformation._PER_CITATION_SURCHARGE_USD * 2)

    def test_surcharge_uses_full_citation_count_even_when_results_are_capped(self, monkeypatch: pytest.MonkeyPatch):
        def fake_get_model_info(model: str, custom_llm_provider: str):
            if model == "xai/grok-4-fast-non-reasoning":
                return {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
            raise Exception("not mapped")

        monkeypatch.setattr(transformation, "get_model_info", fake_get_model_info)
        annotations = [_citation(f"https://example.com/{i}", f"T{i}") for i in range(5)]
        payload = _message_response(
            "claim",
            annotations,
            usage={"input_tokens": 0, "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}},
        )
        resp = _config().transform_search_response(
            _resp(payload), logging_obj=Mock(), optional_params={"max_results": 2}
        )
        assert len(resp.results) == 2
        assert resp._hidden_params["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ] == pytest.approx(transformation._PER_CITATION_SURCHARGE_USD * 5)


class TestXAISearchConfigGetErrorClass:
    def test_attributes_the_provider(self):
        error = _config().get_error_class(error_message="quota exceeded", status_code=429, headers={})
        assert error.status_code == 429
        assert "xAI x_search: quota exceeded" in str(error)
