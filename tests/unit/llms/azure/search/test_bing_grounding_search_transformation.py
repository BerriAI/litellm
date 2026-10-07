import json, litellm
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from litellm.llms.azure.search.transformation import BingGroundingSearchConfig
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome

REAL_FIXTURE = json.loads((Path(__file__).parent / "foundry_responses_web_search_fixture.json").read_text())

RESPONSES_URL = "https://acct.services.ai.azure.com/api/projects/proj/openai/v1/responses"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in (
        "BING_GROUNDING_PROJECT_ENDPOINT",
        "BING_GROUNDING_MODEL",
        "BING_GROUNDING_CONNECTION_ID",
        "BING_GROUNDING_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def _config(entra_token_minter=None) -> BingGroundingSearchConfig:
    return BingGroundingSearchConfig(entra_token_minter=entra_token_minter)


def _resp(payload, status_code: int = 200):
    r = Mock()
    r.status_code = status_code
    r.headers = {}
    r.content = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
    return r


def _message_response(text: str, annotations: list) -> dict:
    return {
        "output": [
            {"type": "web_search_call", "status": "completed"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": annotations}],
            },
        ]
    }


def _citation(url: str, title: str, start: int, end: int) -> dict:
    return {"type": "url_citation", "url": url, "title": title, "start_index": start, "end_index": end}


def test_ui_friendly_name():
    assert _config().ui_friendly_name() == "Grounding with Bing Search"


def test_validate_environment_api_key_uses_api_key_header_not_bearer():
    headers = _config().validate_environment({}, api_key="azure-api-key")
    assert headers["api-key"] == "azure-api-key"
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_reads_env_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BING_GROUNDING_TOKEN", "env-token")
    headers = _config().validate_environment({})
    assert headers["Authorization"] == "Bearer env-token"
    assert "api-key" not in headers


def test_validate_environment_falls_back_to_entra_minter():
    headers = _config(entra_token_minter=lambda: "entra-token").validate_environment({})
    assert headers["Authorization"] == "Bearer entra-token"


def test_validate_environment_api_key_beats_env_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BING_GROUNDING_TOKEN", "env-token")
    minter = Mock(return_value="entra-token")
    headers = _config(entra_token_minter=minter).validate_environment({}, api_key="azure-api-key")
    assert headers["api-key"] == "azure-api-key"
    assert "Authorization" not in headers
    minter.assert_not_called()


def test_validate_environment_env_token_beats_entra_minter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BING_GROUNDING_TOKEN", "env-token")
    minter = Mock(return_value="entra-token")
    assert _config(entra_token_minter=minter).validate_environment({})["Authorization"] == "Bearer env-token"
    minter.assert_not_called()


def test_validate_environment_refuses_entra_token_for_caller_api_base():
    minter = Mock(return_value="entra-token")
    with pytest.raises(ValueError, match="Refusing to send the server-configured"):
        _config(entra_token_minter=minter).validate_environment({}, api_base="https://attacker.example.com")
    minter.assert_not_called()


def test_validate_environment_entra_minter_failure_names_the_options():
    def failing_minter() -> str:
        raise RuntimeError("no az login")

    with pytest.raises(ValueError, match="no credential available") as excinfo:
        _config(entra_token_minter=failing_minter).validate_environment({})
    message = str(excinfo.value)
    assert "BING_GROUNDING_TOKEN" in message
    assert "https://ai.azure.com/.default" in message
    assert "no az login" in message


def test_validate_environment_does_not_mutate_and_is_idempotent():
    config = _config()
    caller_headers = {"X-Custom": "keep-me"}

    once = config.validate_environment(caller_headers, api_key="k")
    twice = config.validate_environment(once, api_key="k")

    assert caller_headers == {"X-Custom": "keep-me"}
    assert once == twice
    assert once["X-Custom"] == "keep-me"


def test_get_complete_url_from_api_base():
    url = _config().get_complete_url("https://acct.services.ai.azure.com/api/projects/proj", {})
    assert url == RESPONSES_URL


def test_get_complete_url_reads_env_endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BING_GROUNDING_PROJECT_ENDPOINT", "https://acct.services.ai.azure.com/api/projects/proj/")
    assert _config().get_complete_url(None, {}) == RESPONSES_URL


def test_get_complete_url_missing_endpoint_raises():
    with pytest.raises(ValueError, match="BING_GROUNDING_PROJECT_ENDPOINT"):
        _config().get_complete_url(None, {})


@pytest.mark.parametrize(
    "api_base",
    [
        "https://acct.services.ai.azure.com/api/projects/proj",
        "https://acct.services.ai.azure.com/api/projects/proj/",
        "https://acct.services.ai.azure.com/api/projects/proj/openai/v1/responses",
        "https://acct.services.ai.azure.com/api/projects/proj/openai/v1/responses/",
    ],
)
def test_get_complete_url_appends_responses_path_exactly_once(api_base: str):
    assert _config().get_complete_url(api_base, {}) == RESPONSES_URL


def test_transform_search_request_missing_model_raises():
    with pytest.raises(ValueError, match="BING_GROUNDING_MODEL"):
        _config().transform_search_request("q", {})


def test_transform_search_request_web_search_mode_exact_body(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BING_GROUNDING_MODEL", "gpt-4.1")
    body = _config().transform_search_request("latest AI developments", {"max_results": 5})
    assert body == {
        "model": "gpt-4.1",
        "input": "latest AI developments",
        "tools": [{"type": "web_search"}],
    }


def test_transform_search_request_web_search_mode_maps_country(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BING_GROUNDING_MODEL", "gpt-4.1")
    body = _config().transform_search_request("q", {"country": "us"})
    assert body["tools"] == [{"type": "web_search", "user_location": {"type": "approximate", "country": "US"}}]


def test_transform_search_request_connection_mode_exact_body(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BING_GROUNDING_MODEL", "gpt-4.1")
    monkeypatch.setenv("BING_GROUNDING_CONNECTION_ID", "conn-id")
    body = _config().transform_search_request("q", {"max_results": 5})
    assert body == {
        "model": "gpt-4.1",
        "input": "q",
        "tools": [
            {
                "type": "bing_grounding",
                "bing_grounding": {"search_configurations": [{"project_connection_id": "conn-id", "count": 5}]},
            }
        ],
    }


def test_transform_search_request_connection_mode_omits_count_without_max_results(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("BING_GROUNDING_MODEL", "gpt-4.1")
    monkeypatch.setenv("BING_GROUNDING_CONNECTION_ID", "conn-id")
    body = _config().transform_search_request("q", {})
    assert body["tools"][0]["bing_grounding"]["search_configurations"] == [{"project_connection_id": "conn-id"}]


@pytest.mark.parametrize("max_results", [True, False, 0, -1])
def test_transform_search_request_connection_mode_omits_count_for_invalid_max_results(
    monkeypatch: pytest.MonkeyPatch, max_results: object
):
    monkeypatch.setenv("BING_GROUNDING_MODEL", "gpt-4.1")
    monkeypatch.setenv("BING_GROUNDING_CONNECTION_ID", "conn-id")
    body = _config().transform_search_request("q", {"max_results": max_results})
    assert body["tools"][0]["bing_grounding"]["search_configurations"] == [{"project_connection_id": "conn-id"}]


def test_transform_search_response_ignores_invalid_max_results_cap():
    annotations = [_citation(f"https://example.com/{i}", f"T{i}", 0, 5) for i in range(3)]
    resp = _config().transform_search_response(
        _resp(_message_response("claim", annotations)), logging_obj=Mock(), optional_params={"max_results": True}
    )
    assert [r.url for r in resp.results] == [f"https://example.com/{i}" for i in range(3)]


def test_transform_search_request_joins_list_query(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BING_GROUNDING_MODEL", "gpt-4.1")
    assert _config().transform_search_request(["foo", "bar"], {})["input"] == "foo bar"


def test_transform_search_response_real_fixture_dedupes_and_preserves_order():
    resp = _config().transform_search_response(_resp(REAL_FIXTURE), logging_obj=Mock())

    assert resp.object == "search"
    assert [r.url for r in resp.results] == [
        "https://github.com/BerriAI/litellm/releases",
        "https://releasealert.dev/github/BerriAI/litellm",
        "https://api.github.com/repos/BerriAI/litellm/releases/latest",
    ]
    assert resp.results[0].title == "Releases · BerriAI/litellm - GitHub"
    assert resp.results[1].title == "BerriAI/litellm on GitHub | Release Alert"


def test_transform_search_response_real_fixture_snippets_are_the_cited_claims():
    resp = _config().transform_search_response(_resp(REAL_FIXTURE), logging_obj=Mock())

    assert resp.results[0].snippet.startswith("- On the GitHub **Releases** page for BerriAI/litellm")
    assert resp.results[1].snippet.startswith("- An external release")
    assert resp.results[2].snippet.startswith("- The GitHub API (via `releases/latest`)")
    for result in resp.results:
        assert "url_citation" not in result.snippet
        assert not result.snippet.startswith("([")


def test_transform_search_response_snippet_falls_back_to_text_head_for_leading_citation():
    text = "([example.com](https://example.com)) trailing prose"
    payload = _message_response(text, [_citation("https://example.com", "Example", 0, 36)])
    resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
    assert resp.results[0].snippet == text


def test_transform_search_response_snippet_without_indices_uses_last_line():
    payload = _message_response(
        "first line\nthe claim on the last line",
        [{"type": "url_citation", "url": "https://example.com", "title": "Example"}],
    )
    resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
    assert resp.results[0].snippet == "the claim on the last line"


def test_transform_search_response_ignores_non_citation_annotations():
    payload = _message_response("text", [{"type": "file_citation", "url": "https://example.com"}])
    assert _config().transform_search_response(_resp(payload), logging_obj=Mock()).results == []


def test_transform_search_response_ignores_citation_without_url():
    payload = _message_response("text", [{"type": "url_citation", "title": "no url"}])
    assert _config().transform_search_response(_resp(payload), logging_obj=Mock()).results == []


def test_transform_search_response_no_message_output():
    payload = {"output": [{"type": "web_search_call", "status": "completed"}]}
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
def test_transform_search_response_malformed_body_raises_instead_of_reporting_empty(body: str):
    with pytest.raises(Exception, match="Grounding with Bing Search"):
        _config().transform_search_response(_resp(body, status_code=502), logging_obj=Mock())


def test_transform_search_response_caps_results_to_max_results():
    annotations = [_citation(f"https://example.com/{i}", f"T{i}", 0, 5) for i in range(5)]
    resp = _config().transform_search_response(
        _resp(_message_response("claim", annotations)), logging_obj=Mock(), optional_params={"max_results": 2}
    )
    assert [r.url for r in resp.results] == ["https://example.com/0", "https://example.com/1"]


def test_transform_search_response_without_max_results_returns_all_citations():
    annotations = [_citation(f"https://example.com/{i}", f"T{i}", 0, 5) for i in range(4)]
    resp = _config().transform_search_response(_resp(_message_response("c", annotations)), logging_obj=Mock())
    assert len(resp.results) == 4


def test_transform_search_response_failed_status_raises_with_error_message():
    payload = {"output": [], "status": "failed", "error": {"message": "content was filtered"}}
    with pytest.raises(Exception, match="content was filtered") as excinfo:
        _config().transform_search_response(_resp(payload), logging_obj=Mock())
    assert excinfo.value.status_code == 502


def test_transform_search_response_incomplete_with_no_results_raises_with_reason():
    payload = {"output": [], "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    with pytest.raises(Exception, match="incomplete: max_output_tokens"):
        _config().transform_search_response(_resp(payload), logging_obj=Mock())


def test_transform_search_response_incomplete_with_partial_results_returns_them():
    payload = _message_response("claim", [_citation("https://example.com", "Example", 0, 5)])
    payload["status"] = "incomplete"
    resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
    assert [r.url for r in resp.results] == ["https://example.com"]


def test_transform_search_response_web_search_mode_zeroes_per_query_cost():
    payload = _message_response("claim", [_citation("https://example.com", "Example", 0, 5)])
    resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
    assert resp._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.0


def test_transform_search_response_connection_mode_leaves_price_to_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BING_GROUNDING_CONNECTION_ID", "conn-id")
    payload = _message_response("claim", [_citation("https://example.com", "Example", 0, 5)])
    resp = _config().transform_search_response(_resp(payload), logging_obj=Mock())
    assert "additional_headers" not in resp._hidden_params


def test_get_error_class_attributes_the_provider():
    error = _config().get_error_class(error_message="quota exceeded", status_code=429, headers={})
    assert error.status_code == 429
    assert "Grounding with Bing Search: quota exceeded" in str(error)
    assert "learn.microsoft.com" in str(error)


def test_get_error_class_unwraps_the_nested_tool_error():
    nested_tool_error = json.dumps(
        {
            "error": "Tool_User_Error",
            "message": (
                "The specified connection ID 'conn-id' in tool config input was not found "
                "in the project or account connections."
            ),
            "code": "invalid_tool_input",
            "tool": "bing_grounding",
        }
    )
    live_400_shape = json.dumps(
        {
            "error": {
                "message": nested_tool_error,
                "type": "invalid_request_error",
                "param": None,
                "code": "tool_user_error",
            }
        }
    )
    error = _config().get_error_class(error_message=live_400_shape, status_code=400, headers={})
    assert (
        "Grounding with Bing Search: The specified connection ID 'conn-id' in tool config input "
        "was not found in the project or account connections" in str(error)
    )
    assert "Tool_User_Error" not in str(error)


def test_get_error_class_unwraps_a_plain_error_envelope():
    error = _config().get_error_class(
        error_message='{"error":{"message":"The api key is invalid.","code":"401"}}',
        status_code=401,
        headers={},
    )
    assert "Grounding with Bing Search: The api key is invalid" in str(error)


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

PROJECT_ENDPOINT = "https://acct.services.ai.azure.com/api/projects/proj"

_ANSWER_TEXT = (
    "LiteLLM is an open source LLM gateway ([github.com](https://github.com/BerriAI/litellm))\n"
    "The docs live on docs.litellm.ai ([docs.litellm.ai](https://docs.litellm.ai/))"
)

def _annotation(marker: str, url: str, title: str) -> dict:
    start = _ANSWER_TEXT.index(marker)
    return {
        "type": "url_citation",
        "url": url,
        "title": title,
        "start_index": start,
        "end_index": start + len(marker),
    }

MOCK_BING_GROUNDING_RESPONSE = {
    "id": "resp_mock",
    "object": "response",
    "status": "completed",
    "model": "gpt-4.1",
    "output": [
        {"type": "web_search_call", "status": "completed"},
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": _ANSWER_TEXT,
                    "annotations": [
                        _annotation(
                            "([github.com](https://github.com/BerriAI/litellm))",
                            "https://github.com/BerriAI/litellm",
                            "BerriAI/litellm - GitHub",
                        ),
                        _annotation(
                            "([docs.litellm.ai](https://docs.litellm.ai/))",
                            "https://docs.litellm.ai/",
                            "LiteLLM Docs",
                        ),
                    ],
                }
            ],
        },
    ],
    "usage": {"input_tokens": 100, "output_tokens": 50},
}

def _mock_response():
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.content = json.dumps(MOCK_BING_GROUNDING_RESPONSE).encode()
    return response

@pytest.mark.usefixtures("_vcr_outcome_gate")
class TestBingGroundingSearchTransformation:
    """
    Full-stack tests through `litellm.search` / `litellm.asearch` with the HTTP layer mocked.
    Transformation details are unit-tested in tests/unit/llms/azure/search/.
    """

    @pytest.fixture(autouse=True)
    def _server_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BING_GROUNDING_PROJECT_ENDPOINT", PROJECT_ENDPOINT)
        monkeypatch.setenv("BING_GROUNDING_MODEL", "gpt-4.1")
        monkeypatch.setenv("BING_GROUNDING_TOKEN", "test-entra-token")
        monkeypatch.delenv("BING_GROUNDING_CONNECTION_ID", raising=False)

    def test_bing_grounding_search_request_and_response(self):
        with patch(  # test-quality-ok: litellm.search has no client injection seam
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ) as mock_post:
            response = litellm.search(
                query="what is litellm",
                search_provider="bing_grounding",
                max_results=5,
                country="us",
            )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["url"] == f"{PROJECT_ENDPOINT}/openai/v1/responses"
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-entra-token"

        request_body = call_kwargs["json"]
        assert request_body["model"] == "gpt-4.1"
        assert request_body["input"] == "what is litellm"
        assert request_body["tools"] == [
            {"type": "web_search", "user_location": {"type": "approximate", "country": "US"}}
        ]

        assert response.object == "search"
        assert len(response.results) == 2
        assert response.results[0].url == "https://github.com/BerriAI/litellm"
        assert response.results[0].title == "BerriAI/litellm - GitHub"
        assert response.results[0].snippet == "LiteLLM is an open source LLM gateway"
        assert response.results[1].url == "https://docs.litellm.ai/"
        assert response.results[1].snippet == "The docs live on docs.litellm.ai"

    def test_connection_mode_sends_the_bing_grounding_tool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "BING_GROUNDING_CONNECTION_ID",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices"
            "/accounts/acct/projects/proj/connections/bing-conn",
        )
        with patch(  # test-quality-ok: litellm.search has no client injection seam
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ) as mock_post:
            litellm.search(
                query="what is litellm",
                search_provider="bing_grounding",
                max_results=3,
            )

        request_body = mock_post.call_args.kwargs["json"]
        assert request_body["tools"] == [
            {
                "type": "bing_grounding",
                "bing_grounding": {
                    "search_configurations": [
                        {
                            "project_connection_id": (
                                "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices"
                                "/accounts/acct/projects/proj/connections/bing-conn"
                            ),
                            "count": 3,
                        }
                    ]
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_bing_grounding_asearch(self):
        with patch(  # test-quality-ok: litellm.asearch has no client injection seam
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=AsyncMock(return_value=_mock_response()),
        ) as mock_post:
            response = await litellm.asearch(
                query="what is litellm",
                search_provider="bing_grounding",
            )

        assert mock_post.call_args.kwargs["json"]["tools"] == [{"type": "web_search"}]
        assert len(response.results) == 2

    def test_web_search_mode_is_not_billed_the_g1_price(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        with patch(  # test-quality-ok: litellm.search has no client injection seam
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ):
            response = litellm.search(query="pricing check", search_provider="bing_grounding")

        assert response._hidden_params["response_cost"] == 0.0

    def test_connection_mode_tracks_the_g1_cost(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BING_GROUNDING_CONNECTION_ID", "conn-id")
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        with patch(  # test-quality-ok: litellm.search has no client injection seam
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ):
            response = litellm.search(query="pricing check", search_provider="bing_grounding")

        # Grounding with Bing Search (G1 SKU): $14 per 1,000 transactions, https://www.microsoft.com/en-us/bing/apis, checked 2026-09-24
        assert response._hidden_params["response_cost"] == pytest.approx(0.014)
