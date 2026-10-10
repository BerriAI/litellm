import json
import logging
from datetime import datetime
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx


import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.rerank import RerankResponse

MARKER_QUERY = "MARKER_QUERY_do_not_log_at_info"
MARKER_DOC = "MARKER_DOC_sensitive_customer_text"
COHERE_RERANK_RESPONSE: Final = {
    "id": "rerank-1",
    "results": [
        {"index": 0, "relevance_score": 0.95, "document": {"text": "hello"}},
        {"index": 1, "relevance_score": 0.1, "document": {"text": "world"}},
    ],
    "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": 1}},
}


class RespxAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, router: respx.MockRouter) -> None:
        self.router = router

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self.router.async_handler(request)


def _mock_cohere_response() -> MagicMock:
    mock_response = MagicMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "meta": {
                "api_version": {"version": "1.0"},
                "billed_units": {"search_units": 1},
            },
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200
    return mock_response


def test_rerank_does_not_log_request_content_at_info(caplog):
    """Regression for #32525: rerank must not emit query/documents to logs at INFO.

    The mapped ``optional_rerank_params`` (which always contains ``query`` and
    ``documents``) bypasses ``turn_off_message_logging`` / ``redact_messages``,
    so logging it at INFO leaks raw request content into stdout and any log sink.
    """
    litellm.cohere_key = "test_api_key"
    caplog.set_level(logging.DEBUG, logger="LiteLLM")

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=_mock_cohere_response(),
    ):
        litellm.rerank(
            model="cohere/rerank-english-v3.0",
            query=MARKER_QUERY,
            documents=[MARKER_DOC, "unrelated"],
            top_n=2,
        )

    litellm_records = [r for r in caplog.records if r.name == "LiteLLM"]

    info_or_above = [
        r.getMessage()
        for r in litellm_records
        if r.levelno >= logging.INFO and (MARKER_QUERY in r.getMessage() or MARKER_DOC in r.getMessage())
    ]
    assert not info_or_above, f"rerank leaked request content at INFO+: {info_or_above}"

    optional_params_logs = [r for r in litellm_records if "optional_rerank_params" in r.getMessage()]
    assert optional_params_logs, "expected the optional_rerank_params line to be logged"
    assert all(r.levelno == logging.DEBUG for r in optional_params_logs), (
        "optional_rerank_params must be logged at DEBUG, not INFO"
    )


TOGETHER_RERANK_BODY = {
    "id": "rerank-mock-id",
    "results": [{"index": 0, "relevance_score": 0.95}],
    "usage": {"prompt_tokens": 10, "total_tokens": 10},
}


def test_together_rerank_defaults_to_together_ai_host(respx_mock: respx.MockRouter, monkeypatch):
    """Regression for the Together host migration: rerank used to hardcode
    https://api.together.xyz/v1/rerank. The default must now be api.together.ai."""
    monkeypatch.delenv("TOGETHER_AI_API_BASE", raising=False)

    mock_route = respx_mock.post("https://api.together.ai/v1/rerank")
    mock_route.return_value = httpx.Response(200, json=TOGETHER_RERANK_BODY)

    response = litellm.rerank(
        model="together_ai/mixedbread-ai/mxbai-rerank-large-v2",
        query=MARKER_QUERY,
        documents=[MARKER_DOC],
        api_key="fake-together-key",
    )

    assert mock_route.called
    assert response.results[0]["relevance_score"] == 0.95


def test_together_rerank_honors_api_base(respx_mock: respx.MockRouter):
    """Regression: a custom api_base was silently ignored by the Together rerank handler."""
    mock_route = respx_mock.post("https://custom-together.example/v1/rerank")
    mock_route.return_value = httpx.Response(200, json=TOGETHER_RERANK_BODY)

    litellm.rerank(
        model="together_ai/mixedbread-ai/mxbai-rerank-large-v2",
        query=MARKER_QUERY,
        documents=[MARKER_DOC],
        api_key="fake-together-key",
        api_base="https://custom-together.example/v1",
    )

    assert mock_route.called
    assert mock_route.calls[0].request.headers["authorization"] == "Bearer fake-together-key"


DASHSCOPE_RERANK_BODY = {
    "object": "list",
    "results": [{"index": 0, "relevance_score": 0.95}],
    "model": "qwen3-rerank",
    "id": "rerank-mock-id",
    "usage": {"total_tokens": 10},
}


def test_dashscope_rerank_defaults_to_live_rerank_route(respx_mock: respx.MockRouter, monkeypatch):
    """Regression for the dead default endpoint: get_llm_provider always returns the
    chat base for dashscope, which used to hijack rerank onto the dead
    /compatible-mode/v1/reranks route."""
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_BASE_RERANK", raising=False)

    mock_route = respx_mock.post("https://dashscope.aliyuncs.com/compatible-api/v1/reranks")
    mock_route.return_value = httpx.Response(200, json=DASHSCOPE_RERANK_BODY)

    response = litellm.rerank(
        model="dashscope/qwen3-rerank",
        query=MARKER_QUERY,
        documents=[MARKER_DOC],
        api_key="fake-dashscope-key",
    )

    assert mock_route.called
    assert response.results[0]["relevance_score"] == 0.95


def test_dashscope_rerank_chat_env_base_keeps_host_and_rerank_route(respx_mock: respx.MockRouter, monkeypatch):
    """Regression: a chat-style DASHSCOPE_API_BASE must not hijack rerank onto the
    chat path, while its host (the region) is preserved."""
    monkeypatch.setenv("DASHSCOPE_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    monkeypatch.delenv("DASHSCOPE_API_BASE_RERANK", raising=False)

    mock_route = respx_mock.post("https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks")
    mock_route.return_value = httpx.Response(200, json=DASHSCOPE_RERANK_BODY)

    litellm.rerank(
        model="dashscope/qwen3-rerank",
        query=MARKER_QUERY,
        documents=[MARKER_DOC],
        api_key="fake-dashscope-key",
    )

    assert mock_route.called


def test_dashscope_rerank_explicit_api_base_wins(respx_mock: respx.MockRouter, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

    mock_route = respx_mock.post("https://custom-rerank.example/v1/reranks")
    mock_route.return_value = httpx.Response(200, json=DASHSCOPE_RERANK_BODY)

    litellm.rerank(
        model="dashscope/qwen3-rerank",
        query=MARKER_QUERY,
        documents=[MARKER_DOC],
        api_key="fake-dashscope-key",
        api_base="https://custom-rerank.example/v1",
    )

    assert mock_route.called


DASHSCOPE_404_BODY = {
    "error": {
        "message": "The model `does-not-exist` does not exist or you do not have access to it.",
        "type": "invalid_request_error",
        "param": None,
        "code": "model_not_found",
    },
    "request_id": "mock-request-id",
}


def test_rerank_error_names_provider_and_keeps_body(respx_mock: respx.MockRouter, monkeypatch):
    """Regression for the rerank error path mapping with the unresolved provider param:
    a provider 404 surfaced as 'None - ' instead of naming the provider and its error body."""
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_BASE_RERANK", raising=False)

    mock_route = respx_mock.post("https://dashscope.example/v1/reranks")
    mock_route.return_value = httpx.Response(404, json=DASHSCOPE_404_BODY)

    with pytest.raises(litellm.NotFoundError) as exc_info:
        litellm.rerank(
            model="dashscope/does-not-exist",
            query=MARKER_QUERY,
            documents=[MARKER_DOC],
            api_key="fake-dashscope-key",
            api_base="https://dashscope.example/v1",
        )

    assert mock_route.called
    assert "DashscopeException" in str(exc_info.value)
    assert "does not exist or you do not have access to it" in str(exc_info.value)
    assert "None - " not in str(exc_info.value)


@pytest.mark.asyncio
async def test_arerank_error_is_mapped_to_litellm_exception(respx_mock: respx.MockRouter, monkeypatch):
    """Regression for arerank's bare re-raise: provider errors escaped as raw
    provider exception classes instead of the mapped litellm exception contract."""
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_BASE_RERANK", raising=False)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")

    mock_route = respx_mock.post("https://dashscope.example/v1/reranks")
    mock_route.return_value = httpx.Response(404, json=DASHSCOPE_404_BODY)

    with pytest.raises(litellm.NotFoundError) as exc_info:
        await litellm.arerank(
            model="dashscope/does-not-exist",
            query=MARKER_QUERY,
            documents=[MARKER_DOC],
            api_key="fake-dashscope-key",
            api_base="https://dashscope.example/v1",
        )

    assert mock_route.called
    assert "DashscopeException" in str(exc_info.value)
    assert "does not exist or you do not have access to it" in str(exc_info.value)
    assert "None - " not in str(exc_info.value)


@pytest.mark.asyncio
async def test_arerank_declared_authenticating_provider_skips_resolution(monkeypatch):
    """Regression for the event-loop hazard in arerank's provider pre-resolution:
    get_llm_provider runs the blocking OAuth device flow for github_copilot/chatgpt,
    so arerank must adopt the declared provider instead of resolving it, while the
    except path still maps with that declared provider."""
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    resolution_calls = []

    def record_resolution(*args, **kwargs):
        resolution_calls.append((args, kwargs))
        return "gpt-4o", "github_copilot", None, None

    def rerank_raises_provider_error(*args, **kwargs):
        raise BaseLLMException(status_code=401, message='{"error":"bad key"}')

    monkeypatch.setattr(litellm, "get_llm_provider", record_resolution)
    monkeypatch.setattr(
        "litellm.litellm_core_utils.llm_response_utils.get_api_base.get_llm_provider", record_resolution
    )
    monkeypatch.setattr("litellm.rerank_api.main.rerank", rerank_raises_provider_error)

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        await litellm.arerank(
            model="github_copilot/gpt-4o",
            query=MARKER_QUERY,
            documents=[MARKER_DOC],
        )

    assert resolution_calls == []
    assert "Github_copilotException" in str(exc_info.value)
    assert "None - " not in str(exc_info.value)


@pytest.mark.asyncio
async def test_together_rerank_async_honors_env_api_base(respx_mock: respx.MockRouter, monkeypatch):
    """Regression: TOGETHER_AI_API_BASE was honored by chat but ignored by rerank."""
    monkeypatch.setenv("TOGETHER_AI_API_BASE", "https://env-together.example/v1")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")

    mock_route = respx_mock.post("https://env-together.example/v1/rerank")
    mock_route.return_value = httpx.Response(200, json=TOGETHER_RERANK_BODY)

    response = await litellm.arerank(
        model="together_ai/mixedbread-ai/mxbai-rerank-large-v2",
        query=MARKER_QUERY,
        documents=[MARKER_DOC],
        api_key="fake-together-key",
    )

    assert mock_route.called
    assert response.results[0]["relevance_score"] == 0.95


def test_cohere_rerank_v2_client():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    litellm.api_base = "http://localhost:4000"
    litellm.set_verbose = True

    text = "Hello there!"
    list_texts = ["Hello there!", "How are you?", "How do you do?"]

    rerank_model = "rerank-multilingual-v3.0"

    with patch.object(client, "post") as mock_post:
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "id": "cmpl-mockid",
                "results": [
                    {"index": 0, "relevance_score": 0.95},
                    {"index": 1, "relevance_score": 0.75},
                    {"index": 2, "relevance_score": 0.65},
                ],
                "usage": {"prompt_tokens": 100, "total_tokens": 150},
            }
        )
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)

        mock_post.return_value = mock_response

        response = litellm.rerank(
            model=rerank_model,
            query=text,
            documents=list_texts,
            custom_llm_provider="cohere",
            max_tokens_per_doc=3,
            top_n=2,
            api_key="fake-api-key",
            client=client,
        )

        # Ensure Cohere API is called with the expected params
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["url"] == "http://localhost:4000/v2/rerank"

        request_data = json.loads(mock_post.call_args.kwargs["data"])
        assert request_data["model"] == rerank_model
        assert request_data["query"] == text
        assert request_data["documents"] == list_texts
        assert request_data["max_tokens_per_doc"] == 3
        assert request_data["top_n"] == 2

        # Ensure litellm response is what we expect
        assert response["results"] == mock_response.json()["results"]


@pytest.mark.usefixtures("fake_provider_credentials")
def test_rerank_infer_region_from_model_arn(monkeypatch):

    mock_response = MagicMock()

    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    args = {
        "model": "bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        "query": "hello",
        "documents": ["hello", "world"],
    }

    def return_val():
        return {
            "results": [
                {"index": 0, "relevanceScore": 0.6716859340667725},
                {"index": 1, "relevanceScore": 0.0004994205664843321},
            ]
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    client = HTTPHandler()

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        litellm.rerank(
            model=args["model"],
            query=args["query"],
            documents=args["documents"],
            client=client,
        )

        mock_post.assert_called_once()
        print(f"mock_post.call_args: {mock_post.call_args.kwargs}")
        assert "us-west-2" in mock_post.call_args.kwargs["url"]
        assert "us-east-1" not in mock_post.call_args.kwargs["url"]


@pytest.mark.respx(assert_all_called=True)
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [True, False])
async def test_cohere_rerank_parses_results_from_provider_response(
    async_mode: bool, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post("https://api.cohere.com/v2/rerank")
    route.return_value = httpx.Response(200, json=COHERE_RERANK_RESPONSE)
    kwargs: Final = {
        "model": "cohere/rerank-v4.0-pro",
        "query": "hello",
        "documents": ["hello", "world"],
        "top_n": 2,
        "api_key": "test-api-key",
    }

    if async_mode:
        client: Final = AsyncHTTPHandler(transport=RespxAsyncTransport(respx_mock))
        response: Final = await litellm.arerank(**kwargs, client=client)
        await client.client.aclose()
    else:
        response = litellm.rerank(**kwargs)

    assert response.model_dump() == COHERE_RERANK_RESPONSE


@pytest.mark.respx(assert_all_called=True)
def test_cohere_rerank_reuses_cached_result_for_identical_request(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.caching.caching import Cache

    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    route: Final = respx_mock.post("https://api.cohere.com/v2/rerank")
    route.return_value = httpx.Response(200, json=COHERE_RERANK_RESPONSE)
    request: Final = {
        "model": "cohere/rerank-v4.0-pro",
        "query": "hello",
        "documents": ["hello", "world"],
        "top_n": 2,
        "api_key": "test-api-key",
    }

    first_response: Final = litellm.rerank(**request)
    second_response: Final = litellm.rerank(**request)

    assert len(route.calls) == 1
    assert first_response.results == second_response.results
    assert "cache_key" in second_response._hidden_params


@pytest.mark.respx(assert_all_called=True)
def test_cohere_rerank_completes_root_api_base_to_v2_route(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post("http://localhost:4000/v2/rerank")
    route.return_value = httpx.Response(200, json=COHERE_RERANK_RESPONSE)

    litellm.rerank(
        model="cohere/rerank-v4.0-pro",
        query="hello",
        documents=["hello", "world"],
        custom_llm_provider="cohere",
        api_base="http://localhost:4000",
        api_key="test-api-key",
    )

    assert route.calls[0].request.url == "http://localhost:4000/v2/rerank"


@pytest.mark.respx(assert_all_called=True)
def test_cohere_rerank_sends_return_documents_and_preserves_text(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post("https://api.cohere.com/v2/rerank")
    route.return_value = httpx.Response(200, json=COHERE_RERANK_RESPONSE)

    response: Final = litellm.rerank(
        model="cohere/rerank-v4.0-pro",
        query="hello",
        documents=["hello", "world"],
        return_documents=True,
        top_n=2,
        api_key="test-api-key",
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "rerank-v4.0-pro",
        "query": "hello",
        "documents": ["hello", "world"],
        "top_n": 2,
        "return_documents": True,
    }
    assert [result["document"]["text"] for result in response.results] == ["hello", "world"]


@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize(
    ("api_base", "endpoint"),
    [
        ("https://cohere.example/v1/rerank", "https://cohere.example/v1/rerank"),
        ("https://cohere.example", "https://cohere.example/v2/rerank"),
    ],
)
def test_cohere_rerank_custom_api_base_selects_versioned_route(
    api_base: str, endpoint: str, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(endpoint)
    route.return_value = httpx.Response(200, json=COHERE_RERANK_RESPONSE)

    litellm.rerank(
        model="cohere/rerank-v4.0-pro",
        query="hello",
        documents=["hello", "world"],
        top_n=2,
        api_base=api_base,
        api_key="test-api-key",
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body["model"] == "rerank-v4.0-pro"
    assert request_body["query"] == "hello"
    assert request_body["documents"] == ["hello", "world"]
    assert request_body["top_n"] == 2


@pytest.mark.asyncio
async def test_cohere_rerank_success_callback_receives_cost() -> None:
    class CostCallback(CustomLogger):
        def __init__(self) -> None:
            self.response_cost: float | None = None
            self.result_index: int | None = None
            super().__init__()

        async def async_log_success_event(
            self,
            kwargs: dict[str, object],
            response_obj: RerankResponse,
            start_time: datetime,
            end_time: datetime,
        ) -> None:
            response_cost: Final = kwargs["response_cost"]
            assert isinstance(response_cost, (int, float))
            self.response_cost = float(response_cost)
            self.result_index = response_obj.results[0]["index"]

    callback: Final = CostCallback()
    logging_obj: Final = Logging(
        model="cohere/rerank-v4.0-pro",
        messages=[],
        stream=False,
        call_type="rerank",
        start_time=datetime(2025, 1, 1),
        litellm_call_id="rerank-test-call",
        function_id="rerank-test-function",
        dynamic_async_success_callbacks=[callback],
        kwargs={"custom_llm_provider": "cohere"},
    )
    response: Final = RerankResponse.model_validate(COHERE_RERANK_RESPONSE)
    logging_obj.model_call_details["response_cost"] = 0.01

    await logging_obj.async_success_handler(
        result=response,
        start_time=datetime(2025, 1, 1),
        end_time=datetime(2025, 1, 1),
    )

    assert callback.response_cost is not None
    assert callback.response_cost > 0
    assert callback.result_index == 0
