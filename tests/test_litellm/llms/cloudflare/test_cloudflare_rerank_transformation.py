import json
import math

import httpx
import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.cloudflare.common_utils import CloudflareError
from litellm.llms.cloudflare.rerank.transformation import CloudflareRerankConfig, sigmoid
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.rerank import RerankResponse

MODEL = "@cf/baai/bge-reranker-base"
ACCOUNT_BASE = "https://api.cloudflare.com/client/v4/accounts/acct"
RUN_URL = f"{ACCOUNT_BASE}/ai/run/{MODEL}"


@pytest.fixture
def config():
    return CloudflareRerankConfig()


@pytest.fixture
def logging_obj():
    return Logging(
        model=MODEL,
        messages=[],
        stream=False,
        call_type="rerank",
        start_time=None,
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )


def cloudflare_response(scores, success=True, status_code=200):
    return httpx.Response(
        status_code,
        json={"result": {"response": scores}, "success": success, "errors": [], "messages": []},
        request=httpx.Request("POST", RUN_URL),
    )


def test_url_defaults_to_account_run_path(config, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")

    assert config.get_complete_url(api_base=None, model=MODEL) == RUN_URL


def test_url_without_account_id_raises(config, monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)

    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        config.get_complete_url(api_base=None, model=MODEL)


@pytest.mark.parametrize(
    "api_base",
    [
        ACCOUNT_BASE,
        f"{ACCOUNT_BASE}/",
        f"{ACCOUNT_BASE}/ai/v1",
        f"{ACCOUNT_BASE}/ai/v1/",
        f"{ACCOUNT_BASE}/ai/run",
        f"{ACCOUNT_BASE}/ai/run/",
        RUN_URL,
    ],
)
def test_url_normalizes_every_accepted_base_form(config, api_base):
    """The OpenAI-compat base is the one users already have, but rerank only lives on /ai/run."""
    assert config.get_complete_url(api_base=api_base, model=MODEL) == RUN_URL


def test_validate_environment_sets_bearer_token(config, monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)

    headers = config.validate_environment(headers={}, model=MODEL, api_key="cf-key")

    assert headers["Authorization"] == "Bearer cf-key"
    assert headers["content-type"] == "application/json"


def test_validate_environment_falls_back_to_env_key(config, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "env-key")

    headers = config.validate_environment(headers={}, model=MODEL, api_key=None)

    assert headers["Authorization"] == "Bearer env-key"


def test_validate_environment_lets_caller_headers_win(config):
    headers = config.validate_environment(headers={"Authorization": "Bearer caller-key"}, model=MODEL, api_key="cf-key")

    assert headers["Authorization"] == "Bearer caller-key"


def test_validate_environment_without_key_raises(config, monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Cloudflare API key"):
        config.validate_environment(headers={}, model=MODEL, api_key=None)


def test_request_maps_documents_to_contexts_and_top_n_to_top_k(config):
    body = config.transform_rerank_request(
        model=MODEL,
        optional_rerank_params={
            "query": "which is cooler",
            "documents": ["a cyberpunk lizzard", {"text": "a cyberpunk cat"}],
            "top_n": 2,
        },
        headers={},
    )

    assert body == {
        "query": "which is cooler",
        "contexts": [{"text": "a cyberpunk lizzard"}, {"text": "a cyberpunk cat"}],
        "top_k": 2,
    }


def test_request_omits_top_k_when_top_n_is_unset(config):
    body = config.transform_rerank_request(
        model=MODEL,
        optional_rerank_params={"query": "q", "documents": ["a"], "top_n": None},
        headers={},
    )

    assert "top_k" not in body


@pytest.mark.parametrize(
    "params, missing_field",
    [
        ({"documents": ["a"]}, "query"),
        ({"query": "q"}, "documents"),
    ],
)
def test_request_requires_query_and_documents(config, params, missing_field):
    with pytest.raises(ValueError, match="Invalid Cloudflare rerank request") as failure:
        config.transform_rerank_request(model=MODEL, optional_rerank_params=params, headers={})

    assert missing_field in str(failure.value)
    assert "Field required" in str(failure.value)


def test_request_rejects_documents_without_text(config):
    with pytest.raises(ValueError, match="'text' field"):
        config.transform_rerank_request(
            model=MODEL,
            optional_rerank_params={"query": "q", "documents": [{"body": "no text key"}]},
            headers={},
        )


def test_map_params_keeps_the_three_supported_params(config):
    mapped = config.map_cohere_rerank_params(
        non_default_params={},
        model=MODEL,
        drop_params=False,
        query="q",
        documents=["a", "b"],
        top_n=1,
    )

    assert mapped == {"query": "q", "documents": ["a", "b"], "top_n": 1}


@pytest.mark.parametrize(
    "unsupported",
    [
        {"rank_fields": ["title"]},
        {"max_chunks_per_doc": 4},
        {"max_tokens_per_doc": 128},
        {"instruction": "rank these"},
    ],
)
def test_map_params_rejects_params_cloudflare_cannot_honour(config, unsupported):
    with pytest.raises(UnsupportedParamsError, match=next(iter(unsupported))):
        config.map_cohere_rerank_params(
            non_default_params={},
            model=MODEL,
            drop_params=False,
            query="q",
            documents=["a"],
            **unsupported,
        )


def test_map_params_drops_unsupported_params_when_asked(config):
    mapped = config.map_cohere_rerank_params(
        non_default_params={},
        model=MODEL,
        drop_params=True,
        query="q",
        documents=["a"],
        rank_fields=["title"],
        instruction="rank these",
    )

    assert mapped == {"query": "q", "documents": ["a"], "top_n": None}


def test_response_sigmoids_logits_and_preserves_cloudflare_ordering(config, logging_obj):
    request_data = {"query": "q", "contexts": [{"text": "bananas"}, {"text": "cars"}, {"text": "a gateway"}]}

    result = config.transform_rerank_response(
        model=MODEL,
        raw_response=cloudflare_response([{"id": 2, "score": 5.1}, {"id": 0, "score": -1.0}]),
        model_response=RerankResponse(),
        logging_obj=logging_obj,
        request_data=request_data,
    )

    assert [r["index"] for r in result.results] == [2, 0]
    assert result.results[0]["relevance_score"] == pytest.approx(1 / (1 + math.exp(-5.1)))
    assert result.results[1]["relevance_score"] == pytest.approx(1 / (1 + math.exp(1.0)))
    assert all(0.0 <= r["relevance_score"] <= 1.0 for r in result.results)
    assert result.results[0]["document"] == {"text": "a gateway"}
    assert result.results[1]["document"] == {"text": "bananas"}


def test_response_omits_document_when_index_is_out_of_range(config, logging_obj):
    result = config.transform_rerank_response(
        model=MODEL,
        raw_response=cloudflare_response([{"id": 7, "score": 1.0}]),
        model_response=RerankResponse(),
        logging_obj=logging_obj,
        request_data={"contexts": [{"text": "only one"}]},
    )

    assert "document" not in result.results[0]


def test_response_rejects_a_failed_envelope(config, logging_obj):
    raw = httpx.Response(
        200,
        json={"result": None, "success": False, "errors": [{"code": 7002, "message": "no route"}]},
        request=httpx.Request("POST", RUN_URL),
    )

    with pytest.raises(CloudflareError, match="Cloudflare rerank request failed") as failure:
        config.transform_rerank_response(
            model=MODEL, raw_response=raw, model_response=RerankResponse(), logging_obj=logging_obj
        )

    assert "no route" in failure.value.message


def test_response_rejects_missing_results(config, logging_obj):
    raw = httpx.Response(200, json={"result": {}, "success": True}, request=httpx.Request("POST", RUN_URL))

    with pytest.raises(CloudflareError, match="No rerank results"):
        config.transform_rerank_response(
            model=MODEL, raw_response=raw, model_response=RerankResponse(), logging_obj=logging_obj
        )


def test_response_rejects_non_json_body(config, logging_obj):
    raw = httpx.Response(502, text="<html>bad gateway</html>", request=httpx.Request("POST", RUN_URL))

    with pytest.raises(CloudflareError, match="Error parsing"):
        config.transform_rerank_response(
            model=MODEL, raw_response=raw, model_response=RerankResponse(), logging_obj=logging_obj
        )


def test_response_rejects_a_score_entry_missing_fields(config, logging_obj):
    """An entry without a score cannot be ranked, so it is a malformed body rather than a gap."""
    with pytest.raises(CloudflareError, match="Error parsing"):
        config.transform_rerank_response(
            model=MODEL,
            raw_response=cloudflare_response([{"id": 0}]),
            model_response=RerankResponse(),
            logging_obj=logging_obj,
        )


@pytest.mark.parametrize("logit", [-800.0, -40.0, 0.0, 40.0, 800.0])
def test_sigmoid_stays_in_range_without_overflowing(logit):
    """math.exp(-logit) overflows well before these magnitudes, so the branchless form would raise."""
    score = sigmoid(logit)

    assert 0.0 <= score <= 1.0


def test_sigmoid_is_monotonic_so_ranking_survives_normalisation():
    logits = [-9.0, -1.5, 0.0, 1.5, 9.0]

    scores = [sigmoid(logit) for logit in logits]

    assert scores == sorted(scores)
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_rerank_routes_cloudflare_models_to_the_workers_ai_run_endpoint(monkeypatch):
    """Covers provider resolution, the rerank_api dispatch branch and the lazy config export."""
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return cloudflare_response([{"id": 2, "score": 5.1}, {"id": 0, "score": -1.0}])

    client = HTTPHandler()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))

    response = litellm.rerank(
        model="cloudflare/@cf/baai/bge-reranker-base",
        query="what is litellm",
        documents=["bananas", "cars", "an LLM gateway"],
        top_n=2,
        client=client,
    )

    assert captured["url"] == RUN_URL
    assert captured["auth"] == "Bearer cf-key"
    assert captured["body"] == {
        "query": "what is litellm",
        "contexts": [{"text": "bananas"}, {"text": "cars"}, {"text": "an LLM gateway"}],
        "top_k": 2,
    }
    assert [r["index"] for r in response.results] == [2, 0]
    assert response.results[0]["relevance_score"] == pytest.approx(1 / (1 + math.exp(-5.1)))


def test_rerank_honours_an_openai_compatible_api_base(monkeypatch):
    """Deployments share one Cloudflare base URL, and for chat/embeddings that is the /ai/v1 form."""
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return cloudflare_response([{"id": 0, "score": 1.0}])

    client = HTTPHandler()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))

    litellm.rerank(
        model="cloudflare/@cf/baai/bge-reranker-base",
        query="q",
        documents=["a"],
        api_base=f"{ACCOUNT_BASE}/ai/v1",
        client=client,
    )

    assert captured["url"] == RUN_URL
