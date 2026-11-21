import json

import httpx
import pytest
import respx

import litellm

URL = "https://api.butlercompute.com/v1/rerank"
DOCUMENTS = ["Paris is the capital of France.", "Berlin is in Germany.", "The Eiffel Tower is in Paris."]
BUTLER_RERANK_BODY = {
    "id": "req_01M6W7V58V0009YTSPV3TVKFAN",
    "model": "cicero",
    "results": [
        {"index": 0, "relevance_score": 0.9909759013641205, "document": {"text": DOCUMENTS[0]}},
        {"index": 2, "relevance_score": 0.5957479814217147, "document": {"text": DOCUMENTS[2]}},
    ],
    "usage": {"total_tokens": 464, "input_tokens": 461, "output_tokens": 3, "cached_tokens": 192, "cost": 7.3e-6},
}


def _rerank(**kwargs):
    return litellm.rerank(model="butler/cicero", query="What is the capital of France?", documents=DOCUMENTS, **kwargs)


@pytest.fixture(autouse=True)
def _butler_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BUTLER_API_BASE", raising=False)
    monkeypatch.delenv("BUTLER_API_KEY", raising=False)


def test_butler_rerank_posts_the_cohere_v2_body_and_maps_the_response(respx_mock: respx.MockRouter):
    route = respx_mock.post(URL)
    route.return_value = httpx.Response(200, json=BUTLER_RERANK_BODY)

    response = _rerank(api_key="b-key", top_n=2, return_documents=True, rank_fields=["title"], max_tokens_per_doc=128)

    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer b-key"
    assert json.loads(request.content) == {
        "model": "cicero",
        "query": "What is the capital of France?",
        "documents": DOCUMENTS,
        "top_n": 2,
        "return_documents": True,
    }
    assert [r["index"] for r in response.results] == [0, 2]
    assert response.results[0]["document"]["text"] == DOCUMENTS[0]
    assert response.id == BUTLER_RERANK_BODY["id"]
    assert response.meta["billed_units"]["total_tokens"] == 464


def test_butler_rerank_reads_the_key_and_base_from_the_environment(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("BUTLER_API_KEY", "env-key")
    monkeypatch.setenv("BUTLER_API_BASE", "https://butler.example/v1")
    route = respx_mock.post("https://butler.example/v1/rerank")
    route.return_value = httpx.Response(200, json=BUTLER_RERANK_BODY)

    _rerank()

    assert route.calls[0].request.headers["authorization"] == "Bearer env-key"


def test_butler_rerank_without_a_key_names_the_env_var():
    with pytest.raises(litellm.APIConnectionError, match="BUTLER_API_KEY"):
        _rerank()


def test_butler_rerank_error_carries_the_api_message(respx_mock: respx.MockRouter):
    error = {"error": {"message": "invalid request: `rank_fields` is not supported", "type": "invalid_request_error"}}
    respx_mock.post(URL).return_value = httpx.Response(400, json=error)

    with pytest.raises(litellm.BadRequestError, match="rank_fields"):
        _rerank(api_key="b-key")
