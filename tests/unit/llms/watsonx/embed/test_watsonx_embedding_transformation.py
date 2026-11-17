

import json
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm

from litellm.llms.watsonx.embed.transformation import IBMWatsonXEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


class TestIBMWatsonXEmbeddingConfig:
    def _config(self) -> IBMWatsonXEmbeddingConfig:
        return IBMWatsonXEmbeddingConfig()

    def test_transform_embedding_request_wraps_string_input(self):
        cfg = self._config()
        request = cfg.transform_embedding_request(
            model="intfloat/multilingual-e5-large",
            input="cdsc",
            optional_params={"project_id": "test-project-id"},
            headers={},
        )

        assert request["inputs"] == ["cdsc"]
        assert request["project_id"] == "test-project-id"

    def test_transform_embedding_request_preserves_list_input(self):
        cfg = self._config()
        request = cfg.transform_embedding_request(
            model="intfloat/multilingual-e5-large",
            input=["first", "second"],
            optional_params={"project_id": "test-project-id"},
            headers={},
        )

        assert request["inputs"] == ["first", "second"]

    def test_transform_embedding_request_rejects_token_input(self):
        cfg = self._config()

        with pytest.raises(ValueError, match="string or list of strings"):
            cfg.transform_embedding_request(
                model="intfloat/multilingual-e5-large",
                input=[[1, 2, 3]],
                optional_params={"project_id": "test-project-id"},
                headers={},
            )


def _transform(payload: object) -> EmbeddingResponse:
    return IBMWatsonXEmbeddingConfig().transform_embedding_response(
        model="ibm/slate-125m-english-rtrvr",
        raw_response=httpx.Response(200, json=payload),
        model_response=EmbeddingResponse(),
        logging_obj=MagicMock(),
        api_key="test-key",
        request_data={},
        optional_params={},
        litellm_params={},
    )


def test_transform_embedding_response_numbers_each_result_and_bills_the_input_tokens():
    response = _transform(
        {
            "model_id": "ibm/slate-125m-english-rtrvr",
            "results": [{"embedding": [0.1, 2], "input": "ignored"}, {"embedding": [0.5]}],
            "input_token_count": 7,
        }
    )

    assert response.object == "list"
    assert response.data == [
        {"object": "embedding", "index": 0, "embedding": [0.1, 2]},
        {"object": "embedding", "index": 1, "embedding": [0.5]},
    ]
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (7, 0, 7)


@pytest.mark.parametrize(
    ("payload", "expected_tokens"),
    [
        ({"input_token_count": None}, 0),
        ({"results": []}, 0),
        ({}, 0),
    ],
)
def test_transform_embedding_response_defaults_missing_results_and_token_count(
    payload: dict[str, object], expected_tokens: int
):
    response = _transform(payload)

    assert response.data == []
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (expected_tokens, expected_tokens)


@pytest.mark.parametrize(
    "payload",
    [
        {"results": None},
        {"results": 7},
        {"results": ["not an object"]},
        {"results": [{"embedding": [0.1]}, None]},
        {"input_token_count": 1.5},
        {"input_token_count": "many"},
    ],
)
def test_transform_embedding_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)


@pytest.mark.parametrize("payload", [["not", "an", "object"], "not an object"])
def test_transform_embedding_response_non_object_body_raises_attribute_error(payload: object):
    with pytest.raises(AttributeError):
        _transform(payload)


def test_transform_embedding_response_result_without_embedding_raises_key_error():
    with pytest.raises(KeyError, match="embedding"):
        _transform({"results": [{"input": "x"}]})


@pytest.mark.parametrize(
    "payload",
    [{"results": ["leaked payload text"]}, {"results": {"leaked payload text": 1}}],
)
def test_transform_embedding_response_shape_errors_do_not_echo_the_payload(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert "leaked payload text" not in str(exc_info.value)


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_watsonx_embedding_sends_the_bearer_token_to_the_regional_endpoint(
    sync_mode: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("WATSONX_API_KEY", "mock-api-key")
    monkeypatch.setenv("WATSONX_TOKEN", "mock-watsonx-token")
    monkeypatch.setenv("WATSONX_API_BASE", "https://us-south.ml.cloud.ibm.com")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "mock-project-id")
    route: Final = respx_mock.post(url__startswith="https://us-south.ml.cloud.ibm.com/ml/v1/text/embeddings").respond(
        json={
            "model_id": "ibm/slate-30m-english-rtrvr",
            "created_at": "2024-01-01T00:00:00.00Z",
            "results": [{"embedding": [0.0, 1.0]}],
            "input_token_count": 8,
        }
    )
    request: Final = {"model": "watsonx/ibm/slate-30m-english-rtrvr", "input": ["good morning from litellm"]}

    response: Final = litellm.embedding(**request) if sync_mode else await litellm.aembedding(**request)

    assert route.call_count == 1
    assert route.calls.last.request.headers["Authorization"] == "Bearer mock-watsonx-token"
    assert json.loads(route.calls.last.request.content)["inputs"] == ["good morning from litellm"]
    assert response.data == [{"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}]
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (8, 8)
