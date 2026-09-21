"""
Regression tests for https://github.com/BerriAI/litellm/issues/34165

The native /v1/ranking endpoint accepts only model, query, passages, and
truncate. Two defects are covered here:
1. structured image documents were json.dumps-stringified into text passages
2. Cohere top_n was mapped to top_k, which /v1/ranking rejects with a 400
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.nvidia_nim.rerank.ranking_transformation import (
    NvidiaNimRankingConfig,
)
from litellm.llms.nvidia_nim.rerank.transformation import NvidiaNimRerankConfig
from litellm.types.rerank import RerankResponse

RANKING_MODEL = "ranking/nvidia/llama-nemotron-rerank-vl-1b-v2"
IMAGE_DOC = {"image": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
TEXT_DOC = {"text": "a plain text passage"}
MIXED_DOC = {"text": "caption for the image", "image": "data:image/png;base64,iVBORw0KGgo="}


def _build_ranking_request(documents, top_n=None, non_default_params=None):
    """Run map_cohere_rerank_params + transform_rerank_request for /v1/ranking."""
    config = NvidiaNimRankingConfig()
    optional_params = config.map_cohere_rerank_params(
        non_default_params=non_default_params,
        model=RANKING_MODEL,
        drop_params=False,
        query="which passage shows a cat?",
        documents=documents,
        top_n=top_n,
    )
    request_data = config.transform_rerank_request(
        model=RANKING_MODEL,
        optional_rerank_params=optional_params,
        headers={},
    )
    return config, request_data


def _build_ranking_response(config, request_data, rankings):
    """Run transform_rerank_response against a mocked raw ranking response."""
    raw_response = MagicMock()
    raw_response.json.return_value = {"rankings": rankings}
    return config.transform_rerank_response(
        model=RANKING_MODEL,
        raw_response=raw_response,
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
        request_data=request_data,
    )


class TestNvidiaNimRankingRequestTransform:
    def test_string_documents(self):
        _, request_data = _build_ranking_request(["passage one", "passage two"])
        assert request_data["passages"] == [
            {"text": "passage one"},
            {"text": "passage two"},
        ]

    def test_text_object_documents(self):
        _, request_data = _build_ranking_request([TEXT_DOC])
        assert request_data["passages"] == [TEXT_DOC]

    def test_image_object_documents_are_preserved(self):
        _, request_data = _build_ranking_request([IMAGE_DOC, TEXT_DOC])
        assert request_data["passages"] == [IMAGE_DOC, TEXT_DOC]

    def test_mixed_text_image_documents_are_preserved(self):
        _, request_data = _build_ranking_request([MIXED_DOC])
        assert request_data["passages"] == [MIXED_DOC]

    def test_unsupported_dict_documents_are_stringified(self):
        doc = {"title": "no supported fields here"}
        _, request_data = _build_ranking_request([doc])
        assert request_data["passages"] == [{"text": json.dumps(doc)}]

    def test_top_n_is_not_sent_to_the_ranking_endpoint(self):
        _, request_data = _build_ranking_request(["a", "b"], top_n=1)
        assert "top_k" not in request_data
        assert "top_n" not in request_data

    def test_provider_specific_top_k_is_stripped(self):
        _, request_data = _build_ranking_request(["a", "b"], non_default_params={"top_k": 2})
        assert "top_k" not in request_data

    @pytest.mark.parametrize("invalid_top_n", [0, -1, 1.5, "2", True])
    def test_invalid_top_n_raises_value_error(self, invalid_top_n):
        with pytest.raises(ValueError, match="top_n"):
            _build_ranking_request(["a", "b"], top_n=invalid_top_n)


class TestNvidiaNimRankingResponseTransform:
    RANKINGS = [
        {"index": 0, "logit": 0.95},
        {"index": 1, "logit": 0.75},
        {"index": 2, "logit": 0.55},
    ]

    def test_top_n_one_truncates_to_best_result(self):
        config, request_data = _build_ranking_request(["a", "b", "c"], top_n=1)
        response = _build_ranking_response(config, request_data, self.RANKINGS)
        assert len(response.results) == 1
        assert response.results[0]["index"] == 0

    def test_top_n_equal_to_document_count_keeps_all_results(self):
        config, request_data = _build_ranking_request(["a", "b", "c"], top_n=3)
        response = _build_ranking_response(config, request_data, self.RANKINGS)
        assert len(response.results) == 3

    def test_top_n_greater_than_document_count_keeps_all_results(self):
        config, request_data = _build_ranking_request(["a", "b", "c"], top_n=10)
        response = _build_ranking_response(config, request_data, self.RANKINGS)
        assert len(response.results) == 3

    def test_top_n_truncation_keeps_most_relevant_results(self):
        unsorted_rankings = [
            {"index": 0, "logit": 0.10},
            {"index": 1, "logit": 0.90},
            {"index": 2, "logit": 0.50},
        ]
        config, request_data = _build_ranking_request(["a", "b", "c"], top_n=2)
        response = _build_ranking_response(config, request_data, unsorted_rankings)
        assert [result["index"] for result in response.results] == [1, 2]

    def test_image_only_passages_do_not_break_document_echo(self):
        config, request_data = _build_ranking_request([IMAGE_DOC, TEXT_DOC])
        response = _build_ranking_response(config, request_data, self.RANKINGS[:2])
        assert len(response.results) == 2
        # Image-only passage has no text to echo back
        assert "document" not in response.results[0]
        assert response.results[1]["document"] == {"text": TEXT_DOC["text"]}


@pytest.mark.asyncio()
async def test_nvidia_nim_ranking_endpoint_image_documents_and_top_n():
    """
    End-to-end (mocked transport): image documents reach /v1/ranking intact
    and top_n is applied client-side instead of being sent as top_k.
    """
    mock_response = AsyncMock()

    def return_val():
        return {
            "rankings": [
                {"index": 0, "logit": 0.95},
                {"index": 1, "logit": 0.75},
            ],
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.arerank(
            model="nvidia_nim/ranking/nvidia/llama-nemotron-rerank-vl-1b-v2",
            query="which passage shows a cat?",
            documents=[IMAGE_DOC, TEXT_DOC],
            top_n=1,
            api_key="fake-api-key",
        )

        mock_post.assert_called_once()
        request_data = json.loads(mock_post.call_args.kwargs["data"])

        assert mock_post.call_args.kwargs["url"] == "https://ai.api.nvidia.com/v1/ranking"
        # Image passage preserved as-is, not stringified into text
        assert request_data["passages"] == [IMAGE_DOC, TEXT_DOC]
        # Neither top_k nor top_n is sent to the native endpoint
        assert "top_k" not in request_data
        assert "top_n" not in request_data
        # top_n applied client-side on the converted response
        assert len(response.results) == 1
        assert response.results[0]["index"] == 0


class TestNvidiaNimRetrievalRerankRequestTransform:
    """
    The default /v1/retrieval/{model}/reranking route keeps its existing
    contract: top_n still maps to top_k, and structured documents keep the
    prior text-only passage behavior.
    """

    def _build_request(self, documents, top_n=None):
        config = NvidiaNimRerankConfig()
        optional_params = config.map_cohere_rerank_params(
            non_default_params=None,
            model="nvidia/llama-3_2-nv-rerankqa-1b-v2",
            drop_params=False,
            query="which passage shows a cat?",
            documents=documents,
            top_n=top_n,
        )
        return config.transform_rerank_request(
            model="nvidia/llama-3_2-nv-rerankqa-1b-v2",
            optional_rerank_params=optional_params,
            headers={},
        )

    def test_top_n_still_maps_to_top_k(self):
        request_data = self._build_request(["a", "b"], top_n=1)
        assert request_data["top_k"] == 1
        assert "top_n" not in request_data

    def test_string_documents_unchanged(self):
        request_data = self._build_request(["passage one", "passage two"])
        assert request_data["passages"] == [
            {"text": "passage one"},
            {"text": "passage two"},
        ]

    def test_text_object_documents_unchanged(self):
        request_data = self._build_request([TEXT_DOC])
        assert request_data["passages"] == [TEXT_DOC]

    def test_image_object_documents_keep_retrieval_behavior(self):
        request_data = self._build_request([IMAGE_DOC, TEXT_DOC])
        assert request_data["passages"] == [
            {"text": json.dumps(IMAGE_DOC)},
            TEXT_DOC,
        ]

    def test_mixed_text_image_documents_keep_text_only(self):
        request_data = self._build_request([MIXED_DOC])
        assert request_data["passages"] == [{"text": MIXED_DOC["text"]}]

    def test_unsupported_dict_documents_are_stringified(self):
        doc = {"title": "no supported fields here"}
        request_data = self._build_request([doc])
        assert request_data["passages"] == [{"text": json.dumps(doc)}]
