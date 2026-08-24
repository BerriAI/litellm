"""Bedrock never echoes document text, so the transformer must back-fill it."""

import pytest

from litellm.llms.bedrock.rerank.transformation import BedrockRerankConfig
from litellm.types.rerank import RerankRequest

BEDROCK_RESPONSE = {
    "results": [
        {"index": 1, "relevanceScore": 0.95},
        {"index": 0, "relevanceScore": 0.42},
    ]
}

DOCUMENTS = ["the first passage", "the second passage"]


def _request(documents=None, return_documents=None):
    return RerankRequest(
        model="bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        query="which passage?",
        documents=DOCUMENTS if documents is None else documents,
        return_documents=return_documents,
    )


class TestBedrockRerankReturnDocuments:
    def setup_method(self):
        self.config = BedrockRerankConfig()

    @pytest.mark.parametrize("return_documents", [True, None])
    def test_back_fills_document_text_by_index(self, return_documents):
        """`None` is the Cohere-compatible default and back-fills like `True`."""
        response = self.config._transform_response(BEDROCK_RESPONSE, _request(return_documents=return_documents))

        # Ranked order is preserved, and each result carries its source text.
        assert response.results[0]["index"] == 1
        assert response.results[0]["document"] == {"text": "the second passage"}
        assert response.results[1]["index"] == 0
        assert response.results[1]["document"] == {"text": "the first passage"}

    def test_omits_documents_when_caller_opts_out(self):
        response = self.config._transform_response(BEDROCK_RESPONSE, _request(return_documents=False))

        assert all("document" not in result for result in response.results)

    def test_accepts_dict_documents(self):
        response = self.config._transform_response(
            BEDROCK_RESPONSE,
            _request(documents=[{"text": "first as dict"}, {"text": "second as dict"}]),
        )

        assert response.results[0]["document"] == {"text": "second as dict"}

    def test_no_request_data_leaves_response_untouched(self):
        """Older callers pass only the response; nothing to back-fill from."""
        response = self.config._transform_response(BEDROCK_RESPONSE)

        assert all("document" not in result for result in response.results)

    def test_out_of_range_index_is_skipped_not_raised(self):
        response = self.config._transform_response({"results": [{"index": 7, "relevanceScore": 0.1}]}, _request())

        assert response.results[0]["index"] == 7
        assert "document" not in response.results[0]

    def test_json_documents_yield_no_document_and_do_not_raise(self):
        """
        A dict document with no `text` key goes to Bedrock as a `jsonDocument`
        (`_transform_sources`), and `RerankResponseDocument` carries only
        `text`, so there is nowhere to put it. Pinned as a known boundary: no
        back-fill, but no crash either. Raised by @kimnamu on #38006.
        """
        response = self.config._transform_response(
            BEDROCK_RESPONSE,
            _request(documents=[{"title": "zero", "body": "b0"}, {"title": "one", "body": "b1"}]),
        )

        assert all("document" not in result for result in response.results)
        assert [r["index"] for r in response.results] == [1, 0]

    def test_non_integer_index_is_tolerated(self):
        """`index` is coerced by pydantic upstream; do not raise on it here."""
        response = self.config._transform_response({"results": [{"index": "2", "relevanceScore": 0.9}]}, _request())

        assert response.results[0]["relevance_score"] == 0.9

    def test_relevance_scores_are_preserved(self):
        response = self.config._transform_response(BEDROCK_RESPONSE, _request())

        assert [r["relevance_score"] for r in response.results] == [0.95, 0.42]
