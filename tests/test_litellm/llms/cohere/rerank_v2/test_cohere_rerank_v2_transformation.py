import pytest

from litellm.llms.cohere.rerank_v2.transformation import CohereRerankV2Config


@pytest.mark.parametrize(
    "api_base, expected",
    [
        # versioned root must not duplicate the version segment (#31167)
        ("https://api.cohere.ai/v2", "https://api.cohere.ai/v2/rerank"),
        ("https://api.cohere.ai/v2/", "https://api.cohere.ai/v2/rerank"),
        # bare host gets the full v2 rerank path
        ("https://api.cohere.ai", "https://api.cohere.ai/v2/rerank"),
        # already-complete url is left untouched
        ("https://api.cohere.ai/v2/rerank", "https://api.cohere.ai/v2/rerank"),
        # self-hosted gateway exposing a versioned root
        ("https://gateway.internal/cohere/v2", "https://gateway.internal/cohere/v2/rerank"),
        # no api_base falls back to the public endpoint
        (None, "https://api.cohere.ai/v2/rerank"),
    ],
)
def test_get_complete_url_does_not_duplicate_version(api_base, expected):
    config = CohereRerankV2Config()
    assert config.get_complete_url(api_base=api_base, model="rerank-v3.5") == expected
