import pytest

from litellm.llms.cohere.rerank_v2.transformation import CohereRerankV2Config


@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        ("https://api.cohere.ai", "https://api.cohere.ai/v2/rerank"),
        ("https://api.cohere.ai/v2", "https://api.cohere.ai/v2/rerank"),
        ("https://api.cohere.ai/v2/rerank/", "https://api.cohere.ai/v2/rerank"),
    ],
)
def test_get_complete_url_normalizes_cohere_rerank_v2_api_base(
    api_base: str, expected_url: str
) -> None:
    assert (
        CohereRerankV2Config().get_complete_url(api_base=api_base, model="rerank-v3.5")
        == expected_url
    )
