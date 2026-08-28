import pytest
import litellm

def test_bedrock_cohere_english_v3_supported_params():
    """Verify cohere.embed-english-v3 routes to BedrockCohereEmbeddingConfig and supports encoding_format."""
    optional_params = litellm.utils.get_optional_params_embeddings(
        model="cohere.embed-english-v3",
        custom_llm_provider="bedrock",
        encoding_format="float",
    )
    assert "encoding_format" in optional_params or optional_params.get("encoding_format") == "float"

