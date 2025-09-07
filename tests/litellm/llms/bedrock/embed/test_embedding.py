
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import patch

import pytest

from litellm.types.utils import Embedding
from litellm.main import bedrock_embedding, embedding, EmbeddingResponse, Usage


_mock_model_id = "arn:aws:bedrock:us-east-1:123412341234:application-inference-profile/abc123123"
_mock_app_ip_url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123412341234%3Aapplication-inference-profile%2Fabc123123/invoke"


def _get_mock_embedding_response(model: str) -> EmbeddingResponse:
    return EmbeddingResponse(
        model=model,
        usage=Usage(
            prompt_tokens=1,
            completion_tokens=0,
            total_tokens=1,
            completion_tokens_details=None,
            prompt_tokens_details=None
        ),
        data=[
            Embedding(
                embedding=[-0.671875, 0.291015625, -0.1826171875, 0.8828125],
                index=0,
                object="embedding"
            )
        ]
    )


@pytest.mark.parametrize(
    "model",
    [
        "amazon.titan-embed-text-v1",
        "amazon.titan-embed-text-v2:0"
    ]
)
def test_bedrock_embedding_titan_app_profile(model: str):
    with patch.object(bedrock_embedding, '_single_func_embeddings') as mock_method:
        mock_method.return_value = _get_mock_embedding_response(model=model)
        resp = embedding(
            custom_llm_provider="bedrock",
            model=model,
            model_id=_mock_model_id,
            input=["tester"],
            aws_region_name="us-east-1",
            aws_access_key_id="mockaws_access_key_id",
            aws_secret_access_key="mockaws_secret_access_key"
        )
        assert mock_method.call_args.kwargs['endpoint_url'] == _mock_app_ip_url
        

@pytest.mark.parametrize(
    "model",
    [
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3"
    ]
)
def test_bedrock_embedding_cohere_app_profile(model: str):
    with patch("litellm.llms.bedrock.embed.embedding.cohere_embedding") as mock_cohere_embedding:
        mock_cohere_embedding.return_value = _get_mock_embedding_response(model=model)
        resp = embedding(
            custom_llm_provider="bedrock",
            model=model,
            model_id=_mock_model_id,
            input=["tester"],
            aws_region_name="us-east-1",
            aws_access_key_id="mockaws_access_key_id",
            aws_secret_access_key="mockaws_secret_access_key"
        )
        assert mock_cohere_embedding.call_args.kwargs['complete_api_base'] == _mock_app_ip_url
        
