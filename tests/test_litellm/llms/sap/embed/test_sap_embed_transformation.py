from unittest.mock import patch, PropertyMock

import pytest

from litellm.llms.sap.embed.transformation import GenAIHubEmbeddingConfig

@pytest.fixture
def fake_token_creator():
    return lambda: "Bearer FAKE_TOKEN", "https://api.ai.moke-sap.com", "fake-group"


@pytest.fixture
def fake_deployment_url():
    return "https://api.ai.moke-sap.com/v2/inference/deployments/mokeid"

def test_basic_config_transform(fake_token_creator, fake_deployment_url):
    expected_dict = {
        'config': {
            'modules': {
                'embeddings': {
                    'model': {
                        'name': 'text-embedding-3-small',
                        'version': 'latest',
                        'params': {}
                    }
                }
            }
        },
        'input': {
            'text': 'Hi',
            'type': 'text'
        }
    }
    with patch(
            "litellm.llms.sap.embed.transformation.GenAIHubEmbeddingConfig.deployment_url",
            new_callable=PropertyMock,
            return_value=fake_deployment_url,
    ), patch(
        "litellm.llms.sap.embed.transformation.get_token_creator",
        return_value=fake_token_creator,
    ):
        body = GenAIHubEmbeddingConfig().transform_embedding_request(
            model="text-embedding-3-small",
            input="Hi",
            optional_params={},
            headers={}
        )
        assert body == expected_dict