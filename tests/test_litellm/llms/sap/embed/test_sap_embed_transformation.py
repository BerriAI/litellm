import pytest

from litellm.llms.sap.embed.transformation import GenAIHubEmbeddingConfig

@pytest.mark.asyncio
async def test_basic_config_transform():
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
    body = GenAIHubEmbeddingConfig().transform_embedding_request(
        model="text-embedding-3-small",
        input="Hi",
        optional_params={},
        headers={}
    )
    assert body == expected_dict