from unittest.mock import patch, PropertyMock

import pytest

from litellm.llms.sap.embed.transformation import GenAIHubEmbeddingConfig


@pytest.fixture
def fake_token_creator():
    return (lambda: "Bearer FAKE_TOKEN", "https://api.ai.moke-sap.com", "fake-group")


@pytest.fixture
def fake_deployment_url():
    return "https://api.ai.moke-sap.com/v2/inference/deployments/mokeid"


def test_basic_config_transform(fake_token_creator, fake_deployment_url):
    expected_dict = {
        "config": {
            "modules": {
                "embeddings": {
                    "model": {
                        "name": "text-embedding-3-small",
                        "version": "latest",
                        "params": {},
                    }
                }
            }
        },
        "input": {"text": "Hi"},
    }
    with (
        patch(
            "litellm.llms.sap.embed.transformation.GenAIHubEmbeddingConfig.deployment_url",
            new_callable=PropertyMock,
            return_value=fake_deployment_url,
        ),
        patch(
            "litellm.llms.sap.embed.transformation.get_token_creator",
            return_value=fake_token_creator,
        ),
    ):
        body = GenAIHubEmbeddingConfig().transform_embedding_request(
            model="text-embedding-3-small", input="Hi", optional_params={}, headers={}
        )
        assert body == expected_dict


def test_model_params(fake_token_creator, fake_deployment_url):
    with (
        patch(
            "litellm.llms.sap.embed.transformation.GenAIHubEmbeddingConfig.deployment_url",
            new_callable=PropertyMock,
            return_value=fake_deployment_url,
        ),
        patch(
            "litellm.llms.sap.embed.transformation.get_token_creator",
            return_value=fake_token_creator,
        ),
    ):
        body = GenAIHubEmbeddingConfig().transform_embedding_request(
            model="text-embedding-3-small",
            input="Hi",
            optional_params={"parameters": {"truncate": "END"}},
            headers={},
        )
        assert body["config"]["modules"]["embeddings"]["model"]["params"] == {
            "truncate": "END"
        }


def test_embed_with_masking(fake_token_creator, fake_deployment_url):
    masking_config = {
        "providers": [
            {
                "type": "sap_data_privacy_integration",
                "method": "anonymization",
                "entities": [
                    {"type": "profile-address"},
                    {"type": "profile-phone"},
                    {"type": "profile-person"},
                    {"type": "profile-location"},
                ],
            }
        ]
    }
    with (
        patch(
            "litellm.llms.sap.embed.transformation.GenAIHubEmbeddingConfig.deployment_url",
            new_callable=PropertyMock,
            return_value=fake_deployment_url,
        ),
        patch(
            "litellm.llms.sap.embed.transformation.get_token_creator",
            return_value=fake_token_creator,
        ),
    ):
        body = GenAIHubEmbeddingConfig().transform_embedding_request(
            model="text-embedding-3-small",
            input="Hi",
            optional_params={
                "parameters": {"truncate": "END"},
                "masking": masking_config,
            },
            headers={},
        )
        assert body["config"]["modules"]["masking"] == masking_config


def test_dimensions_forwarded_to_model_params(fake_token_creator, fake_deployment_url):
    with (
        patch(
            "litellm.llms.sap.embed.transformation.GenAIHubEmbeddingConfig.deployment_url",
            new_callable=PropertyMock,
            return_value=fake_deployment_url,
        ),
        patch(
            "litellm.llms.sap.embed.transformation.get_token_creator",
            return_value=fake_token_creator,
        ),
    ):
        body = GenAIHubEmbeddingConfig().transform_embedding_request(
            model="text-embedding-3-large",
            input="Hi",
            optional_params={"dimensions": 2000},
            headers={},
        )
        assert (
            body["config"]["modules"]["embeddings"]["model"]["params"]["dimensions"]
            == 2000
        )


def test_dimensions_merged_with_existing_parameters(
    fake_token_creator, fake_deployment_url
):
    with (
        patch(
            "litellm.llms.sap.embed.transformation.GenAIHubEmbeddingConfig.deployment_url",
            new_callable=PropertyMock,
            return_value=fake_deployment_url,
        ),
        patch(
            "litellm.llms.sap.embed.transformation.get_token_creator",
            return_value=fake_token_creator,
        ),
    ):
        body = GenAIHubEmbeddingConfig().transform_embedding_request(
            model="text-embedding-3-large",
            input="Hi",
            optional_params={"parameters": {"truncate": "END"}, "dimensions": 512},
            headers={},
        )
        params = body["config"]["modules"]["embeddings"]["model"]["params"]
        assert params["truncate"] == "END"
        assert params["dimensions"] == 512


def test_map_openai_params_forwards_supported_params(fake_token_creator):
    with patch(
        "litellm.llms.sap.embed.transformation.get_token_creator",
        return_value=fake_token_creator,
    ):
        config = GenAIHubEmbeddingConfig()
        result = config.map_openai_params(
            non_default_params={"dimensions": 1024, "user": "alice"},
            optional_params={},
            model="text-embedding-3-large",
            drop_params=False,
        )
        assert result == {"dimensions": 1024}


def test_encoding_format_forwarded_to_model_params(
    fake_token_creator, fake_deployment_url
):
    with (
        patch(
            "litellm.llms.sap.embed.transformation.GenAIHubEmbeddingConfig.deployment_url",
            new_callable=PropertyMock,
            return_value=fake_deployment_url,
        ),
        patch(
            "litellm.llms.sap.embed.transformation.get_token_creator",
            return_value=fake_token_creator,
        ),
    ):
        body = GenAIHubEmbeddingConfig().transform_embedding_request(
            model="text-embedding-3-large",
            input="Hi",
            optional_params={"encoding_format": "float"},
            headers={},
        )
        assert (
            body["config"]["modules"]["embeddings"]["model"]["params"][
                "encoding_format"
            ]
            == "float"
        )


def test_map_openai_params_no_dimensions_for_non_v3_model(fake_token_creator):
    with patch(
        "litellm.llms.sap.embed.transformation.get_token_creator",
        return_value=fake_token_creator,
    ):
        config = GenAIHubEmbeddingConfig()
        result = config.map_openai_params(
            non_default_params={"dimensions": 512, "encoding_format": "float"},
            optional_params={},
            model="text-embedding-ada-002",
            drop_params=False,
        )
        assert "dimensions" not in result
        assert result.get("encoding_format") == "float"
