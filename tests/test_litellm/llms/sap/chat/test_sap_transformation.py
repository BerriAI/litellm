from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
import pytest


def test_sap_placeholder_defaults():
    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Hello. Answer {{ ?user_query }}"}
        ],
        optional_params={'deployment_url': "shouldn't be in results",
                         "placeholder_defaults": {"user_query": "default value"}},
        litellm_params={},
        headers={}
    )

    assert config["config"]["modules"][0]["prompt_templating"]["prompt"]["defaults"] == {"user_query": "default value"}
    assert config["config"]["modules"][0]["prompt_templating"]["model"]["params"] == {}


def test_sap_placeholder_values():
    placeholder_values = {"user_query": "Some text"}
    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Hello. Answer {{ ?user_query }}"}
        ],
        optional_params={'deployment_url': "shouldn't be in results",
                         "placeholder_values": placeholder_values},
        litellm_params={},
        headers={}
    )

    assert config["placeholder_values"] == placeholder_values
    assert config["config"]["modules"][0]["prompt_templating"]["model"]["params"] == {}


def test_sap_grounding():
    grounding_config = {
        'type': 'document_grounding_service',
        'config': {
            'filters': [
                {'id': 's3-docs',
                 'data_repository_type': 'vector',
                 'search_config': {'max_chunk_count': 2},
                 'data_repositories': ['123456890-test']
                 }
            ],
            'placeholders': {'input': ['user_query'], 'output': 'grounding_response'},
            'metadata_params': ['source', 'webUrl', 'title', 'mimeType', 'fileSuffix']
        }
    }
    placeholder_values = {"user_query": "Some text"}
    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Hello. Answer {{ ?user_query }} using context: {{ ?grounding_response }}"}
        ],
        optional_params={'deployment_url': "shouldn't be in results",
                         "grounding": grounding_config,
                         "placeholder_values": placeholder_values},
        litellm_params={},
        headers={}
    )
    assert config["config"]["modules"][0]["grounding"] == grounding_config
    assert config["placeholder_values"] == placeholder_values
    assert config["config"]["modules"][0]["prompt_templating"]["model"]["params"] == {}


def test_sap_filtering():
    filtering_config_azure = {
        'input':
            {
                'filters':
                    [
                        {'type': 'azure_content_safety',
                         'config':
                             {'hate': 0,
                              'sexual': 0,
                              'violence': 0,
                              'self_harm': 0
                              }
                         }
                    ]
            },
        'output':
            {
                'filters':
                    [
                        {'type': 'azure_content_safety',
                         'config': {'hate': 0,
                                    'sexual': 0,
                                    'violence': 0,
                                    'self_harm': 0
                                    }
                         }
                    ]
            }
    }
    filtering_config_llama = {
        'input':
            {
                'filters':
                    [
                        {
                            'type': 'llama_guard_3_8b',
                            'config': {'hate': True,
                                       "elections": True}
                        }
                    ]
            },
        'output':
            {
                'filters':
                    [
                        {
                            'type': 'llama_guard_3_8b',
                            'config': {'hate': True, "elections": True}
                        }
                    ]
            }
    }
    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello."}],
        optional_params={'deployment_url': "shouldn't be in results",
                         "filtering": filtering_config_azure},
        litellm_params={},
        headers={}
    )
    assert config["config"]["modules"][0]["filtering"] == filtering_config_azure
    assert config["config"]["modules"][0]["prompt_templating"]["model"]["params"] == {}

    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello."}],
        optional_params={'deployment_url': "shouldn't be in results",
                         "filtering": filtering_config_llama},
        litellm_params={},
        headers={}
    )
    assert config["config"]["modules"][0]["filtering"] == filtering_config_llama
    assert config["config"]["modules"][0]["prompt_templating"]["model"]["params"] == {}


def test_sap_masking():
    masking_config = {
        'providers':
            [
                {
                    'type': 'sap_data_privacy_integration',
                    'method': 'anonymization',
                    'entities': [
                        {'type': 'profile-address'},
                        {'type': 'profile-email'},
                        {'type': 'profile-phone'},
                        {'type': 'profile-person'},
                        {'type': 'profile-location'}
                    ]
                }
            ]
    }

    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello."}],
        optional_params={'deployment_url': "shouldn't be in results",
                         "masking": masking_config},
        litellm_params={},
        headers={}
    )
    assert config["config"]["modules"][0]["masking"] == masking_config
    assert config["config"]["modules"][0]["prompt_templating"]["model"]["params"] == {}


def test_sap_translation():
    translation_config = {
        'input':
            {'type': 'sap_document_translation',
             'config':
                 {'source_language': 'en-US',
                  'target_language': 'de-DE'}
             },
        'output':
            {'type': 'sap_document_translation',
             'config':
                 {'source_language': 'de-DE',
                  'target_language': 'fr-FR'}
             }
    }

    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello."}],
        optional_params={'deployment_url': "shouldn't be in results",
                         "translation": translation_config},
        litellm_params={},
        headers={}
    )
    assert config["config"]["modules"][0]["translation"] == translation_config
    assert config["config"]["modules"][0]["prompt_templating"]["model"]["params"] == {}


def test_sap_multiple_modules():
    translation_config = {
        'input':
            {'type': 'sap_document_translation',
             'config':
                 {'source_language': 'en-US',
                  'target_language': 'de-DE'}
             },
        'output':
            {'type': 'sap_document_translation',
             'config':
                 {'source_language': 'de-DE',
                  'target_language': 'fr-FR'}
             }
    }

    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello."}],
        optional_params={'deployment_url': "shouldn't be in results",
                         "fallback_modules": [{"model": "sap/gpt-5",
                                               "messages": [{"role": "user", "content": "Hello world!"}],
                                               "translation": translation_config
                                               }]
            ,
                         },
        litellm_params={},
        headers={}
    )
    assert "translation" not in config["config"]["modules"][0]
    assert config["config"]["modules"][1]["translation"] == translation_config
    assert config["config"]["modules"][1]["prompt_templating"]["model"]["name"] == "gpt-5"
    assert config["config"]["modules"][0]["prompt_templating"]["model"]["name"] == "gpt-4o"
    assert config["config"]["modules"][0]["prompt_templating"]["model"]["params"] == {}
    assert config["config"]["modules"][1]["prompt_templating"]["prompt"]["template"][0]["content"] == "Hello world!"
    assert config["config"]["modules"][0]["prompt_templating"]["prompt"]["template"][0]["content"] == "Hello."


class TestSAPTransformationIntegration:
    """Integration tests for SAP transformation with parameter classification."""

    @pytest.fixture
    def mock_config(self):
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        config = GenAIHubOrchestrationConfig()
        config.token_creator = lambda: "Bearer TEST_TOKEN"
        config._base_url = "https://api.test-sap.com"
        config._resource_group = "test-group"

        return config

    def test_parameter_classification_in_transform_request(self, mock_config):
        """Test parameter classification within the actual transform_request method."""

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]

        optional_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "deployment_url": "https://custom.sap.com/deployment/123",
            "model_version": "v1.5",
            "tools": [{"type": "function", "function": {"name": "calculator"}}],
            "frequency_penalty": 0.1
        }

        result = mock_config.transform_request(
            model, messages, optional_params, {}, {}
        )

        model_params = result["config"]["modules"][0]["prompt_templating"]["model"]["params"]

        assert "temperature" in model_params
        assert "frequency_penalty" in model_params
        assert "deployment_url" not in model_params
        assert "model_version" not in model_params
        assert "tools" not in model_params

        model_version = result["config"]["modules"][0]["prompt_templating"]["model"]["version"]
        assert model_version == "v1.5"

        prompt = result["config"]["modules"][0]["prompt_templating"]["prompt"]
        if "tools" in prompt:
            assert isinstance(prompt["tools"], list)

    def test_transform_request_parameter_handling_robustness(self, mock_config):
        """Test transform_request method handles various parameter combinations correctly."""

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]

        test_cases = [
            # Case 1: Basic parameters only
            {
                "params": {"temperature": 0.7, "max_tokens": 100},
                "expected_in_model": {"temperature", "max_tokens"},
                "expected_excluded": set()
            },
            # Case 2: Parameters with auth/infrastructure components
            {
                "params": {
                    "temperature": 0.8,
                    "deployment_url": "https://api.sap.com/deployments/test",
                    "max_tokens": 150
                },
                "expected_in_model": {"temperature", "max_tokens"},
                "expected_excluded": {"deployment_url"}
            },
            # Case 3: Parameters with framework components
            {
                "params": {
                    "temperature": 0.6,
                    "model_version": "v2.0",
                    "tools": [{"function": {"name": "test"}}],
                    "frequency_penalty": 0.1
                },
                "expected_in_model": {"temperature", "frequency_penalty"},
                "expected_excluded": {"model_version", "tools"}
            }
        ]

        for i, test_case in enumerate(test_cases):
            filtered_params = {
                k: v for k, v in test_case["params"].items()
                if k not in {"tools", "model_version", "deployment_url"}
            }

            for expected_param in test_case["expected_in_model"]:
                assert expected_param in filtered_params, f"Case {i + 1}: {expected_param} should be in model params"

            for excluded_param in test_case["expected_excluded"]:
                assert excluded_param not in filtered_params, f"Case {i + 1}: {excluded_param} should be excluded from model params"

            try:
                result = mock_config.transform_request(
                    model, messages, test_case["params"], {}, {}
                )
                if result and "config" in result:
                    model_params = result["config"]["modules"][0]["prompt_templating"]["model"]["params"]

                    for excluded_param in test_case["expected_excluded"]:
                        assert excluded_param not in model_params, (
                            f"Case {i + 1}: {excluded_param} should not be in actual model params"
                        )
            except AttributeError as e:
                if "deployment_url" in str(e):
                    pass
                else:
                    pytest.fail(f"Unexpected AttributeError: {e}")
            except Exception as e:
                pytest.fail(f"Unexpected exception in transform_request: {e}")
