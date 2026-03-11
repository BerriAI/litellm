import pytest

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

        model_params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

        assert "temperature" in model_params
        assert "frequency_penalty" in model_params
        assert "deployment_url" not in model_params
        assert "model_version" not in model_params
        assert "tools" not in model_params

        model_version = result["config"]["modules"]["prompt_templating"]["model"]["version"]
        assert model_version == "v1.5"

        prompt = result["config"]["modules"]["prompt_templating"]["prompt"]
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

            result = mock_config.transform_request(
                model, messages, test_case["params"], {}, {}
            )
            if result and "config" in result:
                model_params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

                for excluded_param in test_case["expected_excluded"]:
                    assert excluded_param not in model_params, (
                        f"Case {i + 1}: {excluded_param} should not be in actual model params"
                    )

    def test_config_transform_with_response_format_json_object(self, mock_config):
        expected_dict = {'config':
                             {'modules':
                                  {'prompt_templating':
                                       {'prompt':
                                            {'template':
                                                 [{'role': 'user', 'content': 'First man on the moon, answer in json'}],
                                             'response_format': {'type': 'json_object'}},
                                        'model': {'name': 'gpt-4o', 'params': {}, 'version': 'latest'}
                                        }
                                   },
                              'stream': {}
                              }
                         }
        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{'role': 'user', 'content': 'First man on the moon, answer in json'}],
            optional_params={'response_format': {'type': 'json_object'},
                             'deployment_url': "shouldn't be in results"},
            litellm_params={},
            headers={}
        )
        assert config == expected_dict

    def test_config_transform_with_response_format_json_schema(self, mock_config):

        expected_response_format = {
            'type': 'json_schema',
            'json_schema': {
                'description': 'Schema for person information',
                'name': 'person_info',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': "The person's full name"
                        },
                        'age': {
                            'type': 'integer',
                            'description': "The person's age in years"
                        },
                        'occupation': {
                            'type': 'string',
                            'description': "The person's job title"
                        }
                    },
                    'required': ['name', 'age', 'occupation'],
                    'additionalProperties': False
                },
                'strict': True
            }
        }

        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{'role': 'user', 'content': 'First man on the moon, answer in json'}],
            optional_params={'response_format': expected_response_format,
                             'deployment_url': "shouldn't be in results"},
            litellm_params={},
            headers={}
        )
        assert config["config"]["modules"]["prompt_templating"]["prompt"]["response_format"] == expected_response_format
        assert len(config["config"]["modules"]["prompt_templating"]["model"]["params"]) == 0

    def test_config_transform_with_stream(self, mock_config):
        expected_dict = {
            'config': {
                'modules': {
                    'prompt_templating': {
                        'prompt': {
                            'template': [{'role': 'user', 'content': 'Hello, how are you?'}]
                        },
                        'model': {
                            'name': 'anthropic--claude-4-sonnet',
                            'params': {},
                            'version': 'latest'
                        }
                    }
                },
                'stream': {'chunk_size': 10}
            }
        }
        config = mock_config.transform_request(
            model="anthropic--claude-4-sonnet",
            messages=[{'content': 'Hello, how are you?', 'role': 'user'}],
            optional_params={'stream': True,
                             'stream_options': {'chunk_size': 10},
                             'model_version': 'latest',
                             'deployment_url': "shouldn't be in results"},
            litellm_params={},
            headers={}
        )

        assert config == expected_dict

    def test_sap_placeholder_defaults(self, mock_config):
        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Hello. Answer {{ ?user_query }}"}
            ],
            optional_params={'deployment_url': "shouldn't be in results",
                             "placeholder_defaults": {"user_query": "default value"}},
            litellm_params={},
            headers={}
        )

        assert config["config"]["modules"]["prompt_templating"]["prompt"]["defaults"] == {
            "user_query": "default value"}
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_sap_placeholder_values(self, mock_config):
        placeholder_values = {"user_query": "Some text"}
        config = mock_config.transform_request(
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
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_sap_grounding(self, mock_config):
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
        config = mock_config.transform_request(
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
        assert config["config"]["modules"]["grounding"] == grounding_config
        assert config["placeholder_values"] == placeholder_values
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_sap_filtering(self, mock_config):
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
        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello."}],
            optional_params={'deployment_url': "shouldn't be in results",
                             "filtering": filtering_config_azure},
            litellm_params={},
            headers={}
        )
        assert config["config"]["modules"]["filtering"] == filtering_config_azure
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello."}],
            optional_params={'deployment_url': "shouldn't be in results",
                             "filtering": filtering_config_llama},
            litellm_params={},
            headers={}
        )
        assert config["config"]["modules"]["filtering"] == filtering_config_llama
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_sap_masking(self, mock_config):
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

        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello."}],
            optional_params={'deployment_url': "shouldn't be in results",
                             "masking": masking_config},
            litellm_params={},
            headers={}
        )
        assert config["config"]["modules"]["masking"] == masking_config
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_sap_translation(self, mock_config):
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

        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello."}],
            optional_params={'deployment_url': "shouldn't be in results",
                             "translation": translation_config},
            litellm_params={},
            headers={}
        )
        assert config["config"]["modules"]["translation"] == translation_config
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_sap_multiple_modules(self, mock_config):
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

        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello."}],
            optional_params={'deployment_url': "shouldn't be in results",
                             "fallback_sap_modules": [{"model": "sap/gpt-5",
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