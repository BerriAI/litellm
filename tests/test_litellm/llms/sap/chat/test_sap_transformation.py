from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

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
                         "fallback_modules":[{"model": "sap/gpt-5",
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