from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

def test_basic_config_transform():
    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"temperature": 1, 'deployment_url': "shouldn't be in results"},
        litellm_params={},
        headers={}
    )
    assert config["config"]["modules"]["prompt_templating"]["prompt"]["template"] == [{"role": "user", "content": "Hello"}]
    assert config["config"]["modules"]["prompt_templating"]["model"]["name"] == "gpt-4o"
    assert config["config"]["modules"]["prompt_templating"]["model"]["params"]["temperature"] == 1
    assert config["config"]["modules"]["prompt_templating"]["model"]["version"] == "latest"
    assert len(config["config"]["modules"]["prompt_templating"]["model"]["params"]) == 1

def test_config_transform_with_response_format_json_object():
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
    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[{'role': 'user', 'content': 'First man on the moon, answer in json'}],
        optional_params={'response_format': {'type': 'json_object'},
                         'deployment_url': "shouldn't be in results"},
        litellm_params={},
        headers={}
    )
    assert config==expected_dict

def test_config_transform_with_response_format_json_schema():

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

    config = GenAIHubOrchestrationConfig().transform_request(
        model="gpt-4o",
        messages=[{'role': 'user', 'content': 'First man on the moon, answer in json'}],
        optional_params={'response_format': expected_response_format,
                         'deployment_url': "shouldn't be in results"},
        litellm_params={},
        headers={}
    )
    assert config["config"]["modules"]["prompt_templating"]["prompt"]["response_format"] == expected_response_format
    assert len(config["config"]["modules"]["prompt_templating"]["model"]["params"]) == 0

def test_config_transform_with_stream():
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
    config = GenAIHubOrchestrationConfig().transform_request(
        model="anthropic--claude-4-sonnet",
        messages=[{'content': 'Hello, how are you?', 'role': 'user'}],
        optional_params={'stream': True,
                         'stream_options': {'chunk_size': 10},
                         'model_version': 'latest',
                         'deployment_url': "shouldn't be in results"},
        litellm_params={},
        headers={}
    )

    assert config==expected_dict