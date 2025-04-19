from unittest.mock import patch
import pytest

from litellm.llms.watsonx.common_utils import WatsonXAIError


@pytest.mark.parametrize(
    "value_from",
    [
        {'params': {'space_id': 'space-id'}, 'secrets': {}},
        {'params': {}, 'secrets': {'WATSONX_DEPLOYMENT_SPACE_ID': 'space-id'}},
        {'params': {}, 'secrets': {'WATSONX_SPACE_ID': 'space-id'}},
        {'params': {}, 'secrets': {'SPACE_ID': 'space-id'}},
    ],
    ids=[
        "field",
        "WATSONX_DEPLOYMENT_SPACE_ID",
        "WATSONX_SPACE_ID",
        "SPACE_ID"
    ]
)
def test_watsonx_chat_handler_with_space_id(value_from):
    with patch('litellm.llms.watsonx.common_utils.get_secret_str') as get_secret_str:
        get_secret_str.return_value = ''
        get_secret_str.side_effect = lambda key: value_from['secrets'][key] if key in value_from['secrets'] else None

        from litellm.llms.watsonx import common_utils

        _get_api_params = getattr(common_utils, '_get_api_params')
        params = _get_api_params(value_from['params'])
        assert params['space_id'] == 'space-id'


@pytest.mark.parametrize(
    "value_from",
    [
        {'params': {'project_id': 'project-id'}, 'secrets': {}},
        {'params': {}, 'secrets': {'WATSONX_PROJECT_ID': 'project-id'}},
        {'params': {}, 'secrets': {'WX_PROJECT_ID': 'project-id'}},
        {'params': {}, 'secrets': {'PROJECT_ID': 'project-id'}},
    ],
    ids=[
        "field",
        "WATSONX_PROJECT_ID",
        "WX_PROJECT_ID",
        "PROJECT_ID"
    ]
)
def test_watsonx_chat_handler_with_project_id(value_from):
    with patch('litellm.llms.watsonx.common_utils.get_secret_str') as get_secret_str:
        get_secret_str.return_value = ''
        get_secret_str.side_effect = lambda key: value_from['secrets'][key] if key in value_from['secrets'] else None

        from litellm.llms.watsonx import common_utils

        _get_api_params = getattr(common_utils, '_get_api_params')
        params = _get_api_params(value_from['params'])
        assert params['project_id'] == 'project-id'


def test_watsonx_chat_handler_with_no_space_id_or_project_id():
    with patch('litellm.llms.watsonx.common_utils.get_secret_str') as get_secret_str:
        get_secret_str.return_value = None

        from litellm.llms.watsonx import common_utils

        _get_api_params = getattr(common_utils, '_get_api_params')
        with pytest.raises(WatsonXAIError) as error:
            params = _get_api_params({})
        assert 'At least one of project_id or space_id must be set.' in error.value.message
