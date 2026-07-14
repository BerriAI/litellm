import pytest

from litellm.exceptions import APIError
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.types.utils import ModelResponse


def test_non_openai_error_preserves_provider_status_and_message():
    response_object = {
        "response": "Token is invalid [2]",
        "status": 400,
        "choices": None,
    }

    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object=response_object,
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == 400
    assert "Token is invalid [2]" in exc_info.value.message


def test_non_openai_error_uses_status_code_and_message_fields():
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object={
                "status_code": 401,
                "message": "Invalid provider credentials",
                "choices": None,
            },
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == 401
    assert "Invalid provider credentials" in exc_info.value.message


@pytest.mark.parametrize("response_object", [{"choices": None}, {}])
def test_missing_or_none_choices_raises_api_error(response_object):
    with pytest.raises(APIError):
        convert_to_model_response_object(
            response_object=response_object,
            model_response_object=ModelResponse(),
        )


def test_missing_provider_status_uses_default_status_and_fallback_message():
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object={"choices": None, "provider_data": "unexpected"},
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == 500
    assert "no 'choices'" in exc_info.value.message
    assert "provider_data" in exc_info.value.message
