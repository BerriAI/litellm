import pytest

from litellm.exceptions import APIError
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
    convert_to_streaming_response,
    convert_to_streaming_response_async,
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


def test_non_openai_error_coerces_numeric_string_status():
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object={
                "status": "400",
                "response": "Token is invalid [2]",
                "choices": None,
            },
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == 400
    assert "Token is invalid [2]" in exc_info.value.message


@pytest.mark.parametrize("code", [401, "401"])
def test_non_openai_error_uses_nested_error_object(code):
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object={
                "error": {"message": "Invalid credentials", "code": code},
                "choices": None,
            },
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in exc_info.value.message


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_message"),
    [
        ({"response": "Provider unavailable", "status": "503"}, 503, "Provider unavailable"),
        ("Provider request failed", 500, "Provider request failed"),
    ],
)
def test_non_openai_error_uses_nested_error_message_variants(error, expected_status, expected_message):
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object={"error": error, "choices": None},
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == expected_status
    assert expected_message in exc_info.value.message


def test_streaming_non_openai_error_uses_provider_status_and_message():
    response_object = {
        "error": {"message": "Invalid credentials", "code": 401},
        "choices": None,
    }

    with pytest.raises(APIError) as exc_info:
        next(convert_to_streaming_response(response_object))

    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in exc_info.value.message


@pytest.mark.asyncio
async def test_async_streaming_non_openai_error_uses_provider_status_and_message():
    response_object = {
        "error": {"message": "Invalid credentials", "code": 401},
        "choices": None,
    }

    with pytest.raises(APIError) as exc_info:
        await convert_to_streaming_response_async(response_object).__anext__()

    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in exc_info.value.message


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
