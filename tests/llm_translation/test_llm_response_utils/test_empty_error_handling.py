
import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import convert_to_model_response_object
from litellm.types.utils import ModelResponse

def test_convert_to_model_response_object_with_empty_error():
    """
    Test that convert_to_model_response_object does NOT raise an exception
    when an 'error' object is present but empty (no message and no code).
    This is seen with providers like minimax-m2.1.
    """
    response_object = {
        "model": "minimax-m2.1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hey! I'm doing well, thanks for asking!"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        },
        "error": {
            "message": "",
            "type": "",
            "param": "",
            "code": None
        }
    }

    # This should NOT raise an exception
    model_response = ModelResponse()
    result = convert_to_model_response_object(
        response_object=response_object,
        model_response_object=model_response,
        response_type="completion"
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hey! I'm doing well, thanks for asking!"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10

def test_convert_to_model_response_object_with_actual_error():
    """
    Test that convert_to_model_response_object STILL raises an exception
    when an 'error' object contains an actual error message.
    """
    response_object = {
        "error": {
            "message": "Invalid API Key",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key"
        }
    }

    model_response = ModelResponse()
    with pytest.raises(Exception) as excinfo:
        convert_to_model_response_object(
            response_object=response_object,
            model_response_object=model_response,
            response_type="completion"
        )
    
    assert "Invalid API Key" in str(excinfo.value)
    assert getattr(excinfo.value, "status_code", None) == "invalid_api_key"

def test_convert_to_model_response_object_with_error_code_only():
    """
    Test that convert_to_model_response_object STILL raises an exception
    when an 'error' object contains a code but no message.
    """
    response_object = {
        "error": {
            "message": "",
            "type": "error",
            "param": None,
            "code": 500
        }
    }

    model_response = ModelResponse()
    with pytest.raises(Exception) as excinfo:
        convert_to_model_response_object(
            response_object=response_object,
            model_response_object=model_response,
            response_type="completion"
        )
    
    assert getattr(excinfo.value, "status_code", None) == 500
