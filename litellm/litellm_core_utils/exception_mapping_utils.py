import json
from typing import Optional


def get_error_message(error_obj) -> Optional[str]:
    """
    OpenAI Returns Error message that is nested, this extract the message

    Example:
    {
        'request': "<Request('POST', 'https://api.openai.com/v1/chat/completions')>",
        'message': "Error code: 400 - {\'error\': {\'message\': \"Invalid 'temperature': decimal above maximum value. Expected a value <= 2, but got 200 instead.\", 'type': 'invalid_request_error', 'param': 'temperature', 'code': 'decimal_above_max_value'}}",
        'body': {
            'message': "Invalid 'temperature': decimal above maximum value. Expected a value <= 2, but got 200 instead.",
            'type': 'invalid_request_error',
            'param': 'temperature',
            'code': 'decimal_above_max_value'
        },
        'code': 'decimal_above_max_value',
        'param': 'temperature',
        'type': 'invalid_request_error',
        'response': "<Response [400 Bad Request]>",
        'status_code': 400,
        'request_id': 'req_f287898caa6364cd42bc01355f74dd2a'
    }
    """
    try:
        # First, try to access the message directly from the 'body' key
        if error_obj is None:
            return None

        if hasattr(error_obj, "body"):
            _error_obj_body = getattr(error_obj, "body")
            if isinstance(_error_obj_body, dict):
                return _error_obj_body.get("message")

        # If all else fails, return None
        return None
    except Exception as e:
        return None
