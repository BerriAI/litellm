"""
The "Invalid response object" error carries only the conversion's own traceback. A conversion that
runs while another error is being handled (a router retry inside the previous attempt's ``except``)
must not copy that error's text into its message, where exception mapping would read it.
"""

import pytest

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import convert_to_model_response_object  # pyright: ignore[reportUnknownVariableType]  # its untyped start_time, end_time and dict parameters
from litellm.types.utils import ModelResponse

# A text_completion body handed to the chat conversion: ``choices[0]`` has no ``message``.
_MALFORMED_CHAT_BODY = {
    "id": "cmpl-1",
    "object": "text_completion",
    "created": 1,
    "model": "gpt-4o",
    "choices": [{"text": "ok", "index": 0, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_OUTER_TEXT = "outer attempt failed: rate limit exceeded"


class _OuterAttemptError(Exception):
    pass


def _convert_while_handling_another_error() -> Exception:
    try:
        raise _OuterAttemptError(_OUTER_TEXT)
    except _OuterAttemptError:
        with pytest.raises(Exception, match="Invalid response object") as exc_info:
            _ = convert_to_model_response_object(
                response_object=dict(_MALFORMED_CHAT_BODY),
                model_response_object=ModelResponse(),
                response_type="completion",
            )
    return exc_info.value


def test_the_message_leaves_out_the_error_being_handled_when_the_conversion_ran() -> None:
    message = str(_convert_while_handling_another_error())

    assert _OUTER_TEXT not in message
    assert "_OuterAttemptError" not in message
    assert "During handling of the above exception" not in message


def test_the_message_names_the_conversion_error_itself() -> None:
    message = str(_convert_while_handling_another_error())

    assert "KeyError: 'message'" in message
    assert "Traceback (most recent call last)" in message
