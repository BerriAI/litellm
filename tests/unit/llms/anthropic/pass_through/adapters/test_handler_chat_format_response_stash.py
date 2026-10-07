"""The Anthropic bridge stashes the pre-translation chat-format response.

Issue #44743: the response /v1/messages returns drops
``completion_tokens_details.image_tokens`` from usage, so a cost recomputed
from it (the response-cost header race) prices image output tokens at the
text rate. ``_stash_chat_format_response`` keeps the original on the logging
object for the recompute branch to prefer.
"""

from litellm.constants import LITELLM_CHAT_FORMAT_RESPONSE_KEY
from litellm.llms.anthropic.pass_through.adapters.handler import _stash_chat_format_response


def _logging_obj():
    import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    return Logging(
        model="gemini/test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.datetime.now(),
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )


def test_sets_key_on_logging_obj_model_call_details():
    logging_obj = _logging_obj()

    _stash_chat_format_response({"litellm_logging_obj": logging_obj}, "original-response")

    assert logging_obj.model_call_details[LITELLM_CHAT_FORMAT_RESPONSE_KEY] == "original-response"


def test_no_op_without_logging_obj():
    _stash_chat_format_response({}, "original-response")  # must not raise
