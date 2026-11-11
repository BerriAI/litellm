import base64
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm.decisions.main import _prepare_decisions_call
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.systemone.main import _prepare_systemone_call

QUESTION_TEXT: Final = "Is this about a billing problem?"
CONNECTION: Final = {
    "model": "hosted_vllm/jev-fake",
    "safety_identifier": None,
    "api_key": "fake-provider-key",
    "api_base": "http://127.0.0.1:4010",
    "timeout": None,
    "custom_llm_provider": None,
    "extra_headers": None,
}


def _logging_obj(call_type: str) -> Logging:
    return Logging(
        model="hosted_vllm/jev-fake",
        messages="placeholder",
        stream=False,
        call_type=call_type,
        start_time=datetime.now(timezone.utc),
        litellm_call_id="lit-9659",
        function_id="lit-9659",
    )


def _logged_request(logging_obj: Logging) -> Mapping[str, object]:
    messages: Final = logging_obj.model_call_details["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return json.loads(messages[0]["content"])


def test_systemone_logs_every_request_field_the_caller_sent():
    logging_obj: Final = _logging_obj("asystemone")

    _prepare_systemone_call(
        **CONNECTION,
        state={"message": "My card was charged twice."},
        questions={"billing": {"type": "noul", "instructions": QUESTION_TEXT}},
        images=None,
        assets=None,
        audio=None,
        videos=None,
        kwargs={"litellm_logging_obj": logging_obj},
    )

    logged: Final = _logged_request(logging_obj)
    assert logged["state"] == {"message": "My card was charged twice."}
    assert logged["questions"]["billing"]["instructions"] == QUESTION_TEXT
    assert set(logged) == {"state", "questions"}


def test_decisions_logs_every_request_field_the_caller_sent():
    logging_obj: Final = _logging_obj("adecisions")

    _prepare_decisions_call(
        **CONNECTION,
        decision_input="My card was charged twice.",
        questions=[{"type": "predicate", "instructions": QUESTION_TEXT}],
        audio=None,
        videos=None,
        kwargs={"litellm_logging_obj": logging_obj},
    )

    logged: Final = _logged_request(logging_obj)
    assert logged["input"] == "My card was charged twice."
    assert logged["questions"][0]["instructions"] == QUESTION_TEXT
    assert set(logged) == {"input", "questions"}


def test_request_is_logged_even_when_it_is_rejected_before_the_provider_call():
    logging_obj: Final = _logging_obj("asystemone")

    with pytest.raises(litellm.BadRequestError):
        _prepare_systemone_call(
            **CONNECTION,
            state={"message": "My card was charged twice."},
            questions={"billing": {"type": "noul", "instructions": QUESTION_TEXT}},
            images=None,
            assets=None,
            audio=None,
            videos=None,
            kwargs={"litellm_logging_obj": logging_obj, "input": "sent to the wrong function"},
        )

    logged: Final = _logged_request(logging_obj)
    assert logged["questions"]["billing"]["instructions"] == QUESTION_TEXT


def test_decisions_request_is_logged_even_when_its_questions_fail_validation():
    logging_obj: Final = _logging_obj("adecisions")

    with pytest.raises(litellm.BadRequestError):
        _prepare_decisions_call(
            **CONNECTION,
            decision_input="My card was charged twice.",
            questions=[{"type": "not-a-question-type", "instructions": QUESTION_TEXT}],
            audio=None,
            videos=None,
            kwargs={"litellm_logging_obj": logging_obj},
        )

    logged: Final = _logged_request(logging_obj)
    assert logged["questions"][0]["instructions"] == QUESTION_TEXT


def test_non_utf8_bytes_in_the_request_are_logged_and_rejected_with_a_400():
    logging_obj: Final = _logging_obj("adecisions")
    raw_input: Final = b"\xff\xfe not utf-8"

    with pytest.raises(litellm.BadRequestError):
        _prepare_decisions_call(
            **CONNECTION,
            decision_input=raw_input,
            questions=[{"type": "predicate", "instructions": QUESTION_TEXT}],
            audio=None,
            videos=None,
            kwargs={"litellm_logging_obj": logging_obj},
        )

    logged: Final = _logged_request(logging_obj)
    assert base64.urlsafe_b64decode(logged["input"]) == raw_input
    assert logged["questions"][0]["instructions"] == QUESTION_TEXT
