import json
from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.decisions.call import OpenAIDecisionRequestFields, SystemOneRequestFields, prepare_call
from litellm.litellm_core_utils.litellm_logging import Logging

QUESTION_TEXT: Final = "Is this about a billing problem?"


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


def _logged_request(logging_obj: Logging) -> dict[str, object]:
    messages: Final = logging_obj.model_call_details["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return json.loads(messages[0]["content"])


@pytest.mark.parametrize(
    ("call_type", "fields", "input_key", "expected_input"),
    [
        (
            "asystemone",
            SystemOneRequestFields(
                state={"message": "My card was charged twice."},
                questions={"billing": {"type": "noul", "instructions": QUESTION_TEXT}},
                images=None,
                assets=None,
                audio=None,
                videos=None,
            ),
            "state",
            {"message": "My card was charged twice."},
        ),
        (
            "adecisions",
            OpenAIDecisionRequestFields(
                input="My card was charged twice.",
                questions=[{"type": "predicate", "instructions": QUESTION_TEXT}],
                audio=None,
                videos=None,
            ),
            "input",
            "My card was charged twice.",
        ),
    ],
)
def test_prepare_call_logs_every_request_field_the_caller_sent(call_type, fields, input_key, expected_input):
    logging_obj: Final = _logging_obj(call_type)

    prepare_call(
        model="hosted_vllm/jev-fake",
        fields=fields,
        safety_identifier=None,
        api_key="fake-provider-key",
        api_base="http://127.0.0.1:4010",
        timeout=None,
        custom_llm_provider=None,
        extra_headers=None,
        kwargs={"litellm_logging_obj": logging_obj},
    )

    logged: Final = _logged_request(logging_obj)
    assert logged[input_key] == expected_input
    assert QUESTION_TEXT in json.dumps(logged["questions"])
    assert set(logged) == {input_key, "questions"}
