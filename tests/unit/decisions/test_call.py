import base64
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm.decisions.call import DecisionsCall, NativeOpenAIDecisionsCall
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


def test_an_openai_body_carries_the_caller_fields_and_none_of_the_settings_litellm_owns(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    question: Final = {"type": "predicate", "instructions": QUESTION_TEXT, "rubric": "strict"}

    call: Final = _prepare_decisions_call(
        **{**CONNECTION, "model": "openai/gpt-6-luna", "api_base": None},
        decision_input="My card was charged twice.",
        questions=[question],
        audio=None,
        videos=None,
        kwargs={
            "prompt_cache_key": "billing-v1",
            "user": "key-hash",
            "drop_params": True,
            "organization": "org-1",
            "base_url": "https://elsewhere.example/v1",
            "additional_drop_params": ["temperature"],
            "aws_region_name": "us-east-1",
            "vertex_project": "project-1",
            "azure_ad_token": "token-1",
            "metadata": {"tags": ["billing"]},
        },
    )

    assert isinstance(call, NativeOpenAIDecisionsCall)
    assert call.body == {
        "model": "gpt-6-luna",
        "input": "My card was charged twice.",
        "questions": [question],
        "prompt_cache_key": "billing-v1",
    }


@pytest.mark.parametrize("drop_params", (False, True), ids=("strict", "drop_params"))
def test_prompt_cache_fields_never_reach_a_provider_that_is_not_openai(
    drop_params: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "drop_params", False)
    image: Final = "data:image/png;base64,AA=="

    call: Final = _prepare_decisions_call(
        **CONNECTION,
        decision_input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "My card was charged twice.",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    },
                    {"type": "input_image", "image_url": image, "prompt_cache_breakpoint": {"mode": "explicit"}},
                ],
            }
        ],
        questions=[{"type": "predicate", "name": "billing", "instructions": QUESTION_TEXT}],
        audio=None,
        videos=None,
        kwargs={
            "prompt_cache_key": "billing-v1",
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "drop_params": drop_params,
        },
    )

    assert isinstance(call, DecisionsCall)
    assert call.provider_config.transform_decisions_request(
        model=call.model, request=call.ir_request, custom_llm_provider=call.custom_llm_provider
    ) == {
        "model": "jev-fake",
        "state": "My card was charged twice.",
        "images": [image],
        "questions": {"billing": {"type": "noul", "instructions": QUESTION_TEXT}},
    }


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
