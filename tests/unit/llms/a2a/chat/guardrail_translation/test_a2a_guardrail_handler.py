"""Tests for litellm/llms/a2a/chat/guardrail_translation/handler.py."""

import json
from typing import Final, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.a2a.chat.guardrail_translation.handler import A2AGuardrailHandler
from litellm.llms.base_llm.guardrail_translation.base_translation import StreamingScanKey
from litellm.types.agents import LiteLLMSendMessageResponse
from litellm.types.utils import GenericGuardrailAPIInputs


def _text_event(text: str) -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "result": {"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": text}]},
        }
    )


def _status_event() -> str:
    return json.dumps({"jsonrpc": "2.0", "id": "req-1", "result": {"kind": "status-update", "status": {}}})


class TestA2AGuardrailHandlerStreamingScanKey:
    def test_key_joins_the_text_of_every_message_event(self):
        key = A2AGuardrailHandler().get_streaming_scan_key([_text_event("hello "), _text_event("world")])
        assert key == StreamingScanKey(texts=("hello world",))

    def test_events_without_text_leave_the_key_unchanged(self):
        handler = A2AGuardrailHandler()
        events = [_text_event("hello")]
        assert handler.get_streaming_scan_key(events + [_status_event()]) == handler.get_streaming_scan_key(events)

    def test_unparseable_items_are_ignored(self):
        key = A2AGuardrailHandler().get_streaming_scan_key([_text_event("hi"), "not json", b"bytes"])
        assert key.texts == ("hi",)

    def test_json_items_that_are_not_objects_are_ignored(self):
        key = A2AGuardrailHandler().get_streaming_scan_key([_text_event("hi"), "[1, 2]", "5"])
        assert key.texts == ("hi",)


SSN_REPLY: Final = {"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": "SSN 123-45-6789"}]}


class _MaskingGuardrail(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Logging | None = None,
    ) -> GenericGuardrailAPIInputs:
        return {"texts": [text.replace("123-45-6789", "[SSN]") for text in inputs.get("texts", [])]}


class _BlockingGuardrail(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Logging | None = None,
    ) -> GenericGuardrailAPIInputs:
        raise ValueError(f"blocked {inputs.get('texts')}")


class TestA2AGuardrailHandlerOutputResponse:
    async def test_dict_response_gets_the_guardrailed_text_in_place(self):
        response: Final = {"jsonrpc": "2.0", "id": "req-1", "result": json.loads(json.dumps(SSN_REPLY))}

        processed: Final = await A2AGuardrailHandler().process_output_response(
            response, _MaskingGuardrail(guardrail_name="mask-ssn")
        )

        assert processed is response
        assert response["result"]["parts"][0]["text"] == "SSN [SSN]"

    async def test_pydantic_response_text_reaches_the_guardrail(self):
        response: Final = LiteLLMSendMessageResponse(id="req-1", result=json.loads(json.dumps(SSN_REPLY)))

        with pytest.raises(ValueError, match=r"blocked \['SSN 123-45-6789'\]"):
            await A2AGuardrailHandler().process_output_response(response, _BlockingGuardrail(guardrail_name="block"))

    async def test_response_of_an_unknown_type_is_returned_without_running_the_guardrail(self):
        processed: Final = await A2AGuardrailHandler().process_output_response(
            "SSN 123-45-6789", _BlockingGuardrail(guardrail_name="block")
        )

        assert processed == "SSN 123-45-6789"

    async def test_tasks_list_03_and_10_dict_and_pydantic_responses_rewrite_all_task_text_in_place(self):
        v03_tasks_list: Final = {
            "tasks": [
                {
                    "kind": "task",
                    "id": "task-1",
                    "status": {
                        "state": "completed",
                        "message": {"kind": "message", "parts": [{"kind": "text", "text": "status 123-45-6789"}]},
                    },
                    "artifacts": [{"artifactId": "a1", "parts": [{"kind": "text", "text": "artifact 123-45-6789"}]}],
                    "history": [{"kind": "message", "parts": [{"kind": "text", "text": "history 123-45-6789"}]}],
                },
                {
                    "id": "task-2",
                    "status": {
                        "state": "TASK_STATE_COMPLETED",
                        "message": {"parts": [{"text": "v1 status 123-45-6789"}]},
                    },
                    "artifacts": [{"artifactId": "a2", "parts": [{"text": "v1 artifact 123-45-6789"}]}],
                    "history": [{"parts": [{"text": "v1 history 123-45-6789"}]}],
                },
            ]
        }
        pydantic_response: Final = LiteLLMSendMessageResponse(id="req-1", result=json.loads(json.dumps(v03_tasks_list)))

        processed: Final = await A2AGuardrailHandler().process_output_response(
            pydantic_response, _MaskingGuardrail(guardrail_name="mask-ssn")
        )

        assert processed is pydantic_response
        dumped: Final = pydantic_response.model_dump()
        task1: Final = dumped["result"]["tasks"][0]
        task2: Final = dumped["result"]["tasks"][1]
        assert task1["status"]["message"]["parts"][0]["text"] == "status [SSN]"
        assert task1["artifacts"][0]["parts"][0]["text"] == "artifact [SSN]"
        assert task1["history"][0]["parts"][0]["text"] == "history [SSN]"
        assert task2["status"]["message"]["parts"][0]["text"] == "v1 status [SSN]"
        assert task2["artifacts"][0]["parts"][0]["text"] == "v1 artifact [SSN]"
        assert task2["history"][0]["parts"][0]["text"] == "v1 history [SSN]"

    async def test_tasks_list_blocking_guardrail_rejects_task_artifacts(self):
        response: Final = LiteLLMSendMessageResponse(
            id="req-2",
            result={
                "tasks": [
                    {
                        "id": "task-1",
                        "artifacts": [{"parts": [{"kind": "text", "text": "secret 123-45-6789"}]}],
                    }
                ]
            },
        )

        with pytest.raises(ValueError, match=r"blocked \['secret 123-45-6789'\]"):
            await A2AGuardrailHandler().process_output_response(response, _BlockingGuardrail(guardrail_name="block"))
