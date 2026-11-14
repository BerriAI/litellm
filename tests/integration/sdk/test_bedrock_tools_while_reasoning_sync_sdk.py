import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.bedrock_runtime_peer import answer, respond, target_of
from integration._support.client import eventually
from integration._support.wire import wire_server

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Choices, ModelResponse

_MODEL_ID: Final = "us.openai.gpt-5.6-sol"
_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "lookup_invoice",
        "description": "Look up an invoice",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    },
}


class _SuccessRecorder(CustomLogger):
    successes: tuple[tuple[object, object], ...] = ()

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.successes = (*self.successes, (kwargs.get("call_type"), getattr(response_obj, "id", None)))


def test_sync_completion_with_tools_while_reasoning_keeps_converse_and_logs_one_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder: Final = _SuccessRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire:
        response: Final = litellm.completion(
            model=f"bedrock/{_MODEL_ID}",
            messages=[{"role": "user", "content": f"synthetic native request marker-{marker}"}],
            tools=[_TOOL],
            reasoning_effort="high",
            api_key="synthetic-bedrock-bearer",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            num_retries=0,
        )
        sent: Final = [(request.method, target_of(request)) for request in wire.drain()]
    assert sent == [("POST", f"/model/{_MODEL_ID}/converse")], sent
    assert isinstance(response, ModelResponse), type(response)
    choice: Final = response.choices[0]
    assert isinstance(choice, Choices), choice
    assert choice.message.content == answer(marker), choice
    logged: Final = eventually(
        lambda: recorder.successes,
        lambda successes: any(logged_id == response.id for _, logged_id in successes),
    )
    assert logged == (("completion", response.id),), logged
