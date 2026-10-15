import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply
from integration.translation.case import SSEFrame
from integration.translation.runner import encode_sse_frame
from pydantic import BaseModel, ConfigDict, JsonValue


class Frame(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: dict[str, JsonValue] | Literal["[DONE]"]
    event: str | None = None
    headers: dict[str, str | int] | None = None

    def encode(self) -> bytes:
        if self.headers is not None:
            return aws_event_stream_frame(self.headers, json.dumps(self.data).encode())
        return encode_sse_frame(SSEFrame(data=self.data, event=self.event))


class Exchange(BaseModel):
    model_config = ConfigDict(frozen=True)
    request: dict[str, JsonValue]
    target: str
    content_type: str = "application/json"
    response: dict[str, JsonValue] | None = None
    frames: tuple[Frame, ...] = ()

    def reply(self) -> Reply:
        if self.response is not None:
            return Reply(body=json.dumps(self.response).encode())
        return Reply(content_type=self.content_type, chunks=tuple(frame.encode() for frame in self.frames))


class AgentCase(BaseModel):
    model_config = ConfigDict(frozen=True)
    tier: Literal["small", "mid", "high"]
    model: str
    tools: Exchange
    stream: Exchange
    followup: Exchange
    reasoning_fields: tuple[str, ...]


DIRECTORY: Final = Path(__file__).parent / "captures"
CASES: Final = tuple(AgentCase.model_validate_json(path.read_text()) for path in sorted(DIRECTORY.glob("*.json")))


def reference_for(case: AgentCase) -> AgentCase:
    provider: Final = "anthropic/" if case.model.startswith("openai/") else "openai/"
    return next(row for row in CASES if row.tier == case.tier and row.model.startswith(provider))


def assert_extensions(body: Mapping[str, JsonValue], case: AgentCase) -> None:
    choices: Final = body["choices"]
    assert isinstance(choices, list) and len(choices) == 1
    choice: Final = choices[0]
    assert isinstance(choice, dict)
    message: Final = choice["message"]
    assert isinstance(message, dict)
    for field in case.reasoning_fields:
        assert message.get(field), f"{case.model} lost {field}"
