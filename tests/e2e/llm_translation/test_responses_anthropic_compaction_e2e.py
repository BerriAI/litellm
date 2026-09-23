"""Live /v1/responses -> Anthropic compaction round-trip through the chat bridge.

Anthropic's ``compact_20260112`` context-management edit triggers once input tokens
cross ``compact_threshold`` (Anthropic requires >= 50000) and returns a signed
``compaction`` block. A stateless Responses client must get that block back as a
``{"type": "compaction", "id": "cmp_...", "encrypted_content": ...}`` output item and
replay it on the next request so Anthropic drops everything before it (#41456). On the
buggy path the block never leaves ``provider_specific_fields`` and replayed items are
dropped, so the client replays the full transcript forever.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from e2e_config import MASTER_KEY, PROXY_BASE_URL, unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel, Field
from transport import HttpTransport

pytestmark = pytest.mark.e2e

ANTHROPIC_COMPACTION_BACKEND: Final = "bedrock/invoke/us.anthropic.claude-sonnet-5"
COMPACT_THRESHOLD: Final = 50000
LONG_DOCUMENT: Final = ("The quick brown fox jumps over the lazy dog. " * 25 + "\n") * 300


class _ContextManagementEdit(BaseModel):
    type: str
    compact_threshold: int


class _ResponsesBody(BaseModel):
    model_config = {"extra": "allow"}
    model: str
    input: list[object]
    context_management: list[_ContextManagementEdit]
    stream: bool | None = None


class _Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class _ResponsesOutputItem(BaseModel):
    model_config = {"extra": "allow"}
    type: str
    id: str | None = None
    encrypted_content: str | None = None


class _ResponsesResult(BaseModel):
    model_config = {"extra": "allow"}
    output: list[_ResponsesOutputItem] = Field(default_factory=list)
    usage: _Usage = _Usage()


class _StreamEventItem(BaseModel):
    model_config = {"extra": "allow"}
    type: str | None = None
    id: str | None = None
    encrypted_content: str | None = None


class _StreamEvent(BaseModel):
    model_config = {"extra": "allow"}
    type: str
    item: _StreamEventItem | None = None


def _transport() -> HttpTransport:
    return HttpTransport(base_url=PROXY_BASE_URL, master_key=MASTER_KEY, request_timeout=600.0)


def _body(model: str, items: list[object], *, stream: bool | None = None) -> _ResponsesBody:
    return _ResponsesBody(
        model=model,
        input=items,
        context_management=[_ContextManagementEdit(type="compaction", compact_threshold=COMPACT_THRESHOLD)],
        stream=stream,
    )


def _first_turn_items() -> list[object]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Here is a long document:\n" + LONG_DOCUMENT + "\nSummarize it in one sentence.",
                }
            ],
        }
    ]


def _post_responses(transport: HttpTransport, key: str, body: _ResponsesBody) -> StreamingResponse:
    return transport.send("/v1/responses", headers=transport.bearer(key), json=body, stream=body.stream is True)


def _compaction_items(result: _ResponsesResult) -> tuple[_ResponsesOutputItem, ...]:
    return tuple(item for item in result.output if item.type == "compaction")


class TestResponsesAnthropicCompaction:
    @pytest.fixture
    def anthropic_model(self, proxy: ProxyClient, resources: ResourceManager) -> str:
        model = f"e2e-responses-compaction-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=ANTHROPIC_COMPACTION_BACKEND,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION_NAME",
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        return model

    @pytest.mark.covers("llm.responses.bedrock_invoke.compaction.nonstream.output_item")
    def test_nonstream_returns_compaction_output_item(
        self, proxy: ProxyClient, resources: ResourceManager, anthropic_model: str
    ) -> None:
        transport = _transport()
        result = _post_responses(transport, resources.key(), _body(anthropic_model, _first_turn_items()))
        assert result.ok, f"/v1/responses failed: {result.status_code} {result.body[:500]}"
        response = _ResponsesResult.model_validate_json(result.body)
        items = _compaction_items(response)
        assert items, f"no compaction item in output: {result.body[:800]}"
        assert all(item.encrypted_content for item in items), f"empty encrypted_content: {result.body[:800]}"

    @pytest.mark.covers("llm.responses.bedrock_invoke.compaction.stream.output_item")
    def test_stream_returns_compaction_item_events(
        self, proxy: ProxyClient, resources: ResourceManager, anthropic_model: str
    ) -> None:
        transport = _transport()
        result = _post_responses(
            transport, resources.key(), _body(anthropic_model, _first_turn_items(), stream=True)
        )
        assert result.ok and result.is_streaming, f"stream not established: {result.status_code} {result.body[:500]}"
        assert result.stream_error is None, f"stream error event: {result.stream_error}"
        events = [_StreamEvent.model_validate_json(payload) for payload in result.stream_events]
        added = [e.item for e in events if e.type == "response.output_item.added" and e.item and e.item.type == "compaction"]
        done = [e.item for e in events if e.type == "response.output_item.done" and e.item and e.item.type == "compaction"]
        assert added, f"no response.output_item.added for compaction: {result.stream_events[:6]}"
        assert done and all(item.encrypted_content for item in done), (
            f"no response.output_item.done compaction item with encrypted_content: {result.stream_events[-6:]}"
        )

    @pytest.mark.covers("llm.responses.bedrock_invoke.compaction.replay.shrinks_input")
    def test_replaying_compaction_item_shrinks_next_input(
        self, proxy: ProxyClient, resources: ResourceManager, anthropic_model: str
    ) -> None:
        transport = _transport()
        key = resources.key()
        first = _post_responses(transport, key, _body(anthropic_model, _first_turn_items()))
        assert first.ok, f"first /v1/responses failed: {first.status_code} {first.body[:500]}"
        first_response = _ResponsesResult.model_validate_json(first.body)
        compaction = _compaction_items(first_response)
        assert compaction, f"first response carried no compaction item to replay: {first.body[:800]}"

        replay_input: list[object] = [
            json.loads(item.model_dump_json(exclude_none=True)) for item in first_response.output
        ] + [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "In one sentence, what did I just ask about?"}],
            }
        ]
        second = _post_responses(transport, key, _body(anthropic_model, replay_input))
        assert second.ok, f"replay /v1/responses failed: {second.status_code} {second.body[:500]}"
        second_response = _ResponsesResult.model_validate_json(second.body)
        assert second_response.usage.input_tokens < first_response.usage.input_tokens, (
            f"replay did not shrink input: first={first_response.usage.input_tokens} "
            f"second={second_response.usage.input_tokens} body={second.body[:500]}"
        )
