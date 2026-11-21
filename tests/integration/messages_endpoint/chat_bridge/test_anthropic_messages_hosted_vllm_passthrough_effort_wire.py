import json
import uuid
from typing import Final

import anthropic
import pytest
from integration._support.client import Gateway
from integration._support.safeguards_rig import native_message, native_stream_frames, text_block
from integration._support.wire import Reply, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "example/reasoning-model"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_THINKING: Final[dict[str, JsonValue]] = {"type": "adaptive"}
_OUTPUT_CONFIG: Final[dict[str, JsonValue]] = {"effort": "low"}


def _reply(identity: str, stream: bool) -> Reply:
    message: Final = native_message(identity, content=(text_block("Done."),), stop_reason="end_turn", model=_BACKEND)
    if stream:
        return Reply(content_type="text/event-stream", chunks=native_stream_frames(message))
    return Reply(body=json.dumps(message).encode())


def _only_body(wire: Wire) -> dict[str, JsonValue]:
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", "/v1/messages")]
    return _JSON_OBJECT.validate_json(received[0].body)


@pytest.mark.parametrize("stream", [False, True])
async def test_native_passthrough_forwards_adaptive_thinking_and_effort_to_hosted_vllm(
    gateway: Gateway, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(lambda _: _reply(marker, stream)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"hosted_vllm/{_BACKEND}",
            api_base=wire.url + "/v1",
            api_key="synthetic-hosted-vllm-key",
            drop_params=True,
            model_info={"supported_endpoints": ["/v1/messages", "/v1/chat/completions"]},
        )
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
        )
        response: Final = await client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": f"Say done {marker}"}],
            thinking=_THINKING,  # pyright: ignore[reportArgumentType]  # plain JSON thinking config
            extra_body={"output_config": _OUTPUT_CONFIG},
            stream=stream,
        )
        if isinstance(response, anthropic.AsyncStream):
            assert [event.type async for event in response][-1] == "message_stop"
        else:
            assert response.id == f"msg_{marker}"
        body: Final = _only_body(wire)
        assert (body.get("thinking"), body.get("output_config")) == (_THINKING, _OUTPUT_CONFIG)
