



from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server

import litellm
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

_MODEL_ID: Final = "global.moonshotai.kimi-k3"


_CONVERSE_MODEL: Final = f"bedrock/converse/{_MODEL_ID}"


@dataclass(frozen=True, slots=True)
class _Consumed:
    text: str
    finish_reasons: tuple[str | None, ...]


def _consume_sync_stream(wire: Wire) -> _Consumed:
    response: Final = litellm.completion(
        model=_CONVERSE_MODEL,
        messages=[{"role": "user", "content": "What does the sync decoder do with this stream?"}],
        max_tokens=16,
        stream=True,
        api_base=wire.url,
        aws_access_key_id="AKIASCRIPTEDPROVIDER",
        aws_secret_access_key="scripted-secret",
        aws_region_name="us-east-1",
        num_retries=0,
    )
    assert isinstance(response, CustomStreamWrapper), type(response)
    chunks: Final = tuple(response)
    return _Consumed(
        text="".join(str(chunk.choices[0].delta.content or "") for chunk in chunks),
        finish_reasons=tuple(chunk.choices[0].finish_reason for chunk in chunks),
    )


@pytest.mark.covers("other.provider_wire.k04_sync_stream_open_400_carries_the_bedrock_body_as_text")
def test_k04_sync_stream_open_400_carries_the_bedrock_body_as_text() -> None:
    body: Final = '{"message":"The request is invalid."}'

    def reject(_: Request) -> Reply:
        return Reply(status=400, body=body.encode())

    with wire_server(reject) as wire:
        with pytest.raises(BadRequestError) as raised:
            _consume_sync_stream(wire)
        assert raised.value.status_code == 400, raised.value
        assert f"BedrockException - {body}" in raised.value.message, raised.value.message
        assert len(wire.drain()) == 1
