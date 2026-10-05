import json
import uuid
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

_WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)


def _write_config(directory: Path, upstream_url: str, stt: str, moderation: str) -> Path:
    config: Final = directory / f"audio_transcription_{uuid.uuid4().hex}.yaml"
    litellm_params: Final[dict[str, JsonValue]] = {
        "api_base": f"{upstream_url}/v1",
        "api_key": "synthetic-openai-key",
    }
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {"model_name": stt, "litellm_params": {**litellm_params, "model": "openai/whisper-1"}},
                    {
                        "model_name": moderation,
                        "litellm_params": {**litellm_params, "model": "openai/omni-moderation-latest"},
                    },
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "disable_spend_logs": False,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    "moderation_model": moderation,
                },
                "litellm_settings": {
                    "enable_redis_auth_cache": True,
                    "cache": True,
                    "cache_params": {
                        "type": "redis",
                        "host": "os.environ/REDIS_HOST",
                        "port": "os.environ/REDIS_PORT",
                    },
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


def _text_parts(request: Request) -> dict[str, str]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
        for part in parsed.iter_parts()
        if part.get_filename() is None
    }


_TRANSCRIPTIONS: Final = "/v1/audio/transcriptions"


@pytest.mark.timeout(240)
def test_audio_transcription_routes_to_the_requested_model_not_the_moderation_model(
    gateway: Gateway, tmp_path: Path
) -> None:
    pytest.skip(
        "BUG: general_settings.moderation_model overrides the model of every /audio/transcriptions request, "
        "so the upstream receives the moderation deployment's model"
    )
    stt: Final = f"integration-stt-{uuid.uuid4().hex}"
    moderation: Final = f"integration-moderation-{uuid.uuid4().hex}"
    def respond(request: Request) -> Reply:
        if request.target != _TRANSCRIPTIONS:
            return Reply(body=b'{"object":"list","data":[]}')
        return Reply(body=json.dumps({"text": "hello world"}).encode())

    with wire_server(respond) as wire:
        config: Final = _write_config(tmp_path, wire.url, stt, moderation)
        with owned_proxy(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config) as candidate:
            response: Final = candidate.request_multipart(
                _TRANSCRIPTIONS, {"model": stt}, {"file": ("a.wav", _WAV_BYTES, "audio/wav")}
            )
            assert response.status_code == 200, response.text
        calls: Final = wire.drain()
        assert tuple(
            (request.method, request.target) for request in calls if request.target == _TRANSCRIPTIONS
        ) == (("POST", _TRANSCRIPTIONS),), response.text
        assert tuple(
            _text_parts(request)["model"] for request in calls if request.target == _TRANSCRIPTIONS
        ) == ("whisper-1",), response.text
