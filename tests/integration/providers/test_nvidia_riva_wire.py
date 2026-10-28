from collections.abc import Mapping
from typing import Final

import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.openai_wire import chat_reply
from integration._support.riva import Rpc, RivaServer, Word, pcm_bytes, pcm_wav, riva_server
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL: Final = "nvidia_riva/nvidia/parakeet-0.6b-tdt"
_WORDS: Final = (Word("riva", 0, 400), Word("offline", 400, 900), Word("works", 900, 1400))
_TRANSCRIPT: Final = "riva offline works"
_CLIP_SECONDS: Final = 0.5
_REFUSED_FIELD: Final = "riva_offline is not allowed in request body"


def _riva_model(scenario: Scenario, server: RivaServer, **parameters: JsonValue) -> str:
    return scenario.model(
        model=_MODEL,
        api_base=server.address,
        model_info={"mode": "audio_transcription"},
        **parameters,
    )


def _transcribe(
    gateway: Gateway, key: str, model: str, seconds: float = _CLIP_SECONDS, **fields: str
) -> httpx.Response:
    return gateway.request_multipart(
        "/v1/audio/transcriptions",
        {"model": model, **fields},
        {"file": ("clip.wav", pcm_wav(seconds), "audio/wav")},
        key=key,
    )


def _sdk(gateway: Gateway, key: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=key, max_retries=0)


def test_a_form_riva_offline_on_transcriptions_is_refused_before_riva_is_called(gateway: Gateway) -> None:
    with riva_server(_WORDS) as server, gateway.scenario() as scenario:
        model: Final = _riva_model(scenario, server)
        refused: Final = _transcribe(gateway, scenario.key(), model, riva_offline="true")
        assert refused.status_code == 401, refused.text
        assert _REFUSED_FIELD in refused.json()["error"]["message"], refused.text
        assert server.drain() == ()


def test_a_json_riva_offline_on_chat_is_refused_before_the_provider_is_called(gateway: Gateway) -> None:
    def respond(_: Request) -> Reply:
        return chat_reply("chatcmpl-riva-audit", "gpt-4o-mini", "served", stream=False)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        refused: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "riva flag on chat"}], "riva_offline": True},
            key=scenario.key(),
        )
        assert refused.status_code == 401, refused.text
        assert _REFUSED_FIELD in refused.json()["error"]["message"], refused.text
        assert wire.drain() == ()


def test_an_offline_deployment_sends_the_whole_clip_in_one_recognize_call(gateway: Gateway) -> None:
    with riva_server(_WORDS, offline_only=True) as server, gateway.scenario() as scenario:
        model: Final = _riva_model(scenario, server, riva_offline=True)
        transcription: Final = _sdk(gateway, scenario.key()).audio.transcriptions.create(
            model=model,
            file=("clip.wav", pcm_wav(_CLIP_SECONDS), "audio/wav"),
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
        assert transcription.text == _TRANSCRIPT
        assert [(word.word, word.start, word.end) for word in transcription.words or ()] == [
            (word.word, word.start_ms / 1000, word.end_ms / 1000) for word in _WORDS
        ]
        assert [(call.rpc, call.audio_bytes) for call in server.drain()] == [(Rpc.RECOGNIZE, pcm_bytes(_CLIP_SECONDS))]


def test_a_deployment_without_riva_offline_still_streams_the_clip(gateway: Gateway) -> None:
    with riva_server(_WORDS) as server, gateway.scenario() as scenario:
        model: Final = _riva_model(scenario, server, timeout=30)
        served: Final = _transcribe(gateway, scenario.key(), model)
        assert served.status_code == 200, served.text
        assert served.json()["text"] == _TRANSCRIPT, served.text
        assert [(call.rpc, call.audio_bytes) for call in server.drain()] == [(Rpc.STREAMING, pcm_bytes(_CLIP_SECONDS))]


@pytest.mark.parametrize(
    ("sent", "deployment", "status", "rpcs"),
    (
        pytest.param("true", {}, 200, (Rpc.RECOGNIZE,), id="true-on-a-streaming-deployment"),
        pytest.param("false", {"riva_offline": True}, 200, (Rpc.STREAMING,), id="false-on-an-offline-deployment"),
        pytest.param("yes", {}, 400, (), id="malformed"),
        pytest.param("yes", {"drop_params": True}, 200, (Rpc.STREAMING,), id="malformed-under-drop-params"),
    ),
)
def test_an_opted_in_caller_riva_offline_picks_the_rpc(
    gateway: Gateway, sent: str, deployment: Mapping[str, JsonValue], status: int, rpcs: tuple[Rpc, ...]
) -> None:
    with riva_server(_WORDS) as server, gateway.scenario() as scenario:
        model: Final = _riva_model(scenario, server, configurable_clientside_auth_params=["riva_offline"], **deployment)
        answered: Final = _transcribe(gateway, scenario.key(), model, riva_offline=sent)
        assert answered.status_code == status, answered.text
        assert tuple(call.rpc for call in server.drain()) == rpcs


def test_offline_audio_over_the_servers_message_limit_answers_413(gateway: Gateway) -> None:
    seconds: Final = 2.0
    limit: Final = pcm_bytes(seconds) // 2
    with (
        riva_server(_WORDS, offline_only=True, max_receive_bytes=limit) as server,
        gateway.scenario() as scenario,
    ):
        model: Final = _riva_model(scenario, server, riva_offline=True)
        refused: Final = _transcribe(gateway, scenario.key(), model, seconds)
        assert refused.status_code == 413, refused.text
        assert server.drain() == ()


def test_a_stalled_offline_recognize_answers_408_at_the_deployment_timeout(gateway: Gateway) -> None:
    with riva_server(_WORDS, offline_only=True, stall=True) as server, gateway.scenario() as scenario:
        model: Final = _riva_model(scenario, server, riva_offline=True, timeout=2)
        timed_out: Final = _transcribe(gateway, scenario.key(), model)
        assert timed_out.status_code == 408, timed_out.text
        calls: Final = eventually(server.drain, lambda observed: len(observed) > 0, seconds=30)
        assert all(call.rpc is Rpc.RECOGNIZE and call.cancelled for call in calls), calls
