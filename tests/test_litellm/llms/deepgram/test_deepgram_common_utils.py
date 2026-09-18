import math
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

import litellm
from litellm.llms.deepgram.common_utils import (
    deepgram_listen_addon_pricing_models,
    deepgram_listen_audio_seconds,
    deepgram_listen_base_pricing_models,
    deepgram_listen_callback_params,
    deepgram_listen_channel_count,
    deepgram_listen_model,
    deepgram_listen_requested_model,
    deepgram_listen_transcript,
    deepgram_listen_websocket_target,
)

NOVA_3_URL: Final = "wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16&sample_rate=16000"


def _results(
    start: object,
    duration: object,
    transcript: str = "",
    is_final: object = True,
    channel_index: object = (0, 1),
) -> dict[str, object]:
    return {
        "type": "Results",
        "start": start,
        "duration": duration,
        "is_final": is_final,
        "channel_index": list(channel_index) if isinstance(channel_index, tuple) else channel_index,
        "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.9}]},
    }


def _metadata(duration: object, channels: object = 1) -> dict[str, object]:
    return {"type": "Metadata", "request_id": "req-1", "duration": duration, "channels": channels}


@pytest.mark.parametrize(
    ("api_base", "query_string", "expected"),
    [
        pytest.param(
            None,
            "model=nova-3&encoding=linear16",
            "wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16",
            id="default",
        ),
        pytest.param(
            None,
            "encoding=linear16&sample_rate=16000",
            "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&model=nova-3",
            id="model added when missing",
        ),
        pytest.param(
            None,
            "model=&encoding=linear16",
            "wss://api.deepgram.com/v1/listen?encoding=linear16&model=nova-3",
            id="empty model replaced",
        ),
        pytest.param(
            "http://localhost:9000/v1/",
            "model=nova-2",
            "ws://localhost:9000/v1/listen?model=nova-2",
            id="custom base becomes ws",
        ),
        pytest.param(
            "wss://dg.internal/v1",
            "model=nova-3&keywords=a&keywords=b",
            "wss://dg.internal/v1/listen?model=nova-3&keywords=a&keywords=b",
            id="repeated keys preserved",
        ),
    ],
)
def test_deepgram_listen_websocket_target(api_base: str | None, query_string: str, expected: str):
    assert deepgram_listen_websocket_target(api_base=api_base, query_string=query_string) == expected


@pytest.mark.parametrize(
    ("query_string", "expected"),
    [
        pytest.param("model=nova-3&encoding=linear16", (), id="no callback"),
        pytest.param("model=nova-3&callback=https%3A%2F%2Fevil.example%2Fsink", ("callback",), id="callback"),
        pytest.param(
            "callback_method=put&model=nova-3&callback=wss%3A%2F%2Fevil.example",
            ("callback", "callback_method"),
            id="callback and method",
        ),
        pytest.param("model=nova-3&callback_method=put", ("callback_method",), id="method alone"),
        pytest.param("model=nova-3&callbacks=x&my_callback=y", (), id="only exact names match"),
    ],
)
def test_deepgram_listen_callback_params(query_string: str, expected: tuple[str, ...]):
    assert deepgram_listen_callback_params(query_string) == expected


@pytest.mark.parametrize(
    ("frames", "expected_seconds"),
    [
        pytest.param((_results(0.0, 2.0), _results(2.0, 3.5), _metadata(6.25)), 6.25, id="metadata wins"),
        pytest.param((_metadata(4.0), _results(0.0, 9.0), _metadata(5.5)), 5.5, id="last metadata wins"),
        pytest.param((_results(0.0, 2.0), _results(2.0, 3.5), _results(1.0, 1.0)), 5.5, id="furthest results end"),
        pytest.param((_results(0.0, 0.0), _metadata(0.0)), 0.0, id="zero metadata is a real zero"),
        pytest.param(
            (_metadata(0.0), _results(0.0, 2.0), _results(2.0, 3.5)),
            5.5,
            id="handshake metadata zero does not hide streamed results",
        ),
        pytest.param((_metadata(0.0), _results(0.0, 2.0), _metadata(0.0)), 2.0, id="only zero metadata frames"),
        pytest.param((_results(0.0, 1.5), _metadata("6.25")), 1.5, id="string metadata is ignored"),
        pytest.param((_results(0.0, 1.5), _metadata(True)), 1.5, id="boolean metadata is ignored"),
        pytest.param((_results(0.0, 1.5), _metadata(-3.0)), 1.5, id="negative metadata is ignored"),
        pytest.param((_results(0.0, 1.5), _metadata(math.nan), _metadata(math.inf)), 1.5, id="nan/inf ignored"),
        pytest.param((_results("0", 2.0), _results(0.0, None), _results(0.0, 0.75)), 0.75, id="malformed results"),
        pytest.param(({"type": "SpeechStarted", "timestamp": 3.0}, {"type": "UtteranceEnd"}), 0.0, id="no usage"),
        pytest.param((), 0.0, id="no frames"),
    ],
)
def test_deepgram_listen_audio_seconds(frames: Sequence[Mapping[str, object]], expected_seconds: float):
    assert deepgram_listen_audio_seconds(frames) == expected_seconds


@pytest.mark.parametrize(
    ("frames", "upstream_url", "expected_channels"),
    [
        pytest.param((_results(0.0, 2.0), _metadata(6.25)), NOVA_3_URL, 1, id="mono"),
        pytest.param((_results(0.0, 2.0, channel_index=(0, 2)), _metadata(6.25, 2)), NOVA_3_URL, 2, id="stereo"),
        pytest.param((_metadata(1.0, 3), _metadata(1.0, 5)), NOVA_3_URL, 5, id="last metadata wins"),
        pytest.param(
            (_metadata(1.0, 20), _results(0.0, 1.0, channel_index=(1, 2))),
            NOVA_3_URL,
            20,
            id="metadata beats channel_index",
        ),
        pytest.param(
            (_results(0.0, 1.0, channel_index=(0, 2)), _results(0.0, 1.0, channel_index=(3, 4))),
            NOVA_3_URL,
            4,
            id="widest channel_index without metadata",
        ),
        pytest.param(
            (_results(0.0, 1.0, channel_index=(0, 2)),),
            f"{NOVA_3_URL}&channels=7&multichannel=true",
            2,
            id="frames beat the declared query",
        ),
        pytest.param((), f"{NOVA_3_URL}&channels=7&multichannel=true", 7, id="declared query when no frames"),
        pytest.param((), f"{NOVA_3_URL}&channels=0", 1, id="zero declared channels"),
        pytest.param((), f"{NOVA_3_URL}&channels=-2", 1, id="negative declared channels"),
        pytest.param((), f"{NOVA_3_URL}&channels=two", 1, id="non numeric declared channels"),
        pytest.param((), NOVA_3_URL, 1, id="nothing declared"),
        pytest.param((_metadata(1.0, "2"), _metadata(1.0, True), _metadata(1.0, 0)), NOVA_3_URL, 1, id="bad metadata"),
        pytest.param((_metadata(1.0, 3), _metadata(1.0, True)), NOVA_3_URL, 3, id="boolean does not shadow a count"),
        pytest.param((_metadata(1.0, 2.0), _metadata(1.0, -1)), NOVA_3_URL, 1, id="float and negative metadata"),
        pytest.param(
            (_metadata(1.0, 2), {**_results(0.0, 1.0), "channels": 9}, {"type": "UtteranceEnd", "channels": 11}),
            NOVA_3_URL,
            2,
            id="channels on non metadata frames ignored",
        ),
        pytest.param(
            (
                _results(0.0, 1.0, channel_index=[0]),
                _results(0.0, 1.0, channel_index=(0, "2")),
                _results(0.0, 1.0, channel_index=(0, 0)),
            ),
            NOVA_3_URL,
            1,
            id="bad channel_index",
        ),
    ],
)
def test_deepgram_listen_channel_count(
    frames: Sequence[Mapping[str, object]], upstream_url: str, expected_channels: int
):
    assert deepgram_listen_channel_count(frames, upstream_url) == expected_channels


def test_deepgram_listen_transcript_joins_final_results_only():
    frames = (
        _results(0.0, 1.0, "hello wor", is_final=False),
        _results(0.0, 1.5, "hello world"),
        _results(1.5, 0.5, "", is_final=True),
        _results(2.0, 1.0, "how are you", is_final="yes"),
        {"type": "Results", "start": 3.0, "duration": 1.0, "is_final": True, "channel": {"alternatives": []}},
        _results(4.0, 1.0, "goodbye"),
        _metadata(5.0),
    )
    assert deepgram_listen_transcript(frames) == "hello world goodbye"


@pytest.mark.parametrize(
    ("upstream_url", "expected_model"),
    [
        (NOVA_3_URL, "nova-3"),
        ("wss://api.deepgram.com/v1/listen?encoding=linear16&model=nova-2-medical", "nova-2-medical"),
        ("wss://api.deepgram.com/v1/listen?model=nova-3&model=nova-2", "nova-3"),
        ("wss://api.deepgram.com/v1/listen?encoding=linear16", litellm.constants.DEEPGRAM_LISTEN_DEFAULT_MODEL),
    ],
)
def test_deepgram_listen_model_comes_from_the_upstream_query(upstream_url: str, expected_model: str):
    assert deepgram_listen_model(upstream_url) == expected_model


@pytest.mark.parametrize(
    "query_string",
    ["model=nova-2&language=en", "language=en", "model=&language=en", "", "model=nova-3-medical"],
)
def test_requested_model_is_the_model_the_upstream_target_will_carry(query_string: str):
    """Authorization runs against ``deepgram_listen_requested_model``; the upstream URL is built separately, so the
    two must always agree or a key could be authorized for one model and reach another."""
    target: Final = deepgram_listen_websocket_target(None, query_string)
    assert deepgram_listen_requested_model(query_string) == deepgram_listen_model(target)


@pytest.mark.parametrize(
    ("upstream_url", "expected"),
    [
        pytest.param(NOVA_3_URL, ("streaming/nova-3", "nova-3"), id="monolingual"),
        pytest.param(f"{NOVA_3_URL}&language=en", ("streaming/nova-3", "nova-3"), id="explicit language"),
        pytest.param(
            f"{NOVA_3_URL}&language=multi",
            ("streaming/nova-3-multilingual", "streaming/nova-3", "nova-3"),
            id="multilingual",
        ),
        pytest.param(
            f"{NOVA_3_URL}&language=MULTI",
            ("streaming/nova-3-multilingual", "streaming/nova-3", "nova-3"),
            id="multilingual any case",
        ),
        pytest.param(
            "wss://api.deepgram.com/v1/listen?model=nova-2&language=multi",
            ("streaming/nova-2-multilingual", "streaming/nova-2", "nova-2"),
            id="other model",
        ),
    ],
)
def test_deepgram_listen_base_pricing_models(upstream_url: str, expected: tuple[str, ...]):
    assert deepgram_listen_base_pricing_models(upstream_url) == expected


@pytest.mark.parametrize(
    ("upstream_url", "expected"),
    [
        pytest.param(NOVA_3_URL, (), id="no add-ons"),
        pytest.param(f"{NOVA_3_URL}&redact=pci", ("streaming/redact",), id="redact"),
        pytest.param(f"{NOVA_3_URL}&redact=pci&redact=ssn", ("streaming/redact",), id="repeated redact once"),
        pytest.param(f"{NOVA_3_URL}&keyterm=a&keyterm=b", ("streaming/keyterm",), id="keyterm"),
        pytest.param(f"{NOVA_3_URL}&detect_entities=true", ("streaming/detect_entities",), id="detect_entities"),
        pytest.param(f"{NOVA_3_URL}&diarize=true", ("streaming/diarize",), id="diarize"),
        pytest.param(f"{NOVA_3_URL}&diarize_model=v1", ("streaming/diarize",), id="diarize_model"),
        pytest.param(f"{NOVA_3_URL}&diarize=true&diarize_model=latest", ("streaming/diarize",), id="diarize both once"),
        pytest.param(f"{NOVA_3_URL}&detect_entities=false&diarize=FALSE&redact=", (), id="disabled"),
        pytest.param(
            f"{NOVA_3_URL}&detect_entities=false&detect_entities=true",
            ("streaming/detect_entities",),
            id="any enabling value wins",
        ),
        pytest.param(
            f"{NOVA_3_URL}&diarize=true&redact=pci&keyterm=x&detect_entities=true",
            ("streaming/detect_entities", "streaming/diarize", "streaming/keyterm", "streaming/redact"),
            id="all, sorted",
        ),
    ],
)
def test_deepgram_listen_addon_pricing_models(upstream_url: str, expected: tuple[str, ...]):
    assert deepgram_listen_addon_pricing_models(upstream_url) == expected
