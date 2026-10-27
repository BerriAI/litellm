import json
import math
from collections.abc import Mapping, Sequence
from typing import Final
from urllib.parse import parse_qs, urlparse

import pytest

import litellm
from litellm.llms.deepgram.common_utils import (
    deepgram_listen_addon_pricing_models,
    deepgram_listen_audio_seconds,
    deepgram_listen_callback_params,
    deepgram_listen_channel_count,
    deepgram_listen_is_priced,
    deepgram_listen_model,
    deepgram_listen_pricing_model,
    deepgram_listen_registry_key,
    deepgram_listen_requested_model,
    deepgram_listen_transcript,
    deepgram_listen_websocket_target,
    deepgram_speak_character_rate,
    deepgram_speak_characters,
    deepgram_speak_is_priced,
    deepgram_speak_model,
    deepgram_speak_registry_key,
    deepgram_speak_requested_model,
    deepgram_speak_websocket_target,
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
        pytest.param(
            None,
            "model=nova-2&encoding=linear16&model=nova-3",
            "wss://api.deepgram.com/v1/listen?model=nova-2&encoding=linear16",
            id="only the authorized first model reaches deepgram",
        ),
        pytest.param(
            None,
            "language=en&model=nova-3&language=multi",
            "wss://api.deepgram.com/v1/listen?language=en&model=nova-3",
            id="only the priced first language reaches deepgram",
        ),
        pytest.param(
            None,
            "model=&model=nova-2",
            "wss://api.deepgram.com/v1/listen?model=nova-3",
            id="blank first model is the default, later models dropped",
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
    [
        "model=nova-2&language=en",
        "language=en",
        "model=&language=en",
        "",
        "model=nova-3-medical",
        "model=nova-2&model=nova-3",
        "model=&model=nova-3-medical",
    ],
)
def test_requested_model_is_the_only_model_the_upstream_target_carries(query_string: str):
    """Authorization runs against ``deepgram_listen_requested_model``; the upstream URL is built separately, so the
    two must always agree or a key could be authorized for one model and reach another. Deepgram reads the last
    repeated ``model``, so the target must carry exactly one."""
    target: Final = deepgram_listen_websocket_target(None, query_string)
    assert parse_qs(urlparse(target).query)["model"] == [deepgram_listen_requested_model(query_string)]
    assert deepgram_listen_requested_model(query_string) == deepgram_listen_model(target)


@pytest.mark.parametrize(
    ("upstream_url", "expected"),
    [
        pytest.param(NOVA_3_URL, "streaming/nova-3", id="monolingual"),
        pytest.param(f"{NOVA_3_URL}&language=en", "streaming/nova-3", id="explicit language"),
        pytest.param(f"{NOVA_3_URL}&language=multi", "streaming/nova-3-multilingual", id="multilingual"),
        pytest.param(f"{NOVA_3_URL}&language=MULTI", "streaming/nova-3-multilingual", id="multilingual any case"),
        pytest.param(
            "wss://api.deepgram.com/v1/listen?model=nova-2&language=multi",
            "streaming/nova-2-multilingual",
            id="other model",
        ),
        pytest.param("wss://api.deepgram.com/v1/listen?encoding=linear16", "streaming/nova-3", id="default model"),
    ],
)
def test_deepgram_listen_pricing_model_is_the_streaming_entry_never_the_prerecorded_one(
    upstream_url: str, expected: str
):
    assert deepgram_listen_pricing_model(upstream_url) == expected
    assert deepgram_listen_registry_key(upstream_url) == f"deepgram/{expected}"


NOVA_2_URL: Final = "wss://api.deepgram.com/v1/listen?model=nova-2"


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    ("upstream_url", "extra_rows", "expected"),
    [
        pytest.param(NOVA_3_URL, (), True, id="streaming entry present"),
        pytest.param(f"{NOVA_3_URL}&language=multi", (), True, id="multilingual entry present"),
        pytest.param(NOVA_2_URL, (), False, id="only the pre-recorded entry"),
        pytest.param(f"{NOVA_2_URL}&language=multi", ("deepgram/streaming/nova-2",), False, id="needs multilingual"),
        pytest.param("wss://api.deepgram.com/v1/listen?model=nova-99-unmapped", (), False, id="nothing priced"),
        pytest.param(NOVA_2_URL, ("deepgram/streaming/nova-2",), True, id="operator-supplied streaming entry"),
        pytest.param(NOVA_2_URL, ("streaming/nova-2",), False, id="a row under another key is not the entry"),
    ],
)
def test_deepgram_listen_is_priced(
    monkeypatch: pytest.MonkeyPatch, upstream_url: str, extra_rows: tuple[str, ...], expected: bool
):
    """The bundled map prices only nova-3 for streaming; nova-2 has a pre-recorded row, which must never count."""
    monkeypatch.delitem(litellm.model_cost, "deepgram/streaming/nova-2", raising=False)
    assert "deepgram/nova-2" in litellm.model_cost
    for row in extra_rows:
        monkeypatch.setitem(litellm.model_cost, row, dict(litellm.model_cost["deepgram/streaming/nova-3"]))

    assert deepgram_listen_is_priced(upstream_url) is expected


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


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    ("pricing_model", "usd_per_thousand_characters"),
    [
        pytest.param("aura-2", 0.030, id="aura-2"),
        pytest.param("aura", 0.015, id="aura-1"),
    ],
)
def test_aura_rows_bill_speech_per_character_at_the_published_rate(
    pricing_model: str, usd_per_thousand_characters: float
) -> None:
    """Pay as you go rates from https://deepgram.com/pricing, checked 2026-10-26"""
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=pricing_model, custom_llm_provider="deepgram", call_type="speech", prompt_characters=1000
    )

    assert prompt_cost == pytest.approx(usd_per_thousand_characters)
    assert completion_cost == 0.0


SPEAK_BASE: Final = "wss://api.deepgram.com/v1/speak"


@pytest.mark.parametrize(
    ("api_base", "query_string", "expected"),
    [
        pytest.param(None, "", f"{SPEAK_BASE}?model=aura-asteria-en", id="default model added"),
        pytest.param(None, "model=", f"{SPEAK_BASE}?model=aura-asteria-en", id="empty model replaced"),
        pytest.param(
            None,
            "model=aura-2-thalia-en&encoding=linear16&sample_rate=24000",
            f"{SPEAK_BASE}?model=aura-2-thalia-en&encoding=linear16&sample_rate=24000",
            id="query kept in order",
        ),
        pytest.param(
            None,
            "model=aura-2-thalia-en&model=aura-2-zeus-en",
            f"{SPEAK_BASE}?model=aura-2-thalia-en",
            id="only the first model",
        ),
        pytest.param(
            "http://127.0.0.1:8080/v1", "model=aura-2-thalia-en", "ws://127.0.0.1:8080/v1/speak?model=aura-2-thalia-en"
        ),
        pytest.param(
            "https://dg.internal/v1/",
            "encoding=mulaw",
            "wss://dg.internal/v1/speak?encoding=mulaw&model=aura-asteria-en",
        ),
    ],
)
def test_deepgram_speak_websocket_target(api_base: str | None, query_string: str, expected: str) -> None:
    assert deepgram_speak_websocket_target(api_base=api_base, query_string=query_string) == expected


@pytest.mark.parametrize(
    "query_string",
    ["", "model=aura-2-thalia-en", "model=aura-2-thalia-en&model=aura-2-zeus-en", "encoding=mulaw&model=aura-luna-en"],
)
def test_the_authorized_speak_model_is_the_only_model_the_upstream_target_carries(query_string: str) -> None:
    """The key is checked against deepgram_speak_requested_model, so the upstream must use exactly that voice"""
    target: Final = deepgram_speak_websocket_target(api_base=None, query_string=query_string)

    assert parse_qs(urlparse(target).query)["model"] == [deepgram_speak_requested_model(query_string)]
    assert deepgram_speak_model(target) == deepgram_speak_requested_model(query_string)


@pytest.mark.parametrize(
    ("model", "expected_family"),
    [
        # Voice names from https://developers.deepgram.com/docs/tts-models, checked 2026-10-26
        pytest.param("aura-2-thalia-en", "aura-2", id="aura-2 english"),
        pytest.param("aura-2-celeste-es", "aura-2", id="aura-2 spanish"),
        pytest.param("aura-2-rhea-nl", "aura-2", id="aura-2 dutch"),
        pytest.param("aura-2-agathe-fr", "aura-2", id="aura-2 french"),
        pytest.param("aura-2-viktoria-de", "aura-2", id="aura-2 german"),
        pytest.param("aura-2-livia-it", "aura-2", id="aura-2 italian"),
        pytest.param("aura-2-izanami-ja", "aura-2", id="aura-2 japanese"),
        pytest.param("aura-asteria-en", "aura", id="aura-1"),
        pytest.param("aura-zeus-en", "aura", id="another aura-1"),
        pytest.param("aura-3-foo-en", "aura-3-foo-en", id="unknown aura generation"),
        pytest.param("aura-2", "aura-2", id="bare aura-2"),
        pytest.param("aura-x", "aura-x", id="aura without a language"),
        pytest.param("aura-2-thalia-en-extra", "aura-2-thalia-en-extra", id="trailing part"),
        pytest.param("flux-tts-en", "flux-tts-en", id="other family"),
        pytest.param("nova-3", "nova-3", id="listen model"),
    ],
)
def test_deepgram_speak_registry_key_names_the_aura_generation_row_else_the_voice(
    model: str, expected_family: str
) -> None:
    target: Final = deepgram_speak_websocket_target(api_base=None, query_string=f"model={model}")

    assert deepgram_speak_registry_key(target) == f"deepgram/{expected_family}"


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param("aura-2-thalia-en", True, id="aura-2"),
        pytest.param("aura-asteria-en", True, id="aura-1"),
        pytest.param("aura-3-foo-en", False, id="unknown aura generation"),
        pytest.param("flux-tts-en", False, id="unpriced family"),
        pytest.param("nova-3", False, id="a listen model has no speech row"),
    ],
)
def test_deepgram_speak_is_priced(model: str, expected: bool) -> None:
    assert deepgram_speak_is_priced(f"{SPEAK_BASE}?model={model}") is expected


@pytest.mark.usefixtures("local_model_cost_map")
def test_deepgram_speak_is_not_priced_once_its_family_row_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(litellm.model_cost, "deepgram/aura-2")

    assert deepgram_speak_is_priced(f"{SPEAK_BASE}?model=aura-2-thalia-en") is False


@pytest.mark.usefixtures("local_model_cost_map")
def test_a_cost_map_row_named_after_a_new_voice_prices_it_at_that_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 1008 refusal tells the operator to add the voice's row, so adding it must be enough"""
    url: Final = f"{SPEAK_BASE}?model=aura-3-foo-en"
    monkeypatch.setitem(
        litellm.model_cost,
        "deepgram/aura-3-foo-en",
        {"input_cost_per_character": 4e-05, "litellm_provider": "deepgram", "mode": "audio_speech"},
    )

    assert deepgram_speak_character_rate(url) == 4e-05
    assert deepgram_speak_is_priced(url) is True


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        pytest.param(
            json.dumps({"type": "Speak", "text": "The quick brown fox jumps over the lazy dog."}), 44, id="speak"
        ),
        pytest.param(json.dumps({"type": "Speak", "text": "Gr\u00fc\u00dfe \U0001f44b"}), 7, id="code points"),
        pytest.param(
            json.dumps({"type": "Speak", "text": "Sent as a binary frame."}).encode(), 23, id="speak in a binary frame"
        ),
        pytest.param(b"\xff\xfe\x00", 0, id="binary frame that is not utf-8"),
        pytest.param(json.dumps({"type": "Speak", "text": ""}), 0, id="empty text"),
        pytest.param(json.dumps({"type": "Flush"}), 0, id="flush"),
        pytest.param(json.dumps({"type": "Clear"}), 0, id="clear"),
        pytest.param(json.dumps({"type": "Close"}), 0, id="close"),
        pytest.param(json.dumps({"type": "Speak"}), 0, id="speak without text"),
        pytest.param(json.dumps({"type": "Speak", "text": 12345}), 0, id="text not a string"),
        pytest.param(json.dumps({"type": "Speak", "text": ["a", "b"]}), 0, id="text a list"),
        pytest.param(json.dumps([{"type": "Speak", "text": "hello"}]), 0, id="not an object"),
        pytest.param("hello there", 0, id="not json"),
        pytest.param("", 0, id="empty frame"),
        pytest.param("[" * 100_000, 0, id="nesting deep enough to recurse"),
        pytest.param("1" * 5_000, 0, id="integer longer than the parser allows"),
    ],
)
def test_deepgram_speak_characters_counts_speak_text_and_never_raises(frame: str | bytes, expected: int) -> None:
    assert deepgram_speak_characters(frame) == expected
