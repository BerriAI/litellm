from collections.abc import Mapping
from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.audio_output_refusal import (
    AudioOnlyModel,
    AudioOutputRefusal,
    AudioOutputRequested,
    forward_without_audio_output,
    params_without_audio_output,
    raise_audio_output_refusal,
)

_TEXT_MODEL: Final = "gemini/gemini-3.8-flash"
_VOICE: Final = {"voice": "Kore", "format": "pcm16"}


@pytest.mark.parametrize(
    "model",
    ["gemini/gemini-3.8-flash-tts", "gemini-3.8-flash-tts", "vertex_ai/gemini-2.5-flash-preview-tts"],
)
@pytest.mark.parametrize(
    "params",
    [{}, {"drop_params": True}, {"audio": _VOICE, "drop_params": True}, {"additional_drop_params": ["audio"]}],
    ids=["plain", "drop-params", "audio-under-drop-params", "additional-drop-params"],
)
def test_a_gemini_tts_model_is_refused_whatever_the_request_drops(model: str, params: Mapping[str, object]) -> None:
    assert params_without_audio_output(params, model) == AudioOnlyModel(model)


@pytest.mark.parametrize(
    ("params", "fields"),
    [
        ({"audio": _VOICE}, ("audio",)),
        ({"modalities": ["text", "audio"]}, ("modalities",)),
        ({"audio": _VOICE, "modalities": ["audio"]}, ("audio", "modalities")),
        ({"audio": _VOICE, "modalities": ["text", "audio"], "additional_drop_params": ["audio"]}, ("modalities",)),
        ({"audio": _VOICE, "drop_params": False}, ("audio",)),
    ],
)
def test_an_audio_output_ask_is_refused_without_drop_params(
    params: Mapping[str, object], fields: tuple[str, ...]
) -> None:
    with patch.object(litellm, "drop_params", False):
        assert params_without_audio_output(params, _TEXT_MODEL) == AudioOutputRequested(fields)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"audio": None},
        {"modalities": ["text", "image"]},
        {"modalities": "audio"},
        {"audio": _VOICE, "additional_drop_params": ["audio"]},
        {"audio": _VOICE, "modalities": ["text", "audio"], "additional_drop_params": ["audio", "modalities"]},
    ],
)
def test_a_request_without_an_audio_output_ask_is_forwarded_as_is(params: Mapping[str, object]) -> None:
    with patch.object(litellm, "drop_params", False):
        assert params_without_audio_output(params, _TEXT_MODEL) is params


@pytest.mark.parametrize(
    ("params", "forwarded"),
    [
        ({"audio": _VOICE, "drop_params": True, "temperature": 0.2}, {"drop_params": True, "temperature": 0.2}),
        ({"audio": _VOICE, "drop_params": "true"}, {"drop_params": "true"}),
        (
            {"modalities": ["text", "image", "audio"], "drop_params": True},
            {"modalities": ["text", "image"], "drop_params": True},
        ),
        ({"audio": _VOICE, "modalities": ["audio"], "drop_params": True}, {"drop_params": True}),
    ],
)
def test_drop_params_forwards_the_request_without_its_audio_output_ask(
    params: Mapping[str, object], forwarded: Mapping[str, object]
) -> None:
    with patch.object(litellm, "drop_params", False):
        assert params_without_audio_output(params, _TEXT_MODEL) == forwarded


def test_the_global_drop_params_setting_drops_the_audio_output_ask() -> None:
    with patch.object(litellm, "drop_params", True):
        assert params_without_audio_output({"audio": _VOICE, "user": "u-1"}, _TEXT_MODEL) == {"user": "u-1"}


@pytest.mark.parametrize(
    ("refusal", "model", "expected"),
    [
        (
            AudioOnlyModel("gemini-3.8-flash-tts"),
            "gemini-3.8-flash-tts",
            "gemini-3.8-flash-tts only answers with audio, which /v1/responses cannot return. "
            "Call /v1/audio/speech or /v1/chat/completions instead.",
        ),
        (
            AudioOutputRequested(("audio", "modalities")),
            "gemini-3.8-flash",
            "/v1/responses cannot return audio output, so it does not support ['audio', 'modalities'] "
            "for model=gemini-3.8-flash. Call /v1/chat/completions for audio, "
            "or set drop_params: true to drop these params.",
        ),
    ],
)
def test_a_refusal_is_raised_as_a_400_naming_the_way_out(
    refusal: AudioOutputRefusal, model: str, expected: str
) -> None:
    with pytest.raises(UnsupportedParamsError) as error:
        raise_audio_output_refusal(refusal, endpoint="/v1/responses", model=model, custom_llm_provider="gemini")

    assert error.value.status_code == 400
    assert error.value.llm_provider == "gemini"
    assert error.value.model == model
    assert error.value.message == f"litellm.UnsupportedParamsError: {expected}"


def test_forwarding_returns_the_cleaned_params_and_raises_a_refusal() -> None:
    with patch.object(litellm, "drop_params", False):
        forwarded: Final = forward_without_audio_output(
            {"audio": _VOICE, "drop_params": True}, endpoint="/v1/messages", model=_TEXT_MODEL, custom_llm_provider=None
        )
        with pytest.raises(UnsupportedParamsError) as error:
            forward_without_audio_output(
                {"audio": _VOICE}, endpoint="/v1/messages", model=_TEXT_MODEL, custom_llm_provider="gemini"
            )

    assert forwarded == {"drop_params": True}
    assert error.value.message.startswith("litellm.UnsupportedParamsError: /v1/messages cannot return audio output")
