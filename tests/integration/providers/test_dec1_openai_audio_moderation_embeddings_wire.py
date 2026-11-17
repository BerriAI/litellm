"""Contract: OpenAI audio transcription, moderation and the embeddings gaps, through the proxy to a scripted peer.

Transcription is the one OpenAI route that is multipart in and multipart out: the caller uploads a file to
`/v1/audio/transcriptions` and the proxy re-sends it to `<api_base>/audio/transcriptions` with the upstream
model name and the optional form fields (`language`, `prompt`, `response_format`, `temperature`) unchanged, then
returns the provider's `{"text": ...}`. Moderation forwards `input` and the upstream model and returns the
provider's `results` unchanged. For embeddings, `dimensions` and `encoding_format` each have tests elsewhere
(`test_dec1_textapis_embeddings_openai.py`, `test_openai_embedding_wire.py`); here they are pinned together and on
an OpenAI-compatible provider.
"""

from __future__ import annotations

from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_openai_support import (
    PROVIDER_KEY,
    TRANSCRIPT,
    Call,
    deployment,
    error_message,
    json_reply,
    marker,
    openai_error,
    peer,
    served,
)

_AUDIO: Final = b"RIFF\x24\x00\x00\x00WAVEfmt dec1 synthetic audio bytes"


def _transcribe(gateway: Gateway, model: str, **fields: str) -> httpx.Response:
    return gateway.request_multipart(
        "/v1/audio/transcriptions", {"model": model, **fields}, {"file": ("dec1.wav", _AUDIO, "audio/wav")}
    )


def test_a_transcription_upload_reaches_the_provider_as_multipart_with_the_file(gateway: Gateway) -> None:
    """The caller uploads a small WAV to an `openai/whisper-1` deployment.

    The provider receives one `POST /v1/audio/transcriptions` with `authorization: Bearer <deployment key>`, a
    `multipart/form-data` content type, the upstream model `whisper-1` as a form field, and the uploaded bytes
    unchanged. The caller gets `{"text": "dec1 synthetic transcript"}`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/whisper-1", model_info={"mode": "audio_transcription"})
        body: Final = served(_transcribe(gateway, model))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/audio/transcriptions")
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.headers["content-type"].startswith("multipart/form-data"), call.headers
        assert _AUDIO in call.raw and b'name="model"\r\n\r\nwhisper-1' in call.raw, call.raw[:400]
        assert body["text"] == TRANSCRIPT, body


@pytest.mark.parametrize(
    ("name", "value"),
    [("language", "de"), ("prompt", "dec1 glossary"), ("temperature", "0.2"), ("response_format", "json")],
)
def test_optional_transcription_fields_are_forwarded(gateway: Gateway, name: str, value: str) -> None:
    """Each optional OpenAI transcription field sent as a form field reaches the provider as the same form field
    with the same value (`map_openai_transcription_params`). UNVERIFIED: `temperature` is re-encoded from the float
    the proxy parses; this pins the text `0.2`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/whisper-1", model_info={"mode": "audio_transcription"})
        served(_transcribe(gateway, model, **{name: value}))
        assert f'name="{name}"\r\n\r\n{value}'.encode() in provider.only().raw


def test_a_transcription_provider_error_keeps_its_status(gateway: Gateway) -> None:
    """The provider answers 400 `dec1 audio too short`; the caller gets 400 with that text in the message."""
    with peer(lambda call: openai_error(400, "dec1 audio too short")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/whisper-1", model_info={"mode": "audio_transcription"})
        response: Final = _transcribe(gateway, model)
        assert response.status_code == 400, response.text
        assert "dec1 audio too short" in error_message(response)


def test_a_flagged_moderation_result_is_returned_unchanged(gateway: Gateway) -> None:
    """The provider flags the input with `categories.violence: true` and score 0.93.

    The caller's `results[0]` has `flagged: true`, the same category flag and the same score, and the provider
    body was `{"model": "omni-moderation-latest", "input": "<marker>"}`.
    """
    text: Final = marker()

    def respond(call: Call) -> Reply:
        return json_reply(
            {
                "id": "modr-dec1",
                "model": "omni-moderation-latest",
                "results": [{"flagged": True, "categories": {"violence": True}, "category_scores": {"violence": 0.93}}],
            }
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, "openai/omni-moderation-latest", model_info={"mode": "moderation"}
        )
        body: Final = served(gateway.request("POST", "/v1/moderations", {"model": model, "input": text}))
        assert provider.only().body == {"model": "omni-moderation-latest", "input": text}
        result: Final = object_value(list_value(body["results"])[0])
        assert result["flagged"] is True and object_value(result["categories"])["violence"] is True, body
        assert object_value(result["category_scores"])["violence"] == 0.93, body


def test_a_moderation_provider_429_keeps_its_status(gateway: Gateway) -> None:
    """The moderation provider answers 429 `dec1 moderation busy`; the caller gets 429 with that text."""
    with peer(lambda call: openai_error(429, "dec1 moderation busy")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, "openai/omni-moderation-latest", model_info={"mode": "moderation"}
        )
        response: Final = gateway.request("POST", "/v1/moderations", {"model": model, "input": marker()})
        assert response.status_code == 429, response.text
        assert "dec1 moderation busy" in error_message(response)


def test_dimensions_and_base64_encoding_are_forwarded_together(gateway: Gateway) -> None:
    """`dimensions: 256` and `encoding_format: "base64"` on `openai/text-embedding-3-small`.

    The provider body is `{"model": "text-embedding-3-small", "input": "<marker>", "dimensions": 256,
    "encoding_format": "base64"}`; neither param suppresses the other.
    """
    text: Final = marker()

    def respond(call: Call) -> Reply:
        return json_reply(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": "AACAPgAAAD8AAEA/"}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/text-embedding-3-small")
        body: Final = served(
            gateway.request(
                "POST",
                "/v1/embeddings",
                {"model": model, "input": text, "dimensions": 256, "encoding_format": "base64"},
            )
        )
        assert provider.only().body == {
            "model": "text-embedding-3-small",
            "input": text,
            "dimensions": 256,
            "encoding_format": "base64",
        }
        assert object_value(list_value(body["data"])[0])["embedding"] == "AACAPgAAAD8AAEA/", body


def test_mistral_embeddings_post_to_its_v1_embeddings_route(gateway: Gateway) -> None:
    """A `mistral/mistral-embed` deployment with a bare `api_base`. The provider receives
    `POST /v1/embeddings` with `model: "mistral-embed"` and the caller's input, and `Bearer <deployment key>`.
    UNVERIFIED: the embeddings route reuses `get_mistral_chat_credentials`' `/v1` normalisation."""
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, "mistral/mistral-embed", api_base_suffix="", model_info={"mode": "embedding"}
        )
        served(gateway.request("POST", "/v1/embeddings", {"model": model, "input": [text]}))
        call: Final = provider.only()
        assert call.path == "/v1/embeddings", call.path
        assert (call.body["model"], call.body["input"]) == ("mistral-embed", [text]), call.body
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
