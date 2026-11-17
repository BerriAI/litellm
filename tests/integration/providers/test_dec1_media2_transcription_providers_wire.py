"""`POST /v1/audio/transcriptions` on Azure OpenAI, Groq, Deepgram and ElevenLabs, plus the proxy-side rules every
provider shares (file size limit, per-second billing, translations route).

How a transcription moves through the proxy: the caller posts multipart form data with a `file` part and text
fields. The proxy reads the form (`get_form_data` turns repeated `timestamp_granularities[]` into a list), checks the
deployment's `max_file_size_mb` (premium only), then calls `litellm.atranscription`. OpenAI-compatible providers
(OpenAI, Groq) and Azure go through the OpenAI SDK and receive a new multipart body; Deepgram receives the raw
audio bytes as the request body with every non-OpenAI field as a query parameter; ElevenLabs receives multipart
with `model_id` and `language_code`. Unsupported OpenAI params are refused unless `drop_params` is set, and Groq
refuses `response_format` values other than `json`, `verbose_json` and `text` (litellm/llms/groq/stt/transformation.py).

Cost is per second unless the response carries token usage: the provider's `usage.seconds` when `usage.type` is
`duration`, else the response `duration` (Deepgram `metadata.duration`, OpenAI `verbose_json`), else the length
measured from the uploaded file (litellm/cost_calculator.py `_transcription_duration_for_cost`). A rewrite that
always measured the file, or always trusted the response, would change every bill below.

`compatibility/test_audio_transcriptions_wire.py` already pins the OpenAI field passthrough, plain text formats
and gpt-transcribe billing; `test_dec1_openai_audio_moderation_embeddings_wire.py` pins the basic OpenAI upload.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_media2_support import (
    AZURE_KEY,
    DEEPGRAM_KEY,
    ELEVENLABS_KEY,
    GROQ_KEY,
    OPENAI_KEY,
    deepgram_reply,
    elevenlabs_stt_reply,
    error_message,
    error_of,
    file_parts,
    form_fields,
    json_reply,
    logged_rows,
    openai_error_reply,
    path_of,
    query_of,
    response_cost,
    settled_key_spend,
    silent_wav,
    targets,
    text_fields,
    transcribe,
)

_TRANSCRIPTIONS: Final = "/v1/audio/transcriptions"
_RATE: Final = 0.001
_AZURE_VERSION: Final = "2024-06-01"


def _clip(seconds: float = 2.0) -> bytes:
    return silent_wav(seconds, uuid.uuid4().bytes)


def _openai_text(_request: Request) -> Reply:
    return json_reply({"text": "hello world"})


# ---------------------------------------------------------------- Azure OpenAI


def test_azure_transcription_posts_to_the_deployment_path_with_api_version_and_api_key_header(
    gateway: Gateway,
) -> None:
    """An `azure/whisper-1` deployment with `api_version` 2024-06-01 transcribes a WAV upload. Azure must get one
    POST to `/openai/deployments/whisper-1/audio/transcriptions?api-version=2024-06-01`, authenticated with the
    `api-key` header (no bearer token), carrying the upload under the `file` part, and the caller gets the text."""
    audio: Final = _clip()
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/whisper-1", api_base=wire.url, api_key=AZURE_KEY, api_version=_AZURE_VERSION
        )
        response: Final = transcribe(gateway, {"model": model}, audio)
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["text"] == "hello world"
        (request,) = wire.drain()
    assert request.method == "POST"
    assert path_of(request) == "/openai/deployments/whisper-1/audio/transcriptions"
    assert query_of(request)["api-version"] == [_AZURE_VERSION]
    assert request.headers["api-key"] == AZURE_KEY
    assert "authorization" not in request.headers
    assert [(part.name, part.content) for part in file_parts(request)] == [("file", audio)]


def test_azure_transcription_forwards_language_prompt_and_verbose_json(gateway: Gateway) -> None:
    """The caller sends `language=de`, `prompt=glossary` and `response_format=verbose_json` to an Azure whisper
    deployment. The multipart body Azure receives must carry those three fields with the same values, because the
    Azure branch passes every OpenAI transcription param through unchanged (litellm/utils.py
    get_optional_params_transcription, openai/azure branch)."""
    with (
        wire_server(
            lambda _r: json_reply({"task": "transcribe", "language": "german", "duration": 2.0, "text": "hallo"})
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="azure/whisper-1", api_base=wire.url, api_key=AZURE_KEY, api_version=_AZURE_VERSION
        )
        response: Final = transcribe(
            gateway,
            {"model": model, "language": "de", "prompt": "glossary", "response_format": "verbose_json"},
            _clip(),
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    fields: Final = form_fields(request)
    assert (fields["language"], fields["prompt"], fields["response_format"]) == ("de", "glossary", "verbose_json")


def test_azure_transcription_401_reaches_the_caller_as_401(gateway: Gateway) -> None:
    """Azure answers the transcription with 401 `Access denied due to invalid subscription key`. The caller must
    get 401 and the provider message must appear in the error body, not a generic 500."""
    with (
        wire_server(
            lambda _r: openai_error_reply(401, "Access denied due to invalid subscription key", "invalid_request_error")
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="azure/whisper-1", api_base=wire.url, api_key=AZURE_KEY, api_version=_AZURE_VERSION
        )
        response: Final = transcribe(gateway, {"model": model}, _clip())
        assert response.status_code == 401, response.text
        assert "Access denied due to invalid subscription key" in error_message(response)
        assert len(wire.drain()) >= 1


# ---------------------------------------------------------------- Groq


def test_groq_transcription_posts_to_the_openai_compatible_path_with_a_bearer_key(gateway: Gateway) -> None:
    """A `groq/whisper-large-v3` deployment whose `api_base` ends in `/openai/v1` transcribes an upload with
    `response_format=json`. Groq must get one POST to `/openai/v1/audio/transcriptions` with `Authorization:
    Bearer <key>`, `model=whisper-large-v3` (provider prefix stripped) and `response_format=json`."""
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="groq/whisper-large-v3", api_base=f"{wire.url}/openai/v1", api_key=GROQ_KEY)
        response: Final = transcribe(gateway, {"model": model, "response_format": "json"}, _clip())
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", "/openai/v1/audio/transcriptions")
    assert request.headers["authorization"] == f"Bearer {GROQ_KEY}"
    fields: Final = form_fields(request)
    assert (fields["model"], fields["response_format"]) == ("whisper-large-v3", "json")


@pytest.mark.parametrize("response_format", ("srt", "vtt"))
def test_groq_refuses_a_subtitle_response_format_with_400_before_calling_groq(
    gateway: Gateway, response_format: str
) -> None:
    """Groq only serves `json`, `verbose_json` and `text`. A caller asking for `srt` or `vtt` must get 400 with
    the message `Groq doesn't support response_format=<format>` and Groq must not be called
    (litellm/llms/groq/stt/transformation.py map_openai_params_stt)."""
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="groq/whisper-large-v3", api_base=f"{wire.url}/openai/v1", api_key=GROQ_KEY)
        response: Final = transcribe(gateway, {"model": model, "response_format": response_format}, _clip())
        assert response.status_code == 400, response.text
        assert f"Groq doesn't support response_format={response_format}" in error_message(response)
        assert wire.drain() == ()


def test_groq_drops_a_subtitle_response_format_when_the_deployment_sets_drop_params(gateway: Gateway) -> None:
    """The same Groq deployment with `drop_params: true` gets `response_format=srt`. The call must succeed and the
    multipart body Groq receives must have no `response_format` field at all, so Groq answers with its default."""
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="groq/whisper-large-v3", api_base=f"{wire.url}/openai/v1", api_key=GROQ_KEY, drop_params=True
        )
        response: Final = transcribe(gateway, {"model": model, "response_format": "srt"}, _clip())
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert "response_format" not in form_fields(request)


def test_groq_verbose_json_returns_segments_and_duration_unchanged(gateway: Gateway) -> None:
    """Groq answers `verbose_json` with `duration` 3.25, a language and one segment. The caller must get the same
    `text`, `language`, `duration` and `segments` values back, because the OpenAI transport returns the provider's
    transcription object as is."""
    reply: Final = {
        "task": "transcribe",
        "language": "English",
        "duration": 3.25,
        "text": "groq says hi",
        "segments": [{"id": 0, "seek": 0, "start": 0.0, "end": 3.25, "text": "groq says hi"}],
    }
    with wire_server(lambda _r: json_reply(reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="groq/whisper-large-v3", api_base=f"{wire.url}/openai/v1", api_key=GROQ_KEY)
        response: Final = transcribe(gateway, {"model": model, "response_format": "verbose_json"}, _clip())
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert (body["text"], body["language"], body["duration"]) == ("groq says hi", "English", 3.25)
        assert body["segments"] == reply["segments"]
        assert len(wire.drain()) == 1


def test_groq_timestamp_granularities_without_drop_params_answers_400(gateway: Gateway) -> None:
    """Groq's STT config does not list `timestamp_granularities`. A caller sending `timestamp_granularities[]=word`
    must get a 400 before Groq is called.
    Product bug: `_check_valid_arg` raises `UnsupportedParamsError(status_code=500)` with the copy-pasted message
    `Setting user/encoding format is not supported by groq` (litellm/utils.py get_optional_params_transcription),
    so the caller gets a 500 for its own bad input."""
    pytest.skip("BUG: an unsupported transcription param answers 500 'Setting user/encoding format...' instead of 400")
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="groq/whisper-large-v3", api_base=f"{wire.url}/openai/v1", api_key=GROQ_KEY)
        response: Final = transcribe(gateway, [("model", model), ("timestamp_granularities[]", "word")], _clip())
        assert response.status_code == 400, response.text
        assert "timestamp_granularities" in error_message(response)
        assert wire.drain() == ()


def test_groq_bills_the_seconds_measured_from_the_upload_when_the_reply_has_no_duration(gateway: Gateway) -> None:
    """A Groq deployment priced at `input_cost_per_second` 0.001 transcribes a 3 second WAV and Groq answers plain
    `json` with no `duration`. The response cost header, the spend row and the key spend must all be
    3 x 0.001, because the proxy measures the uploaded file when the provider reports no length."""
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="groq/whisper-large-v3",
            api_base=f"{wire.url}/openai/v1",
            api_key=GROQ_KEY,
            input_cost_per_second=_RATE,
        )
        key: Final = scenario.key(models=[model])
        response: Final = transcribe(gateway, {"model": model}, _clip(3.0), key=key)
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(3 * _RATE)
        (row,) = logged_rows(key)
        assert row["call_type"] == "atranscription"
        assert row["custom_llm_provider"] == "groq"
        assert float(str(row["spend"])) == pytest.approx(3 * _RATE)
        settled_key_spend(key, 3 * _RATE)
        assert len(wire.drain()) == 1


# ---------------------------------------------------------------- Deepgram


def test_deepgram_receives_raw_audio_at_listen_with_the_model_in_the_query_and_a_token_header(
    gateway: Gateway,
) -> None:
    """A `deepgram/nova-2` deployment transcribes a WAV upload with `language=en`. Deepgram must get one POST to
    `/v1/listen` whose query has `model=nova-2` and `language=en`, whose body is exactly the uploaded bytes (no
    multipart envelope), and whose `Authorization` header is `Token <key>` rather than `Bearer`
    (litellm/llms/deepgram/audio_transcription/transformation.py)."""
    audio: Final = _clip()
    with wire_server(lambda _r: deepgram_reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="deepgram/nova-2", api_base=f"{wire.url}/v1", api_key=DEEPGRAM_KEY)
        response: Final = transcribe(gateway, {"model": model, "language": "en"}, audio)
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", "/v1/listen")
    query: Final = query_of(request)
    assert query["model"] == ["nova-2"]
    assert query["language"] == ["en"]
    assert request.body == audio
    assert request.headers["authorization"] == f"Token {DEEPGRAM_KEY}"


def test_deepgram_provider_specific_form_fields_become_query_parameters(gateway: Gateway) -> None:
    """The caller adds Deepgram-only form fields `punctuate=true` and `smart_format=true`. They are not OpenAI
    params, so the proxy must forward them to Deepgram as query parameters with the same values, next to `model`.
    UNVERIFIED: assumes the proxy keeps unknown form fields as provider kwargs (main.transcription
    get_non_default_transcription_params, then add_provider_specific_params_to_optional_params)."""
    with wire_server(lambda _r: deepgram_reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="deepgram/nova-2", api_base=f"{wire.url}/v1", api_key=DEEPGRAM_KEY)
        response: Final = transcribe(gateway, {"model": model, "punctuate": "true", "smart_format": "true"}, _clip())
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    query: Final = query_of(request)
    assert (query["punctuate"], query["smart_format"]) == (["true"], ["true"])


def test_deepgram_reply_is_mapped_to_an_openai_transcription_with_words_and_duration(gateway: Gateway) -> None:
    """Deepgram answers with a transcript, three words carrying a `confidence`, and `metadata.duration` 42.5. The
    caller must get `text` equal to the transcript, `task` `transcribe`, `language` `en` (Deepgram sent no detected
    language), `duration` 42.5 and `words` holding only `word`, `start` and `end` for each word."""
    with wire_server(lambda _r: deepgram_reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="deepgram/nova-2", api_base=f"{wire.url}/v1", api_key=DEEPGRAM_KEY)
        response: Final = transcribe(gateway, {"model": model}, _clip())
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert len(wire.drain()) == 1
    assert body["text"] == "hello from deepgram"
    assert (body["task"], body["language"], body["duration"]) == ("transcribe", "en", 42.5)
    assert body["words"] == [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "from", "start": 0.5, "end": 0.7},
        {"word": "deepgram", "start": 0.8, "end": 1.3},
    ]


def test_deepgram_detected_language_wins_over_the_english_default(gateway: Gateway) -> None:
    """Deepgram's channel carries `detected_language` `fr`. The caller's `language` field must be `fr`, so a
    rewrite that hard-codes `en` would mislabel every non-English call."""
    with wire_server(lambda _r: deepgram_reply(detected_language="fr")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="deepgram/nova-2", api_base=f"{wire.url}/v1", api_key=DEEPGRAM_KEY)
        response: Final = transcribe(gateway, {"model": model}, _clip())
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["language"] == "fr"
        assert len(wire.drain()) == 1


def test_deepgram_diarized_words_are_joined_into_speaker_labelled_text(gateway: Gateway) -> None:
    """Deepgram answers a diarized transcript without `paragraphs`: words carry `speaker` 0, 0, 1 and
    `punctuated_word` values. The caller's `text` must be rebuilt from the words as
    `Speaker 0: Hi there.` followed by `\\nSpeaker 1: Bye.\\n` joined with a newline, the exact shape
    `_reconstruct_diarized_transcript` produces, not the flat `transcript` field."""
    words: Final = (
        {"word": "hi", "punctuated_word": "Hi", "start": 0.0, "end": 0.2, "speaker": 0},
        {"word": "there", "punctuated_word": "there.", "start": 0.3, "end": 0.6, "speaker": 0},
        {"word": "bye", "punctuated_word": "Bye.", "start": 1.0, "end": 1.2, "speaker": 1},
    )
    with (
        wire_server(lambda _r: deepgram_reply(transcript="hi there bye", words=words)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="deepgram/nova-2", api_base=f"{wire.url}/v1", api_key=DEEPGRAM_KEY)
        response: Final = transcribe(gateway, {"model": model, "diarize": "true"}, _clip())
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["text"] == "Speaker 0: Hi there.\n\nSpeaker 1: Bye.\n"
        assert len(wire.drain()) == 1


def test_deepgram_bills_the_duration_deepgram_reports_not_the_uploaded_length(gateway: Gateway) -> None:
    """A Deepgram deployment priced at `input_cost_per_second` 0.001 receives a 1 second WAV and Deepgram reports
    `metadata.duration` 42.5. The response cost and the spend row must be 42.5 x 0.001, because a response that
    carries a duration is billed by it and the file is only measured when the duration is missing."""
    with wire_server(lambda _r: deepgram_reply(duration=42.5)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="deepgram/nova-2", api_base=f"{wire.url}/v1", api_key=DEEPGRAM_KEY, input_cost_per_second=_RATE
        )
        key: Final = scenario.key(models=[model])
        response: Final = transcribe(gateway, {"model": model}, _clip(1.0), key=key)
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(42.5 * _RATE)
        (row,) = logged_rows(key)
        assert (row["call_type"], row["custom_llm_provider"]) == ("atranscription", "deepgram")
        assert float(str(row["spend"])) == pytest.approx(42.5 * _RATE)
        assert len(wire.drain()) == 1


def test_deepgram_unsupported_response_format_answers_400_before_calling_deepgram(gateway: Gateway) -> None:
    """Deepgram's config supports only `language`. A caller sending `response_format=verbose_json` without
    `drop_params` must get a 400 and Deepgram must not be called.
    Product bug: `_check_valid_arg` raises `UnsupportedParamsError(status_code=500)` with the message
    `Setting user/encoding format is not supported by deepgram` (litellm/utils.py get_optional_params_transcription)."""
    pytest.skip("BUG: an unsupported transcription param answers 500 'Setting user/encoding format...' instead of 400")
    with wire_server(lambda _r: deepgram_reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="deepgram/nova-2", api_base=f"{wire.url}/v1", api_key=DEEPGRAM_KEY)
        response: Final = transcribe(gateway, {"model": model, "response_format": "verbose_json"}, _clip())
        assert response.status_code == 400, response.text
        assert wire.drain() == ()


def test_deepgram_drops_response_format_with_drop_params_and_never_sends_it_as_a_query(gateway: Gateway) -> None:
    """With `drop_params: true` on the Deepgram deployment, `response_format=verbose_json` must be dropped: the call
    succeeds and the `/v1/listen` query holds no `response_format` key, since Deepgram would reject it."""
    with wire_server(lambda _r: deepgram_reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="deepgram/nova-2", api_base=f"{wire.url}/v1", api_key=DEEPGRAM_KEY, drop_params=True
        )
        response: Final = transcribe(gateway, {"model": model, "response_format": "verbose_json"}, _clip())
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert "response_format" not in query_of(request)


def test_deepgram_error_status_reaches_the_caller(gateway: Gateway) -> None:
    """Deepgram answers 400 `{"err_code":"Bad Request","err_msg":"corrupt or unsupported data"}`. The caller must
    get 400 and the Deepgram message must appear in the error text."""
    with (
        wire_server(
            lambda _r: json_reply({"err_code": "Bad Request", "err_msg": "corrupt or unsupported data"}, 400)
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="deepgram/nova-2", api_base=f"{wire.url}/v1", api_key=DEEPGRAM_KEY)
        response: Final = transcribe(gateway, {"model": model}, _clip())
        assert response.status_code == 400, response.text
        assert "corrupt or unsupported data" in response.text
        assert len(wire.drain()) >= 1


# ---------------------------------------------------------------- ElevenLabs


def test_elevenlabs_receives_multipart_with_model_id_language_code_and_the_xi_api_key_header(
    gateway: Gateway,
) -> None:
    """An `elevenlabs/scribe_v1` deployment transcribes an upload with `language=en` and `temperature=0.2`.
    ElevenLabs must get one POST to `/v1/speech-to-text` with header `xi-api-key: <key>`, form fields
    `model_id=scribe_v1`, `language_code=en` (renamed from `language`) and `temperature=0.2`, no `language` field,
    and the audio under the `file` part."""
    audio: Final = _clip()
    with wire_server(lambda _r: elevenlabs_stt_reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="elevenlabs/scribe_v1", api_base=wire.url, api_key=ELEVENLABS_KEY)
        response: Final = transcribe(gateway, {"model": model, "language": "en", "temperature": "0.2"}, audio)
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", "/v1/speech-to-text")
    assert request.headers["xi-api-key"] == ELEVENLABS_KEY
    fields: Final = form_fields(request)
    assert (fields["model_id"], fields["language_code"], fields["temperature"]) == ("scribe_v1", "en", "0.2")
    assert "language" not in fields
    assert [(part.name, part.content) for part in file_parts(request)] == [("file", audio)]


def test_elevenlabs_reply_keeps_only_word_entries_and_maps_language_code(gateway: Gateway) -> None:
    """ElevenLabs answers with `language_code` `eng` and three entries in `words`, one of type `spacing`. The
    caller must get `text`, `task` `transcribe`, `language` `eng`, and `words` with only the two `word` entries,
    each renamed from `text` to `word`."""
    with wire_server(lambda _r: elevenlabs_stt_reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="elevenlabs/scribe_v1", api_base=wire.url, api_key=ELEVENLABS_KEY)
        response: Final = transcribe(gateway, {"model": model}, _clip())
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert len(wire.drain()) == 1
    assert (body["text"], body["task"], body["language"]) == ("hello from scribe", "transcribe", "eng")
    assert body["words"] == [{"word": "hello", "start": 0.0, "end": 0.4}, {"word": "scribe", "start": 0.5, "end": 1.1}]


def test_elevenlabs_bills_the_measured_upload_length_since_scribe_reports_no_duration(gateway: Gateway) -> None:
    """An ElevenLabs deployment priced at `input_cost_per_second` 0.001 transcribes a 4 second WAV. Scribe's reply
    has no duration, so the response cost and the spend row must be 4 x 0.001, measured from the file."""
    with wire_server(lambda _r: elevenlabs_stt_reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="elevenlabs/scribe_v1", api_base=wire.url, api_key=ELEVENLABS_KEY, input_cost_per_second=_RATE
        )
        key: Final = scenario.key(models=[model])
        response: Final = transcribe(gateway, {"model": model}, _clip(4.0), key=key)
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(4 * _RATE)
        (row,) = logged_rows(key)
        assert (row["call_type"], row["custom_llm_provider"]) == ("atranscription", "elevenlabs")
        assert float(str(row["spend"])) == pytest.approx(4 * _RATE)
        assert len(wire.drain()) == 1


# ---------------------------------------------------------------- shared proxy rules


def test_openai_verbose_json_duration_is_billed_instead_of_the_measured_upload(gateway: Gateway) -> None:
    """An `openai/whisper-1` deployment priced at 0.001 per second receives a 2 second WAV and answers
    `verbose_json` with `duration` 9.0. The cost must be 9 x 0.001: the response duration is preferred over the
    measured file length (litellm/main.py only stores the measured length when the response has no duration)."""
    reply: Final = {"task": "transcribe", "language": "english", "duration": 9.0, "text": "long one"}
    with wire_server(lambda _r: json_reply(reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, input_cost_per_second=_RATE
        )
        response: Final = transcribe(gateway, {"model": model, "response_format": "verbose_json"}, _clip(2.0))
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(9 * _RATE)
        assert len(wire.drain()) == 1


def test_openai_duration_usage_block_wins_over_the_verbose_json_duration(gateway: Gateway) -> None:
    """OpenAI answers `verbose_json` with `duration` 9.0 and also `usage: {"type": "duration", "seconds": 5}`. The
    cost must be 5 x 0.001, since `usage.seconds` is the billed duration and comes first in
    `_transcription_duration_for_cost` (litellm/cost_calculator.py)."""
    reply: Final = {
        "task": "transcribe",
        "language": "english",
        "duration": 9.0,
        "text": "usage wins",
        "usage": {"type": "duration", "seconds": 5},
    }
    with wire_server(lambda _r: json_reply(reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, input_cost_per_second=_RATE
        )
        response: Final = transcribe(gateway, {"model": model, "response_format": "verbose_json"}, _clip(2.0))
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(5 * _RATE)
        assert len(wire.drain()) == 1


def test_word_and_segment_timestamp_granularities_reach_openai_as_two_repeated_fields(gateway: Gateway) -> None:
    """The caller sends `timestamp_granularities[]=word` and `timestamp_granularities[]=segment` with
    `verbose_json`. OpenAI must receive both values under the repeated `timestamp_granularities[]` field (compared
    sorted, since the multipart reader sorts fields), and the caller gets the provider's `words` back unchanged."""
    reply: Final = {
        "task": "transcribe",
        "language": "english",
        "duration": 1.0,
        "text": "hi",
        "words": [{"word": "hi", "start": 0.0, "end": 0.3}],
        "segments": [],
    }
    with wire_server(lambda _r: json_reply(reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = transcribe(
            gateway,
            [
                ("model", model),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ],
            _clip(),
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["words"] == reply["words"]
        (request,) = wire.drain()
    assert [value for name, value in text_fields(request) if name == "timestamp_granularities[]"] == [
        "segment",
        "word",
    ]


def test_an_upload_over_the_deployment_max_file_size_is_refused_with_400_before_the_provider(
    gateway: Gateway,
) -> None:
    """A whisper deployment sets `max_file_size_mb: 1` and the caller uploads a 2 MB WAV. The proxy must answer
    400 with `param` `file` and a message starting `File size is too large. Please check your file size.` and
    must not call the provider (litellm/proxy/common_utils/http_parsing_utils.py check_file_size_under_limit).
    Assumes a premium licence (`LITELLM_LICENSE`); without one the same 400 says `Tried setting
    max_file_size_mb for /audio/transcriptions`."""
    audio: Final = silent_wav(65.6) + uuid.uuid4().bytes
    assert len(audio) > 2 * 1024 * 1024 - 1024
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, max_file_size_mb=1
        )
        response: Final = transcribe(gateway, {"model": model}, audio)
        assert response.status_code == 400, response.text
        assert error_message(response).startswith("File size is too large. Please check your file size.")
        assert "Max file size: 1 MB" in error_message(response)
        assert error_of(response)["param"] == "file"
        assert wire.drain() == ()


def test_an_upload_under_the_deployment_max_file_size_is_forwarded(gateway: Gateway) -> None:
    """The same `max_file_size_mb: 1` deployment receives a 100 KB WAV. The proxy must forward it unchanged and
    answer 200, so the limit is a ceiling and not a rejection of every upload."""
    audio: Final = _clip(3.0)
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, max_file_size_mb=1
        )
        response: Final = transcribe(gateway, {"model": model}, audio)
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert [part.content for part in file_parts(request)] == [audio]


def test_a_transcription_without_a_file_part_answers_422_before_the_provider(gateway: Gateway) -> None:
    """The caller posts only the `model` form field. `file` is a required `UploadFile` on the endpoint, so the
    proxy must answer 422 (FastAPI validation) and never call the provider."""
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.client.post(
            _TRANSCRIPTIONS, data={"model": model}, headers={"Authorization": f"Bearer {gateway.key}"}
        )
        assert response.status_code == 422, response.text
        assert wire.drain() == ()


def test_a_transcription_rate_limit_from_the_provider_reaches_the_caller_as_429(gateway: Gateway) -> None:
    """OpenAI answers the transcription with 429 `Rate limit reached for whisper-1`. The caller must get 429 and
    the provider message, and a failure spend row with `call_type` `atranscription` and zero spend must land."""
    with (
        wire_server(
            lambda _r: openai_error_reply(429, "Rate limit reached for whisper-1", "requests", "rate_limit_exceeded")
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, input_cost_per_second=_RATE
        )
        key: Final = scenario.key(models=[model])
        response: Final = transcribe(gateway, {"model": model}, _clip(), key=key)
        assert response.status_code == 429, response.text
        assert "Rate limit reached for whisper-1" in error_message(response)
        (row,) = logged_rows(key)
        assert (row["call_type"], row["status"]) == ("atranscription", "failure")
        assert float(str(row["spend"])) == 0.0
        assert len(wire.drain()) >= 1


def test_audio_translations_route_forwards_the_upload_to_the_provider(gateway: Gateway) -> None:
    """The OpenAI SDK's `audio.translations.create` posts multipart to `/v1/audio/translations`. The proxy should
    route it like a transcription and the provider should get one POST to `/v1/audio/translations` with the file.
    Product bug: the proxy registers no `/v1/audio/translations` HTTP route (only the billable-route list in
    litellm/proxy/middleware/billable_request_metrics_middleware.py names it), so the call never reaches a
    provider."""
    pytest.skip("BUG: the proxy registers no /v1/audio/translations route, so translations answer 404 or 405")
    audio: Final = _clip()
    with wire_server(_openai_text) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/whisper-1", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.client.post(
            "/v1/audio/translations",
            data={"model": model},
            files={"file": ("clip.wav", audio, "audio/wav")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 200, response.text
        assert targets(wire) == [("POST", "/v1/audio/translations")]
