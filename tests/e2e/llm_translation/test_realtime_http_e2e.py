"""Vendor §9.19: realtime client_secrets + calls HTTP surface (LIT-4778).

Websocket coverage already lives under realtime/; this file pins the HTTP
client-secret mint (realtime, translation and transcription sessions), the
translation token scope, and the missing-auth contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import NoBody, PartialBody, UnknownApiError, assert_auth_denied, unwrap
from e2e_metadata import Domain, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

REALTIME_BACKEND = "openai/gpt-realtime"
TRANSLATIONS_CLIENT_SECRETS: Final = "/v1/realtime/translations/client_secrets"


class RealtimeSession(BaseModel):
    type: str = "realtime"
    model: str | None = None
    instructions: str | None = None
    output_modalities: list[str] | None = None


class RealtimeExpiresAfter(BaseModel):
    anchor: str = "created_at"
    seconds: int = 600


class RealtimeClientSecretRequest(BaseModel):
    model: str
    expires_after: RealtimeExpiresAfter | None = None
    session: RealtimeSession | None = None


class TurnDetection(BaseModel):
    type: str


class SessionTranscription(BaseModel):
    model: str | None = None


class SessionAudioInput(BaseModel):
    transcription: SessionTranscription | None = None
    turn_detection: TurnDetection | None = None


class SessionAudioOutput(BaseModel):
    language: str | None = None


class SessionAudio(BaseModel):
    input: SessionAudioInput | None = None
    output: SessionAudioOutput | None = None


class TranslationSession(BaseModel):
    model: str
    audio: SessionAudio


class TranscriptionSession(BaseModel):
    type: str = "transcription"
    audio: SessionAudio


class SessionClientSecretRequest(BaseModel):
    session: TranslationSession | TranscriptionSession


class NullableAudioInput(PartialBody):
    turn_detection: TurnDetection | None = None


class NullableAudio(PartialBody):
    input: NullableAudioInput


class NullableRealtimeSession(PartialBody):
    type: str
    model: str
    instructions: str | None = None
    audio: NullableAudio


class NullableClientSecretRequest(PartialBody):
    model: str
    session: NullableRealtimeSession


class RealtimeClientSecretSession(BaseModel):
    type: str | None = None
    audio: SessionAudio | None = None


class RealtimeClientSecretResponse(BaseModel):
    value: str | None = None
    expires_at: int | None = None
    session: RealtimeClientSecretSession | None = None


def _register(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    return _register_backend(
        proxy, resources, LiteLLMParamsBody(model=REALTIME_BACKEND, api_key="os.environ/OPENAI_API_KEY")
    )


def _register_backend(
    proxy: ProxyClient, resources: ResourceManager, litellm_params: LiteLLMParamsBody
) -> tuple[str, str]:
    model: Final = f"e2e-realtime-http-{unique_marker()}"
    model_id: Final = proxy.create_model(model, litellm_params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


@dataclass(frozen=True, slots=True)
class _AudioBackend:
    id: str
    registry_route: str
    provider: Provider
    litellm_params: LiteLLMParamsBody

    @property
    def upstream_model(self) -> str:
        return self.litellm_params.model.split("/", 1)[1]


def _openai_backend(model: str) -> _AudioBackend:
    return _AudioBackend(
        "openai", "openai", Provider.OPENAI, LiteLLMParamsBody(model=model, api_key="os.environ/OPENAI_API_KEY")
    )


def _azure_backend(model: str) -> _AudioBackend:
    return _AudioBackend(
        "azure",
        "azure_openai",
        Provider.AZURE,
        LiteLLMParamsBody(model=model, api_base="os.environ/AZURE_API_BASE", api_key="os.environ/AZURE_API_KEY"),
    )


TRANSLATION_BACKENDS: Final = (
    _openai_backend("openai/gpt-realtime-translate"),
    _azure_backend("azure/gpt-realtime-translate"),
)
TRANSCRIPTION_BACKENDS: Final = (
    _openai_backend("openai/gpt-live-transcribe"),
    _azure_backend("azure/gpt-live-transcribe"),
)


def _subject(backend: _AudioBackend) -> pytest.MarkDecorator:
    return meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.REALTIME,
            providers=(backend.provider,),
            models=(backend.litellm_params.model,),
        )
    )


TRANSLATION_PARAMS: Final = tuple(
    pytest.param(
        b,
        id=b.id,
        marks=(pytest.mark.covers(f"llm.realtime.{b.registry_route}.translation.stream.works"), _subject(b)),
    )
    for b in TRANSLATION_BACKENDS
)
TRANSCRIPTION_PARAMS: Final = tuple(pytest.param(b, id=b.id, marks=_subject(b)) for b in TRANSCRIPTION_BACKENDS)


def _french_translation_request(model: str) -> SessionClientSecretRequest:
    return SessionClientSecretRequest(
        session=TranslationSession(model=model, audio=SessionAudio(output=SessionAudioOutput(language="fr")))
    )


class TestRealtimeHttp:
    @pytest.mark.covers("llm.realtime.openai.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.REALTIME,
            providers=(Provider.OPENAI,),
            models=(REALTIME_BACKEND,),
        )
    )
    def test_create_client_secret(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register(proxy, resources)
        secret = unwrap(
            proxy.transport.post(
                "/v1/realtime/client_secrets",
                headers=proxy.transport.bearer(key),
                json=RealtimeClientSecretRequest(
                    model=model,
                    expires_after=RealtimeExpiresAfter(),
                    session=RealtimeSession(
                        model=REALTIME_BACKEND,
                        instructions="You are a helpful assistant.",
                        output_modalities=["text"],
                    ),
                ),
                response_type=RealtimeClientSecretResponse,
            )
        )
        assert secret.value, f"client secret value missing: {secret}"
        if secret.session is not None:
            assert secret.session.type in (None, "realtime"), f"unexpected session type: {secret.session.type}"

    @pytest.mark.covers("other.auth.realtime.missing_header_denied")
    @meta(Subject(domain=Domain.PROXY_AUTH, route=Route.REALTIME))
    def test_client_secret_missing_auth_is_denied(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, _ = _register(proxy, resources)
        result = proxy.transport.send(
            "/v1/realtime/client_secrets",
            headers=NoBody(),
            json=RealtimeClientSecretRequest(model=model),
        )
        assert_auth_denied(result, "realtime client_secrets missing auth")

    @pytest.mark.covers("other.auth.realtime.missing_header_denied")
    @meta(Subject(domain=Domain.PROXY_AUTH, route=Route.REALTIME))
    def test_calls_without_auth_is_denied(self, proxy: ProxyClient) -> None:
        result = proxy.transport.send(
            "/v1/realtime/calls",
            headers=NoBody(),
            json=NoBody(),
        )
        assert_auth_denied(result, "realtime calls missing auth")


class TestRealtimeTranslationClientSecrets:
    @pytest.mark.parametrize("backend", TRANSLATION_PARAMS)
    def test_translations_route_mints_a_french_translation_session(
        self, proxy: ProxyClient, resources: ResourceManager, backend: _AudioBackend
    ) -> None:
        model, key = _register_backend(proxy, resources, backend.litellm_params)
        secret: Final = unwrap(
            proxy.transport.post(
                TRANSLATIONS_CLIENT_SECRETS,
                headers=proxy.transport.bearer(key),
                json=_french_translation_request(model),
                response_type=RealtimeClientSecretResponse,
            )
        )
        assert secret.value, f"client secret value missing: {secret}"
        assert secret.session is not None and secret.session.type == "translation", (
            f"the translations route did not mint a translation session: {secret.session}"
        )
        output: Final = secret.session.audio.output if secret.session.audio else None
        assert output is not None and output.language == "fr", (
            f"the requested target language did not reach the session: {secret.session}"
        )

    @meta(
        Subject(
            domain=Domain.PROXY_AUTH,
            route=Route.REALTIME,
            providers=(Provider.OPENAI,),
            models=(TRANSLATION_BACKENDS[0].litellm_params.model,),
        )
    )
    def test_translation_token_is_refused_by_the_realtime_calls_route(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register_backend(proxy, resources, TRANSLATION_BACKENDS[0].litellm_params)
        secret: Final = unwrap(
            proxy.transport.post(
                TRANSLATIONS_CLIENT_SECRETS,
                headers=proxy.transport.bearer(key),
                json=_french_translation_request(model),
                response_type=RealtimeClientSecretResponse,
            )
        )
        assert secret.value, f"client secret value missing: {secret}"
        result: Final = proxy.transport.send(
            "/v1/realtime/calls", headers=proxy.transport.bearer(secret.value), json=NoBody()
        )
        assert result.status_code == 401, (
            f"a translation token opened a plain realtime call: {result.status_code} {result.body[:300]}"
        )
        assert "not valid for this Realtime endpoint" in result.body, (
            f"the refusal does not say the token belongs to another endpoint: {result.body[:300]}"
        )

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.REALTIME,
            providers=(Provider.OPENAI,),
            models=(TRANSLATION_BACKENDS[0].litellm_params.model,),
        )
    )
    def test_translation_session_on_the_realtime_route_is_rejected(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register_backend(proxy, resources, TRANSLATION_BACKENDS[0].litellm_params)
        result: Final = proxy.transport.post(
            "/v1/realtime/client_secrets",
            headers=proxy.transport.bearer(key),
            json=RealtimeClientSecretRequest(model=model, session=RealtimeSession(type="translation", model=model)),
            response_type=RealtimeClientSecretResponse,
        )
        match result:
            case UnknownApiError(status_code=400, body=body):
                assert "translations endpoint" in body, f"the 400 does not point at the translations route: {body}"
            case other:
                pytest.fail(f"a translation session on /v1/realtime/client_secrets expected a 400, got {other!r}")


class TestRealtimeTranscriptionClientSecrets:
    @pytest.mark.parametrize("backend", TRANSCRIPTION_PARAMS)
    def test_transcription_session_runs_on_the_deployment_model(
        self, proxy: ProxyClient, resources: ResourceManager, backend: _AudioBackend
    ) -> None:
        model, key = _register_backend(proxy, resources, backend.litellm_params)
        secret: Final = unwrap(
            proxy.transport.post(
                "/v1/realtime/client_secrets",
                headers=proxy.transport.bearer(key),
                json=SessionClientSecretRequest(
                    session=TranscriptionSession(
                        audio=SessionAudio(input=SessionAudioInput(transcription=SessionTranscription(model=model)))
                    )
                ),
                response_type=RealtimeClientSecretResponse,
            )
        )
        assert secret.value, f"client secret value missing: {secret}"
        assert secret.session is not None and secret.session.type == "transcription", (
            f"the client secret did not mint a transcription session: {secret.session}"
        )
        audio_input: Final = secret.session.audio.input if secret.session.audio else None
        transcription: Final = audio_input.transcription if audio_input else None
        assert transcription is not None and transcription.model == backend.upstream_model, (
            f"the session does not transcribe with the deployment's model {backend.upstream_model}: {secret.session}"
        )


class TestRealtimeClientSecretNulls:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.REALTIME,
            providers=(Provider.OPENAI,),
            models=(REALTIME_BACKEND,),
        )
    )
    def test_top_level_null_is_dropped_and_nested_null_is_kept(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register(proxy, resources)
        secret: Final = unwrap(
            proxy.transport.post(
                "/v1/realtime/client_secrets",
                headers=proxy.transport.bearer(key),
                json=NullableClientSecretRequest(
                    model=model,
                    session=NullableRealtimeSession(
                        type="realtime",
                        model=model,
                        instructions=None,
                        audio=NullableAudio(input=NullableAudioInput(turn_detection=None)),
                    ),
                ),
                response_type=RealtimeClientSecretResponse,
            )
        )
        assert secret.value, f"a session with instructions: null was not minted: {secret}"
        audio_input: Final = secret.session.audio.input if secret.session and secret.session.audio else None
        assert audio_input is not None and "turn_detection" in audio_input.model_fields_set, (
            f"the session does not report turn detection: {secret.session}"
        )
        assert audio_input.turn_detection is None, (
            f"turn_detection: null was dropped, so the session fell back to {audio_input.turn_detection}"
        )
