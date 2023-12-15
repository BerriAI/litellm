# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

from typing import TYPE_CHECKING

from .speech import (
    Speech,
    AsyncSpeech,
    SpeechWithRawResponse,
    AsyncSpeechWithRawResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from .translations import (
    Translations,
    AsyncTranslations,
    TranslationsWithRawResponse,
    AsyncTranslationsWithRawResponse,
)
from .transcriptions import (
    Transcriptions,
    AsyncTranscriptions,
    TranscriptionsWithRawResponse,
    AsyncTranscriptionsWithRawResponse,
)

if TYPE_CHECKING:
    from ..._client import OpenAI, AsyncOpenAI

__all__ = ["Audio", "AsyncAudio"]


class Audio(SyncAPIResource):
    transcriptions: Transcriptions
    translations: Translations
    speech: Speech
    with_raw_response: AudioWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self.transcriptions = Transcriptions(client)
        self.translations = Translations(client)
        self.speech = Speech(client)
        self.with_raw_response = AudioWithRawResponse(self)


class AsyncAudio(AsyncAPIResource):
    transcriptions: AsyncTranscriptions
    translations: AsyncTranslations
    speech: AsyncSpeech
    with_raw_response: AsyncAudioWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        super().__init__(client)
        self.transcriptions = AsyncTranscriptions(client)
        self.translations = AsyncTranslations(client)
        self.speech = AsyncSpeech(client)
        self.with_raw_response = AsyncAudioWithRawResponse(self)


class AudioWithRawResponse:
    def __init__(self, audio: Audio) -> None:
        self.transcriptions = TranscriptionsWithRawResponse(audio.transcriptions)
        self.translations = TranslationsWithRawResponse(audio.translations)
        self.speech = SpeechWithRawResponse(audio.speech)


class AsyncAudioWithRawResponse:
    def __init__(self, audio: AsyncAudio) -> None:
        self.transcriptions = AsyncTranscriptionsWithRawResponse(audio.transcriptions)
        self.translations = AsyncTranslationsWithRawResponse(audio.translations)
        self.speech = AsyncSpeechWithRawResponse(audio.speech)
