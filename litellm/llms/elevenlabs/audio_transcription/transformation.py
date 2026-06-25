"""
Translates from OpenAI's `/v1/audio/transcriptions` to ElevenLabs's `/v1/speech-to-text`
"""

from itertools import groupby
from typing import List, Optional, Tuple, Union

from httpx import Headers, Response

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.elevenlabs import (
    ElevenLabsSTTChunk,
    ElevenLabsSTTMultichannelResponse,
    ElevenLabsSTTWord,
    OpenAIDiarizedSegment,
    OpenAITranscriptionWord,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import (
    FileTypes,
    TranscriptionResponse,
    TranscriptionUsageDurationObject,
)

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from ..common_utils import ElevenLabsException


class ElevenLabsAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    @property
    def custom_llm_provider(self) -> str:
        return litellm.LlmProviders.ELEVENLABS.value

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "temperature", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k not in supported_params:
                continue
            if k == "language":
                # Map OpenAI language format to ElevenLabs language_code
                optional_params["language_code"] = v
            elif k == "response_format":
                # ElevenLabs always returns JSON; the only response_format that
                # changes the request is diarized_json, which maps to diarization.
                # Other formats are shaped from the same JSON at response time.
                if v == "diarized_json":
                    optional_params["diarize"] = True
            else:
                optional_params[k] = v
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return ElevenLabsException(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Transforms the audio transcription request for ElevenLabs API.

        Returns AudioTranscriptionRequestData with both form data and files.

        Returns:
            AudioTranscriptionRequestData: Structured data with form data and files
        """

        # Use common utility to process the audio file
        processed_audio = process_audio_file(audio_file)

        # Prepare form data
        form_data = {"model_id": model}

        #########################################################
        # Add OpenAI Compatible Parameters
        #########################################################
        for key, value in optional_params.items():
            if key in self.get_supported_openai_params(model) and value is not None:
                # Convert values to strings for form data, but skip None values
                form_data[key] = str(value)

        #########################################################
        # Add Provider Specific Parameters
        #########################################################
        provider_specific_params = self.get_provider_specific_params(
            model=model,
            optional_params=optional_params,
            openai_params=self.get_supported_openai_params(model),
        )

        for key, value in provider_specific_params.items():
            form_data[key] = str(value)
        #########################################################
        #########################################################

        # Prepare files
        files = {
            "file": (
                processed_audio.filename,
                processed_audio.file_content,
                processed_audio.content_type,
            )
        }

        return AudioTranscriptionRequestData(data=form_data, files=files)

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        """
        Transforms the raw ElevenLabs speech-to-text response.

        When the response carries speaker information (diarize=true) or per-channel
        transcripts (use_multi_channel=true), it is shaped into OpenAI's
        diarized_json form (segments with speaker labels + usage). Otherwise it
        falls back to a flat transcript with word-level timestamps.
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise ValueError(
                f"Error transforming ElevenLabs response: {str(e)}\nResponse: {raw_response.text}"
            )

        if "transcripts" in response_json:
            parsed = ElevenLabsSTTMultichannelResponse.model_validate(response_json)
            words = _words_from_channels(parsed.transcripts)
            text = " ".join(t.text for t in parsed.transcripts if t.text)
            language = next(
                (t.language_code for t in parsed.transcripts if t.language_code), None
            )
            duration = parsed.audio_duration_secs
            diarized = True
        else:
            chunk = ElevenLabsSTTChunk.model_validate(response_json)
            words = tuple(chunk.words)
            text = chunk.text
            language = chunk.language_code
            duration = chunk.audio_duration_secs
            diarized = any(w.speaker_id is not None for w in words)

        response = TranscriptionResponse(text=text)
        response["task"] = "transcribe"
        response["language"] = language or "unknown"

        if diarized:
            response["segments"] = _build_segments(words)
            if duration is not None:
                response["duration"] = duration
                response["usage"] = TranscriptionUsageDurationObject(
                    type="duration", seconds=duration
                )
        else:
            response["words"] = _openai_words(words)

        response._hidden_params = response_json
        return response

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            api_base = (
                get_secret_str("ELEVENLABS_API_BASE") or "https://api.elevenlabs.io"
            )
        api_base = api_base.rstrip("/")  # Remove trailing slash if present

        # ElevenLabs speech-to-text endpoint
        url = f"{api_base}/v1/speech-to-text"

        return url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        api_key = api_key or get_secret_str("ELEVENLABS_API_KEY")
        if api_key is None:
            raise ValueError(
                "ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable."
            )

        auth_header = {
            "xi-api-key": api_key,
        }

        headers.update(auth_header)
        return headers


def _words_from_channels(
    transcripts: List[ElevenLabsSTTChunk],
) -> Tuple[ElevenLabsSTTWord, ...]:
    """
    Flatten per-channel transcripts into a single time-ordered word stream, using
    the channel as the speaker so the two sides of e.g. a stereo call don't collide
    on the per-channel speaker ids ElevenLabs assigns independently.
    """
    words = tuple(
        word.model_copy(update={"speaker_id": f"speaker_{transcript.channel_index}"})
        for transcript in transcripts
        for word in transcript.words
    )
    return tuple(sorted(words, key=lambda w: w.start if w.start is not None else 0.0))


def _build_segments(
    words: Tuple[ElevenLabsSTTWord, ...],
) -> List[OpenAIDiarizedSegment]:
    spoken = (w for w in words if w.type == "word" and w.start is not None)
    groups = (tuple(group) for _, group in groupby(spoken, key=lambda w: w.speaker_id))
    return [
        OpenAIDiarizedSegment(
            id=f"segment_{index}",
            start=group[0].start if group[0].start is not None else 0.0,
            end=group[-1].end if group[-1].end is not None else 0.0,
            speaker=group[0].speaker_id or "speaker_0",
            text=" ".join(w.text for w in group),
            type="transcript.text.segment",
        )
        for index, group in enumerate(groups)
    ]


def _openai_words(
    words: Tuple[ElevenLabsSTTWord, ...],
) -> List[OpenAITranscriptionWord]:
    return [
        OpenAITranscriptionWord(
            word=w.text,
            start=w.start if w.start is not None else 0.0,
            end=w.end if w.end is not None else 0.0,
        )
        for w in words
        if w.type == "word"
    ]
