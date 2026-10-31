from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import AudioTranscriptionRequestData
from litellm.llms.openai.transcriptions.whisper_transformation import OpenAIWhisperAudioTranscriptionConfig
from litellm.types.llms.openai import AllMessageValues, OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes


class FoundryLocalAudioTranscriptionConfig(OpenAIWhisperAudioTranscriptionConfig):
    @property
    def has_native_transcription_endpoint(self) -> bool:
        return True

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: base class contract
        return ["language", "prompt", "response_format", "temperature"]

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: the shared HTTP handler adds the request Content-Type header
        return {**headers, **({"Authorization": f"Bearer {api_key}"} if api_key else {})}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        if not api_base:
            raise ValueError("Foundry Local transcription requires api_base or FOUNDRY_LOCAL_API_BASE")
        return f"{api_base.rstrip('/')}/audio/transcriptions"

    @contextmanager
    def audio_file_context(self, audio_file: FileTypes) -> Generator[FileTypes, None, None]:
        processed: Final = process_audio_file(audio_file)
        # Foundry reads a filename rather than an upload. Close it before HTTP for Windows.
        with TemporaryDirectory(prefix="litellm-foundry-audio-") as directory:
            path: Final = Path(directory) / f"audio{Path(processed.filename).suffix or '.wav'}"
            path.write_bytes(processed.file_content)
            yield path

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        if not isinstance(audio_file, Path):
            raise TypeError("Foundry Local audio must be prepared using audio_file_context")
        if optional_params.get("stream"):
            raise ValueError("Streaming Foundry Local transcription is not supported by this integration")
        return AudioTranscriptionRequestData(
            data={**optional_params, "model": model, "filename": str(audio_file), "stream": False},
            content_type="application/json",
        )
