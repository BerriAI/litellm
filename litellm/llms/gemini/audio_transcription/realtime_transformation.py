from litellm.types.llms.openai import OpenAIAudioTranscriptionOptionalParams

from .transformation import GeminiAudioTranscriptionConfig


class GeminiRealtimeAudioTranscriptionConfig(GeminiAudioTranscriptionConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "keywords"]
