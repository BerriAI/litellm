import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.hosted_vllm.transcriptions.transformation import (
    HostedVLLMAudioTranscriptionConfig,
)


def _complete_url(api_base: str | None) -> str:
    return HostedVLLMAudioTranscriptionConfig().get_complete_url(
        api_base=api_base,
        api_key=None,
        model="whisper-1",
        optional_params={},
        litellm_params={},
    )


class TestHostedVLLMTranscriptionUrl:
    @pytest.mark.parametrize(
        "api_base,expected",
        [
            (
                "http://vllm.example.com",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
            (
                "http://vllm.example.com/",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
            (
                "http://vllm.example.com/v1",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
            (
                "http://vllm.example.com/v1/",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
            (
                "http://proxy.example.com/qwen3-asr/v1",
                "http://proxy.example.com/qwen3-asr/v1/audio/transcriptions",
            ),
            (
                "http://vllm.example.com/v1/audio/transcriptions",
                "http://vllm.example.com/v1/audio/transcriptions",
            ),
        ],
    )
    def test_get_complete_url(self, api_base: str, expected: str) -> None:
        assert _complete_url(api_base) == expected

    def test_get_complete_url_requires_api_base(self) -> None:
        with pytest.raises(ValueError, match="api_base must be provided"):
            _complete_url(None)
