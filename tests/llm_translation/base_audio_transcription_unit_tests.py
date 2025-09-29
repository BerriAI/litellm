import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os
from litellm._uuid import uuid

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import transcription
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.utils import ProviderConfigManager
from abc import ABC, abstractmethod

pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

file_path = os.path.join(pwd, "gettysburg.wav")

audio_file = open(file_path, "rb")


class BaseLLMAudioTranscriptionTest(ABC):
    @abstractmethod
    def get_base_audio_transcription_call_args(self) -> dict:
        """Must return the base audio transcription call args"""
        pass

    @abstractmethod
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        """Must return the custom llm provider"""
        pass

    def test_audio_transcription(self):
        """
        Test that the audio transcription is translated correctly.
        """
        litellm.set_verbose = True
        transcription_call_args = self.get_base_audio_transcription_call_args()
        transcript = transcription(**transcription_call_args, file=audio_file)
        print(f"transcript: {transcript.model_dump()}")
        print(f"transcript hidden params: {transcript._hidden_params}")

        assert transcript.text is not None

    @pytest.mark.asyncio
    async def test_audio_transcription_async(self):
        """
        Test that the audio transcription is translated correctly.
        """

        litellm.set_verbose = True
        litellm._turn_on_debug()
        AUDIO_FILE = open(file_path, "rb")
        transcription_call_args = self.get_base_audio_transcription_call_args()
        transcript = await litellm.atranscription(**transcription_call_args, file=AUDIO_FILE)
        print(f"transcript: {transcript.model_dump()}")
        print(f"transcript hidden params: {transcript._hidden_params}")

        assert transcript.text is not None

    def test_audio_transcription_optional_params(self):
        """
        Test that the audio transcription is translated correctly.
        """
        transcription_args = self.get_base_audio_transcription_call_args()
        model = transcription_args["model"]
        custom_llm_provider = self.get_custom_llm_provider()
        optional_params = get_supported_openai_params(
            model=model,
            custom_llm_provider=custom_llm_provider.value,
            request_type="transcription",
        )
        print(f"optional_params: {optional_params}")
        assert optional_params is not None
        assert (
            "max_completion_tokens" not in optional_params
        )  # assert default chat completion response not returned

    def test_audio_transcription_config(self):
        """
        Test that the audio transcription config is implemented and correctly instrumented.
        """
        transcription_args = self.get_base_audio_transcription_call_args()
        model = transcription_args["model"]
        custom_llm_provider = self.get_custom_llm_provider()
        config = ProviderConfigManager.get_provider_audio_transcription_config(
            model=model,
            provider=custom_llm_provider,
        )
        print(f"config: {config}")
        assert config is not None
        assert isinstance(config, BaseAudioTranscriptionConfig)
