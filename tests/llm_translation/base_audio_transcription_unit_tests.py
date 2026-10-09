import httpx
import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os
from litellm._uuid import uuid

import litellm
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

