"""Live e2e: POST /v1/audio/speech returns audio.

Registers an OpenAI text-to-speech deployment at runtime and asserts the response
is an audio body (binary, not JSON). Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class TestAudioSpeech:
    def test_audio_speech_returns_audio(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-speech-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-4o-mini-tts", api_key="os.environ/OPENAI_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.audio_speech(key, model, "Hello!")
        require_successful_call(result)
        assert "audio" in (result.content_type or ""), (
            f"/audio/speech content-type is not audio: {result.content_type!r}"
        )
        assert result.body, "/audio/speech returned an empty body"
