"""
Vertex AI Realtime (BidiGenerateContent) config.

Extends GeminiRealtimeConfig but adapts the WSS URL and auth header for the
Vertex AI endpoint instead of Google AI Studio.

URL pattern:
  wss://{location}-aiplatform.googleapis.com/ws/
      google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent

Auth: OAuth2 Bearer token (not an API key).
"""

import json
from typing import List, Optional

from litellm.llms.gemini.realtime.transformation import GeminiRealtimeConfig


class VertexAIRealtimeConfig(GeminiRealtimeConfig):
    """
    Realtime config for Vertex AI (BidiGenerateContent).

    ``access_token`` and ``project`` must be pre-resolved by the caller
    (they require async I/O) and injected at construction time.
    """

    def __init__(self, access_token: str, project: str, location: str) -> None:
        self._access_token = access_token
        self._project = project
        self._location = location

    # ------------------------------------------------------------------
    # URL
    # ------------------------------------------------------------------

    def get_complete_url(
        self, api_base: Optional[str], model: str, api_key: Optional[str] = None  # noqa: ARG002
    ) -> str:
        """
        Build the Vertex AI Live WSS endpoint URL.

        If *api_base* is provided it overrides the default aiplatform host,
        allowing enterprise / VPC-SC deployments to point at a custom gateway.
        """
        if api_base:
            # Allow callers to supply a fully-qualified wss:// base URL.
            base = api_base.rstrip("/")
            base = base.replace("https://", "wss://").replace("http://", "ws://")
            return f"{base}/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

        location = self._location
        if location == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{location}-aiplatform.googleapis.com"

        return f"wss://{host}/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

    # ------------------------------------------------------------------
    # Auth headers
    # ------------------------------------------------------------------

    def validate_environment(
        self,
        headers: dict,
        model: str,  # noqa: ARG002
        api_key: Optional[str] = None,  # noqa: ARG002
    ) -> dict:
        """
        Return headers with a Bearer token for Vertex AI.

        ``api_key`` is intentionally ignored — Vertex AI uses OAuth2 tokens,
        not API keys.  The token was resolved at config-construction time.
        """
        headers = dict(headers)
        headers["Authorization"] = f"Bearer {self._access_token}"
        if self._project:
            headers["x-goog-user-project"] = self._project
        return headers

    # ------------------------------------------------------------------
    # Audio MIME type — Vertex AI needs the sample rate in the MIME string
    # ------------------------------------------------------------------

    def get_audio_mime_type(self, input_audio_format: str = "pcm16") -> str:
        mime_types = {
            "pcm16": "audio/pcm;rate=16000",
            "g711_ulaw": "audio/pcmu",
            "g711_alaw": "audio/pcma",
        }
        return mime_types.get(input_audio_format, "application/octet-stream")

    # ------------------------------------------------------------------
    # Session setup message
    # ------------------------------------------------------------------

    def session_configuration_request(self, model: str) -> str:
        """
        Return the JSON setup message for Vertex AI Live.

        Vertex AI requires the fully-qualified model path:
        ``projects/{project}/locations/{location}/publishers/google/models/{model}``

        Also enables automatic activity detection (server VAD) and output
        audio transcription so the proxy forwards transcript events.
        """
        from litellm.types.llms.gemini import BidiGenerateContentSetup
        from litellm.types.llms.vertex_ai import GeminiResponseModalities

        response_modalities: list[GeminiResponseModalities] = ["AUDIO"]
        full_model_path = (
            f"projects/{self._project}"
            f"/locations/{self._location}"
            f"/publishers/google/models/{model}"
        )
        setup_config: BidiGenerateContentSetup = {
            "model": full_model_path,
            "generationConfig": {"responseModalities": response_modalities},
            # Enable server-side VAD with sensible defaults for voice sessions.
            "realtimeInputConfig": {
                "automaticActivityDetection": {
                    "disabled": False,
                    "silenceDurationMs": 800,
                }
            },
            # Return input transcript so guardrails can inspect user speech.
            "inputAudioTranscription": {},
            # Return output transcript so clients can read what the model said.
            "outputAudioTranscription": {},
        }
        return json.dumps({"setup": setup_config})

    # ------------------------------------------------------------------
    # Request translation
    # ------------------------------------------------------------------

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: Optional[str] = None,
    ) -> List[str]:
        """
        Translate OpenAI realtime client messages to Vertex AI format.

        ``session.update`` is intentionally ignored (returns []) because
        Vertex AI only accepts a single ``setup`` message at the start of
        the connection — sending a second one causes a 1007 close error.
        The initial setup (sent automatically before bidirectional_forward)
        already includes AUDIO modality and server VAD, so there is nothing
        more to configure.
        """
        json_message = json.loads(message)
        if json_message.get("type") == "session.update":
            # Do not forward as a second setup — Vertex AI rejects it.
            return []

        return super().transform_realtime_request(
            message, model, session_configuration_request
        )
