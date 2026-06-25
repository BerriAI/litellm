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

from litellm import verbose_logger
from litellm.llms.gemini.realtime.transformation import GeminiRealtimeConfig


class VertexAIRealtimeConfig(GeminiRealtimeConfig):
    """
    Realtime config for Vertex AI (BidiGenerateContent).

    ``access_token`` and ``project`` must be pre-resolved by the caller
    (they require async I/O) and injected at construction time.
    """

    def __init__(self, access_token: str, project: str, location: str) -> None:
        super().__init__()
        self._access_token = access_token
        self._project = project
        self._location = location

    def _include_function_response_id(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # URL
    # ------------------------------------------------------------------

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        api_key: Optional[str] = None,  # noqa: ARG002
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
            # Gemini Live native audio (OpenAI GA realtime default) is 24kHz PCM.
            "pcm16": "audio/pcm;rate=24000",
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

    def _vertex_model_path(self, model: str) -> str:
        """Return the fully-qualified Vertex AI model resource path."""
        return (
            f"projects/{self._project}"
            f"/locations/{self._location}"
            f"/publishers/google/models/{model}"
        )

    def _build_vertex_ai_setup_config(self, model: str, session_params: dict) -> dict:
        """Build Vertex AI setup configuration with proper model path and defaults."""
        # Normalize GA-remapped fields (``output_modalities``, nested
        # ``audio.input.transcription``, ``audio.input.turn_detection``) back to
        # their flat beta keys so ``map_openai_params`` picks them up. Without
        # this, GA clients' explicit modality / transcription / turn-detection
        # settings would be silently dropped because ``map_openai_params`` only
        # recognises the flat OpenAI-beta key names.
        session_params = self._normalize_session_payload_for_mapping(session_params)
        setup_config = self.map_openai_params(
            optional_params={}, non_default_params=session_params
        )

        # Use full Vertex AI model path
        setup_config["model"] = self._vertex_model_path(model)

        # Add Vertex AI specific defaults if not provided
        generation_config = setup_config.setdefault("generationConfig", {})
        generation_config.setdefault("responseModalities", ["AUDIO"])

        # Ensure Vertex defaults for realtimeInputConfig apply even when
        # the client provided a partial ``turn_detection`` (e.g. only
        # ``silence_duration_ms``). ``map_automatic_turn_detection`` sets
        # ``disabled=True`` whenever ``create_response`` is absent or
        # ``False``. Force ``disabled=False`` only when the client did
        # not explicitly request ``create_response: False`` — that path
        # is how transcription guardrails suppress automatic responses,
        # and overriding it here would silently bypass the guardrail.
        # Vertex Live has no "VAD on, no auto-response" mode, so callers
        # that need that behaviour must accept that VAD is off.
        client_turn_detection = session_params.get("turn_detection")
        client_disabled_auto_response = (
            isinstance(client_turn_detection, dict)
            and client_turn_detection.get("create_response") is False
        )
        realtime_input_config = setup_config.setdefault("realtimeInputConfig", {})
        automatic_detection = realtime_input_config.setdefault(
            "automaticActivityDetection", {}
        )
        if not client_disabled_auto_response:
            automatic_detection["disabled"] = False
        automatic_detection.setdefault("silenceDurationMs", 800)

        setup_config.setdefault("inputAudioTranscription", {})
        setup_config.setdefault("outputAudioTranscription", {})

        return setup_config

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: Optional[str] = None,
    ) -> List[str]:
        """
        Translate OpenAI realtime client messages to Vertex AI format.

        On the first ``session.update`` (when no setup has been sent yet) the
        full ``BidiGenerateContentSetup`` is built with Vertex AI's model path
        and forwarded. Any later ``session.update`` is dropped: Vertex AI
        documents ``setup`` as the first-and-only client message, and a second
        ``setup`` closes the connection with a 1007 policy error.
        """
        json_message = json.loads(message)
        msg_type = json_message.get("type")

        if msg_type == "session.update":
            if session_configuration_request is None:
                setup_config = self._build_vertex_ai_setup_config(
                    model, json_message.get("session") or {}
                )
                gemini_setup_msg = json.dumps({"setup": setup_config})

                verbose_logger.debug(
                    "Vertex AI Realtime: Sending initial setup with tools to backend"
                )
                return [gemini_setup_msg]

            # A follow-up session.update can't be forwarded as a second setup
            # (Vertex Live closes the WebSocket with 1007). If this drop is
            # silencing the audio-transcription guardrail's create_response
            # disable, surface a warning so operators know the model will
            # auto-respond before the guardrail can gate it on Vertex AI.
            client_turn_detection = GeminiRealtimeConfig._extract_turn_detection(
                json_message.get("session") or {}
            )
            if (
                isinstance(client_turn_detection, dict)
                and client_turn_detection.get("create_response") is False
            ):
                verbose_logger.warning(
                    "Vertex AI Realtime: Dropping subsequent session.update "
                    "(turn_detection.create_response=False) — Vertex Live "
                    "rejects a second setup message. Audio-transcription "
                    "guardrails cannot suppress the model's auto-response on "
                    "Vertex AI in non-deferred mode."
                )
            else:
                verbose_logger.debug(
                    "Vertex AI Realtime: Ignoring session.update (setup already sent)"
                )
            return []

        return super().transform_realtime_request(
            message, model, session_configuration_request
        )
