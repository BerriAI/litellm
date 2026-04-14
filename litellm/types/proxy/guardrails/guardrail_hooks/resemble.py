from typing import Literal, Optional

from pydantic import Field

from .base import GuardrailConfigModel


class ResembleGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "The Resemble AI API token. If not provided, the `RESEMBLE_API_KEY` "
            "environment variable is checked."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Override the Resemble API base URL. If not provided, the "
            "`RESEMBLE_API_BASE` environment variable is checked and falls "
            "back to `https://app.resemble.ai/api/v2`."
        ),
    )
    resemble_threshold: Optional[float] = Field(
        default=0.5,
        description=(
            "Aggregated score above which media is treated as fake (0.0–1.0). "
            "Default 0.5."
        ),
    )
    resemble_media_type: Optional[Literal["audio", "video", "image"]] = Field(
        default=None,
        description=(
            "Optionally force audio / video / image. If omitted, Resemble "
            "auto-detects from the file extension or content type."
        ),
    )
    resemble_audio_source_tracing: Optional[bool] = Field(
        default=False,
        description=(
            "Identify which TTS vendor (elevenlabs, resemble_ai, etc.) generated "
            "the audio when it is flagged as fake."
        ),
    )
    resemble_use_reverse_search: Optional[bool] = Field(
        default=False,
        description=(
            "For image detections, search the web for matching images to "
            "improve accuracy."
        ),
    )
    resemble_zero_retention_mode: Optional[bool] = Field(
        default=False,
        description=(
            "Automatically delete submitted media after detection completes. "
            "URLs are redacted and filenames are tokenized."
        ),
    )
    resemble_metadata_key: Optional[str] = Field(
        default="mediaUrl",
        description=(
            "Key in request `metadata` to read the media URL from when it is "
            "not present in the message content. Default `mediaUrl`."
        ),
    )
    resemble_poll_interval_seconds: Optional[float] = Field(
        default=2.0,
        description=(
            "How often to poll Resemble for the detection result. Default 2s."
        ),
    )
    resemble_poll_timeout_seconds: Optional[float] = Field(
        default=60.0,
        description=(
            "Maximum total time (in seconds) to wait for a detection result "
            "before failing open. Default 60s."
        ),
    )
    resemble_fail_closed: Optional[bool] = Field(
        default=False,
        description=(
            "If true, Resemble API errors (network, auth, timeout) will BLOCK "
            "the request. If false (default), errors are logged but the "
            "request passes through."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Resemble AI Detect"
