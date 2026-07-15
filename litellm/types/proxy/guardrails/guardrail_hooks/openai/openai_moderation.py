from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..base import GuardrailConfigModel


class BaseOpenAIModerationGuardrailConfigModel(GuardrailConfigModel):
    """Base configuration model for the OpenAI Moderation guardrail"""

    model: Optional[Literal["omni-moderation-latest", "text-moderation-latest"]] = Field(
        default="omni-moderation-latest",
        description="The OpenAI moderation model to use. 'omni-moderation-latest' supports more categorization options and multi-modal inputs. Defaults to 'omni-moderation-latest'.",
    )


class OpenAIModerationGuardrailConfigModel(BaseOpenAIModerationGuardrailConfigModel):
    """Configuration model for the OpenAI Moderation guardrail"""

    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.",
    )

    api_base: Optional[str] = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL. Defaults to 'https://api.openai.com/v1'.",
    )

    streaming_end_of_stream_only: Optional[bool] = Field(
        default=False,
        description="If False (default), moderation runs on sampled chunks during the stream at the cadence set by streaming_sampling_rate, and an in-flight violation stops further chunks from streaming. If True, moderation runs once at end of stream over the assembled response — lower cost and latency, but flagged content has already streamed to the client before the terminal block.",
    )

    streaming_sampling_rate: Optional[int] = Field(
        default=5,
        description="When streaming_end_of_stream_only is False, moderation runs every Nth streamed chunk. Ignored when streaming_end_of_stream_only is True.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "OpenAI Moderation"
