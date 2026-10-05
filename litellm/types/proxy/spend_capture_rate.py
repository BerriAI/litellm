"""The captured-spend to provider-bill ratio: the share of a provider's bill that went through LiteLLM and was priced.

``capture_rate = captured_spend / provider_spend`` over the same UTC days. 1.0 means LiteLLM saw and priced every
dollar the provider billed, lower means traffic reaches the provider outside LiteLLM or cost tracking drops spend,
higher means LiteLLM prices above the bill. ``None`` means the provider billed nothing, so there is no ratio.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.constants import SPEND_CAPTURE_RATE_MAX_RANGE_DAYS

SpendCaptureProvider = Literal["openai"]


class SpendCaptureRateCheckSettings(BaseModel):
    """``general_settings.spend_capture_rate_check``: the daily check of captured spend against the provider bill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    providers: tuple[SpendCaptureProvider, ...] = Field(("openai",), min_length=1)
    threshold: float = Field(0.9, gt=0, le=1)
    lookback_days: int = Field(7, ge=1, le=SPEND_CAPTURE_RATE_MAX_RANGE_DAYS)
    openai_project_ids: tuple[str, ...] = Field(
        (),
        description=(
            "Scope the OpenAI bill to these project ids; empty compares against the whole organization. Captured "
            "spend is never scoped, so list every project LiteLLM's OpenAI keys belong to"
        ),
    )


class CaptureRateDay(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: str
    captured_spend: float
    provider_spend: float
    capture_rate: float | None


class CaptureRateReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: SpendCaptureProvider
    start_date: str
    end_date: str
    captured_spend: float
    provider_spend: float
    capture_rate: float | None
    threshold: float
    below_threshold: bool
    days: tuple[CaptureRateDay, ...]
