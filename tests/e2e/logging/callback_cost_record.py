"""Capture row shared by callback_cost_capture (proxy side) and the test; must stay importable without litellm."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

CAPTURE_FILE_ENV: Final = "E2E_CALLBACK_COST_CAPTURE_FILE"


class CapturedCost(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    request_id: str
    call_type: str | None
    hook: str
    response_cost: float | None
