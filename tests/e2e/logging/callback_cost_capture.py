"""Config-registered CustomLogger that records the cost each success callback is handed.

Registered on the e2e gateway as ``callbacks: ["tests.e2e.logging.callback_cost_capture.capture"]``.
Every success event appends one JSON line to ``E2E_CALLBACK_COST_CAPTURE_FILE`` with the request id
and ``standard_logging_object.response_cost`` exactly as a customer's Langfuse, Datadog, or custom
CustomLogger would see it. Tests read the file back and compare against the SpendLogs row.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict

from litellm.integrations.custom_logger import CustomLogger

CAPTURE_FILE_ENV: Final = "E2E_CALLBACK_COST_CAPTURE_FILE"


class CapturedCost(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    request_id: str
    call_type: str | None
    hook: str
    response_cost: float | None


class _StandardLoggingCostFields(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    call_type: str | None = None
    response_cost: float | None = None


def _append(hook: str, kwargs: dict[str, object]) -> None:
    path: Final = os.environ.get(CAPTURE_FILE_ENV)
    payload: Final = kwargs.get("standard_logging_object")
    if path is None or not isinstance(payload, dict):
        return
    fields: Final = _StandardLoggingCostFields.model_validate(payload)
    row: Final = CapturedCost(
        request_id=fields.id, call_type=fields.call_type, hook=hook, response_cost=fields.response_cost
    )
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row.model_dump()) + "\n")


class CallbackCostCapture(CustomLogger):
    def log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _append("log_success_event", kwargs)

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        _append("async_log_success_event", kwargs)


capture: Final = CallbackCostCapture()
