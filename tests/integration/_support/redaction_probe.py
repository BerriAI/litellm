import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import httpx
from pydantic import BaseModel

from litellm.integrations.custom_logger import CustomLogger


def _payload(kwargs: Mapping[str, object], response_obj: object, logger: str, hook: str) -> str:
    params: Final = kwargs.get("litellm_params")
    request: Final = params.get("proxy_server_request") if isinstance(params, Mapping) else None
    standard: Final = kwargs.get("standard_logging_object")
    return json.dumps(
        {
            "logger": logger,
            "hook": hook,
            "standard": standard,
            "request": request,
            "response": response_obj.model_dump() if isinstance(response_obj, BaseModel) else response_obj,
            "standard_response": standard.get("response") if isinstance(standard, Mapping) else None,
        },
        default=str,
    )


class SyncRedactionProbe(CustomLogger):
    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        with httpx.Client(trust_env=False) as client:
            response: Final = client.post(
                os.environ["REDACTION_PROBE_URL"], content=_payload(kwargs, response_obj, type(self).__name__, "sync")
            )
            response.raise_for_status()


class RedactionProbe(CustomLogger):
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            response: Final = await client.post(
                os.environ["REDACTION_PROBE_URL"], content=_payload(kwargs, response_obj, type(self).__name__, "async")
            )
            response.raise_for_status()


class DualRedactionProbe(CustomLogger):
    log_success_event = SyncRedactionProbe.log_success_event
    async_log_success_event = RedactionProbe.async_log_success_event


probe: Final = RedactionProbe()
sync_probe: Final = SyncRedactionProbe()
dual_probe: Final = DualRedactionProbe()
