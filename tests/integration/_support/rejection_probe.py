import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import httpx
from fastapi import HTTPException

from litellm.integrations.custom_logger import CustomLogger

REJECT_HEADER: Final = "x-rejection-probe"
REJECTION_DETAIL: Final = "rejected by the rejection probe"


class RejectionProbe(CustomLogger):
    """Rejects a request that carries `REJECT_HEADER` in its pre-call hook with a 429, and reports the failure
    log of every rejected request: the callback `input` and the standard logging payload."""

    async def async_pre_call_hook(
        self, user_api_key_dict: object, cache: object, data: Mapping[str, object], call_type: str
    ) -> None:
        request: Final = data.get("proxy_server_request")
        headers: Final = request.get("headers") if isinstance(request, Mapping) else None
        if isinstance(headers, Mapping) and REJECT_HEADER in headers:
            raise HTTPException(status_code=429, detail=REJECTION_DETAIL)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            response: Final = await client.post(
                os.environ["REJECTION_PROBE_URL"],
                content=json.dumps(
                    {"input": kwargs.get("input"), "standard": kwargs.get("standard_logging_object")}, default=str
                ),
            )
            response.raise_for_status()


probe: Final = RejectionProbe()
