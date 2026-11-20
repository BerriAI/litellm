"""A logging callback that relays the messages it is handed, after logging hooks ran, to `LOGGED_MESSAGES_URL`."""

import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import httpx

from litellm.integrations.custom_logger import CustomLogger


class LoggedMessagesProbe(CustomLogger):
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            response: Final = await client.post(
                os.environ["LOGGED_MESSAGES_URL"],
                content=json.dumps({"messages": kwargs.get("messages")}, default=str),
            )
            response.raise_for_status()


probe: Final = LoggedMessagesProbe()
