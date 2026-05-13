"""TokenJam logger for LiteLLM.

TokenJam (https://tokenjam.dev) is an open-source, local-first, OTel-native
observability and token-economics platform for autonomous AI agents and coding
agents. This logger ships LiteLLM call data to a TokenJam server via the
``tokenjam`` Python SDK.

Configuration via environment variables:
    TJ_ENDPOINT       — TokenJam server URL (default: http://localhost:7391)
    TJ_INGEST_SECRET  — optional ingest secret if the server requires auth

Usage:
    import litellm
    litellm.success_callback = ["tokenjam"]
    litellm.failure_callback = ["tokenjam"]
"""

import asyncio
import os
from typing import Any, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger


class TokenJamLogger(CustomLogger):
    """Named-callback adapter: delegates to the ``tokenjam`` SDK if installed."""

    def __init__(self) -> None:
        super().__init__()
        self._client: Optional[Any] = None

        try:
            from tokenjam.sdk import TokenJamClient
        except ImportError:
            verbose_logger.warning(
                "TokenJam logger configured but 'tokenjam' package is not "
                "installed. Install with: pip install tokenjam"
            )
            return

        try:
            self._client = TokenJamClient(
                endpoint=os.getenv("TJ_ENDPOINT", "http://localhost:7391"),
                ingest_secret=os.getenv("TJ_INGEST_SECRET"),
            )
        except Exception as e:
            verbose_logger.warning(
                "TokenJam client initialization failed: %s. Events will be dropped.",
                e,
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        if self._client is None:
            return
        try:
            self._client.emit_litellm_span(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                success=True,
            )
        except Exception as e:
            verbose_logger.debug(
                "[Non-Blocking] TokenJam log_success_event: %s",
                e,
            )

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        if self._client is None:
            return
        try:
            self._client.emit_litellm_span(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                success=False,
            )
        except Exception as e:
            verbose_logger.debug(
                "[Non-Blocking] TokenJam log_failure_event: %s",
                e,
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # emit_litellm_span makes a blocking HTTP call; offload to a thread
        # so we don't stall the event loop on every LiteLLM async call.
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.log_success_event,
            kwargs,
            response_obj,
            start_time,
            end_time,
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.log_failure_event,
            kwargs,
            response_obj,
            start_time,
            end_time,
        )
