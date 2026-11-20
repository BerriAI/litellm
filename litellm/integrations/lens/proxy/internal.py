from typing import Final

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.integrations.clickhouse.context import lens_analysis
from litellm.integrations.lens.lens_trace_adapter import agent_name
from litellm.litellm_core_utils.initialize_dynamic_callback_params import inherit_message_logging_privacy

LENS_AGENT: Final = "lens-signals"


class LensInternalMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app: Final = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or agent_name(Headers(scope=scope).get("user-agent")) != LENS_AGENT:
            await self.app(scope, receive, send)
            return
        with lens_analysis(), inherit_message_logging_privacy(True):
            await self.app(scope, receive, send)
