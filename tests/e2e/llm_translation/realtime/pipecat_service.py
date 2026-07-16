"""Shared pipecat realtime service for the proxy realtime e2e suite.

Both pipecat suites drive the proxy through pipecat's GA realtime service. The
stock ``OpenAIRealtimeLLMService`` sends websocket keepalive pings at its default
interval, and the LiteLLM proxy does not answer them, so the connection is closed
with a 1011 before the run completes. ``LiteLLMRealtimeLLMService`` carries the
three overrides from bot.py needed to talk to the proxy, the keepalive-disabling
``_connect`` being the load-bearing one for every provider.

Importing this module skips the collecting test when pipecat is not installed:

    uv pip install "pipecat-ai[openai]<1.5"

pipecat 1.5.0 broke the azure/gemini/vertex realtime paths through the proxy
(openai still passes; raw-ws passes for every provider), so imports fail loudly
on >=1.5 instead of letting the suites fail as opaque no-response timeouts.
"""

# pipecat is an optional, dynamically typed dependency loaded behind importorskip,
# so its symbols are Unknown to the type checker; relax those rules for this file.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUntypedBaseClass=false, reportUnknownParameterType=false, reportMissingParameterType=false

from importlib.metadata import version as _distribution_version

import pytest

pytest.importorskip("pipecat", reason="pipecat-ai not installed")

_PIPECAT_VERSION = _distribution_version("pipecat-ai")
assert tuple(int(part) for part in _PIPECAT_VERSION.split(".")[:2]) < (1, 5), (
    f"pipecat-ai {_PIPECAT_VERSION} is installed, but >=1.5 breaks the "
    'azure/gemini/vertex realtime paths; install "pipecat-ai[openai]<1.5"'
)

from pipecat.services.openai.realtime.llm import (  # noqa: E402
    OpenAIRealtimeLLMService,
)
from websockets.asyncio.client import connect as websocket_connect  # noqa: E402


class LiteLLMRealtimeLLMService(OpenAIRealtimeLLMService):
    """Minimal LiteLLM-aware realtime service for tests.

    Three overrides carried from bot.py:
      1. _connect   - disables websockets keepalive pings (LiteLLM proxy
                      does not respond to pings, causing 1011 errors).
      2. _create_response - sends session.update with tools BEFORE history
                      items so that Gemini's deferred-setup logic in the
                      proxy can include tools in the very first setup
                      message it forwards to the backend.
      3. _handle_evt_session_created - immediately marks the session ready
                      without waiting for a session.updated echo (the
                      LiteLLM Gemini bridge does not send one).
    """

    async def _connect(self) -> None:
        if self._websocket:
            return
        try:
            # self.base_url already carries the ?model=<alias> the proxy routes on:
            # the parent __init__ sets self.base_url = f"{base_url}?model={settings.model}"
            # before _connect runs, so passing it through preserves the query param.
            self._websocket = await websocket_connect(
                uri=self.base_url,
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                ping_interval=None,
                close_timeout=10,
                max_size=None,
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as exc:
            await self.push_error(error_msg=f"Error connecting: {exc}", exception=exc)
            self._websocket = None

    async def _create_response(self) -> None:
        if self._llm_needs_conversation_setup and self._context:
            await self._send_session_update()
        await super()._create_response()

    async def _handle_evt_session_created(self, evt: object) -> None:
        await self._send_session_update()
        self._api_session_ready = True
        if self._run_llm_when_api_session_ready:
            self._run_llm_when_api_session_ready = False
            await self._create_response()
