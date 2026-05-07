"""
Event types emitted by an ``AgentRuntime`` while driving an LLM tool loop.

Events are the canonical wire shape between a runtime and the calling
``Session`` — they get persisted to ``LiteLLM_AgentRunEvent`` (one row per
event, monotonically increasing ``seq``) and streamed back to clients via
the existing /v2/sessions/{id}/runs/{rid}/events SSE endpoint.

Wire shape (snake_case, matches what the integration branch already serves):

    {"seq": 1, "event_type": "assistant_message", "payload": {"content": "..."}}

The Python ``Event`` dataclass below is a small in-memory wrapper. The
runtime yields ``Event`` instances; the ``Session`` is responsible for
turning them into rows.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Event type constants — keep in sync with
# litellm/proxy/agent_session_endpoints/constants.py for the lifecycle ones
# (run_started, run_finished, run_cancelled, run_error, user_message). The
# runtime-specific event types below are new and only flow through Epic C.
# ---------------------------------------------------------------------------

# Lifecycle (also defined in agent_session_endpoints.constants)
EVENT_TYPE_RUN_STARTED = "run_started"
EVENT_TYPE_RUN_FINISHED = "run_finished"
EVENT_TYPE_RUN_CANCELLED = "run_cancelled"
EVENT_TYPE_RUN_ERROR = "run_error"
EVENT_TYPE_USER_MESSAGE = "user_message"

# Runtime-emitted (the actual LLM tool-loop events)
EVENT_TYPE_ASSISTANT_MESSAGE = "assistant_message"
EVENT_TYPE_TOOL_USE = "tool_use"
EVENT_TYPE_TOOL_RESULT = "tool_result"
EVENT_TYPE_THINKING = "thinking"
EVENT_TYPE_SYSTEM = "system"


@dataclass
class Event:
    """A single event from a runtime.

    ``type`` is the canonical event type string (snake_case). ``data`` is a
    free-form dict that gets persisted as the JSON ``payload`` column on
    ``LiteLLM_AgentRunEvent``.

    Example::

        Event(type="tool_use", data={"tool": "Bash", "input": {"command": "ls"}})
    """

    type: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Return the wire payload — currently identity, but reserved for
        future field renames so call sites have one place to change.
        """
        return dict(self.data)
