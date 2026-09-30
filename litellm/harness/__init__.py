"""litellm.harness: run coding-agent runtimes (Claude Code, Codex, OpenCode, Deep Agents)
through one API, with every model call routed via LiteLLM.

    from litellm import harness, sandbox

    result = harness.run(
        harness.Harness.CLAUDE_CODE,
        "fix the failing test",
        sandbox=sandbox.local("."),
        model="claude-sonnet-4-5",
    )
"""

from litellm.harness.errors import (
    CapabilityUnsupported,
    HarnessError,
    HarnessInstallFailed,
    OptionsMismatch,
    OutputInvalid,
    SandboxError,
    SessionClosed,
    StateIncompatible,
)
from litellm.harness.options import (
    ClaudeCodeOptions,
    CodexOptions,
    DeepAgentsOptions,
    OpenCodeOptions,
)
from litellm.harness.runtime import (
    AsyncEventStream,
    AsyncSession,
    aresume,
    arun,
    asession,
    astream,
    capabilities,
)
from litellm.harness.sync import EventStream, Session, resume, run, session, stream
from litellm.harness.types import (
    Approval,
    Capabilities,
    Compaction,
    Done,
    Event,
    FileChange,
    Gateway,
    Harness,
    Reasoning,
    Result,
    State,
    Text,
    ToolCall,
    ToolResult,
    Usage,
)

__all__ = [
    "Approval",
    "AsyncEventStream",
    "AsyncSession",
    "Capabilities",
    "CapabilityUnsupported",
    "ClaudeCodeOptions",
    "CodexOptions",
    "Compaction",
    "DeepAgentsOptions",
    "Done",
    "Event",
    "EventStream",
    "FileChange",
    "Gateway",
    "Harness",
    "HarnessError",
    "HarnessInstallFailed",
    "OpenCodeOptions",
    "OptionsMismatch",
    "OutputInvalid",
    "Reasoning",
    "Result",
    "SandboxError",
    "Session",
    "SessionClosed",
    "State",
    "StateIncompatible",
    "Text",
    "ToolCall",
    "ToolResult",
    "Usage",
    "aresume",
    "arun",
    "asession",
    "astream",
    "capabilities",
    "resume",
    "run",
    "session",
    "stream",
]
