"""Agent harnesses: run Claude Code, Codex, OpenCode, pi, Deep Agents or Tool Loop on any LiteLLM model.

The entrypoints live on the top-level package:

    import litellm
    from litellm import Harness, sandbox

    result = litellm.agent(
        Harness.CLAUDE_CODE,
        "fix the failing test",
        sandbox=sandbox.local("."),
        model="litellm_proxy/claude-sonnet-4-5",  # a model group on your AI Gateway
    )

This module holds the types you get back: events, Result, State, errors.
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
from litellm.harness.evaluation import (
    AgentInput,
    AgentOutput,
    CheckResult,
    CommandCheck,
    EvalAgent,
    EvalCheck,
    EvalOutput,
    EvalRun,
    EvalScorecard,
    EvalStatus,
    EvalSummary,
    EvalTarget,
    EvalTask,
    JudgeCheck,
    PredicateCheck,
    aagent_eval,
    aagent_eval_attempt,
    agent_eval,
)
from litellm.harness.options import (
    ClaudeCodeOptions,
    CodexOptions,
    DeepAgentsOptions,
    OpenCodeOptions,
    PiOptions,
    ToolLoopOptions,
)
from litellm.harness.runtime import (
    AsyncEventStream,
    AsyncSession,
    aagent,
    aagent_resume,
    aagent_session,
    agent_capabilities,
)
from litellm.harness.sync import EventStream, Session, agent, agent_resume, agent_session
from litellm.harness.types import (
    Approval,
    Capabilities,
    Compaction,
    Done,
    Event,
    FileChange,
    Harness,
    Reasoning,
    Result,
    State,
    Text,
    ToolCall,
    ToolResult,
    Usage,
)

__all__ = (
    "AgentInput",
    "AgentOutput",
    "Approval",
    "AsyncEventStream",
    "AsyncSession",
    "Capabilities",
    "CapabilityUnsupported",
    "CheckResult",
    "ClaudeCodeOptions",
    "CodexOptions",
    "CommandCheck",
    "Compaction",
    "DeepAgentsOptions",
    "Done",
    "EvalAgent",
    "EvalCheck",
    "EvalOutput",
    "EvalRun",
    "EvalScorecard",
    "EvalStatus",
    "EvalSummary",
    "EvalTarget",
    "EvalTask",
    "Event",
    "EventStream",
    "FileChange",
    "Harness",
    "HarnessError",
    "HarnessInstallFailed",
    "JudgeCheck",
    "OpenCodeOptions",
    "OptionsMismatch",
    "OutputInvalid",
    "PiOptions",
    "PredicateCheck",
    "Reasoning",
    "Result",
    "SandboxError",
    "Session",
    "SessionClosed",
    "State",
    "StateIncompatible",
    "Text",
    "ToolCall",
    "ToolLoopOptions",
    "ToolResult",
    "Usage",
    "aagent",
    "aagent_eval",
    "aagent_eval_attempt",
    "aagent_resume",
    "aagent_session",
    "agent",
    "agent_capabilities",
    "agent_eval",
    "agent_resume",
    "agent_session",
)
