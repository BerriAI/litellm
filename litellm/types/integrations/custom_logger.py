from typing import Any, Final

from pydantic import BaseModel, Field

CHAT_COMPLETION_AGENTIC_SURFACE: Final = "chat_completions"
RESPONSES_AGENTIC_SURFACE: Final = "responses"
CODE_INTERPRETER_INTERCEPTION_PREFIX: Final = "_code_interpreter_interception"
NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES: Final = frozenset(
    ("_websearch_interception", "_compression_interception")
)
INTERCEPTION_INTERNAL_PREFIXES: Final = frozenset(
    (
        *NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
        CODE_INTERPRETER_INTERCEPTION_PREFIX,
    )
)


def is_interception_internal_key(
    key: str,
    prefixes: frozenset[str] = INTERCEPTION_INTERNAL_PREFIXES,
) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


class AgenticLoopSafetyError(ValueError):
    """
    Raised when an agentic-loop safety rail refuses a rerun.

    Covers both rails: the bounded-loop cap (``max_agentic_loops``) and the
    repeated tool-call fingerprint cycle break. Subclasses ``ValueError`` so
    callers that already catch the broader type keep working.

    Only the anthropic messages loop raises this today. The chat completions
    loop in ``litellm_core_utils/chat_completion_agentic_loop.py`` still raises
    a plain ``ValueError`` from its own copy of the same rails, so catching
    this type alone will not cover that surface until it is moved over.
    """


class StandardCustomLoggerInitParams(BaseModel):
    """
    Params for initializing a CustomLogger.
    """

    turn_off_message_logging: bool | None = False


class AgenticLoopRequestPatch(BaseModel):
    """
    Patch returned by callbacks to request a follow-up LLM call.
    """

    model: str | None = None
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    max_tokens: int | None = None
    optional_params: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AgenticLoopPlan(BaseModel):
    """
    Typed callback response for agentic-loop reruns.
    """

    run_agentic_loop: bool = False
    request_patch: AgenticLoopRequestPatch | None = None
    response_override: Any | None = None
    terminate: bool = False
    stop_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
