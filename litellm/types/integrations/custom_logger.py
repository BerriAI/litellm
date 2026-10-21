from collections.abc import Mapping
from typing import Any, Final

from pydantic import Field

from litellm.types.litellm_params import AgenticSurface
from litellm.types.llms.base import LiteLLMBaseModel

CHAT_COMPLETION_AGENTIC_SURFACE: Final[AgenticSurface] = "chat_completions"
RESPONSES_AGENTIC_SURFACE: Final[AgenticSurface] = "responses"
CODE_INTERPRETER_INTERCEPTION_PREFIX: Final = "_code_interpreter_interception"
HEADROOM_INTERCEPTION_PREFIX: Final = "_headroom_interception"
WEBSEARCH_INTERCEPTION_PREFIX: Final = "_websearch_interception"
HEADROOM_CONVERTED_STREAM_KEY: Final = f"{HEADROOM_INTERCEPTION_PREFIX}_converted_stream"
WEBSEARCH_CONVERTED_STREAM_KEY: Final = f"{WEBSEARCH_INTERCEPTION_PREFIX}_converted_stream"
WEBSEARCH_STREAM_OPTIONS_KEY: Final = f"{WEBSEARCH_INTERCEPTION_PREFIX}_stream_options"
CODE_INTERPRETER_STREAM_OPTIONS_KEY: Final = f"{CODE_INTERPRETER_INTERCEPTION_PREFIX}_stream_options"
HEADROOM_STREAM_OPTIONS_KEY: Final = f"{HEADROOM_INTERCEPTION_PREFIX}_stream_options"
STREAM_OPTIONS_STASH_KEYS: Final = (
    CODE_INTERPRETER_STREAM_OPTIONS_KEY,
    HEADROOM_STREAM_OPTIONS_KEY,
    WEBSEARCH_STREAM_OPTIONS_KEY,
)
NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES: Final = frozenset(
    (
        WEBSEARCH_INTERCEPTION_PREFIX,
        "_compression_interception",
        HEADROOM_INTERCEPTION_PREFIX,
    )
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


CONVERTED_STREAM_KEYS: Final = frozenset(f"{prefix}_converted_stream" for prefix in INTERCEPTION_INTERNAL_PREFIXES)


def converted_stream_requested(params: Mapping[str, object]) -> bool:
    return any(bool(params.get(key)) for key in CONVERTED_STREAM_KEYS)


def as_converted_stream(
    params: Mapping[str, object], prefix: str
) -> dict[str, object]:  # mutable-ok: deployment hooks hand back the kwargs dict their caller keeps editing
    stream_options: Final = params.get("stream_options")
    stash: Final = {} if stream_options is None else {f"{prefix}_stream_options": stream_options}
    return {
        **{key: value for key, value in params.items() if key != "stream_options"},
        "stream": False,
        f"{prefix}_converted_stream": True,
        **stash,
    }


def stashed_stream_options(params: Mapping[str, object]) -> object:
    return next((params[key] for key in STREAM_OPTIONS_STASH_KEYS if params.get(key) is not None), None)


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


class StandardCustomLoggerInitParams(LiteLLMBaseModel):
    """
    Params for initializing a CustomLogger.
    """

    turn_off_message_logging: bool | None = False


class AgenticLoopRequestPatch(LiteLLMBaseModel):
    """
    Patch returned by callbacks to request a follow-up LLM call.
    """

    model: str | None = None
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    max_tokens: int | None = None
    optional_params: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AgenticLoopPlan(LiteLLMBaseModel):
    """
    Typed callback response for agentic-loop reruns.
    """

    run_agentic_loop: bool = False
    request_patch: AgenticLoopRequestPatch | None = None
    response_override: Any | None = None
    terminate: bool = False
    stop_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
