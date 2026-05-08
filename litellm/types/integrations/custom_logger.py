from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StandardCustomLoggerInitParams(BaseModel):
    """
    Params for initializing a CustomLogger.
    """

    turn_off_message_logging: Optional[bool] = False


class AgenticLoopRequestPatch(BaseModel):
    """
    Patch returned by callbacks to request a follow-up LLM call.
    """

    model: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    optional_params: Dict[str, Any] = Field(default_factory=dict)
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class AgenticLoopPlan(BaseModel):
    """
    Typed callback response for agentic-loop reruns.
    """

    run_agentic_loop: bool = False
    request_patch: Optional[AgenticLoopRequestPatch] = None
    response_override: Optional[Any] = None
    terminate: bool = False
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
