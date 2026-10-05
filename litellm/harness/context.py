"""Per-session state shared by the runtime, handlers and harness configs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel

from litellm.harness.options import HarnessOptions
from litellm.harness.sandbox.base import Sandbox
from litellm.harness.types import Approval, Harness, PermissionMode

if TYPE_CHECKING:
    from litellm.harness.endpoint import ModelEndpoint

ApprovalHandler: TypeAlias = Callable[
    [Approval], bool | Awaitable[bool]  # mutable-ok: Callable parameter list in a type alias, not a runtime collection
]


@dataclass(frozen=True)
class GatewayTarget:
    """Resolved LiteLLM AI Gateway for `litellm_proxy/` models. Internal, not exported."""

    api_base: str
    api_key: str


@dataclass
class SessionContext:
    """Everything a handler and config need for a session. Owned by the runtime."""

    harness: Harness
    sandbox: Sandbox
    session_id: str
    # Model name as sent to the runtime (litellm_proxy/ prefix already stripped).
    model: str | None = None
    gateway: GatewayTarget | None = None
    api_key: str | None = None
    api_base: str | None = None
    endpoint: ModelEndpoint | None = None
    instructions: str | None = None
    tools: Sequence[Callable[..., Any]] = ()
    skills: Sequence[str] = ()
    disable_tools: Sequence[str] = ()
    permissions: PermissionMode = "full"
    on_approval: ApprovalHandler | None = None
    output: type[BaseModel] | None = None
    max_turns: int | None = None
    timeout: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    options: HarnessOptions | None = None
    # Set by the handler after each turn.
    final_text: str = ""
    output_json: str | None = None
    # Usage for in-process harnesses that call LiteLLM directly (no model endpoint).
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    calls: int = 0
