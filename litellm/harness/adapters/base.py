"""The adapter contract every harness implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

from pydantic import BaseModel

from litellm.harness.errors import CapabilityUnsupported
from litellm.harness.options import HarnessOptions
from litellm.harness.sandbox.base import Sandbox
from litellm.harness.types import (
    Approval,
    Capabilities,
    Event,
    Gateway,
    Harness,
    PermissionMode,
)

if TYPE_CHECKING:
    from litellm.harness.endpoint import ModelEndpoint

ApprovalHandler = Callable[[Approval], bool | Awaitable[bool]]


@dataclass
class SessionContext:
    """Everything an adapter needs for a session. Owned by core; adapters read it."""

    harness: Harness
    sandbox: Sandbox
    session_id: str
    model: str | None = None
    gateway: Gateway | None = None
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
    # Set by the adapter during a turn.
    final_text: str = ""
    output_json: str | None = None
    # Filled by adapters that account usage themselves (endpoint-less harnesses).
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    calls: int = 0


class HarnessAdapter(ABC):
    """One runtime. Core owns validation, endpoint, timeouts, files and results."""

    harness: ClassVar[Harness]
    options_type: ClassVar[type]
    capabilities: ClassVar[Capabilities]
    uses_endpoint: ClassVar[bool] = True

    @abstractmethod
    async def start(self, ctx: SessionContext) -> None:
        """Prepare the runtime for this session (config files, skills, agent build)."""

    @abstractmethod
    def turn(self, ctx: SessionContext, prompt: str) -> AsyncIterator[Event]:
        """Run one turn and yield events. Must not yield Done. Sets ctx.final_text."""

    @abstractmethod
    async def stop(self, ctx: SessionContext) -> None:
        """Stop anything this adapter started. Safe to call twice."""

    def native_session_id(self) -> str | None:
        return None

    async def resume(self, ctx: SessionContext, native_session_id: str) -> None:
        raise CapabilityUnsupported(f"{self.harness.name} does not support resume")

    async def history(self, ctx: SessionContext) -> list[dict[str, Any]]:
        raise CapabilityUnsupported(f"{self.harness.name} does not expose history")
