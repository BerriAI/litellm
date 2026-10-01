"""The handler interface the runtime drives. A handler owns I/O for one session."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from litellm.harness.context import SessionContext
from litellm.harness.errors import CapabilityUnsupported
from litellm.harness.types import Event
from litellm.llms.base_llm.harness.transformation import BaseHarnessConfig


class BaseHarnessHandler(ABC):
    """Runs one harness session. The config decides what to run; the handler runs it."""

    def __init__(self, config: BaseHarnessConfig) -> None:
        self.config = config

    @abstractmethod
    async def start(self, ctx: SessionContext) -> None:
        """Prepare the runtime (config files, skills, agent build). Called again after an interrupt."""

    @abstractmethod
    def turn(self, ctx: SessionContext, prompt: str) -> AsyncIterator[Event]:
        """Run one turn and yield events (never Done). Sets ctx.final_text / ctx.output_json."""

    @abstractmethod
    async def stop(self, ctx: SessionContext) -> None:
        """Stop anything this handler started. Safe to call twice."""

    @abstractmethod
    def native_session_id(self) -> str | None:
        """The runtime's own session id, for State / resume."""

    @abstractmethod
    async def resume(self, ctx: SessionContext, native_session_id: str) -> None:
        """Continue the runtime's own session on the next turn."""

    async def history(self, ctx: SessionContext) -> list[dict[str, Any]]:  # mutable-ok: public history() API shape
        raise CapabilityUnsupported(f"Harness.{self.config.harness.name} does not expose history")
