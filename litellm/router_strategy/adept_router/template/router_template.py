from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypedDict


class AdeptTemplateMatch(TypedDict):
    """Result of matching a prompt to a stored template."""

    template_id: str
    template: str
    target_model: str | None
    metadata: Mapping[str, object] | None


class BaseTemplateRouter(ABC):
    """Abstract base class for template-based prompt routing."""

    @abstractmethod
    async def route(self, prompt: str, system_prompt: str | None = None) -> AdeptTemplateMatch | None:
        """
        Match a prompt to a stored template.

        Returns a dict with template details if matched, None otherwise.
        """
        ...

    @abstractmethod
    async def store_conversation(
        self,
        prompt: str,
        response: str,
        model: str | None = None,
        token_usage: Mapping[str, object] | None = None,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        system_prompt: str | None = None,
        routed_to_slm: bool | None = None,
    ) -> None:
        """Persist a prompt-response pair with its template and per-call metrics."""
        ...
