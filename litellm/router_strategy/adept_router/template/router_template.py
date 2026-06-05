from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTemplateRouter(ABC):
    """Abstract base class for template-based prompt routing."""

    @abstractmethod
    def route(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Match a prompt to a stored template.

        Returns a dict with template details if matched, None otherwise.
        """
        ...

    @abstractmethod
    def store_conversation(
        self,
        prompt: str,
        response: str,
        model: Optional[str] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        cost_usd: Optional[float] = None,
        latency_ms: Optional[float] = None,
        system_prompt: Optional[str] = None,
        routed_to_slm: Optional[bool] = None,
    ) -> None:
        """Persist a prompt-response pair with its template and per-call metrics."""
        ...
