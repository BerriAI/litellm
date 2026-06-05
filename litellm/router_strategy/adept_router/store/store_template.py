from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AdeptTemplateStore(ABC):
    """Abstract interface for storing and retrieving ADEPT prompt templates and conversations."""

    @abstractmethod
    def match_by_hash(self, template_hash: str, router_id: str) -> Optional[str]:
        """
        Look up a template ID by the SHA-256 hash of its masked template string.

        Args:
            template_hash: SHA-256 hex digest of the masked template.
            router_id: The router that owns this template.

        Returns:
            The template ID if found, None otherwise.
        """
        ...

    @abstractmethod
    def store_conversation(
        self,
        prompt: str,
        response: str,
        template_id: Optional[str] = None,
        additional_information: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a prompt-response pair linked to a template."""
        ...

    @abstractmethod
    def store_template(
        self,
        template_id: str,
        template: str,
        template_hash: str,
        target_model: str,
        router_id: str,
        additional_information: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a new template row. Returns the surviving template_id (ours or a concurrent
        insert's) so the caller can use it without a follow-up query.
        """
        ...

    @abstractmethod
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific template by ID."""
        ...

    @abstractmethod
    def count_conversation_by_template_id(self, template_id: str) -> Optional[int]:
        """Count conversations associated with a template."""
        ...
