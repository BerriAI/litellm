from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class StoredTemplate:
    """A template row read back from the store."""

    id: str
    template: str
    template_hash: str | None
    router_id: str
    target_model: str | None
    additional_information: Mapping[str, object] | None
    created_at: datetime | None


class AdeptTemplateStore(ABC):
    """Abstract interface for storing and retrieving ADEPT prompt templates and conversations.

    Implementations talk to the user's own database, so every method is async.
    """

    @abstractmethod
    async def match_by_hash(self, template_hash: str, router_id: str) -> str | None:
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
    async def store_conversation(
        self,
        prompt: str,
        response: str,
        template_id: str | None = None,
        additional_information: Mapping[str, object] | None = None,
    ) -> bool:
        """Store a prompt-response pair linked to a template."""
        ...

    @abstractmethod
    async def store_template(
        self,
        template_id: str,
        template: str,
        template_hash: str,
        target_model: str,
        router_id: str,
        additional_information: Mapping[str, object] | None = None,
    ) -> str | None:
        """
        Store a new template row. Returns the surviving template_id (ours or a concurrent
        insert's) so the caller can use it without a follow-up query.
        """
        ...

    @abstractmethod
    async def get_template(self, template_id: str) -> StoredTemplate | None:
        """Retrieve metadata for a specific template by ID."""
        ...

    @abstractmethod
    async def count_conversation_by_template_id(self, template_id: str) -> int | None:
        """Count conversations associated with a template."""
        ...
