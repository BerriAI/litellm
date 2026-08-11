"""
Attachment Registry - Manages policy attachments from YAML config.

Attachments define WHERE policies apply, separate from the policy definitions.
This allows the same policy to be attached to multiple scopes.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.repositories.table_repositories import PolicyAttachmentRepository
from litellm.types.proxy.policy_engine import (
    PolicyAttachment,
    PolicyAttachmentCreateRequest,
    PolicyAttachmentDBResponse,
    PolicyMatchContext,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prisma.models import LiteLLM_PolicyAttachmentTable

    from litellm.proxy.utils import PrismaClient


class PolicyAttachmentMatch(TypedDict):
    policy_name: str
    matched_via: str


class AttachmentRegistry:
    """
    In-memory registry for storing and managing policy attachments.

    Attachments define the relationship between policies and their scopes.
    A single policy can have multiple attachments (applied to different scopes).

    Example YAML:
    ```yaml
    attachments:
      - policy: global-baseline
        scope: "*"
      - policy: healthcare-compliance
        teams: [healthcare-team]
      - policy: dev-safety
        keys: ["dev-key-*"]
    ```
    """

    def __init__(self) -> None:
        self._attachments: list[PolicyAttachment] = []
        self._config_attachments: tuple[PolicyAttachment, ...] = ()
        self._initialized: bool = False

    def load_attachments(self, attachments_config: list[dict[str, Any]]) -> None:
        """
        Load attachments from a configuration list.

        Args:
            attachments_config: List of attachment dictionaries from YAML.
        """
        self._attachments = []

        for attachment_data in attachments_config:
            try:
                attachment = self._parse_attachment(attachment_data)
                self._attachments.append(attachment)
                verbose_proxy_logger.debug("Loaded attachment for policy: %s", attachment.policy)
            except Exception as e:
                verbose_proxy_logger.error("Error loading attachment: %s", e)
                raise ValueError(f"Invalid attachment: {e}") from e

        self._config_attachments = tuple(self._attachments)
        self._initialized = True
        verbose_proxy_logger.info("Loaded %s policy attachments", len(self._attachments))

    def _parse_attachment(self, attachment_data: dict[str, Any]) -> PolicyAttachment:
        """
        Parse an attachment from raw configuration data.

        Args:
            attachment_data: Raw attachment configuration

        Returns:
            Parsed PolicyAttachment object
        """
        return PolicyAttachment(
            policy=attachment_data.get("policy", ""),
            scope=attachment_data.get("scope"),
            teams=attachment_data.get("teams"),
            keys=attachment_data.get("keys"),
            models=attachment_data.get("models"),
            tags=attachment_data.get("tags"),
        )

    def get_attached_policies(self, context: PolicyMatchContext) -> list[str]:
        """
        Get list of policy names attached to the given context.

        Args:
            context: The request context to match against

        Returns:
            List of policy names that are attached to matching scopes
        """
        return [r["policy_name"] for r in self.get_attached_policies_with_reasons(context)]

    def get_attached_policies_with_reasons(self, context: PolicyMatchContext) -> list[PolicyAttachmentMatch]:
        """
        Get list of policy names and match reasons for the given context.

        Returns a list of dicts with 'policy_name' and 'matched_via' keys.
        The 'matched_via' describes which dimension caused the match.
        """
        from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher

        results: Final[list[PolicyAttachmentMatch]] = []
        seen_policies: Final[set[str]] = set()

        for attachment in self._attachments:
            scope = attachment.to_policy_scope()
            if PolicyMatcher.scope_matches(scope=scope, context=context):
                if attachment.policy not in seen_policies:
                    seen_policies.add(attachment.policy)
                    matched_via = self._describe_match_reason(attachment, context)
                    results.append(
                        {
                            "policy_name": attachment.policy,
                            "matched_via": matched_via,
                        }
                    )
                    verbose_proxy_logger.debug(
                        "Attachment matched: policy=%s, matched_via=%s, context=(team=%s, key=%s, model=%s)",
                        attachment.policy,
                        matched_via,
                        context.team_alias,
                        context.key_alias,
                        context.model,
                    )

        return results

    @staticmethod
    def _describe_match_reason(attachment: PolicyAttachment, context: PolicyMatchContext) -> str:
        """Describe why an attachment matched the context."""
        from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher

        if attachment.is_global():
            return "scope:*"

        reasons: Final = []
        if attachment.tags and context.tags:
            matching_tags: Final = [t for t in context.tags if PolicyMatcher.matches_pattern(t, attachment.tags)]
            if matching_tags:
                reasons.append(f"tag:{matching_tags[0]}")
        if attachment.teams and context.team_alias:
            reasons.append(f"team:{context.team_alias}")
        if attachment.keys and context.key_alias:
            reasons.append(f"key:{context.key_alias}")
        if attachment.models and context.model:
            reasons.append(f"model:{context.model}")

        return "+".join(reasons) if reasons else "scope:default"

    def is_policy_attached(self, policy_name: str, context: PolicyMatchContext) -> bool:
        """
        Check if a specific policy is attached to the given context.

        Args:
            policy_name: Name of the policy to check
            context: The request context to match against

        Returns:
            True if the policy is attached to a matching scope
        """
        attached: Final = self.get_attached_policies(context)
        return policy_name in attached

    def get_all_attachments(self) -> list[PolicyAttachment]:
        """
        Get all loaded attachments.

        Returns:
            List of all PolicyAttachment objects
        """
        return self._attachments.copy()

    def get_config_attachments(self) -> tuple[PolicyAttachment, ...]:
        """
        Get the attachments loaded from config.yaml.

        Returns:
            Tuple of config-defined PolicyAttachment objects
        """
        return self._config_attachments

    def get_attachments_for_policy(self, policy_name: str) -> list[PolicyAttachment]:
        """
        Get all attachments for a specific policy.

        Args:
            policy_name: Name of the policy

        Returns:
            List of attachments for the policy
        """
        return [a for a in self._attachments if a.policy == policy_name]

    def is_initialized(self) -> bool:
        """
        Check if the registry has been initialized with attachments.

        Returns:
            True if attachments have been loaded, False otherwise
        """
        return self._initialized

    def clear(self) -> None:
        """
        Clear all attachments from the registry.
        """
        self._attachments = []
        self._config_attachments = ()
        self._initialized = False

    def add_attachment(self, attachment: PolicyAttachment) -> None:
        """
        Add a single attachment.

        Args:
            attachment: PolicyAttachment object to add
        """
        self._attachments.append(attachment)
        self._initialized = True
        verbose_proxy_logger.debug("Added attachment for policy: %s", attachment.policy)

    def remove_attachments_for_policy(self, policy_name: str) -> int:
        """
        Remove all attachments for a specific policy.

        Args:
            policy_name: Name of the policy

        Returns:
            Number of attachments removed
        """
        original_count: Final = len(self._attachments)
        self._attachments = [a for a in self._attachments if a.policy != policy_name]
        removed_count: Final = original_count - len(self._attachments)
        if removed_count > 0:
            verbose_proxy_logger.debug("Removed %s attachment(s) for policy: %s", removed_count, policy_name)
        return removed_count

    def remove_attachment_by_id(self, attachment_id: str) -> bool:
        """
        Remove an attachment by its ID (for DB-synced attachments).

        Args:
            attachment_id: The ID of the attachment to remove

        Returns:
            True if removed, False if not found
        """
        # Note: In-memory attachments don't have IDs, so this is primarily
        # for consistency after DB operations
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Database CRUD Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def add_attachment_to_db(
        self,
        attachment_request: PolicyAttachmentCreateRequest,
        prisma_client: "PrismaClient",
        created_by: str | None = None,
    ) -> PolicyAttachmentDBResponse:
        """
        Add a policy attachment to the database.

        Args:
            attachment_request: The attachment creation request
            prisma_client: The Prisma client instance
            created_by: User who created the attachment

        Returns:
            PolicyAttachmentDBResponse with the created attachment
        """
        try:
            created_attachment: Final[LiteLLM_PolicyAttachmentTable] = await PolicyAttachmentRepository(
                prisma_client
            ).table.create(
                data={
                    "policy_name": attachment_request.policy_name,
                    "scope": attachment_request.scope,
                    "teams": attachment_request.teams or [],
                    "keys": attachment_request.keys or [],
                    "models": attachment_request.models or [],
                    "tags": attachment_request.tags or [],
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "created_by": created_by,
                    "updated_by": created_by,
                }
            )

            # Also add to in-memory registry
            attachment: Final = PolicyAttachment(
                policy=attachment_request.policy_name,
                scope=attachment_request.scope,
                teams=attachment_request.teams,
                keys=attachment_request.keys,
                models=attachment_request.models,
                tags=attachment_request.tags,
            )
            self.add_attachment(attachment)

            return PolicyAttachmentDBResponse(
                attachment_id=created_attachment.attachment_id,
                policy_name=created_attachment.policy_name,
                scope=created_attachment.scope,
                teams=created_attachment.teams or [],
                keys=created_attachment.keys or [],
                models=created_attachment.models or [],
                tags=created_attachment.tags or [],
                created_at=created_attachment.created_at,
                updated_at=created_attachment.updated_at,
                created_by=created_attachment.created_by,
                updated_by=created_attachment.updated_by,
            )
        except Exception as e:
            verbose_proxy_logger.exception("Error adding attachment to DB: %s", e)
            raise Exception(f"Error adding attachment to DB: {e}")

    async def delete_attachment_from_db(
        self,
        attachment_id: str,
        prisma_client: "PrismaClient",
    ) -> dict[str, str]:
        """
        Delete a policy attachment from the database.

        Args:
            attachment_id: The ID of the attachment to delete
            prisma_client: The Prisma client instance

        Returns:
            Dict with success message
        """
        try:
            # Get attachment before deleting
            attachment: Final[LiteLLM_PolicyAttachmentTable | None] = await PolicyAttachmentRepository(
                prisma_client
            ).table.find_unique(where={"attachment_id": attachment_id})

            if attachment is None:
                raise Exception(f"Attachment with ID {attachment_id} not found")

            # Delete from DB
            await PolicyAttachmentRepository(prisma_client).table.delete(where={"attachment_id": attachment_id})

            # Note: In-memory attachments don't have IDs, so we need to sync from DB
            # to properly update in-memory state
            await self.sync_attachments_from_db(prisma_client)

            return {"message": f"Attachment {attachment_id} deleted successfully"}
        except Exception as e:
            verbose_proxy_logger.exception("Error deleting attachment from DB: %s", e)
            raise Exception(f"Error deleting attachment from DB: {e}")

    async def get_attachment_by_id_from_db(
        self,
        attachment_id: str,
        prisma_client: "PrismaClient",
    ) -> PolicyAttachmentDBResponse | None:
        """
        Get a policy attachment by ID from the database.

        Args:
            attachment_id: The ID of the attachment to retrieve
            prisma_client: The Prisma client instance

        Returns:
            PolicyAttachmentDBResponse if found, None otherwise
        """
        try:
            attachment: Final[LiteLLM_PolicyAttachmentTable | None] = await PolicyAttachmentRepository(
                prisma_client
            ).table.find_unique(where={"attachment_id": attachment_id})

            if attachment is None:
                return None

            return PolicyAttachmentDBResponse(
                attachment_id=attachment.attachment_id,
                policy_name=attachment.policy_name,
                scope=attachment.scope,
                teams=attachment.teams or [],
                keys=attachment.keys or [],
                models=attachment.models or [],
                tags=attachment.tags or [],
                created_at=attachment.created_at,
                updated_at=attachment.updated_at,
                created_by=attachment.created_by,
                updated_by=attachment.updated_by,
            )
        except Exception as e:
            verbose_proxy_logger.exception("Error getting attachment from DB: %s", e)
            raise Exception(f"Error getting attachment from DB: {e}")

    async def get_all_attachments_from_db(
        self,
        prisma_client: "PrismaClient",
    ) -> list[PolicyAttachmentDBResponse]:
        """
        Get all policy attachments from the database.

        Args:
            prisma_client: The Prisma client instance

        Returns:
            List of PolicyAttachmentDBResponse objects
        """
        try:
            attachments: Final[Sequence[LiteLLM_PolicyAttachmentTable]] = await PolicyAttachmentRepository(
                prisma_client
            ).table.find_many(
                order={"created_at": "desc"},
            )

            return [
                PolicyAttachmentDBResponse(
                    attachment_id=a.attachment_id,
                    policy_name=a.policy_name,
                    scope=a.scope,
                    teams=a.teams or [],
                    keys=a.keys or [],
                    models=a.models or [],
                    tags=a.tags or [],
                    created_at=a.created_at,
                    updated_at=a.updated_at,
                    created_by=a.created_by,
                    updated_by=a.updated_by,
                )
                for a in attachments
            ]
        except Exception as e:
            verbose_proxy_logger.exception("Error getting attachments from DB: %s", e)
            raise Exception(f"Error getting attachments from DB: {e}")

    async def sync_attachments_from_db(
        self,
        prisma_client: "PrismaClient",
    ) -> None:
        """
        Sync policy attachments from the database to in-memory registry.
        Config-loaded attachments are preserved.

        Args:
            prisma_client: The Prisma client instance
        """
        try:
            attachments: Final = await self.get_all_attachments_from_db(prisma_client)

            db_attachments: Final = [
                PolicyAttachment(
                    policy=attachment_response.policy_name,
                    scope=attachment_response.scope,
                    teams=(attachment_response.teams if attachment_response.teams else None),
                    keys=attachment_response.keys if attachment_response.keys else None,
                    models=(attachment_response.models if attachment_response.models else None),
                    tags=attachment_response.tags if attachment_response.tags else None,
                )
                for attachment_response in attachments
            ]
            self._attachments = [*self._config_attachments, *db_attachments]

            self._initialized = True
            verbose_proxy_logger.info(
                "Synced %s attachments from DB to in-memory registry (%s config-defined attachments preserved)",
                len(attachments),
                len(self._config_attachments),
            )
        except Exception as e:
            verbose_proxy_logger.exception("Error syncing attachments from DB: %s", e)
            raise Exception(f"Error syncing attachments from DB: {e}")


# Global singleton instance
_attachment_registry: AttachmentRegistry | None = None


def get_attachment_registry() -> AttachmentRegistry:
    """
    Get the global AttachmentRegistry singleton.

    Returns:
        The global AttachmentRegistry instance
    """
    global _attachment_registry
    if _attachment_registry is None:
        _attachment_registry = AttachmentRegistry()
    return _attachment_registry
