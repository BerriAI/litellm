"""
Attachment Registry - Manages policy attachments from YAML config.

Attachments define WHERE policies apply, separate from the policy definitions.
This allows the same policy to be attached to multiple scopes.
"""

from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.policy_engine import (
    PolicyAttachment,
    PolicyMatchContext,
    PolicyScope,
)


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

    def __init__(self):
        self._attachments: List[PolicyAttachment] = []
        self._initialized: bool = False

    def load_attachments(self, attachments_config: List[Dict[str, Any]]) -> None:
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
                verbose_proxy_logger.debug(
                    f"Loaded attachment for policy: {attachment.policy}"
                )
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error loading attachment: {str(e)}"
                )
                raise ValueError(f"Invalid attachment: {str(e)}") from e

        self._initialized = True
        verbose_proxy_logger.info(f"Loaded {len(self._attachments)} policy attachments")

    def _parse_attachment(self, attachment_data: Dict[str, Any]) -> PolicyAttachment:
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
        )

    def get_attached_policies(self, context: PolicyMatchContext) -> List[str]:
        """
        Get list of policy names attached to the given context.

        Args:
            context: The request context to match against

        Returns:
            List of policy names that are attached to matching scopes
        """
        from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher

        attached_policies: List[str] = []

        for attachment in self._attachments:
            scope = attachment.to_policy_scope()
            if PolicyMatcher.scope_matches(scope=scope, context=context):
                if attachment.policy not in attached_policies:
                    attached_policies.append(attachment.policy)
                    verbose_proxy_logger.debug(
                        f"Attachment matched: policy={attachment.policy}, "
                        f"context=(team={context.team_alias}, key={context.key_alias}, model={context.model})"
                    )

        return attached_policies

    def is_policy_attached(
        self, policy_name: str, context: PolicyMatchContext
    ) -> bool:
        """
        Check if a specific policy is attached to the given context.

        Args:
            policy_name: Name of the policy to check
            context: The request context to match against

        Returns:
            True if the policy is attached to a matching scope
        """
        attached = self.get_attached_policies(context)
        return policy_name in attached

    def get_all_attachments(self) -> List[PolicyAttachment]:
        """
        Get all loaded attachments.

        Returns:
            List of all PolicyAttachment objects
        """
        return self._attachments.copy()

    def get_attachments_for_policy(self, policy_name: str) -> List[PolicyAttachment]:
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
        self._initialized = False

    def add_attachment(self, attachment: PolicyAttachment) -> None:
        """
        Add a single attachment.

        Args:
            attachment: PolicyAttachment object to add
        """
        self._attachments.append(attachment)
        verbose_proxy_logger.debug(f"Added attachment for policy: {attachment.policy}")

    def remove_attachments_for_policy(self, policy_name: str) -> int:
        """
        Remove all attachments for a specific policy.

        Args:
            policy_name: Name of the policy

        Returns:
            Number of attachments removed
        """
        original_count = len(self._attachments)
        self._attachments = [a for a in self._attachments if a.policy != policy_name]
        removed_count = original_count - len(self._attachments)
        if removed_count > 0:
            verbose_proxy_logger.debug(
                f"Removed {removed_count} attachment(s) for policy: {policy_name}"
            )
        return removed_count


# Global singleton instance
_attachment_registry: Optional[AttachmentRegistry] = None


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
