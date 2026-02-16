"""
Policy Registry - In-memory storage for policies.

Handles storing, retrieving, and managing policies.

Policies define WHAT guardrails to apply. WHERE they apply is defined
by policy_attachments (see AttachmentRegistry).
"""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.policy_engine import (
    GuardrailPipeline,
    PipelineStep,
    Policy,
    PolicyCondition,
    PolicyCreateRequest,
    PolicyDBResponse,
    PolicyGuardrails,
    PolicyUpdateRequest,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class PolicyRegistry:
    """
    In-memory registry for storing and managing policies.

    This is a singleton that holds all loaded policies and provides
    methods to access them.

    Policies define WHAT guardrails to apply:
    - Base guardrails via guardrails.add/remove
    - Inheritance via inherit field
    - Conditional guardrails via condition.model
    """

    def __init__(self):
        self._policies: Dict[str, Policy] = {}
        self._initialized: bool = False

    def load_policies(self, policies_config: Dict[str, Any]) -> None:
        """
        Load policies from a configuration dictionary.

        Args:
            policies_config: Dictionary mapping policy names to policy definitions.
                            This is the raw config from the YAML file.
        """
        self._policies = {}

        for policy_name, policy_data in policies_config.items():
            try:
                policy = self._parse_policy(policy_name, policy_data)
                self._policies[policy_name] = policy
                verbose_proxy_logger.debug(f"Loaded policy: {policy_name}")
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error loading policy '{policy_name}': {str(e)}"
                )
                raise ValueError(f"Invalid policy '{policy_name}': {str(e)}") from e

        self._initialized = True
        verbose_proxy_logger.info(f"Loaded {len(self._policies)} policies")

    def _parse_policy(self, policy_name: str, policy_data: Dict[str, Any]) -> Policy:
        """
        Parse a policy from raw configuration data.

        Args:
            policy_name: Name of the policy
            policy_data: Raw policy configuration

        Returns:
            Parsed Policy object
        """
        # Parse guardrails
        guardrails_data = policy_data.get("guardrails", {})
        if isinstance(guardrails_data, dict):
            guardrails = PolicyGuardrails(
                add=guardrails_data.get("add"),
                remove=guardrails_data.get("remove"),
            )
        else:
            # Handle legacy format where guardrails might be a list
            guardrails = PolicyGuardrails(add=guardrails_data if guardrails_data else None)

        # Parse condition (simple model-based condition)
        condition = None
        condition_data = policy_data.get("condition")
        if condition_data:
            condition = PolicyCondition(model=condition_data.get("model"))

        # Parse pipeline (optional ordered guardrail execution)
        pipeline = PolicyRegistry._parse_pipeline(policy_data.get("pipeline"))

        return Policy(
            inherit=policy_data.get("inherit"),
            description=policy_data.get("description"),
            guardrails=guardrails,
            condition=condition,
            pipeline=pipeline,
        )

    @staticmethod
    def _parse_pipeline(pipeline_data: Optional[Dict[str, Any]]) -> Optional[GuardrailPipeline]:
        """Parse a pipeline configuration from raw data."""
        if pipeline_data is None:
            return None

        steps_data = pipeline_data.get("steps", [])
        steps = [
            PipelineStep(**step_data) if isinstance(step_data, dict) else step_data
            for step_data in steps_data
        ]

        return GuardrailPipeline(
            mode=pipeline_data.get("mode", "pre_call"),
            steps=steps,
        )

    def get_policy(self, policy_name: str) -> Optional[Policy]:
        """
        Get a policy by name.

        Args:
            policy_name: Name of the policy to retrieve

        Returns:
            Policy object if found, None otherwise
        """
        return self._policies.get(policy_name)

    def get_all_policies(self) -> Dict[str, Policy]:
        """
        Get all loaded policies.

        Returns:
            Dictionary mapping policy names to Policy objects
        """
        return self._policies.copy()

    def get_policy_names(self) -> List[str]:
        """
        Get list of all policy names.

        Returns:
            List of policy names
        """
        return list(self._policies.keys())

    def has_policy(self, policy_name: str) -> bool:
        """
        Check if a policy exists.

        Args:
            policy_name: Name of the policy to check

        Returns:
            True if policy exists, False otherwise
        """
        return policy_name in self._policies

    def is_initialized(self) -> bool:
        """
        Check if the registry has been initialized with policies.

        Returns:
            True if policies have been loaded, False otherwise
        """
        return self._initialized

    def clear(self) -> None:
        """
        Clear all policies from the registry.
        """
        self._policies = {}
        self._initialized = False

    def add_policy(self, policy_name: str, policy: Policy) -> None:
        """
        Add or update a single policy.

        Args:
            policy_name: Name of the policy
            policy: Policy object to add
        """
        self._policies[policy_name] = policy
        verbose_proxy_logger.debug(f"Added/updated policy: {policy_name}")

    def remove_policy(self, policy_name: str) -> bool:
        """
        Remove a policy by name.

        Args:
            policy_name: Name of the policy to remove

        Returns:
            True if policy was removed, False if it didn't exist
        """
        if policy_name in self._policies:
            del self._policies[policy_name]
            verbose_proxy_logger.debug(f"Removed policy: {policy_name}")
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Database CRUD Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def add_policy_to_db(
        self,
        policy_request: PolicyCreateRequest,
        prisma_client: "PrismaClient",
        created_by: Optional[str] = None,
    ) -> PolicyDBResponse:
        """
        Add a policy to the database.

        Args:
            policy_request: The policy creation request
            prisma_client: The Prisma client instance
            created_by: User who created the policy

        Returns:
            PolicyDBResponse with the created policy
        """
        try:
            # Build data dict, only include condition if it's set
            data: Dict[str, Any] = {
                "policy_name": policy_request.policy_name,
                "guardrails_add": policy_request.guardrails_add or [],
                "guardrails_remove": policy_request.guardrails_remove or [],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            # Only add optional fields if they have values
            if policy_request.inherit is not None:
                data["inherit"] = policy_request.inherit
            if policy_request.description is not None:
                data["description"] = policy_request.description
            if created_by is not None:
                data["created_by"] = created_by
                data["updated_by"] = created_by
            if policy_request.condition is not None:
                data["condition"] = json.dumps(policy_request.condition.model_dump())
            if policy_request.pipeline is not None:
                validated_pipeline = GuardrailPipeline(**policy_request.pipeline)
                data["pipeline"] = json.dumps(validated_pipeline.model_dump())

            created_policy = await prisma_client.db.litellm_policytable.create(
                data=data
            )

            # Also add to in-memory registry
            policy = self._parse_policy(
                policy_request.policy_name,
                {
                    "inherit": policy_request.inherit,
                    "description": policy_request.description,
                    "guardrails": {
                        "add": policy_request.guardrails_add,
                        "remove": policy_request.guardrails_remove,
                    },
                    "condition": policy_request.condition.model_dump()
                    if policy_request.condition
                    else None,
                    "pipeline": policy_request.pipeline,
                },
            )
            self.add_policy(policy_request.policy_name, policy)

            return PolicyDBResponse(
                policy_id=created_policy.policy_id,
                policy_name=created_policy.policy_name,
                inherit=created_policy.inherit,
                description=created_policy.description,
                guardrails_add=created_policy.guardrails_add or [],
                guardrails_remove=created_policy.guardrails_remove or [],
                condition=created_policy.condition,
                pipeline=created_policy.pipeline,
                created_at=created_policy.created_at,
                updated_at=created_policy.updated_at,
                created_by=created_policy.created_by,
                updated_by=created_policy.updated_by,
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error adding policy to DB: {e}")
            raise Exception(f"Error adding policy to DB: {str(e)}")

    async def update_policy_in_db(
        self,
        policy_id: str,
        policy_request: PolicyUpdateRequest,
        prisma_client: "PrismaClient",
        updated_by: Optional[str] = None,
    ) -> PolicyDBResponse:
        """
        Update a policy in the database.

        Args:
            policy_id: The ID of the policy to update
            policy_request: The policy update request
            prisma_client: The Prisma client instance
            updated_by: User who updated the policy

        Returns:
            PolicyDBResponse with the updated policy
        """
        try:
            # Build update data - only include fields that are set
            update_data: Dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc),
                "updated_by": updated_by,
            }

            if policy_request.policy_name is not None:
                update_data["policy_name"] = policy_request.policy_name
            if policy_request.inherit is not None:
                update_data["inherit"] = policy_request.inherit
            if policy_request.description is not None:
                update_data["description"] = policy_request.description
            if policy_request.guardrails_add is not None:
                update_data["guardrails_add"] = policy_request.guardrails_add
            if policy_request.guardrails_remove is not None:
                update_data["guardrails_remove"] = policy_request.guardrails_remove
            if policy_request.condition is not None:
                update_data["condition"] = json.dumps(policy_request.condition.model_dump())
            if policy_request.pipeline is not None:
                validated_pipeline = GuardrailPipeline(**policy_request.pipeline)
                update_data["pipeline"] = json.dumps(validated_pipeline.model_dump())

            updated_policy = await prisma_client.db.litellm_policytable.update(
                where={"policy_id": policy_id},
                data=update_data,
            )

            # Update in-memory registry
            policy = self._parse_policy(
                updated_policy.policy_name,
                {
                    "inherit": updated_policy.inherit,
                    "description": updated_policy.description,
                    "guardrails": {
                        "add": updated_policy.guardrails_add,
                        "remove": updated_policy.guardrails_remove,
                    },
                    "condition": updated_policy.condition,
                    "pipeline": updated_policy.pipeline,
                },
            )
            self.add_policy(updated_policy.policy_name, policy)

            return PolicyDBResponse(
                policy_id=updated_policy.policy_id,
                policy_name=updated_policy.policy_name,
                inherit=updated_policy.inherit,
                description=updated_policy.description,
                guardrails_add=updated_policy.guardrails_add or [],
                guardrails_remove=updated_policy.guardrails_remove or [],
                condition=updated_policy.condition,
                pipeline=updated_policy.pipeline,
                created_at=updated_policy.created_at,
                updated_at=updated_policy.updated_at,
                created_by=updated_policy.created_by,
                updated_by=updated_policy.updated_by,
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error updating policy in DB: {e}")
            raise Exception(f"Error updating policy in DB: {str(e)}")

    async def delete_policy_from_db(
        self,
        policy_id: str,
        prisma_client: "PrismaClient",
    ) -> Dict[str, str]:
        """
        Delete a policy from the database.

        Args:
            policy_id: The ID of the policy to delete
            prisma_client: The Prisma client instance

        Returns:
            Dict with success message
        """
        try:
            # Get policy name before deleting
            policy = await prisma_client.db.litellm_policytable.find_unique(
                where={"policy_id": policy_id}
            )

            if policy is None:
                raise Exception(f"Policy with ID {policy_id} not found")

            # Delete from DB
            await prisma_client.db.litellm_policytable.delete(
                where={"policy_id": policy_id}
            )

            # Remove from in-memory registry
            self.remove_policy(policy.policy_name)

            return {"message": f"Policy {policy_id} deleted successfully"}
        except Exception as e:
            verbose_proxy_logger.exception(f"Error deleting policy from DB: {e}")
            raise Exception(f"Error deleting policy from DB: {str(e)}")

    async def get_policy_by_id_from_db(
        self,
        policy_id: str,
        prisma_client: "PrismaClient",
    ) -> Optional[PolicyDBResponse]:
        """
        Get a policy by ID from the database.

        Args:
            policy_id: The ID of the policy to retrieve
            prisma_client: The Prisma client instance

        Returns:
            PolicyDBResponse if found, None otherwise
        """
        try:
            policy = await prisma_client.db.litellm_policytable.find_unique(
                where={"policy_id": policy_id}
            )

            if policy is None:
                return None

            return PolicyDBResponse(
                policy_id=policy.policy_id,
                policy_name=policy.policy_name,
                inherit=policy.inherit,
                description=policy.description,
                guardrails_add=policy.guardrails_add or [],
                guardrails_remove=policy.guardrails_remove or [],
                condition=policy.condition,
                pipeline=policy.pipeline,
                created_at=policy.created_at,
                updated_at=policy.updated_at,
                created_by=policy.created_by,
                updated_by=policy.updated_by,
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error getting policy from DB: {e}")
            raise Exception(f"Error getting policy from DB: {str(e)}")

    async def get_all_policies_from_db(
        self,
        prisma_client: "PrismaClient",
    ) -> List[PolicyDBResponse]:
        """
        Get all policies from the database.

        Args:
            prisma_client: The Prisma client instance

        Returns:
            List of PolicyDBResponse objects
        """
        try:
            policies = await prisma_client.db.litellm_policytable.find_many(
                order={"created_at": "desc"},
            )

            return [
                PolicyDBResponse(
                    policy_id=p.policy_id,
                    policy_name=p.policy_name,
                    inherit=p.inherit,
                    description=p.description,
                    guardrails_add=p.guardrails_add or [],
                    guardrails_remove=p.guardrails_remove or [],
                    condition=p.condition,
                    pipeline=p.pipeline,
                    created_at=p.created_at,
                    updated_at=p.updated_at,
                    created_by=p.created_by,
                    updated_by=p.updated_by,
                )
                for p in policies
            ]
        except Exception as e:
            verbose_proxy_logger.exception(f"Error getting policies from DB: {e}")
            raise Exception(f"Error getting policies from DB: {str(e)}")

    async def sync_policies_from_db(
        self,
        prisma_client: "PrismaClient",
    ) -> None:
        """
        Sync policies from the database to in-memory registry.

        Args:
            prisma_client: The Prisma client instance
        """
        try:
            policies = await self.get_all_policies_from_db(prisma_client)

            for policy_response in policies:
                policy = self._parse_policy(
                    policy_response.policy_name,
                    {
                        "inherit": policy_response.inherit,
                        "description": policy_response.description,
                        "guardrails": {
                            "add": policy_response.guardrails_add,
                            "remove": policy_response.guardrails_remove,
                        },
                        "condition": policy_response.condition,
                        "pipeline": policy_response.pipeline,
                    },
                )
                self.add_policy(policy_response.policy_name, policy)

            self._initialized = True
            verbose_proxy_logger.info(
                f"Synced {len(policies)} policies from DB to in-memory registry"
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error syncing policies from DB: {e}")
            raise Exception(f"Error syncing policies from DB: {str(e)}")

    async def resolve_guardrails_from_db(
        self,
        policy_name: str,
        prisma_client: "PrismaClient",
    ) -> List[str]:
        """
        Resolve all guardrails for a policy from the database.
        
        Uses the existing PolicyResolver to handle inheritance chain resolution.
        
        Args:
            policy_name: Name of the policy to resolve
            prisma_client: The Prisma client instance
            
        Returns:
            List of resolved guardrail names
        """
        from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
        
        try:
            # Load all policies from DB to ensure we have the full inheritance chain
            policies = await self.get_all_policies_from_db(prisma_client)
            
            # Build a temporary in-memory map for resolution
            temp_policies = {}
            for policy_response in policies:
                policy = self._parse_policy(
                    policy_response.policy_name,
                    {
                        "inherit": policy_response.inherit,
                        "description": policy_response.description,
                        "guardrails": {
                            "add": policy_response.guardrails_add,
                            "remove": policy_response.guardrails_remove,
                        },
                        "condition": policy_response.condition,
                        "pipeline": policy_response.pipeline,
                    },
                )
                temp_policies[policy_response.policy_name] = policy
            
            # Use the existing PolicyResolver to resolve guardrails
            resolved_policy = PolicyResolver.resolve_policy_guardrails(
                policy_name=policy_name,
                policies=temp_policies,
                context=None,  # No context needed for simple resolution
            )
            
            return sorted(resolved_policy.guardrails)
        except Exception as e:
            verbose_proxy_logger.exception(f"Error resolving guardrails from DB: {e}")
            raise Exception(f"Error resolving guardrails from DB: {str(e)}")


# Global singleton instance
_policy_registry: Optional[PolicyRegistry] = None


def get_policy_registry() -> PolicyRegistry:
    """
    Get the global PolicyRegistry singleton.

    Returns:
        The global PolicyRegistry instance
    """
    global _policy_registry
    if _policy_registry is None:
        _policy_registry = PolicyRegistry()
    return _policy_registry
