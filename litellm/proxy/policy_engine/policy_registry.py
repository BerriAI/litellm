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
                version_number=getattr(created_policy, "version_number", 1),
                version_status=getattr(created_policy, "version_status", "production"),
                parent_version_id=getattr(created_policy, "parent_version_id", None),
                is_latest=getattr(created_policy, "is_latest", True),
                published_at=getattr(created_policy, "published_at", None),
                production_at=getattr(created_policy, "production_at", None),
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
                version_number=getattr(updated_policy, "version_number", 1),
                version_status=getattr(updated_policy, "version_status", "production"),
                parent_version_id=getattr(updated_policy, "parent_version_id", None),
                is_latest=getattr(updated_policy, "is_latest", True),
                published_at=getattr(updated_policy, "published_at", None),
                production_at=getattr(updated_policy, "production_at", None),
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
                version_number=getattr(policy, "version_number", 1),
                version_status=getattr(policy, "version_status", "production"),
                parent_version_id=getattr(policy, "parent_version_id", None),
                is_latest=getattr(policy, "is_latest", True),
                published_at=getattr(policy, "published_at", None),
                production_at=getattr(policy, "production_at", None),
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
                    version_number=getattr(p, "version_number", 1),
                    version_status=getattr(p, "version_status", "production"),
                    parent_version_id=getattr(p, "parent_version_id", None),
                    is_latest=getattr(p, "is_latest", True),
                    published_at=getattr(p, "published_at", None),
                    production_at=getattr(p, "production_at", None),
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

    # ─────────────────────────────────────────────────────────────────────────
    # Policy Versioning Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def create_policy_version(
        self,
        policy_id: str,
        prisma_client: "PrismaClient",
        created_by: Optional[str] = None,
    ) -> PolicyDBResponse:
        """
        Create a new version from an existing policy.

        The new version will:
        - Be created as a draft
        - Have version_number incremented from the latest version
        - Copy all configuration from the source policy
        - Set parent_version_id to the source policy_id

        Args:
            policy_id: ID of the policy to create a version from
            prisma_client: The Prisma client instance
            created_by: User who created the version

        Returns:
            PolicyDBResponse with the new version
        """
        try:
            # Get the source policy
            source_policy = await prisma_client.db.litellm_policytable.find_unique(
                where={"policy_id": policy_id}
            )

            if source_policy is None:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=404, detail=f"Policy with ID {policy_id} not found"
                )

            # Get the latest version number for this policy name
            existing_versions = await prisma_client.db.litellm_policytable.find_many(
                where={"policy_name": source_policy.policy_name},
                order_by={"version_number": "desc"},
            )

            new_version_number = (
                existing_versions[0].version_number + 1 if existing_versions else 1
            )

            # Mark previous versions as not latest
            for version in existing_versions:
                if version.is_latest:
                    await prisma_client.db.litellm_policytable.update(
                        where={"policy_id": version.policy_id},
                        data={"is_latest": False},
                    )

            # Create new version with copied configuration
            new_version_data = {
                "policy_name": source_policy.policy_name,
                "inherit": source_policy.inherit,
                "description": source_policy.description,
                "guardrails_add": source_policy.guardrails_add or [],
                "guardrails_remove": source_policy.guardrails_remove or [],
                "condition": source_policy.condition,
                "pipeline": source_policy.pipeline,
                "version_number": new_version_number,
                "version_status": "draft",
                "parent_version_id": policy_id,
                "is_latest": True,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            if created_by is not None:
                new_version_data["created_by"] = created_by
                new_version_data["updated_by"] = created_by

            created_version = await prisma_client.db.litellm_policytable.create(
                data=new_version_data
            )

            return PolicyDBResponse(
                policy_id=created_version.policy_id,
                policy_name=created_version.policy_name,
                inherit=created_version.inherit,
                description=created_version.description,
                guardrails_add=created_version.guardrails_add or [],
                guardrails_remove=created_version.guardrails_remove or [],
                condition=created_version.condition,
                pipeline=created_version.pipeline,
                version_number=created_version.version_number,
                version_status=created_version.version_status,
                parent_version_id=created_version.parent_version_id,
                is_latest=created_version.is_latest,
                published_at=created_version.published_at,
                production_at=created_version.production_at,
                created_at=created_version.created_at,
                updated_at=created_version.updated_at,
                created_by=created_version.created_by,
                updated_by=created_version.updated_by,
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error creating policy version: {e}")
            raise

    async def get_policy_versions(
        self,
        policy_name: str,
        prisma_client: "PrismaClient",
    ) -> List[PolicyDBResponse]:
        """
        Get all versions of a policy by policy name.

        Returns empty list when the policy has no versions (e.g. older policies
        created before versioning) or when the query fails (e.g. schema mismatch),
        so callers always get a valid response instead of 404/500.
        """
        try:
            versions = await prisma_client.db.litellm_policytable.find_many(
                where={"policy_name": policy_name},
                order_by={"version_number": "desc"},
            )

            return [
                PolicyDBResponse(
                    policy_id=v.policy_id,
                    policy_name=v.policy_name,
                    inherit=v.inherit,
                    description=v.description,
                    guardrails_add=v.guardrails_add or [],
                    guardrails_remove=v.guardrails_remove or [],
                    condition=v.condition,
                    pipeline=v.pipeline,
                    version_number=v.version_number,
                    version_status=v.version_status,
                    parent_version_id=v.parent_version_id,
                    is_latest=v.is_latest,
                    published_at=v.published_at,
                    production_at=v.production_at,
                    created_at=v.created_at,
                    updated_at=v.updated_at,
                    created_by=v.created_by,
                    updated_by=v.updated_by,
                )
                for v in versions
            ]
        except Exception as e:
            verbose_proxy_logger.warning(
                "Error getting policy versions (returning empty list): %s", e
            )
            return []

    async def update_policy_status(
        self,
        policy_id: str,
        new_status: str,
        prisma_client: "PrismaClient",
        updated_by: Optional[str] = None,
    ) -> PolicyDBResponse:
        """
        Update the status of a policy version.

        Valid transitions:
        - draft → published
        - published → production
        - production → published (demote)

        When promoting to production, previous production versions are demoted to published.

        Args:
            policy_id: ID of the policy to update
            new_status: New status (draft, published, or production)
            prisma_client: The Prisma client instance
            updated_by: User who updated the status

        Returns:
            PolicyDBResponse with the updated policy

        Raises:
            ValueError: If the status transition is invalid
        """
        try:
            # Get the policy
            policy = await prisma_client.db.litellm_policytable.find_unique(
                where={"policy_id": policy_id}
            )

            if policy is None:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=404, detail=f"Policy with ID {policy_id} not found"
                )

            current_status = policy.version_status

            # Validate status transition
            valid_transitions = {
                "draft": {"published"},
                "published": {"production", "draft"},
                "production": {"published"},
            }

            if (
                new_status != current_status
                and new_status not in valid_transitions.get(current_status, set())
            ):
                raise ValueError(
                    f"Invalid status transition from '{current_status}' to '{new_status}'. "
                    f"Valid transitions: {valid_transitions.get(current_status, set())}"
                )

            # Build update data
            update_data: Dict[str, Any] = {
                "version_status": new_status,
                "updated_at": datetime.now(timezone.utc),
            }

            if updated_by is not None:
                update_data["updated_by"] = updated_by

            # Set timestamp fields based on new status
            if new_status == "published" and current_status == "draft":
                update_data["published_at"] = datetime.now(timezone.utc)
            elif new_status == "production":
                update_data["production_at"] = datetime.now(timezone.utc)

                # Demote other production versions of this policy to published
                await prisma_client.db.litellm_policytable.update_many(
                    where={
                        "policy_name": policy.policy_name,
                        "version_status": "production",
                    },
                    data={"version_status": "published"},
                )

            # Update the policy
            updated_policy = await prisma_client.db.litellm_policytable.update(
                where={"policy_id": policy_id},
                data=update_data,
            )

            # If promoted to production, sync to in-memory registry
            if new_status == "production":
                policy_obj = self._parse_policy(
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
                self.add_policy(updated_policy.policy_name, policy_obj)

            return PolicyDBResponse(
                policy_id=updated_policy.policy_id,
                policy_name=updated_policy.policy_name,
                inherit=updated_policy.inherit,
                description=updated_policy.description,
                guardrails_add=updated_policy.guardrails_add or [],
                guardrails_remove=updated_policy.guardrails_remove or [],
                condition=updated_policy.condition,
                pipeline=updated_policy.pipeline,
                version_number=updated_policy.version_number,
                version_status=updated_policy.version_status,
                parent_version_id=updated_policy.parent_version_id,
                is_latest=updated_policy.is_latest,
                published_at=updated_policy.published_at,
                production_at=updated_policy.production_at,
                created_at=updated_policy.created_at,
                updated_at=updated_policy.updated_at,
                created_by=updated_policy.created_by,
                updated_by=updated_policy.updated_by,
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error updating policy status: {e}")
            raise

    async def compare_policy_versions(
        self,
        policy_id_1: str,
        policy_id_2: str,
        prisma_client: "PrismaClient",
    ) -> Dict[str, Any]:
        """
        Compare two policy versions and return their differences.

        Args:
            policy_id_1: ID of the first policy
            policy_id_2: ID of the second policy
            prisma_client: The Prisma client instance

        Returns:
            Dict with comparison results showing differences
        """
        try:
            # Get both policies
            policy1 = await self.get_policy_by_id_from_db(policy_id_1, prisma_client)
            policy2 = await self.get_policy_by_id_from_db(policy_id_2, prisma_client)

            if policy1 is None:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=404, detail=f"Policy with ID {policy_id_1} not found"
                )
            if policy2 is None:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=404, detail=f"Policy with ID {policy_id_2} not found"
                )

            # Helper to compare lists
            def compare_lists(list1: List[str], list2: List[str]) -> Dict[str, Any]:
                set1, set2 = set(list1), set(list2)
                return {
                    "added": sorted(set1 - set2),
                    "removed": sorted(set2 - set1),
                    "unchanged": sorted(set1 & set2),
                }

            # Build differences
            differences: Dict[str, Any] = {}

            # Compare guardrails_add
            if policy1.guardrails_add != policy2.guardrails_add:
                differences["guardrails_add"] = compare_lists(
                    policy1.guardrails_add, policy2.guardrails_add
                )

            # Compare guardrails_remove
            if policy1.guardrails_remove != policy2.guardrails_remove:
                differences["guardrails_remove"] = compare_lists(
                    policy1.guardrails_remove, policy2.guardrails_remove
                )

            # Compare description
            if policy1.description != policy2.description:
                differences["description"] = {
                    "changed": True,
                    "new": policy1.description,
                    "old": policy2.description,
                }

            # Compare inherit
            if policy1.inherit != policy2.inherit:
                differences["inherit"] = {
                    "changed": True,
                    "new": policy1.inherit,
                    "old": policy2.inherit,
                }

            # Compare condition
            if policy1.condition != policy2.condition:
                differences["condition"] = {
                    "changed": True,
                    "new": policy1.condition,
                    "old": policy2.condition,
                }

            # Compare pipeline
            if policy1.pipeline != policy2.pipeline:
                differences["pipeline"] = {
                    "changed": True,
                    "new": policy1.pipeline,
                    "old": policy2.pipeline,
                }

            return {
                "policy_1": {
                    "policy_id": policy1.policy_id,
                    "policy_name": policy1.policy_name,
                    "version_number": policy1.version_number,
                    "version_status": policy1.version_status,
                },
                "policy_2": {
                    "policy_id": policy2.policy_id,
                    "policy_name": policy2.policy_name,
                    "version_number": policy2.version_number,
                    "version_status": policy2.version_status,
                },
                "differences": differences,
            }
        except Exception as e:
            verbose_proxy_logger.exception(f"Error comparing policy versions: {e}")
            raise


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
