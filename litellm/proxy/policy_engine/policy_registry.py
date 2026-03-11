"""
Policy Registry - In-memory storage for policies.

Handles storing, retrieving, and managing policies.

Policies define WHAT guardrails to apply. WHERE they apply is defined
by policy_attachments (see AttachmentRegistry).
"""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.policy_engine import (GuardrailPipeline, PipelineStep,
                                               Policy, PolicyCondition,
                                               PolicyCreateRequest,
                                               PolicyDBResponse,
                                               PolicyGuardrails,
                                               PolicyUpdateRequest,
                                               PolicyVersionCompareResponse,
                                               PolicyVersionListResponse)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

# Prefix for policy version IDs in request body. Use policy_<uuid> to execute a specific version.
POLICY_VERSION_ID_PREFIX = "policy_"


def _row_to_policy_db_response(row: Any) -> PolicyDBResponse:
    """Build PolicyDBResponse from a Prisma LiteLLM_PolicyTable row."""
    return PolicyDBResponse(
        policy_id=row.policy_id,
        policy_name=row.policy_name,
        version_number=getattr(row, "version_number", 1),
        version_status=getattr(row, "version_status", "production"),
        parent_version_id=getattr(row, "parent_version_id", None),
        is_latest=getattr(row, "is_latest", True),
        published_at=getattr(row, "published_at", None),
        production_at=getattr(row, "production_at", None),
        inherit=row.inherit,
        description=row.description,
        guardrails_add=row.guardrails_add or [],
        guardrails_remove=row.guardrails_remove or [],
        condition=row.condition,
        pipeline=row.pipeline,
        created_at=row.created_at,
        updated_at=row.updated_at,
        created_by=row.created_by,
        updated_by=row.updated_by,
    )


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
        self._policies_by_id: Dict[str, Tuple[str, Policy]] = {}
        self._initialized: bool = False

    def load_policies(self, policies_config: Dict[str, Any]) -> None:
        """
        Load policies from a configuration dictionary.

        Args:
            policies_config: Dictionary mapping policy names to policy definitions.
                            This is the raw config from the YAML file.
        """
        self._policies = {}
        self._policies_by_id = {}

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
            guardrails = PolicyGuardrails(
                add=guardrails_data if guardrails_data else None
            )

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
    def _parse_pipeline(
        pipeline_data: Optional[Dict[str, Any]],
    ) -> Optional[GuardrailPipeline]:
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
            now = datetime.now(timezone.utc)
            # Build data dict; new policy is v1 production
            data: Dict[str, Any] = {
                "policy_name": policy_request.policy_name,
                "version_number": 1,
                "version_status": "production",
                "is_latest": True,
                "production_at": now,
                "guardrails_add": policy_request.guardrails_add or [],
                "guardrails_remove": policy_request.guardrails_remove or [],
                "created_at": now,
                "updated_at": now,
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
                    "condition": (
                        policy_request.condition.model_dump()
                        if policy_request.condition
                        else None
                    ),
                    "pipeline": policy_request.pipeline,
                },
            )
            self.add_policy(policy_request.policy_name, policy)

            return _row_to_policy_db_response(created_policy)
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
        Update a policy in the database. Only draft versions can be updated.

        Args:
            policy_id: The ID of the policy to update
            policy_request: The policy update request
            prisma_client: The Prisma client instance
            updated_by: User who updated the policy

        Returns:
            PolicyDBResponse with the updated policy

        Raises:
            Exception: If policy is not in draft status (only drafts are editable).
        """
        try:
            existing = await prisma_client.db.litellm_policytable.find_unique(
                where={"policy_id": policy_id}
            )
            if existing is None:
                raise Exception(f"Policy with ID {policy_id} not found")
            version_status = getattr(existing, "version_status", "production")
            if version_status != "draft":
                raise Exception(
                    f"Only draft versions can be updated. This policy has status '{version_status}'."
                )

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
                update_data["condition"] = json.dumps(
                    policy_request.condition.model_dump()
                )
            if policy_request.pipeline is not None:
                validated_pipeline = GuardrailPipeline(**policy_request.pipeline)
                update_data["pipeline"] = json.dumps(validated_pipeline.model_dump())

            updated_policy = await prisma_client.db.litellm_policytable.update(
                where={"policy_id": policy_id},
                data=update_data,
            )

            # Do NOT update in-memory registry: drafts are not loaded into memory.

            return _row_to_policy_db_response(updated_policy)
        except Exception as e:
            verbose_proxy_logger.exception(f"Error updating policy in DB: {e}")
            raise Exception(f"Error updating policy in DB: {str(e)}")

    async def delete_policy_from_db(
        self,
        policy_id: str,
        prisma_client: "PrismaClient",
    ) -> Dict[str, Any]:
        """
        Delete a policy version from the database.

        If the deleted version was production, it is removed from the in-memory
        registry. No other version is auto-promoted; admin must explicitly promote.

        Args:
            policy_id: The ID of the policy version to delete
            prisma_client: The Prisma client instance

        Returns:
            Dict with "message" and optional "warning" if production was deleted.
        """
        try:
            policy = await prisma_client.db.litellm_policytable.find_unique(
                where={"policy_id": policy_id}
            )

            if policy is None:
                raise Exception(f"Policy with ID {policy_id} not found")

            version_status = getattr(policy, "version_status", "production")
            policy_name = policy.policy_name

            # Delete from DB
            await prisma_client.db.litellm_policytable.delete(
                where={"policy_id": policy_id}
            )

            result: Dict[str, Any] = {
                "message": f"Policy {policy_id} deleted successfully"
            }

            # Remove from in-memory registry only if this was the production version
            if version_status == "production":
                self.remove_policy(policy_name)
                result["warning"] = (
                    "Production version was deleted. No other version was promoted. "
                    "Promote another version to production if this policy should remain active."
                )

            return result
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

            return _row_to_policy_db_response(policy)
        except Exception as e:
            verbose_proxy_logger.exception(f"Error getting policy from DB: {e}")
            raise Exception(f"Error getting policy from DB: {str(e)}")

    def get_policy_by_id_for_request(self, policy_id: str) -> Optional[Tuple[str, Policy]]:
        """
        Return a policy version by ID from in-memory cache (no DB access).

        Used when the request body specifies policy_<uuid> to execute a specific version
        (e.g. published or draft). The cache is populated by sync_policies_from_db,
        which loads draft and published versions keyed by policy_id.

        Args:
            policy_id: The policy version ID (raw UUID, no prefix)

        Returns:
            (policy_name, Policy) if found, None otherwise
        """
        return self._policies_by_id.get(policy_id)

    async def get_all_policies_from_db(
        self,
        prisma_client: "PrismaClient",
        version_status: Optional[str] = None,
    ) -> List[PolicyDBResponse]:
        """
        Get all policies from the database, optionally filtered by version_status.

        Args:
            prisma_client: The Prisma client instance
            version_status: If set, only return policies with this status
                           ("draft", "published", "production").

        Returns:
            List of PolicyDBResponse objects
        """
        try:
            where: Dict[str, Any] = {}
            if version_status is not None:
                where["version_status"] = version_status

            policies = await prisma_client.db.litellm_policytable.find_many(
                where=where if where else None,
                order={"created_at": "desc"},
            )

            return [_row_to_policy_db_response(p) for p in policies]
        except Exception as e:
            verbose_proxy_logger.exception(f"Error getting policies from DB: {e}")
            raise Exception(f"Error getting policies from DB: {str(e)}")

    async def sync_policies_from_db(
        self,
        prisma_client: "PrismaClient",
    ) -> None:
        """
        Sync policies from the database to in-memory registry.
        - Production versions are loaded into _policies (by policy name) for resolution.
        - Draft and published versions are loaded into _policies_by_id so request-body
          policy_<uuid> overrides can be resolved without DB access in the hot path.
        """
        try:
            self._policies = {}
            production = await self.get_all_policies_from_db(
                prisma_client, version_status="production"
            )
            for policy_response in production:
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

            self._policies_by_id = {}
            non_production = await prisma_client.db.litellm_policytable.find_many(
                where={"version_status": {"in": ["draft", "published"]}},
                order={"created_at": "desc"},
            )
            for row in non_production:
                policy = self._parse_policy(
                    row.policy_name,
                    {
                        "inherit": row.inherit,
                        "description": row.description,
                        "guardrails": {
                            "add": row.guardrails_add or [],
                            "remove": row.guardrails_remove or [],
                        },
                        "condition": row.condition,
                        "pipeline": row.pipeline,
                    },
                )
                self._policies_by_id[row.policy_id] = (row.policy_name, policy)

            self._initialized = True
            verbose_proxy_logger.info(
                f"Synced {len(production)} production policies and {len(non_production)} "
                "draft/published (by ID) from DB to in-memory registry"
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
            # Load only production versions so inheritance resolves against production
            policies = await self.get_all_policies_from_db(
                prisma_client, version_status="production"
            )

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

    async def get_versions_by_policy_name(
        self,
        policy_name: str,
        prisma_client: "PrismaClient",
    ) -> PolicyVersionListResponse:
        """
        Get all versions of a policy by name, ordered by version_number descending.

        Args:
            policy_name: Name of the policy
            prisma_client: The Prisma client instance

        Returns:
            PolicyVersionListResponse with policy_name and list of versions
        """
        try:
            rows = await prisma_client.db.litellm_policytable.find_many(
                where={"policy_name": policy_name},
                order={"version_number": "desc"},
            )
            versions = [_row_to_policy_db_response(r) for r in rows]
            return PolicyVersionListResponse(
                policy_name=policy_name,
                versions=versions,
                total_count=len(versions),
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error getting versions: {e}")
            raise Exception(f"Error getting versions: {str(e)}")

    async def create_new_version(
        self,
        policy_name: str,
        prisma_client: "PrismaClient",
        source_policy_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> PolicyDBResponse:
        """
        Create a new draft version of a policy. Copies all fields from the source.
        Source is current production if source_policy_id is None.

        Args:
            policy_name: Name of the policy
            prisma_client: The Prisma client instance
            source_policy_id: Policy ID to clone from; if None, use current production
            created_by: User who created the version

        Returns:
            PolicyDBResponse for the new draft version
        """
        try:
            if source_policy_id is not None:
                source = await prisma_client.db.litellm_policytable.find_unique(
                    where={"policy_id": source_policy_id}
                )
                if source is None:
                    raise Exception(f"Source policy {source_policy_id} not found")
                if source.policy_name != policy_name:
                    raise Exception(
                        f"Source policy name '{source.policy_name}' does not match '{policy_name}'"
                    )
            else:
                # Find current production version for this policy_name
                prod = await prisma_client.db.litellm_policytable.find_first(
                    where={
                        "policy_name": policy_name,
                        "version_status": "production",
                    }
                )
                if prod is None:
                    raise Exception(
                        f"No production version found for policy '{policy_name}'"
                    )
                source = prod

            # Next version number
            latest = await prisma_client.db.litellm_policytable.find_first(
                where={"policy_name": policy_name},
                order={"version_number": "desc"},
            )
            next_num = (latest.version_number + 1) if latest else 1

            now = datetime.now(timezone.utc)
            # Set is_latest=False on all existing versions for this policy_name
            await prisma_client.db.litellm_policytable.update_many(
                where={"policy_name": policy_name},
                data={"is_latest": False},
            )

            data: Dict[str, Any] = {
                "policy_name": policy_name,
                "version_number": next_num,
                "version_status": "draft",
                "parent_version_id": source.policy_id,
                "is_latest": True,
                "published_at": None,
                "production_at": None,
                "inherit": source.inherit,
                "description": source.description,
                "guardrails_add": source.guardrails_add or [],
                "guardrails_remove": source.guardrails_remove or [],
                "created_at": now,
                "updated_at": now,
                "created_by": created_by,
                "updated_by": created_by,
            }
            # Prisma expects Json fields as JSON strings on create (same as add_policy_to_db)
            if source.condition is not None:
                data["condition"] = (
                    json.dumps(source.condition)
                    if isinstance(source.condition, dict)
                    else source.condition
                )
            if source.pipeline is not None:
                data["pipeline"] = (
                    json.dumps(source.pipeline)
                    if isinstance(source.pipeline, dict)
                    else source.pipeline
                )

            created = await prisma_client.db.litellm_policytable.create(data=data)
            return _row_to_policy_db_response(created)
        except Exception as e:
            verbose_proxy_logger.exception(f"Error creating new version: {e}")
            raise Exception(f"Error creating new version: {str(e)}")

    async def update_version_status(
        self,
        policy_id: str,
        new_status: str,
        prisma_client: "PrismaClient",
        updated_by: Optional[str] = None,
    ) -> PolicyDBResponse:
        """
        Update a policy version's status. Valid transitions:
        - draft -> published (sets published_at)
        - published -> production (sets production_at, demotes current production to published, updates in-memory)
        - production -> published (demotes, removes from in-memory)
        - draft -> production: NOT allowed (must publish first)
        - published -> draft: NOT allowed

        Args:
            policy_id: The policy version ID
            new_status: "published" or "production"
            prisma_client: The Prisma client instance
            updated_by: User who updated

        Returns:
            PolicyDBResponse for the updated version
        """
        try:
            if new_status not in ("published", "production"):
                raise Exception(
                    f"Invalid status '{new_status}'. Use 'published' or 'production'."
                )

            row = await prisma_client.db.litellm_policytable.find_unique(
                where={"policy_id": policy_id}
            )
            if row is None:
                raise Exception(f"Policy with ID {policy_id} not found")

            current = getattr(row, "version_status", "production")
            policy_name = row.policy_name
            now = datetime.now(timezone.utc)

            if new_status == "published":
                if current != "draft":
                    raise Exception(
                        f"Only draft versions can be published. Current status: '{current}'."
                    )
                updated = await prisma_client.db.litellm_policytable.update(
                    where={"policy_id": policy_id},
                    data={
                        "version_status": "published",
                        "published_at": now,
                        "updated_at": now,
                        "updated_by": updated_by,
                    },
                )
                return _row_to_policy_db_response(updated)

            # new_status == "production"
            if current not in ("draft", "published"):
                raise Exception(
                    f"Only draft or published versions can be promoted to production. Current: '{current}'."
                )
            # Plan: "draft -> production" NOT allowed
            if current == "draft":
                raise Exception(
                    "Cannot promote draft directly to production. Publish the version first."
                )

            # Demote current production to published
            await prisma_client.db.litellm_policytable.update_many(
                where={
                    "policy_name": policy_name,
                    "version_status": "production",
                },
                data={
                    "version_status": "published",
                    "updated_at": now,
                    "updated_by": updated_by,
                },
            )

            # Promote this version to production
            updated = await prisma_client.db.litellm_policytable.update(
                where={"policy_id": policy_id},
                data={
                    "version_status": "production",
                    "production_at": now,
                    "updated_at": now,
                    "updated_by": updated_by,
                },
            )

            # Update in-memory registry: remove old production (by name), add this one
            self.remove_policy(policy_name)
            policy = self._parse_policy(
                policy_name,
                {
                    "inherit": updated.inherit,
                    "description": updated.description,
                    "guardrails": {
                        "add": updated.guardrails_add or [],
                        "remove": updated.guardrails_remove or [],
                    },
                    "condition": updated.condition,
                    "pipeline": updated.pipeline,
                },
            )
            self.add_policy(policy_name, policy)

            return _row_to_policy_db_response(updated)
        except Exception as e:
            verbose_proxy_logger.exception(f"Error updating version status: {e}")
            raise Exception(f"Error updating version status: {str(e)}")

    async def compare_versions(
        self,
        policy_id_a: str,
        policy_id_b: str,
        prisma_client: "PrismaClient",
    ) -> PolicyVersionCompareResponse:
        """
        Compare two policy versions and return field-by-field diffs.

        Args:
            policy_id_a: First policy version ID
            policy_id_b: Second policy version ID
            prisma_client: The Prisma client instance

        Returns:
            PolicyVersionCompareResponse with both versions and field_diffs
        """
        try:
            a = await prisma_client.db.litellm_policytable.find_unique(
                where={"policy_id": policy_id_a}
            )
            b = await prisma_client.db.litellm_policytable.find_unique(
                where={"policy_id": policy_id_b}
            )
            if a is None:
                raise Exception(f"Policy {policy_id_a} not found")
            if b is None:
                raise Exception(f"Policy {policy_id_b} not found")

            resp_a = _row_to_policy_db_response(a)
            resp_b = _row_to_policy_db_response(b)

            # Compare fields that are part of policy content (not metadata)
            compare_fields = [
                "inherit",
                "description",
                "guardrails_add",
                "guardrails_remove",
                "condition",
                "pipeline",
            ]
            field_diffs: Dict[str, Dict[str, Any]] = {}
            for field in compare_fields:
                val_a = getattr(resp_a, field)
                val_b = getattr(resp_b, field)
                if val_a != val_b:
                    field_diffs[field] = {"version_a": val_a, "version_b": val_b}

            return PolicyVersionCompareResponse(
                version_a=resp_a,
                version_b=resp_b,
                field_diffs=field_diffs,
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error comparing versions: {e}")
            raise Exception(f"Error comparing versions: {str(e)}")

    async def delete_all_versions(
        self,
        policy_name: str,
        prisma_client: "PrismaClient",
    ) -> Dict[str, str]:
        """
        Delete all versions of a policy. Also removes from in-memory registry.

        Args:
            policy_name: Name of the policy
            prisma_client: The Prisma client instance

        Returns:
            Dict with success message
        """
        try:
            await prisma_client.db.litellm_policytable.delete_many(
                where={"policy_name": policy_name}
            )
            self.remove_policy(policy_name)
            return {
                "message": f"All versions of policy '{policy_name}' deleted successfully"
            }
        except Exception as e:
            verbose_proxy_logger.exception(f"Error deleting all versions: {e}")
            raise Exception(f"Error deleting all versions: {str(e)}")


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
