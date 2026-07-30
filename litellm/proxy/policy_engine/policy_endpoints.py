"""
CRUD ENDPOINTS FOR POLICIES

Provides REST API endpoints for managing policies and policy attachments.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
from litellm.proxy.policy_engine.pipeline_executor import PipelineExecutor
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.types.proxy.policy_engine import (
    GuardrailPipeline,
    PipelineTestRequest,
    PolicyAttachmentCreateRequest,
    PolicyAttachmentDBResponse,
    PolicyAttachmentListResponse,
    PolicyCreateRequest,
    PolicyDBResponse,
    PolicyListDBResponse,
    PolicyUpdateRequest,
    PolicyVersionCompareResponse,
    PolicyVersionCreateRequest,
    PolicyVersionListResponse,
    PolicyVersionStatusUpdateRequest,
)

router = APIRouter()

CONFIG_ATTACHMENT_ID_PREFIX = "config-attachment-"


def _config_policies_as_responses() -> list[PolicyDBResponse]:
    """
    Render config.yaml policies in the same shape as DB-backed ones.

    Config policies are not versioned, so they are reported as the production version
    and keyed by their name; they have no DB row to take a UUID from.
    """
    return [
        PolicyDBResponse(
            policy_id=policy_name,
            policy_name=policy_name,
            inherit=policy.inherit,
            description=policy.description,
            guardrails_add=policy.guardrails.get_add(),
            guardrails_remove=policy.guardrails.get_remove(),
            condition=policy.condition.model_dump() if policy.condition else None,
            pipeline=policy.pipeline.model_dump() if policy.pipeline else None,
            policy_definition_location="config",
        )
        for policy_name, policy in get_policy_registry().get_config_policies().items()
    ]


def _config_policy_response(policy_id: str) -> PolicyDBResponse | None:
    """
    Look up a config.yaml policy by the ID the list endpoint reports for it.
    """
    return next(
        (policy for policy in _config_policies_as_responses() if policy.policy_id == policy_id),
        None,
    )


def _reject_if_config_policy(policy_id: str) -> None:
    """
    Config-defined policies have no DB row to mutate; say so instead of 404ing.
    """
    if _config_policy_response(policy_id) is None:
        return
    raise HTTPException(
        status_code=400,
        detail=f"Policy '{policy_id}' is defined in config.yaml and can only be changed there",
    )


def _config_attachments_as_responses() -> list[PolicyAttachmentDBResponse]:
    """
    Render config.yaml policy attachments in the same shape as DB-backed ones.
    """
    return [
        PolicyAttachmentDBResponse(
            attachment_id=f"{CONFIG_ATTACHMENT_ID_PREFIX}{index}",
            policy_name=attachment.policy,
            scope=attachment.scope,
            teams=attachment.teams or [],
            keys=attachment.keys or [],
            models=attachment.models or [],
            tags=attachment.tags or [],
            policy_definition_location="config",
        )
        for index, attachment in enumerate(get_attachment_registry().get_config_attachments())
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Policy CRUD Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/policies/list",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyListDBResponse,
)
async def list_policies(version_status: Optional[str] = None):
    """
    List all policies from the database and from config.yaml. Optionally filter by version_status.

    Config-defined policies are reported with policy_definition_location="config"; they have no
    versions, so they are only included when version_status is unset or "production".

    Query params:
    - version_status: Optional. One of "draft", "published", "production".
      If omitted, all versions are returned.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/policies/list" \\
        -H "Authorization: Bearer <your_api_key>"
    curl -X GET "http://localhost:4000/policies/list?version_status=production" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "policies": [
            {
                "policy_id": "123e4567-e89b-12d3-a456-426614174000",
                "policy_name": "global-baseline",
                "version_number": 1,
                "version_status": "production",
                "inherit": null,
                "description": "Base guardrails for all requests",
                "guardrails_add": ["pii_masking"],
                "guardrails_remove": [],
                "condition": null,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        ],
        "total_count": 1
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    config_policies = _config_policies_as_responses() if version_status in (None, "production") else []

    if prisma_client is None:
        return PolicyListDBResponse(policies=config_policies, total_count=len(config_policies))

    try:
        db_policies = await get_policy_registry().get_all_policies_from_db(prisma_client, version_status=version_status)
        policies = db_policies + config_policies
        return PolicyListDBResponse(policies=policies, total_count=len(policies))
    except Exception as e:
        verbose_proxy_logger.exception(f"Error listing policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/policies",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyDBResponse,
)
async def create_policy(
    request: PolicyCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new policy.

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/policies" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "policy_name": "global-baseline",
            "description": "Base guardrails for all requests",
            "guardrails_add": ["pii_masking", "prompt_injection"],
            "guardrails_remove": []
        }'
    ```

    Example Response:
    ```json
    {
        "policy_id": "123e4567-e89b-12d3-a456-426614174000",
        "policy_name": "global-baseline",
        "inherit": null,
        "description": "Base guardrails for all requests",
        "guardrails_add": ["pii_masking", "prompt_injection"],
        "guardrails_remove": [],
        "condition": null,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        created_by = user_api_key_dict.user_id
        result = await get_policy_registry().add_policy_to_db(
            policy_request=request,
            prisma_client=prisma_client,
            created_by=created_by,
        )
        return result
    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating policy: {e}")
        if "unique constraint" in str(e).lower():
            raise HTTPException(
                status_code=400,
                detail=f"Policy with name '{request.policy_name}' already exists",
            )
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Policy Versioning Endpoints (must be before /policies/{policy_id} to avoid path conflicts)
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/policies/name/{policy_name}/versions",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyVersionListResponse,
)
async def list_policy_versions(policy_name: str):
    """
    List all versions of a policy by name, ordered by version_number descending.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        return await get_policy_registry().get_versions_by_policy_name(
            policy_name=policy_name,
            prisma_client=prisma_client,
        )
    except Exception as e:
        verbose_proxy_logger.exception(f"Error listing policy versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/policies/name/{policy_name}/versions",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyDBResponse,
)
async def create_policy_version(
    policy_name: str,
    request: PolicyVersionCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new draft version of a policy. Copies all fields from the source.
    Source is current production if source_policy_id is not provided.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        created_by = user_api_key_dict.user_id
        return await get_policy_registry().create_new_version(
            policy_name=policy_name,
            prisma_client=prisma_client,
            source_policy_id=request.source_policy_id,
            created_by=created_by,
        )
    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating policy version: {e}")
        if "not found" in str(e).lower() or "no production" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/policies/{policy_id}/status",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyDBResponse,
)
async def update_policy_version_status(
    policy_id: str,
    request: PolicyVersionStatusUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update a policy version's status. Valid transitions:
    - draft -> published
    - published -> production (demotes current production to published)
    - production -> published (demotes, policy becomes inactive)
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        updated_by = user_api_key_dict.user_id
        return await get_policy_registry().update_version_status(
            policy_id=policy_id,
            new_status=request.version_status,
            prisma_client=prisma_client,
            updated_by=updated_by,
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating version status: {e}")
        if "invalid status" in str(e).lower() or "only draft" in str(e).lower() or "cannot promote" in str(e).lower():
            raise HTTPException(status_code=400, detail=str(e))
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/policies/compare",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyVersionCompareResponse,
)
async def compare_policy_versions(
    version_a: str,
    version_b: str,
):
    """
    Compare two policy versions. Query params: version_a, version_b (policy version IDs).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        return await get_policy_registry().compare_versions(
            policy_id_a=version_a,
            policy_id_b=version_b,
            prisma_client=prisma_client,
        )
    except Exception as e:
        verbose_proxy_logger.exception(f"Error comparing versions: {e}")
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/policies/name/{policy_name}/all-versions",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_all_policy_versions(policy_name: str):
    """
    Delete all versions of a policy. Also removes from in-memory registry.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        return await get_policy_registry().delete_all_versions(
            policy_name=policy_name,
            prisma_client=prisma_client,
        )
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting all versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Policy CRUD by ID
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/policies/{policy_id}",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyDBResponse,
)
async def get_policy(policy_id: str):
    """
    Get a policy by ID. Config-defined policies are looked up by their name.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/policies/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    config_policy = _config_policy_response(policy_id)

    if prisma_client is None:
        if config_policy is None:
            raise HTTPException(status_code=404, detail=f"Policy with ID {policy_id} not found")
        return config_policy

    try:
        result = await get_policy_registry().get_policy_by_id_from_db(
            policy_id=policy_id,
            prisma_client=prisma_client,
        )
        if result is None:
            result = config_policy
        if result is None:
            raise HTTPException(status_code=404, detail=f"Policy with ID {policy_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/policies/{policy_id}",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyDBResponse,
)
async def update_policy(
    policy_id: str,
    request: PolicyUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing policy.

    Example Request:
    ```bash
    curl -X PUT "http://localhost:4000/policies/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "description": "Updated description",
            "guardrails_add": ["pii_masking", "toxicity_filter"]
        }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    _reject_if_config_policy(policy_id)

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if policy exists and is draft (only drafts can be updated)
        existing = await get_policy_registry().get_policy_by_id_from_db(
            policy_id=policy_id,
            prisma_client=prisma_client,
        )
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Policy with ID {policy_id} not found")
        if getattr(existing, "version_status", "production") != "draft":
            raise HTTPException(
                status_code=400,
                detail="Only draft versions can be updated. Publish or create a new version to change published/production.",
            )

        updated_by = user_api_key_dict.user_id
        result = await get_policy_registry().update_policy_in_db(
            policy_id=policy_id,
            policy_request=request,
            prisma_client=prisma_client,
            updated_by=updated_by,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/policies/{policy_id}",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_policy(policy_id: str):
    """
    Delete a policy.

    Example Request:
    ```bash
    curl -X DELETE "http://localhost:4000/policies/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "message": "Policy 123e4567-e89b-12d3-a456-426614174000 deleted successfully"
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    _reject_if_config_policy(policy_id)

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if policy exists
        existing = await get_policy_registry().get_policy_by_id_from_db(
            policy_id=policy_id,
            prisma_client=prisma_client,
        )
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Policy with ID {policy_id} not found")

        result = await get_policy_registry().delete_policy_from_db(
            policy_id=policy_id,
            prisma_client=prisma_client,
        )
        # Result may include "warning" if production was deleted
        return result
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/policies/{policy_id}/resolved-guardrails",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_resolved_guardrails(policy_id: str):
    """
    Get the resolved guardrails for a policy (including inherited guardrails).

    This endpoint resolves the full inheritance chain and returns the final
    set of guardrails that would be applied for this policy.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/policies/123e4567-e89b-12d3-a456-426614174000/resolved-guardrails" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "policy_id": "123e4567-e89b-12d3-a456-426614174000",
        "policy_name": "healthcare-compliance",
        "resolved_guardrails": ["pii_masking", "prompt_injection", "toxicity_filter"]
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Get the policy
        policy = await get_policy_registry().get_policy_by_id_from_db(
            policy_id=policy_id,
            prisma_client=prisma_client,
        )
        if policy is None:
            raise HTTPException(status_code=404, detail=f"Policy with ID {policy_id} not found")

        # Resolve guardrails
        resolved = await get_policy_registry().resolve_guardrails_from_db(
            policy_name=policy.policy_name,
            prisma_client=prisma_client,
        )

        return {
            "policy_id": policy.policy_id,
            "policy_name": policy.policy_name,
            "resolved_guardrails": resolved,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        verbose_proxy_logger.exception(f"Error resolving guardrails: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Test Endpoint
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/policies/test-pipeline",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
)
async def test_pipeline(
    request: PipelineTestRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Test a guardrail pipeline with sample messages.

    Executes the pipeline steps against the provided test messages and returns
    step-by-step results showing which guardrails passed/failed, actions taken,
    and timing information.

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/policies/test-pipeline" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "pipeline": {
                "mode": "pre_call",
                "steps": [
                    {"guardrail": "pii-guard", "on_pass": "next", "on_fail": "block"}
                ]
            },
            "test_messages": [{"role": "user", "content": "My SSN is 123-45-6789"}]
        }'
    ```
    """
    try:
        validated_pipeline = GuardrailPipeline(**request.pipeline)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline: {e}")

    data = {
        "messages": request.test_messages,
        "model": "test",
        "metadata": {},
    }

    try:
        result = await PipelineExecutor.execute_steps(
            steps=validated_pipeline.steps,
            mode=validated_pipeline.mode,
            data=data,
            user_api_key_dict=user_api_key_dict,
            call_type="completion",
            policy_name="test-pipeline",
        )
        return result.model_dump()
    except Exception as e:
        verbose_proxy_logger.exception(f"Error testing pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Policy Attachment CRUD Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/policies/attachments/list",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyAttachmentListResponse,
)
async def list_policy_attachments():
    """
    List all policy attachments from the database and from config.yaml.

    Config-defined attachments are reported with policy_definition_location="config".

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/policies/attachments/list" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "attachments": [
            {
                "attachment_id": "123e4567-e89b-12d3-a456-426614174000",
                "policy_name": "global-baseline",
                "scope": "*",
                "teams": [],
                "keys": [],
                "models": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        ],
        "total_count": 1
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    config_attachments = _config_attachments_as_responses()

    if prisma_client is None:
        return PolicyAttachmentListResponse(attachments=config_attachments, total_count=len(config_attachments))

    try:
        db_attachments = await get_attachment_registry().get_all_attachments_from_db(prisma_client)
        attachments = db_attachments + config_attachments
        return PolicyAttachmentListResponse(attachments=attachments, total_count=len(attachments))
    except Exception as e:
        verbose_proxy_logger.exception(f"Error listing policy attachments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/policies/attachments",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyAttachmentDBResponse,
)
async def create_policy_attachment(
    request: PolicyAttachmentCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new policy attachment.

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/policies/attachments" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "policy_name": "global-baseline",
            "scope": "*"
        }'
    ```

    Example with team-specific attachment:
    ```bash
    curl -X POST "http://localhost:4000/policies/attachments" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "policy_name": "healthcare-compliance",
            "teams": ["healthcare-team", "medical-research"]
        }'
    ```

    Example Response:
    ```json
    {
        "attachment_id": "123e4567-e89b-12d3-a456-426614174000",
        "policy_name": "global-baseline",
        "scope": "*",
        "teams": [],
        "keys": [],
        "models": [],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
    }
    ```
    """
    from litellm.proxy.policy_engine.policy_validator import PolicyValidator
    from litellm.proxy.proxy_server import llm_router, prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Verify the policy has a production version (attachments resolve against production)
        policies = await get_policy_registry().get_all_policies_from_db(prisma_client, version_status="production")
        policy_names = {p.policy_name for p in policies}
        if request.policy_name not in policy_names:
            raise HTTPException(
                status_code=404,
                detail=f"Policy '{request.policy_name}' not found. Create the policy first.",
            )

        # Reject concrete team/key/model scope entries that don't resolve to a real
        # entity. Wildcard patterns are allowed through (they may match zero today).
        scope_errors = await PolicyValidator(
            prisma_client=prisma_client, llm_router=llm_router
        ).find_invalid_scope_entries(
            policy_name=request.policy_name,
            teams=request.teams,
            keys=request.keys,
            models=request.models,
        )
        if scope_errors:
            raise HTTPException(status_code=400, detail=" | ".join(e.message for e in scope_errors))

        created_by = user_api_key_dict.user_id
        result = await get_attachment_registry().add_attachment_to_db(
            attachment_request=request,
            prisma_client=prisma_client,
            created_by=created_by,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating policy attachment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/policies/attachments/{attachment_id}",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyAttachmentDBResponse,
)
async def get_policy_attachment(attachment_id: str):
    """
    Get a policy attachment by ID.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/policies/attachments/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    config_attachment = next(
        (a for a in _config_attachments_as_responses() if a.attachment_id == attachment_id),
        None,
    )

    if prisma_client is None or config_attachment is not None:
        if config_attachment is None:
            raise HTTPException(
                status_code=404,
                detail=f"Attachment with ID {attachment_id} not found",
            )
        return config_attachment

    try:
        result = await get_attachment_registry().get_attachment_by_id_from_db(
            attachment_id=attachment_id,
            prisma_client=prisma_client,
        )
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Attachment with ID {attachment_id} not found",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting policy attachment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/policies/attachments/{attachment_id}",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_policy_attachment(attachment_id: str):
    """
    Delete a policy attachment.

    Example Request:
    ```bash
    curl -X DELETE "http://localhost:4000/policies/attachments/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "message": "Attachment 123e4567-e89b-12d3-a456-426614174000 deleted successfully"
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if attachment_id.startswith(CONFIG_ATTACHMENT_ID_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=f"Attachment '{attachment_id}' is defined in config.yaml and can only be removed there",
        )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if attachment exists
        existing = await get_attachment_registry().get_attachment_by_id_from_db(
            attachment_id=attachment_id,
            prisma_client=prisma_client,
        )
        if existing is None:
            raise HTTPException(
                status_code=404,
                detail=f"Attachment with ID {attachment_id} not found",
            )

        result = await get_attachment_registry().delete_attachment_from_db(
            attachment_id=attachment_id,
            prisma_client=prisma_client,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting policy attachment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
