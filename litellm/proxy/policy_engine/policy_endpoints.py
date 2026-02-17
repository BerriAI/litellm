"""
CRUD ENDPOINTS FOR POLICIES

Provides REST API endpoints for managing policies and policy attachments.
"""

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
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Policy CRUD Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/policies/list",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyListDBResponse,
)
async def list_policies():
    """
    List all policies from the database.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/policies/list" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "policies": [
            {
                "policy_id": "123e4567-e89b-12d3-a456-426614174000",
                "policy_name": "global-baseline",
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

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        policies = await get_policy_registry().get_all_policies_from_db(prisma_client)
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


@router.get(
    "/policies/{policy_id}",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyDBResponse,
)
async def get_policy(policy_id: str):
    """
    Get a policy by ID.

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/policies/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        result = await get_policy_registry().get_policy_by_id_from_db(
            policy_id=policy_id,
            prisma_client=prisma_client,
        )
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Policy with ID {policy_id} not found"
            )
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

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if policy exists
        existing = await get_policy_registry().get_policy_by_id_from_db(
            policy_id=policy_id,
            prisma_client=prisma_client,
        )
        if existing is None:
            raise HTTPException(
                status_code=404, detail=f"Policy with ID {policy_id} not found"
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

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if policy exists
        existing = await get_policy_registry().get_policy_by_id_from_db(
            policy_id=policy_id,
            prisma_client=prisma_client,
        )
        if existing is None:
            raise HTTPException(
                status_code=404, detail=f"Policy with ID {policy_id} not found"
            )

        result = await get_policy_registry().delete_policy_from_db(
            policy_id=policy_id,
            prisma_client=prisma_client,
        )
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
            raise HTTPException(
                status_code=404, detail=f"Policy with ID {policy_id} not found"
            )

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
    List all policy attachments from the database.

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

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        attachments = await get_attachment_registry().get_all_attachments_from_db(
            prisma_client
        )
        return PolicyAttachmentListResponse(
            attachments=attachments, total_count=len(attachments)
        )
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
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Verify the policy exists
        policy = await get_policy_registry().get_all_policies_from_db(prisma_client)
        policy_names = [p.policy_name for p in policy]
        if request.policy_name not in policy_names:
            raise HTTPException(
                status_code=404,
                detail=f"Policy '{request.policy_name}' not found. Create the policy first.",
            )

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

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

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
