import asyncio
from typing import Any, Dict, List

import boto3
from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.managed_agents_endpoints import config_loader as _config_loader
from litellm.proxy.managed_agents_endpoints.config_loader import (
    get_dockerfile,
    list_dockerfiles,
)
from litellm.proxy.managed_agents_endpoints.fargate.build import provision_template
from litellm.proxy.managed_agents_endpoints.git_validation import (
    encrypt_and_store_git_token,
    validate_repo_branch,
)
from litellm.proxy.managed_agents_endpoints.lifecycle import stop_sessions_for_template
from litellm.proxy.managed_agents_endpoints.types import (
    AwsOverrides,
    DockerfileOut,
    TemplateCreate,
    TemplateOut,
)
from litellm.proxy.utils import jsonify_object

router = APIRouter(prefix="/v1/managed_agents", tags=["managed_agents"])


def _template_row_to_out(row) -> TemplateOut:
    return TemplateOut(
        id=row.template_id,
        name=row.template_name,
        dockerfile_id=row.dockerfile_id,
        container_port=row.container_port,
        repo_url=row.repo_url,
        default_branch=row.default_branch,
        visibility=row.visibility,
        image_uri=row.image_uri,
        task_def_arn=row.task_def_arn,
        build_status=row.build_status,
        build_error=row.build_error,
    )


def _resolve_region() -> str:
    cfg = _config_loader.MANAGED_AGENTS_CONFIG
    if cfg is not None and cfg.aws_region:
        return cfg.aws_region
    return "us-east-1"


def _resolve_aws_overrides() -> AwsOverrides:
    cfg = _config_loader.MANAGED_AGENTS_CONFIG
    if cfg is not None:
        return cfg.aws
    return AwsOverrides()


def _deregister_task_def_sync(region: str, task_def_arn: str) -> None:
    """Synchronous boto3 wrapper, intended to be called via asyncio.to_thread."""
    ecs = boto3.client("ecs", region_name=region)
    ecs.deregister_task_definition(taskDefinition=task_def_arn)


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="admin role required")


def _is_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN


def _assert_owner_or_admin(
    user_api_key_dict: UserAPIKeyAuth,
    created_by: object,
    resource_kind: str,
    resource_id: str,
) -> None:
    """Reject callers that neither own the resource nor are PROXY_ADMIN.

    A 404 is returned for non-owners (rather than 403) so that resource IDs
    cannot be enumerated by unauthorized callers.
    """
    if _is_admin(user_api_key_dict):
        return
    caller = user_api_key_dict.user_id
    if caller is not None and created_by == caller:
        return
    raise HTTPException(
        status_code=404, detail=f"{resource_kind} '{resource_id}' not found"
    )


@router.get("/dockerfiles", response_model=List[DockerfileOut])
async def list_dockerfiles_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[DockerfileOut]:
    return [
        DockerfileOut(id=entry.dockerfile_id, container_port=entry.container_port)
        for entry in list_dockerfiles()
    ]


@router.post("/sandbox-templates", response_model=TemplateOut)
async def create_sandbox_template(
    body: TemplateCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> TemplateOut:
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    try:
        dockerfile_entry = get_dockerfile(body.dockerfile_id)
    except KeyError:
        available = [entry.dockerfile_id for entry in list_dockerfiles()]
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"unknown dockerfile_id '{body.dockerfile_id}'",
                "available_dockerfile_ids": available,
            },
        )

    if body.visibility == "private" and not body.git_token:
        raise HTTPException(
            status_code=400,
            detail="visibility=private requires git_token",
        )
    if body.visibility == "public" and body.git_token:
        raise HTTPException(
            status_code=400,
            detail="visibility=public must not include git_token",
        )

    await asyncio.to_thread(
        validate_repo_branch, body.repo_url, body.default_branch, body.git_token
    )

    git_credential_id = None
    if body.git_token:
        git_credential_id = await encrypt_and_store_git_token(
            prisma_client,
            raw_token=body.git_token,
            created_by=user_api_key_dict.user_id or "",
        )

    create_data = jsonify_object(
        {
            "template_name": body.name,
            "dockerfile_id": body.dockerfile_id,
            "container_port": dockerfile_entry.container_port,
            "repo_url": body.repo_url,
            "default_branch": body.default_branch,
            "visibility": body.visibility,
            "git_credential_id": git_credential_id,
            "description": None,
            "build_status": "pending",
            "created_by": user_api_key_dict.user_id,
            "updated_by": user_api_key_dict.user_id,
        }
    )

    row = await prisma_client.db.litellm_managedagentsandboxtemplatetable.create(
        data=create_data
    )

    region = _resolve_region()
    aws_overrides = _resolve_aws_overrides()

    try:
        provisioned = await provision_template(
            dockerfile_id=dockerfile_entry.dockerfile_id,
            dockerfile_path=dockerfile_entry.path,
            context_dir=dockerfile_entry.context_dir,
            container_port=dockerfile_entry.container_port,
            region=region,
            aws_overrides=aws_overrides,
            build_platform=dockerfile_entry.build_platform,
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            "managed_agents: provision_template failed for template_id=%s: %s",
            row.template_id,
            e,
        )
        failed_row = (
            await prisma_client.db.litellm_managedagentsandboxtemplatetable.update(
                where={"template_id": row.template_id},
                data={
                    "build_status": "failed",
                    "build_error": str(e),
                    "updated_by": user_api_key_dict.user_id,
                },
            )
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "provision_template failed",
                "template": _template_row_to_out(failed_row).model_dump(),
            },
        )

    updated_row = (
        await prisma_client.db.litellm_managedagentsandboxtemplatetable.update(
            where={"template_id": row.template_id},
            data={
                "image_uri": provisioned.image_uri,
                "task_def_arn": provisioned.task_def_arn,
                "image_hash": provisioned.image_hash,
                "build_status": "ready",
                "updated_by": user_api_key_dict.user_id,
            },
        )
    )

    return _template_row_to_out(updated_row)


def _template_visible_to(row, user_api_key_dict: UserAPIKeyAuth) -> bool:
    """A private template is only visible to its creator and admins."""
    if row.visibility == "public":
        return True
    if _is_admin(user_api_key_dict):
        return True
    caller = user_api_key_dict.user_id
    return caller is not None and row.created_by == caller


@router.get("/sandbox-templates", response_model=List[TemplateOut])
async def list_sandbox_templates(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[TemplateOut]:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    if _is_admin(user_api_key_dict):
        where_clause = {}
    else:
        caller = user_api_key_dict.user_id
        visibility_filters: List[Dict[str, Any]] = [{"visibility": "public"}]
        if caller is not None:
            visibility_filters.append({"created_by": caller})
        where_clause = {"OR": visibility_filters}

    rows = await prisma_client.db.litellm_managedagentsandboxtemplatetable.find_many(
        where=where_clause
    )
    return [_template_row_to_out(row) for row in rows]


@router.get("/sandbox-templates/{template_id}", response_model=TemplateOut)
async def get_sandbox_template(
    template_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> TemplateOut:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    row = await prisma_client.db.litellm_managedagentsandboxtemplatetable.find_unique(
        where={"template_id": template_id}
    )
    if row is None or not _template_visible_to(row, user_api_key_dict):
        raise HTTPException(
            status_code=404, detail=f"template '{template_id}' not found"
        )
    return _template_row_to_out(row)


@router.delete("/sandbox-templates/{template_id}")
async def delete_sandbox_template(
    template_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    row = await prisma_client.db.litellm_managedagentsandboxtemplatetable.find_unique(
        where={"template_id": template_id}
    )
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"template '{template_id}' not found"
        )

    agent_count = await prisma_client.db.litellm_managedagenttable.count(
        where={"template_id": template_id}
    )
    if agent_count > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                f"cannot delete template '{template_id}': "
                f"{agent_count} agent(s) still reference it"
            ),
        )

    region = _resolve_region()
    aws_overrides = _resolve_aws_overrides()
    cluster = aws_overrides.cluster or "litellm-agents"

    try:
        await stop_sessions_for_template(
            prisma_client=prisma_client,
            region=region,
            cluster=cluster,
            template_id=template_id,
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            "managed_agents: stop_sessions_for_template failed for template_id=%s: %s",
            template_id,
            e,
        )

    if row.task_def_arn:
        try:
            await asyncio.to_thread(_deregister_task_def_sync, region, row.task_def_arn)
        except Exception as e:
            verbose_proxy_logger.warning(
                "managed_agents: deregister_task_definition failed for arn=%s: %s",
                row.task_def_arn,
                e,
            )

    await prisma_client.db.litellm_managedagentsandboxtemplatetable.delete(
        where={"template_id": template_id}
    )

    return {"id": template_id, "status": "deleted"}
