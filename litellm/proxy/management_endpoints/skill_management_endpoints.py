"""
SKILL MANAGEMENT

Provider-agnostic SKILL.md storage and retrieval.
Skills are scoped to virtual keys via object_permission.skills on LiteLLM_ObjectPermissionTable.

/skill/new              - POST - Create a skill (admin)
/skill/update           - POST - Update a skill (admin)
/skill/delete           - POST - Delete a skill (admin)
/skill/list             - GET  - List skills visible to the calling key (frontmatter only)
/skill/{skill_name}     - GET  - Get full skill content (key-scoped)
"""

import json

import yaml
from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.skill_permission_handler import (
    SkillPermissionHandler,
)
from litellm.types.skill_management import (
    SkillConfig,
    SkillDeleteRequest,
    SkillFullConfig,
    SkillNewRequest,
    SkillUpdateRequest,
)

router = APIRouter()


def _check_skill_management_permission(user_api_key_dict: UserAPIKeyAuth) -> None:
    """
    Raises HTTP 403 if the caller does not have permission to create, update,
    or delete skills.  Only PROXY_ADMIN users are allowed to perform these
    write operations.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Only proxy admins can create, update, or delete skills. Your role={}".format(
                    user_api_key_dict.user_role
                )
            },
        )


def _parse_frontmatter(content: str) -> dict:
    """
    Parse YAML frontmatter from a SKILL.md string.

    Expects the document to start with '---', followed by YAML,
    followed by '---', followed by the markdown body.
    Returns the parsed YAML as a dict, or {} if no frontmatter found.
    """
    content = content.strip()
    if not content.startswith("---"):
        return {}
    # Find the closing '---' (skip the opening one)
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}
    frontmatter_str = content[3:end_idx].strip()
    try:
        return yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError:
        return {}


@router.post(
    "/skill/new",
    tags=["skill management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_skill(
    skill: SkillNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new skill from a SKILL.md document.

    Parameters:
    - name: str        - Unique skill identifier (used as DB primary key)
    - description: Optional[str] - Short description (overrides frontmatter if set)
    - content: str     - Full SKILL.md content (YAML frontmatter + markdown body)
    """
    from litellm.proxy._types import CommonProxyErrors
    from litellm.proxy.proxy_server import prisma_client

    _check_skill_management_permission(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    try:
        existing = await prisma_client.db.litellm_skilltable.find_unique(
            where={"skill_name": skill.name}
        )
        if existing is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Skill '{skill.name}' already exists",
            )

        parsed_meta = _parse_frontmatter(skill.content)
        description = skill.description or parsed_meta.get("description")

        record = await prisma_client.db.litellm_skilltable.create(
            data={
                "skill_name": skill.name,
                "description": description,
                "content": skill.content,
                "metadata": json.dumps(parsed_meta),
                "created_by": user_api_key_dict.user_id,
            }
        )

        return {
            "message": f"Skill '{skill.name}' created successfully",
            "skill": SkillConfig(
                name=record.skill_name,
                description=record.description,
                metadata=parsed_meta,
                created_at=record.created_at.isoformat(),
                updated_at=record.updated_at.isoformat(),
                created_by=record.created_by,
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating skill: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/skill/update",
    tags=["skill management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_skill(
    skill: SkillUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing skill.

    Parameters:
    - name: str - The skill to update (must already exist)
    - description: Optional[str] - Updated description
    - content: Optional[str] - Replacement SKILL.md content
    """
    from litellm.proxy.proxy_server import prisma_client

    _check_skill_management_permission(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        existing = await prisma_client.db.litellm_skilltable.find_unique(
            where={"skill_name": skill.name}
        )
        if existing is None:
            raise HTTPException(
                status_code=404, detail=f"Skill '{skill.name}' not found"
            )

        update_data = {}
        if skill.content is not None:
            update_data["content"] = skill.content
            parsed_meta = _parse_frontmatter(skill.content)
            update_data["metadata"] = json.dumps(parsed_meta)
            if skill.description is None:
                update_data["description"] = parsed_meta.get("description", "")

        if skill.description is not None:
            update_data["description"] = skill.description

        if not update_data:
            raise HTTPException(
                status_code=400,
                detail="No fields to update. Provide 'content' or 'description'.",
            )

        record = await prisma_client.db.litellm_skilltable.update(
            where={"skill_name": skill.name},
            data=update_data,
        )

        metadata = {}
        if record.metadata:
            metadata = (
                json.loads(record.metadata)
                if isinstance(record.metadata, str)
                else record.metadata
            )

        return {
            "message": f"Skill '{skill.name}' updated successfully",
            "skill": SkillConfig(
                name=record.skill_name,
                description=record.description,
                metadata=metadata,
                created_at=record.created_at.isoformat(),
                updated_at=record.updated_at.isoformat(),
                created_by=record.created_by,
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating skill: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/skill/delete",
    tags=["skill management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_skill(
    data: SkillDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a skill by name.

    Parameters:
    - name: str - The skill name to delete
    """
    from litellm.proxy.proxy_server import prisma_client

    _check_skill_management_permission(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        existing = await prisma_client.db.litellm_skilltable.find_unique(
            where={"skill_name": data.name}
        )
        if existing is None:
            raise HTTPException(
                status_code=404, detail=f"Skill '{data.name}' not found"
            )

        await prisma_client.db.litellm_skilltable.delete(
            where={"skill_name": data.name}
        )

        return {"message": f"Skill '{data.name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting skill: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/skill/list",
    tags=["skill management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_skills(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all skills the calling key is allowed to access.
    Returns frontmatter/metadata only (not the full SKILL.md body).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        allowed = await SkillPermissionHandler.get_allowed_skills(
            user_api_key_auth=user_api_key_dict,
        )

        # Query only allowed skills if scoped, otherwise get all
        # Semantics: [] = all allowed, non-empty = restricted
        if allowed:
            records = await prisma_client.db.litellm_skilltable.find_many(
                where={"skill_name": {"in": allowed}}
            )
        else:
            records = await prisma_client.db.litellm_skilltable.find_many()

        results = []
        for record in records:
            metadata = {}
            if record.metadata:
                metadata = (
                    json.loads(record.metadata)
                    if isinstance(record.metadata, str)
                    else record.metadata
                )

            results.append(
                SkillConfig(
                    name=record.skill_name,
                    description=record.description,
                    metadata=metadata,
                    created_at=record.created_at.isoformat(),
                    updated_at=record.updated_at.isoformat(),
                    created_by=record.created_by,
                ).model_dump()
            )

        return results
    except Exception as e:
        verbose_proxy_logger.exception(f"Error listing skills: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/skill/{skill_name}",
    tags=["skill management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_skill(
    skill_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get full skill content by name (key-scoped).

    Returns the complete SKILL.md content plus parsed metadata.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    # Enforce scoping via object_permission.skills
    if not await SkillPermissionHandler.is_skill_allowed(
        skill_name=skill_name,
        user_api_key_auth=user_api_key_dict,
    ):
        raise HTTPException(
            status_code=403,
            detail=f"Skill '{skill_name}' is not allowed for this API key.",
        )

    try:
        record = await prisma_client.db.litellm_skilltable.find_unique(
            where={"skill_name": skill_name}
        )
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Skill '{skill_name}' not found"
            )

        metadata = {}
        if record.metadata:
            metadata = (
                json.loads(record.metadata)
                if isinstance(record.metadata, str)
                else record.metadata
            )

        return SkillFullConfig(
            name=record.skill_name,
            description=record.description,
            content=record.content,
            metadata=metadata,
            created_at=record.created_at.isoformat(),
            updated_at=record.updated_at.isoformat(),
            created_by=record.created_by,
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting skill: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
