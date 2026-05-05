"""
Handler for LiteLLM database-backed skills operations.

This module contains the actual database operations for skills CRUD.
Used by the transformation layer and skills injection hook.
"""

import uuid
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import LiteLLM_SkillsTable, NewSkillRequest, UserAPIKeyAuth
from litellm.proxy.common_utils.resource_ownership import (
    get_primary_resource_owner_scope,
    get_resource_owner_scopes,
    is_proxy_admin,
    user_can_access_resource_owner,
)

# Skills are looked up on every chat completion that has skills enabled
# (`SkillsInjectionHook` calls ``fetch_skill_from_db``). 60s LRU/TTL cache
# absorbs the hot read before it reaches Prisma. ``_NEGATIVE_SKILL_SENTINEL``
# lets us cache a true "skill does not exist" so repeated misses also
# avoid the DB — ``InMemoryCache`` returns ``None`` indistinguishably for
# "miss" and "cached as None".
_NEGATIVE_SKILL_SENTINEL = "__litellm_skill_not_found__"
_SKILL_CACHE = InMemoryCache(max_size_in_memory=10000, default_ttl=60)


def _prisma_skill_to_litellm(prisma_skill) -> LiteLLM_SkillsTable:
    """Convert a Prisma skill record to LiteLLM_SkillsTable.

    Handles Base64 decoding of file_content field — model_dump() converts
    Base64 fields to base64-encoded strings.
    """
    import base64

    data = prisma_skill.model_dump()

    if data.get("file_content") is not None:
        if isinstance(data["file_content"], str):
            data["file_content"] = base64.b64decode(data["file_content"])

    return LiteLLM_SkillsTable(**data)


class LiteLLMSkillsHandler:
    """CRUD for skills stored in ``litellm_skillstable``."""

    @staticmethod
    async def _get_prisma_client():
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise ValueError(
                "Prisma client is not initialized. "
                "Database connection required for LiteLLM skills."
            )
        return prisma_client

    @staticmethod
    async def create_skill(
        data: NewSkillRequest,
        user_id: Optional[str] = None,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> LiteLLM_SkillsTable:
        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()

        skill_id = f"litellm_skill_{uuid.uuid4()}"
        owner = get_primary_resource_owner_scope(user_api_key_dict) or user_id
        if owner is None:
            # Identity-less callers (no user_id / team_id / org_id /
            # api_key / token) can't be uniquely stamped on the row.
            # Stamping a placeholder would let any two such callers see
            # each other's skills via the shared owner. ValueError keeps
            # this module FastAPI-free per the project layering rule.
            raise ValueError(
                "Unable to record skill ownership: caller has no identity scope."
            )

        skill_data: Dict[str, Any] = {
            "skill_id": skill_id,
            "display_title": data.display_title,
            "description": data.description,
            "instructions": data.instructions,
            "source": "custom",
            "created_by": owner,
            "updated_by": owner,
        }

        if data.metadata is not None:
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            skill_data["metadata"] = safe_dumps(data.metadata)

        if data.file_content is not None:
            from prisma.fields import Base64

            skill_data["file_content"] = Base64.encode(data.file_content)
        if data.file_name is not None:
            skill_data["file_name"] = data.file_name
        if data.file_type is not None:
            skill_data["file_type"] = data.file_type

        verbose_logger.debug(
            f"LiteLLMSkillsHandler: Creating skill {skill_id} with title={data.display_title}"
        )

        new_skill = await prisma_client.db.litellm_skillstable.create(data=skill_data)
        return _prisma_skill_to_litellm(new_skill)

    @staticmethod
    async def list_skills(
        limit: int = 20,
        offset: int = 0,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> List[LiteLLM_SkillsTable]:
        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()

        verbose_logger.debug(
            f"LiteLLMSkillsHandler: Listing skills with limit={limit}, offset={offset}"
        )

        find_many_kwargs: Dict[str, Any] = {
            "take": limit,
            "skip": offset,
            "order": {"created_at": "desc"},
        }
        if user_api_key_dict is not None and not is_proxy_admin(user_api_key_dict):
            owner_scopes = get_resource_owner_scopes(user_api_key_dict)
            if not owner_scopes:
                return []
            find_many_kwargs["where"] = {"created_by": {"in": owner_scopes}}

        skills = await prisma_client.db.litellm_skillstable.find_many(
            **find_many_kwargs
        )
        return [_prisma_skill_to_litellm(s) for s in skills]

    @staticmethod
    async def _load_skill(skill_id: str) -> Optional[Any]:
        """Cache-first read of the Prisma skill row. Owner-scope filtering
        happens on the cached row, so the cache is per-skill not per-caller.
        """
        cached = _SKILL_CACHE.get_cache(skill_id)
        if cached == _NEGATIVE_SKILL_SENTINEL:
            return None
        if cached is not None:
            return cached

        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()
        skill = await prisma_client.db.litellm_skillstable.find_unique(
            where={"skill_id": skill_id}
        )
        _SKILL_CACHE.set_cache(
            skill_id, skill if skill is not None else _NEGATIVE_SKILL_SENTINEL
        )
        return skill

    @staticmethod
    async def get_skill(
        skill_id: str,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> LiteLLM_SkillsTable:
        verbose_logger.debug(f"LiteLLMSkillsHandler: Getting skill {skill_id}")

        skill = await LiteLLMSkillsHandler._load_skill(skill_id)
        # Same "not found" message for both "missing" and "cross-tenant"
        # so callers can't enumerate skill IDs they don't own.
        if skill is None or not user_can_access_resource_owner(
            getattr(skill, "created_by", None), user_api_key_dict
        ):
            raise ValueError(f"Skill not found: {skill_id}")

        return _prisma_skill_to_litellm(skill)

    @staticmethod
    async def delete_skill(
        skill_id: str,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> Dict[str, str]:
        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()
        verbose_logger.debug(f"LiteLLMSkillsHandler: Deleting skill {skill_id}")

        skill = await LiteLLMSkillsHandler._load_skill(skill_id)
        if skill is None or not user_can_access_resource_owner(
            getattr(skill, "created_by", None), user_api_key_dict
        ):
            raise ValueError(f"Skill not found: {skill_id}")

        await prisma_client.db.litellm_skillstable.delete(where={"skill_id": skill_id})
        _SKILL_CACHE.set_cache(skill_id, _NEGATIVE_SKILL_SENTINEL)

        return {"id": skill_id, "type": "skill_deleted"}

    @staticmethod
    async def fetch_skill_from_db(
        skill_id: str,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> Optional[LiteLLM_SkillsTable]:
        """Skills-injection-hook helper: returns None instead of raising on
        not-found / not-authorized so the hook can silently skip."""
        try:
            return await LiteLLMSkillsHandler.get_skill(
                skill_id, user_api_key_dict=user_api_key_dict
            )
        except ValueError:
            return None
        except Exception as e:
            verbose_logger.warning(
                f"LiteLLMSkillsHandler: Error fetching skill {skill_id}: {e}"
            )
            return None
