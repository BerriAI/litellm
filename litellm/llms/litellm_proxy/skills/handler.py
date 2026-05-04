"""
Handler for LiteLLM database-backed skills operations.

This module contains the actual database operations for skills CRUD.
Used by the transformation layer and skills injection hook.
"""

import os
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.llms.litellm_proxy.skills.store import LiteLLMSkillsStore
from litellm.proxy._types import LiteLLM_SkillsTable, NewSkillRequest, UserAPIKeyAuth
from litellm.proxy.common_utils.resource_ownership import (
    get_primary_resource_owner_scope,
    get_resource_owner_scopes,
    is_proxy_admin,
    user_can_access_resource_owner,
)

ALLOW_UNOWNED_SKILL_ACCESS_ENV = "LITELLM_ALLOW_UNOWNED_SKILL_ACCESS"

# Skills are looked up on every chat completion that has skills enabled
# (`LiteLLMSkillsHandler.fetch_skill_from_db` in the injection hook). Cache
# the Prisma skill row for a short window so the hot path doesn't issue a DB
# round-trip per request. Same shape as `_byok_cred_cache` and the container
# ownership cache: (value, monotonic_timestamp). `None` is cached as a true
# negative ("skill does not exist") so repeated misses also avoid the DB.
_SKILL_CACHE: "OrderedDict[str, Tuple[Optional[Any], float]]" = OrderedDict()
_SKILL_CACHE_TTL = 60  # seconds
_SKILL_CACHE_MAX_SIZE = 10000


def _read_skill_cache(skill_id: str) -> Tuple[bool, Optional[Any]]:
    """Return (hit, value). hit=False means caller must consult the DB."""
    entry = _SKILL_CACHE.get(skill_id)
    if entry is None:
        return False, None
    value, timestamp = entry
    if time.monotonic() - timestamp > _SKILL_CACHE_TTL:
        _SKILL_CACHE.pop(skill_id, None)
        return False, None
    _SKILL_CACHE.move_to_end(skill_id)
    return True, value


def _write_skill_cache(skill_id: str, skill: Optional[Any]) -> None:
    # LRU eviction (popitem(last=False)) instead of full ``clear()`` —
    # see container ownership cache for rationale.
    if skill_id in _SKILL_CACHE:
        _SKILL_CACHE.move_to_end(skill_id)
    _SKILL_CACHE[skill_id] = (skill, time.monotonic())
    while len(_SKILL_CACHE) > _SKILL_CACHE_MAX_SIZE:
        _SKILL_CACHE.popitem(last=False)


def _invalidate_skill_cache(skill_id: str) -> None:
    """Drop the cache entry after a write so the next read sees the new row."""
    _SKILL_CACHE.pop(skill_id, None)


def _allow_unowned_skill_access() -> bool:
    return os.getenv(ALLOW_UNOWNED_SKILL_ACCESS_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }


def _user_can_access_skill_owner(
    owner: Optional[str],
    user_api_key_dict: Optional[UserAPIKeyAuth],
) -> bool:
    if owner is None and user_api_key_dict is not None:
        if is_proxy_admin(user_api_key_dict):
            return True
        if _allow_unowned_skill_access():
            verbose_logger.warning(
                "Allowing unowned skill access because %s is enabled",
                ALLOW_UNOWNED_SKILL_ACCESS_ENV,
            )
            return True
    return user_can_access_resource_owner(owner, user_api_key_dict)


def _prisma_skill_to_litellm(prisma_skill) -> LiteLLM_SkillsTable:
    """
    Convert a Prisma skill record to LiteLLM_SkillsTable.

    Handles Base64 decoding of file_content field.
    """
    import base64

    data = prisma_skill.model_dump()

    # Decode Base64 file_content back to bytes
    # model_dump() converts Base64 field to base64-encoded string
    if data.get("file_content") is not None:
        if isinstance(data["file_content"], str):
            data["file_content"] = base64.b64decode(data["file_content"])
        elif isinstance(data["file_content"], bytes):
            # Already bytes, no conversion needed
            pass

    return LiteLLM_SkillsTable(**data)


class LiteLLMSkillsHandler:
    """
    Handler for LiteLLM database-backed skills operations.

    This class provides static methods for CRUD operations on skills
    stored in the LiteLLM proxy database (LiteLLM_SkillsTable).
    """

    @staticmethod
    async def _get_prisma_client():
        """Get the prisma client from proxy server."""
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
        """
        Create a new skill in the LiteLLM database.

        Args:
            data: NewSkillRequest with skill details
            user_id: Optional user ID for tracking

        Returns:
            LiteLLM_SkillsTable record
        """
        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()
        store = LiteLLMSkillsStore(prisma_client)

        skill_id = f"litellm_skill_{uuid.uuid4()}"
        owner = get_primary_resource_owner_scope(user_api_key_dict) or user_id

        skill_data: Dict[str, Any] = {
            "skill_id": skill_id,
            "display_title": data.display_title,
            "description": data.description,
            "instructions": data.instructions,
            "source": "custom",
            "created_by": owner,
            "updated_by": owner,
        }

        # Handle metadata
        if data.metadata is not None:
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            skill_data["metadata"] = safe_dumps(data.metadata)

        # Handle file content - wrap bytes in Base64 for Prisma
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

        new_skill = await store.create_skill(skill_data)

        return _prisma_skill_to_litellm(new_skill)

    @staticmethod
    async def list_skills(
        limit: int = 20,
        offset: int = 0,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> List[LiteLLM_SkillsTable]:
        """
        List skills from the LiteLLM database.

        Args:
            limit: Maximum number of skills to return
            offset: Number of skills to skip

        Returns:
            List of LiteLLM_SkillsTable records
        """
        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()
        store = LiteLLMSkillsStore(prisma_client)

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
            if _allow_unowned_skill_access():
                find_many_kwargs["where"] = {
                    "OR": [
                        {"created_by": {"in": owner_scopes}},
                        {"created_by": None},
                    ]
                }
            else:
                find_many_kwargs["where"] = {"created_by": {"in": owner_scopes}}

        skills = await store.list_skills(find_many_kwargs)

        return [_prisma_skill_to_litellm(s) for s in skills]

    @staticmethod
    async def _load_skill(skill_id: str) -> Optional[Any]:
        """Cache-first read of the Prisma skill row.

        Caching here keeps `fetch_skill_from_db` (called per chat completion in
        the skills injection hook) off the DB. Owner-scope filtering happens
        on the cached row, so the cache is per-skill and not per-caller.
        """
        cached_hit, cached_skill = _read_skill_cache(skill_id)
        if cached_hit:
            return cached_skill

        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()
        store = LiteLLMSkillsStore(prisma_client)
        skill = await store.find_skill(skill_id)
        _write_skill_cache(skill_id, skill)
        return skill

    @staticmethod
    async def get_skill(
        skill_id: str,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> LiteLLM_SkillsTable:
        """
        Get a skill by ID from the LiteLLM database.

        Args:
            skill_id: The skill ID to retrieve

        Returns:
            LiteLLM_SkillsTable record

        Raises:
            ValueError: If skill not found
        """
        verbose_logger.debug(f"LiteLLMSkillsHandler: Getting skill {skill_id}")

        skill = await LiteLLMSkillsHandler._load_skill(skill_id)

        if skill is None:
            raise ValueError(f"Skill not found: {skill_id}")

        if not _user_can_access_skill_owner(
            getattr(skill, "created_by", None), user_api_key_dict
        ):
            raise ValueError(f"Skill not found: {skill_id}")

        return _prisma_skill_to_litellm(skill)

    @staticmethod
    async def delete_skill(
        skill_id: str,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> Dict[str, str]:
        """
        Delete a skill by ID from the LiteLLM database.

        Args:
            skill_id: The skill ID to delete

        Returns:
            Dict with id and type of deleted skill

        Raises:
            ValueError: If skill not found
        """
        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()
        store = LiteLLMSkillsStore(prisma_client)

        verbose_logger.debug(f"LiteLLMSkillsHandler: Deleting skill {skill_id}")

        # Check if skill exists
        skill = await LiteLLMSkillsHandler._load_skill(skill_id)

        if skill is None:
            raise ValueError(f"Skill not found: {skill_id}")

        if not _user_can_access_skill_owner(
            getattr(skill, "created_by", None), user_api_key_dict
        ):
            raise ValueError(f"Skill not found: {skill_id}")

        # Delete the skill
        await store.delete_skill(skill_id)
        _invalidate_skill_cache(skill_id)

        return {"id": skill_id, "type": "skill_deleted"}

    @staticmethod
    async def fetch_skill_from_db(
        skill_id: str,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> Optional[LiteLLM_SkillsTable]:
        """
        Fetch a skill from the database (used by skills injection hook).

        This is a convenience method that returns None instead of raising
        an exception if the skill is not found.

        Args:
            skill_id: The skill ID to fetch

        Returns:
            LiteLLM_SkillsTable or None if not found
        """
        try:
            return await LiteLLMSkillsHandler.get_skill(
                skill_id,
                user_api_key_dict=user_api_key_dict,
            )
        except ValueError:
            return None
        except Exception as e:
            verbose_logger.warning(
                f"LiteLLMSkillsHandler: Error fetching skill {skill_id}: {e}"
            )
            return None
