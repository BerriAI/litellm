"""
Handler for LiteLLM database-backed skills operations.

This module contains the actual database operations for skills CRUD.
Used by the transformation layer and skills injection hook.
"""

import uuid
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.proxy._types import LiteLLM_SkillsTable, NewSkillRequest


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

        skill_id = f"litellm_skill_{uuid.uuid4()}"

        skill_data: Dict[str, Any] = {
            "skill_id": skill_id,
            "display_title": data.display_title,
            "description": data.description,
            "instructions": data.instructions,
            "source": "custom",
            "created_by": user_id,
            "updated_by": user_id,
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

        new_skill = await prisma_client.db.litellm_skillstable.create(data=skill_data)

        return _prisma_skill_to_litellm(new_skill)

    @staticmethod
    async def list_skills(
        limit: int = 20,
        offset: int = 0,
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

        verbose_logger.debug(
            f"LiteLLMSkillsHandler: Listing skills with limit={limit}, offset={offset}"
        )

        skills = await prisma_client.db.litellm_skillstable.find_many(
            take=limit,
            skip=offset,
            order={"created_at": "desc"},
        )

        return [_prisma_skill_to_litellm(s) for s in skills]

    @staticmethod
    async def get_skill(skill_id: str) -> LiteLLM_SkillsTable:
        """
        Get a skill by ID from the LiteLLM database.

        Args:
            skill_id: The skill ID to retrieve

        Returns:
            LiteLLM_SkillsTable record

        Raises:
            ValueError: If skill not found
        """
        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()

        verbose_logger.debug(f"LiteLLMSkillsHandler: Getting skill {skill_id}")

        skill = await prisma_client.db.litellm_skillstable.find_unique(
            where={"skill_id": skill_id}
        )

        if skill is None:
            raise ValueError(f"Skill not found: {skill_id}")

        return _prisma_skill_to_litellm(skill)

    @staticmethod
    async def delete_skill(skill_id: str) -> Dict[str, str]:
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

        verbose_logger.debug(f"LiteLLMSkillsHandler: Deleting skill {skill_id}")

        # Check if skill exists
        skill = await prisma_client.db.litellm_skillstable.find_unique(
            where={"skill_id": skill_id}
        )

        if skill is None:
            raise ValueError(f"Skill not found: {skill_id}")

        # Delete the skill
        await prisma_client.db.litellm_skillstable.delete(where={"skill_id": skill_id})

        return {"id": skill_id, "type": "skill_deleted"}

    @staticmethod
    async def fetch_skill_from_db(skill_id: str) -> Optional[LiteLLM_SkillsTable]:
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
            return await LiteLLMSkillsHandler.get_skill(skill_id)
        except ValueError:
            return None
        except Exception as e:
            verbose_logger.warning(
                f"LiteLLMSkillsHandler: Error fetching skill {skill_id}: {e}"
            )
            return None
