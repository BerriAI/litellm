"""
Handler for LiteLLM database-backed skills operations.

This module contains the actual database operations for skills CRUD.
Used by the transformation layer and skills injection hook.
"""

import re
import uuid
from typing import Any, Dict, List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.proxy._types import LiteLLM_SkillsTable, NewSkillRequest


def _redact_sensitive_metadata(
    metadata: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Strip secrets that must never leave the server from a skill metadata dict."""
    if not metadata:
        return metadata
    return {k: v for k, v in metadata.items() if k != "github_pat"}


def _prisma_skill_to_litellm(prisma_skill) -> LiteLLM_SkillsTable:
    """
    Convert a Prisma skill record to LiteLLM_SkillsTable.

    Handles Base64 decoding of file_content and redacts the encrypted PAT.
    """
    import base64

    data = prisma_skill.model_dump()

    # Decode Base64 file_content back to bytes
    # model_dump() converts Base64 field to base64-encoded string
    if data.get("file_content") is not None:
        if isinstance(data["file_content"], str):
            data["file_content"] = base64.b64decode(data["file_content"])
        elif isinstance(data["file_content"], bytes):
            pass

    # Never expose the encrypted PAT ciphertext in API responses.
    if data.get("metadata"):
        data["metadata"] = _redact_sensitive_metadata(data["metadata"])

    return LiteLLM_SkillsTable(**data)


def _parse_github_owner_repo(repo_url: str):
    """Extract (owner, repo) from a GitHub URL."""
    match = re.search(
        r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", repo_url.rstrip("/")
    )
    if not match:
        raise ValueError(
            f"Cannot parse GitHub owner/repo from URL: {repo_url}. "
            "Expected format: https://github.com/owner/repo"
        )
    return match.group(1), match.group(2)


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
    async def _fetch_github_zip(repo_url: str, pat: str) -> tuple:
        """
        Fetch ZIP archive of a GitHub repo using a PAT.

        Returns (zip_bytes, file_name, file_type).
        Raises ValueError on auth failure or missing repo.
        """
        owner, repo = _parse_github_owner_repo(repo_url)
        zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
        headers = {
            "Authorization": f"token {pat}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await client.get(zip_url, headers=headers)

        if response.status_code == 401:
            raise ValueError("GitHub PAT authentication failed. Check the token.")
        if response.status_code == 404:
            raise ValueError(f"GitHub repo not found: {owner}/{repo}")
        if response.status_code != 200:
            raise ValueError(
                f"GitHub API error {response.status_code}: {response.text[:200]}"
            )

        file_name = f"{owner}-{repo}.zip"
        return response.content, file_name, "application/zip"

    @staticmethod
    async def test_github_connection(repo_url: str, pat: str) -> Dict[str, str]:
        """
        Verify a GitHub PAT can access the given repo without storing anything.

        Returns {"status": "ok"} or {"status": "error", "message": "..."}.
        """
        try:
            owner, repo = _parse_github_owner_repo(repo_url)
            headers = {
                "Authorization": f"token {pat}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}",
                    headers=headers,
                )
            if response.status_code == 200:
                return {"status": "ok"}
            if response.status_code == 401:
                return {
                    "status": "error",
                    "message": "PAT authentication failed. Check the token.",
                }
            if response.status_code == 404:
                return {"status": "error", "message": f"Repo not found: {owner}/{repo}"}
            return {
                "status": "error",
                "message": f"GitHub returned {response.status_code}",
            }
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": f"Connection failed: {e}"}

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

        # GitHub import: fetch ZIP and encrypt PAT before building skill_data
        file_content = data.file_content
        file_name = data.file_name
        file_type = data.file_type
        metadata = dict(data.metadata) if data.metadata else {}

        if data.github_repo_url:
            if not data.github_pat:
                raise ValueError(
                    "github_pat is required when github_repo_url is provided"
                )
            file_content, file_name, file_type = (
                await LiteLLMSkillsHandler._fetch_github_zip(
                    data.github_repo_url, data.github_pat
                )
            )
            from litellm.proxy.common_utils.encrypt_decrypt_utils import (
                encrypt_value_helper,
            )

            metadata["github_repo_url"] = data.github_repo_url
            metadata["github_pat"] = encrypt_value_helper(data.github_pat)

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
        if metadata:
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            skill_data["metadata"] = safe_dumps(metadata)
        elif data.metadata is not None:
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            skill_data["metadata"] = safe_dumps(data.metadata)

        # Handle file content - wrap bytes in Base64 for Prisma
        if file_content is not None:
            from prisma.fields import Base64

            skill_data["file_content"] = Base64.encode(file_content)
        if file_name is not None:
            skill_data["file_name"] = file_name
        if file_type is not None:
            skill_data["file_type"] = file_type

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
    async def list_skills_for_registry(
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Return lightweight registry entries for agent discovery.

        Each entry has: skill_id (litellm: prefixed), display_title,
        description, examples, tags.
        """
        prisma_client = await LiteLLMSkillsHandler._get_prisma_client()

        skills = await prisma_client.db.litellm_skillstable.find_many(
            take=limit,
            skip=offset,
            order={"created_at": "desc"},
            include={},
        )

        result = []
        for s in skills:
            meta: Dict[str, Any] = {}
            if s.metadata:
                meta = s.metadata if isinstance(s.metadata, dict) else {}

            result.append(
                {
                    "skill_id": f"litellm:{s.skill_id}",
                    "display_title": s.display_title,
                    "description": s.description or meta.get("description"),
                    "examples": meta.get("examples", []),
                    "tags": meta.get("tags", []),
                }
            )
        return result

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
