"""
Transformation handler for LiteLLM database-backed skills.

This module provides the SDK-level transformation layer that converts
API requests to database operations via LiteLLMSkillsHandler.

Pattern follows litellm/llms/litellm_proxy/responses/transformation.py
"""

from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Optional, Union

from litellm.types.llms.anthropic_skills import (
    DeleteSkillResponse,
    ListSkillsResponse,
    Skill,
)
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class LiteLLMSkillsTransformationHandler:
    """
    Transformation handler for skills API requests to LiteLLM database operations.
    
    This is used when custom_llm_provider="litellm_proxy" to store/retrieve skills
    from the LiteLLM proxy database instead of calling an external API.
    """

    @property
    def custom_llm_provider(self) -> str:
        """Return the provider name for logging."""
        return LlmProviders.LITELLM_PROXY.value

    def create_skill_handler(
        self,
        display_title: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        files: Optional[List[Any]] = None,
        file_content: Optional[bytes] = None,
        file_name: Optional[str] = None,
        file_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        _is_async: bool = False,
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
        litellm_call_id: Optional[str] = None,
        **kwargs,
    ) -> Union[Skill, Coroutine[Any, Any, Skill]]:
        """
        Create a skill in LiteLLM database.
        
        Args:
            display_title: Display title for the skill
            description: Description of the skill
            instructions: Instructions/prompt for the skill
            files: Files to upload - list of tuples (filename, content, content_type)
            file_content: Binary content of skill files (alternative to files)
            file_name: Original filename (alternative to files)
            file_type: MIME type (alternative to files)
            metadata: Additional metadata
            user_id: User ID for tracking
            _is_async: Whether to return a coroutine
            
        Returns:
            Skill object or coroutine that returns Skill
        """
        # Pre-call logging
        if logging_obj:
            logging_obj.update_environment_variables(
                model=None,
                optional_params={"display_title": display_title},
                litellm_params={"litellm_call_id": litellm_call_id},
                custom_llm_provider=self.custom_llm_provider,
            )

        # Extract file content from files parameter if provided
        # files is a list of tuples: [(filename, content, content_type), ...]
        if files and not file_content:
            if isinstance(files, list) and len(files) > 0:
                first_file = files[0]
                if isinstance(first_file, tuple) and len(first_file) >= 2:
                    file_name = first_file[0]
                    file_content = first_file[1]
                    file_type = first_file[2] if len(first_file) > 2 else "application/zip"

        if _is_async:
            return self._async_create_skill(
                display_title=display_title,
                description=description,
                instructions=instructions,
                file_content=file_content,
                file_name=file_name,
                file_type=file_type,
                metadata=metadata,
                user_id=user_id,
            )
        
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._async_create_skill(
                display_title=display_title,
                description=description,
                instructions=instructions,
                file_content=file_content,
                file_name=file_name,
                file_type=file_type,
                metadata=metadata,
                user_id=user_id,
            )
        )

    async def _async_create_skill(
        self,
        display_title: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        file_content: Optional[bytes] = None,
        file_name: Optional[str] = None,
        file_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Skill:
        """Async implementation of create_skill."""
        # Lazy import to avoid SDK dependency on proxy
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
        from litellm.proxy._types import NewSkillRequest

        skill_request = NewSkillRequest(
            display_title=display_title,
            description=description,
            instructions=instructions,
            file_content=file_content,
            file_name=file_name,
            file_type=file_type,
            metadata=metadata,
        )

        db_skill = await LiteLLMSkillsHandler.create_skill(
            data=skill_request,
            user_id=user_id,
        )

        return self._db_skill_to_response(db_skill)

    def list_skills_handler(
        self,
        limit: int = 20,
        offset: int = 0,
        _is_async: bool = False,
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
        litellm_call_id: Optional[str] = None,
        **kwargs,
    ) -> Union[ListSkillsResponse, Coroutine[Any, Any, ListSkillsResponse]]:
        """
        List skills from LiteLLM database.
        
        Args:
            limit: Maximum number of skills to return
            offset: Number of skills to skip
            _is_async: Whether to return a coroutine
            logging_obj: LiteLLM logging object
            litellm_call_id: Call ID for logging
            
        Returns:
            ListSkillsResponse or coroutine that returns ListSkillsResponse
        """
        # Pre-call logging
        if logging_obj:
            logging_obj.update_environment_variables(
                model=None,
                optional_params={"limit": limit, "offset": offset},
                litellm_params={"litellm_call_id": litellm_call_id},
                custom_llm_provider=self.custom_llm_provider,
            )

        if _is_async:
            return self._async_list_skills(limit=limit, offset=offset)
        
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._async_list_skills(limit=limit, offset=offset)
        )

    async def _async_list_skills(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> ListSkillsResponse:
        """Async implementation of list_skills."""
        # Lazy import to avoid SDK dependency on proxy
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        db_skills = await LiteLLMSkillsHandler.list_skills(
            limit=limit,
            offset=offset,
        )

        skills = [self._db_skill_to_response(s) for s in db_skills]
        return ListSkillsResponse(
            data=skills,
            has_more=len(skills) >= limit,
            next_page=None,
        )

    def get_skill_handler(
        self,
        skill_id: str,
        _is_async: bool = False,
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
        litellm_call_id: Optional[str] = None,
        **kwargs,
    ) -> Union[Skill, Coroutine[Any, Any, Skill]]:
        """
        Get a skill from LiteLLM database.
        
        Args:
            skill_id: The skill ID to retrieve
            _is_async: Whether to return a coroutine
            logging_obj: LiteLLM logging object
            litellm_call_id: Call ID for logging
            
        Returns:
            Skill or coroutine that returns Skill
        """
        # Pre-call logging
        if logging_obj:
            logging_obj.update_environment_variables(
                model=None,
                optional_params={"skill_id": skill_id},
                litellm_params={"litellm_call_id": litellm_call_id},
                custom_llm_provider=self.custom_llm_provider,
            )

        if _is_async:
            return self._async_get_skill(skill_id=skill_id)
        
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._async_get_skill(skill_id=skill_id)
        )

    async def _async_get_skill(self, skill_id: str) -> Skill:
        """Async implementation of get_skill."""
        # Lazy import to avoid SDK dependency on proxy
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        db_skill = await LiteLLMSkillsHandler.get_skill(skill_id=skill_id)
        return self._db_skill_to_response(db_skill)

    def delete_skill_handler(
        self,
        skill_id: str,
        _is_async: bool = False,
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
        litellm_call_id: Optional[str] = None,
        **kwargs,
    ) -> Union[DeleteSkillResponse, Coroutine[Any, Any, DeleteSkillResponse]]:
        """
        Delete a skill from LiteLLM database.
        
        Args:
            skill_id: The skill ID to delete
            _is_async: Whether to return a coroutine
            logging_obj: LiteLLM logging object
            litellm_call_id: Call ID for logging
            
        Returns:
            DeleteSkillResponse or coroutine that returns DeleteSkillResponse
        """
        # Pre-call logging
        if logging_obj:
            logging_obj.update_environment_variables(
                model=None,
                optional_params={"skill_id": skill_id},
                litellm_params={"litellm_call_id": litellm_call_id},
                custom_llm_provider=self.custom_llm_provider,
            )

        if _is_async:
            return self._async_delete_skill(skill_id=skill_id)
        
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._async_delete_skill(skill_id=skill_id)
        )

    async def _async_delete_skill(self, skill_id: str) -> DeleteSkillResponse:
        """Async implementation of delete_skill."""
        # Lazy import to avoid SDK dependency on proxy
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        result = await LiteLLMSkillsHandler.delete_skill(skill_id=skill_id)
        return DeleteSkillResponse(
            id=result["id"],
            type=result.get("type", "skill_deleted"),
        )

    def _db_skill_to_response(self, db_skill: Any) -> Skill:
        """
        Convert a database skill record to Anthropic-compatible Skill response.
        
        Args:
            db_skill: LiteLLM_SkillsTable record
            
        Returns:
            Skill object
        """
        created_at = ""
        updated_at = ""
        
        if hasattr(db_skill, "created_at") and db_skill.created_at:
            created_at = (
                db_skill.created_at.isoformat()
                if hasattr(db_skill.created_at, "isoformat")
                else str(db_skill.created_at)
            )
        if hasattr(db_skill, "updated_at") and db_skill.updated_at:
            updated_at = (
                db_skill.updated_at.isoformat()
                if hasattr(db_skill.updated_at, "isoformat")
                else str(db_skill.updated_at)
            )

        return Skill(
            id=db_skill.skill_id,
            created_at=created_at,
            updated_at=updated_at,
            display_title=db_skill.display_title,
            latest_version=db_skill.latest_version,
            source=db_skill.source or "custom",
            type="skill",
        )

