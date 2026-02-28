"""
Base configuration class for Skills API
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.anthropic_skills import (
    CreateSkillRequest,
    DeleteSkillResponse,
    ListSkillsParams,
    ListSkillsResponse,
    Skill,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseSkillsAPIConfig(ABC):
    """Base configuration for Skills API providers"""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def custom_llm_provider(self) -> LlmProviders:
        pass

    @abstractmethod
    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Validate and update headers with provider-specific requirements
        
        Args:
            headers: Base headers dictionary
            litellm_params: LiteLLM parameters
            
        Returns:
            Updated headers dictionary
        """
        return headers

    def get_api_base(
        self, litellm_params: Optional[GenericLiteLLMParams]
    ) -> str:
        """
        Resolve the API base URL for this provider.
        
        Subclasses should override to provide provider-specific defaults.
        
        Args:
            litellm_params: LiteLLM parameters (may contain api_base override)
            
        Returns:
            API base URL string
        """
        if litellm_params and litellm_params.api_base:
            return litellm_params.api_base
        raise ValueError("api_base is required")

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str],
        endpoint: str,
        skill_id: Optional[str] = None,
    ) -> str:
        """
        Get the complete URL for the API request
        
        Args:
            api_base: Base API URL
            endpoint: API endpoint (e.g., 'skills', 'skills/{id}')
            skill_id: Optional skill ID for specific skill operations
            
        Returns:
            Complete URL
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return f"{api_base}/v1/{endpoint}"

    @abstractmethod
    def transform_create_skill_request(
        self,
        create_request: CreateSkillRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Transform create skill request to provider-specific format
        
        Args:
            create_request: Skill creation parameters
            litellm_params: LiteLLM parameters
            headers: Request headers
            
        Returns:
            Provider-specific request body
        """
        pass

    @abstractmethod
    def transform_create_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """
        Transform provider response to Skill object
        
        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object
            
        Returns:
            Skill object
        """
        pass

    @abstractmethod
    def transform_list_skills_request(
        self,
        list_params: ListSkillsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform list skills request parameters
        
        Args:
            list_params: List parameters (pagination, filters)
            litellm_params: LiteLLM parameters
            headers: Request headers
            
        Returns:
            Tuple of (url, query_params)
        """
        pass

    @abstractmethod
    def transform_list_skills_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListSkillsResponse:
        """
        Transform provider response to ListSkillsResponse
        
        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object
            
        Returns:
            ListSkillsResponse object
        """
        pass

    @abstractmethod
    def transform_get_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform get skill request
        
        Args:
            skill_id: Skill ID
            api_base: Base API URL
            litellm_params: LiteLLM parameters
            headers: Request headers
            
        Returns:
            Tuple of (url, headers)
        """
        pass

    @abstractmethod
    def transform_get_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """
        Transform provider response to Skill object
        
        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object
            
        Returns:
            Skill object
        """
        pass

    @abstractmethod
    def transform_delete_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform delete skill request
        
        Args:
            skill_id: Skill ID
            api_base: Base API URL
            litellm_params: LiteLLM parameters
            headers: Request headers
            
        Returns:
            Tuple of (url, headers)
        """
        pass

    @abstractmethod
    def transform_delete_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteSkillResponse:
        """
        Transform provider response to DeleteSkillResponse
        
        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object
            
        Returns:
            DeleteSkillResponse object
        """
        pass

    # ──────────────────────────────────────────────
    # Optional methods for providers that support them
    # (OpenAI-specific: update, content, versions)
    # Default: raise NotImplementedError
    # ──────────────────────────────────────────────

    def transform_update_skill_request(
        self,
        skill_id: str,
        update_data: Dict,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict, Dict]:
        """
        Transform update skill request (e.g. set default_version).
        
        Returns:
            Tuple of (url, headers, request_body)
        """
        raise NotImplementedError(
            f"update_skill is not supported for {self.custom_llm_provider}"
        )

    def transform_update_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """Transform update skill response to Skill object."""
        raise NotImplementedError(
            f"update_skill is not supported for {self.custom_llm_provider}"
        )

    def transform_get_skill_content_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform get skill content request.
        
        Returns:
            Tuple of (url, headers)
        """
        raise NotImplementedError(
            f"get_skill_content is not supported for {self.custom_llm_provider}"
        )

    def transform_get_skill_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Dict:
        """Transform get skill content response."""
        raise NotImplementedError(
            f"get_skill_content is not supported for {self.custom_llm_provider}"
        )

    def transform_create_skill_version_request(
        self,
        skill_id: str,
        create_request: Dict,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict, Dict]:
        """
        Transform create skill version request.
        
        Returns:
            Tuple of (url, headers, request_body)
        """
        raise NotImplementedError(
            f"create_skill_version is not supported for {self.custom_llm_provider}"
        )

    def transform_create_skill_version_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Dict:
        """Transform create skill version response."""
        raise NotImplementedError(
            f"create_skill_version is not supported for {self.custom_llm_provider}"
        )

    def transform_list_skill_versions_request(
        self,
        skill_id: str,
        list_params: Dict,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict, Dict]:
        """
        Transform list skill versions request.
        
        Returns:
            Tuple of (url, headers, query_params)
        """
        raise NotImplementedError(
            f"list_skill_versions is not supported for {self.custom_llm_provider}"
        )

    def transform_list_skill_versions_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Dict:
        """Transform list skill versions response."""
        raise NotImplementedError(
            f"list_skill_versions is not supported for {self.custom_llm_provider}"
        )

    def transform_get_skill_version_request(
        self,
        skill_id: str,
        version: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform get skill version request.
        
        Returns:
            Tuple of (url, headers)
        """
        raise NotImplementedError(
            f"get_skill_version is not supported for {self.custom_llm_provider}"
        )

    def transform_get_skill_version_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Dict:
        """Transform get skill version response."""
        raise NotImplementedError(
            f"get_skill_version is not supported for {self.custom_llm_provider}"
        )

    def transform_delete_skill_version_request(
        self,
        skill_id: str,
        version: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform delete skill version request.
        
        Returns:
            Tuple of (url, headers)
        """
        raise NotImplementedError(
            f"delete_skill_version is not supported for {self.custom_llm_provider}"
        )

    def transform_delete_skill_version_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Dict:
        """Transform delete skill version response."""
        raise NotImplementedError(
            f"delete_skill_version is not supported for {self.custom_llm_provider}"
        )

    def transform_get_skill_version_content_request(
        self,
        skill_id: str,
        version: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform get skill version content request.
        
        Returns:
            Tuple of (url, headers)
        """
        raise NotImplementedError(
            f"get_skill_version_content is not supported for {self.custom_llm_provider}"
        )

    def transform_get_skill_version_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Dict:
        """Transform get skill version content response."""
        raise NotImplementedError(
            f"get_skill_version_content is not supported for {self.custom_llm_provider}"
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,
    ) -> Exception:
        """Get appropriate error class for the provider."""
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

