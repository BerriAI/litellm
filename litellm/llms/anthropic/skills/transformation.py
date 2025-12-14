"""
Anthropic Skills API configuration and transformations
"""

from typing import Any, Dict, Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.skills.transformation import (
    BaseSkillsAPIConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.anthropic_skills import (
    CreateSkillRequest,
    DeleteSkillResponse,
    ListSkillsParams,
    ListSkillsResponse,
    Skill,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class AnthropicSkillsConfig(BaseSkillsAPIConfig):
    """Anthropic-specific Skills API configuration"""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.ANTHROPIC

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """Add Anthropic-specific headers"""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        # Get API key
        api_key = None
        if litellm_params:
            api_key = litellm_params.api_key
        api_key = AnthropicModelInfo.get_api_key(api_key)

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Skills API")

        # Add required headers
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        
        # Add beta header for skills API
        from litellm.constants import ANTHROPIC_SKILLS_API_BETA_VERSION
        
        if "anthropic-beta" not in headers:
            headers["anthropic-beta"] = ANTHROPIC_SKILLS_API_BETA_VERSION
        elif isinstance(headers["anthropic-beta"], list):
            if ANTHROPIC_SKILLS_API_BETA_VERSION not in headers["anthropic-beta"]:
                headers["anthropic-beta"].append(ANTHROPIC_SKILLS_API_BETA_VERSION)
        elif isinstance(headers["anthropic-beta"], str):
            if ANTHROPIC_SKILLS_API_BETA_VERSION not in headers["anthropic-beta"]:
                headers["anthropic-beta"] = [headers["anthropic-beta"], ANTHROPIC_SKILLS_API_BETA_VERSION]
        
        headers["content-type"] = "application/json"

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        endpoint: str,
        skill_id: Optional[str] = None,
    ) -> str:
        """Get complete URL for Anthropic Skills API"""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        if api_base is None:
            api_base = AnthropicModelInfo.get_api_base()

        if skill_id:
            return f"{api_base}/v1/skills/{skill_id}?beta=true"
        return f"{api_base}/v1/{endpoint}?beta=true"

    def transform_create_skill_request(
        self,
        create_request: CreateSkillRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """Transform create skill request for Anthropic"""
        verbose_logger.debug(
            "Transforming create skill request: %s", create_request
        )
        
        # Anthropic expects the request body directly
        request_body = {k: v for k, v in create_request.items() if v is not None}
        
        return request_body

    def transform_create_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """Transform Anthropic response to Skill object"""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming create skill response: %s", response_json
        )
        
        return Skill(**response_json)

    def transform_list_skills_request(
        self,
        list_params: ListSkillsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform list skills request for Anthropic"""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        api_base = AnthropicModelInfo.get_api_base(
            litellm_params.api_base if litellm_params else None
        )
        url = self.get_complete_url(api_base=api_base, endpoint="skills")
        
        # Build query parameters
        query_params: Dict[str, Any] = {}
        if "limit" in list_params and list_params["limit"]:
            query_params["limit"] = list_params["limit"]
        if "page" in list_params and list_params["page"]:
            query_params["page"] = list_params["page"]
        if "source" in list_params and list_params["source"]:
            query_params["source"] = list_params["source"]
        
        verbose_logger.debug(
            "List skills request made to Anthropic Skills endpoint with params: %s", query_params
        )
        
        return url, query_params

    def transform_list_skills_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListSkillsResponse:
        """Transform Anthropic response to ListSkillsResponse"""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming list skills response: %s", response_json
        )
        
        return ListSkillsResponse(**response_json)

    def transform_get_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform get skill request for Anthropic"""
        url = self.get_complete_url(
            api_base=api_base, endpoint="skills", skill_id=skill_id
        )
        
        verbose_logger.debug("Get skill request - URL: %s", url)
        
        return url, headers

    def transform_get_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """Transform Anthropic response to Skill object"""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming get skill response: %s", response_json
        )
        
        return Skill(**response_json)

    def transform_delete_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform delete skill request for Anthropic"""
        url = self.get_complete_url(
            api_base=api_base, endpoint="skills", skill_id=skill_id
        )
        
        verbose_logger.debug("Delete skill request - URL: %s", url)
        
        return url, headers

    def transform_delete_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteSkillResponse:
        """Transform Anthropic response to DeleteSkillResponse"""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming delete skill response: %s", response_json
        )
        
        return DeleteSkillResponse(**response_json)

