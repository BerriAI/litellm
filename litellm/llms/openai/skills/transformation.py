"""
OpenAI Skills API configuration and transformations
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


class OpenAISkillsConfig(BaseSkillsAPIConfig):
    """OpenAI-specific Skills API configuration"""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """Add OpenAI-specific headers"""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        # Get API key
        api_key = None
        if litellm_params:
            api_key = litellm_params.api_key
        api_key = OpenAIGPTConfig.get_api_key(api_key)

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for Skills API")

        # Add required headers
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        endpoint: str,
        skill_id: Optional[str] = None,
    ) -> str:
        """Get complete URL for OpenAI Skills API"""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        if api_base is None:
            api_base = OpenAIGPTConfig.get_api_base()
        
        # Default to OpenAI's API base if not provided
        if api_base is None:
            api_base = "https://api.openai.com"

        if skill_id:
            return f"{api_base}/v1/skills/{skill_id}"
        return f"{api_base}/v1/{endpoint}"

    def transform_create_skill_request(
        self,
        create_request: CreateSkillRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """Transform create skill request for OpenAI"""
        verbose_logger.debug(
            "Transforming create skill request: %s", create_request
        )
        
        # OpenAI expects the request body with files as an array or single zip
        request_body = {k: v for k, v in create_request.items() if v is not None}
        
        return request_body

    def transform_create_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """Transform OpenAI response to Skill object"""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming create skill response: %s", response_json
        )
        
        # Map OpenAI response fields to our Skill model
        # OpenAI returns: id, created_at, default_version, description, latest_version, name, object
        skill_data = {
            "id": response_json.get("id"),
            "created_at": str(response_json.get("created_at", "")),
            "display_title": response_json.get("name"),  # Map name to display_title
            "latest_version": response_json.get("latest_version"),
            "source": "custom",  # OpenAI skills are custom
            "type": "skill",
            "updated_at": str(response_json.get("created_at", "")),  # Use created_at as updated_at if not provided
        }
        
        return Skill(**skill_data)

    def transform_list_skills_request(
        self,
        list_params: ListSkillsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform list skills request for OpenAI"""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        api_base = OpenAIGPTConfig.get_api_base(
            litellm_params.api_base if litellm_params else None
        )
        
        if api_base is None:
            api_base = "https://api.openai.com"
            
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
            "List skills request made to OpenAI Skills endpoint with params: %s", query_params
        )
        
        return url, query_params

    def transform_list_skills_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListSkillsResponse:
        """Transform OpenAI response to ListSkillsResponse"""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming list skills response: %s", response_json
        )
        
        # OpenAI returns a list format, we need to map it to our response format
        # Assuming OpenAI returns similar structure with data array
        skills_data = []
        for skill_json in response_json.get("data", []):
            skill_data = {
                "id": skill_json.get("id"),
                "created_at": str(skill_json.get("created_at", "")),
                "display_title": skill_json.get("name"),
                "latest_version": skill_json.get("latest_version"),
                "source": "custom",
                "type": "skill",
                "updated_at": str(skill_json.get("created_at", "")),
            }
            skills_data.append(Skill(**skill_data))
        
        return ListSkillsResponse(
            data=skills_data,
            next_page=response_json.get("next_page"),
            has_more=response_json.get("has_more", False),
        )

    def transform_get_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform get skill request for OpenAI"""
        if api_base is None:
            api_base = "https://api.openai.com"
            
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
        """Transform OpenAI response to Skill object"""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming get skill response: %s", response_json
        )
        
        skill_data = {
            "id": response_json.get("id"),
            "created_at": str(response_json.get("created_at", "")),
            "display_title": response_json.get("name"),
            "latest_version": response_json.get("latest_version"),
            "source": "custom",
            "type": "skill",
            "updated_at": str(response_json.get("created_at", "")),
        }
        
        return Skill(**skill_data)

    def transform_delete_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform delete skill request for OpenAI"""
        if api_base is None:
            api_base = "https://api.openai.com"
            
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
        """Transform OpenAI response to DeleteSkillResponse"""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming delete skill response: %s", response_json
        )
        
        # OpenAI likely returns similar structure
        return DeleteSkillResponse(
            id=response_json.get("id"),
            type="skill_deleted",
        )
