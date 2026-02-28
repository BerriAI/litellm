"""
OpenAI Skills API configuration and transformations
"""

from typing import Any, Dict, Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.skills.transformation import (
    BaseSkillsAPIConfig,
    LiteLLMLoggingObj,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.anthropic_skills import (
    CreateSkillRequest,
    DeleteSkillResponse,
    ListSkillsParams,
    ListSkillsResponse,
    Skill,
)
from litellm.types.llms.openai_skills import (
    OpenAIDeletedSkill,
    OpenAISkill,
    OpenAISkillList,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

DEFAULT_OPENAI_API_BASE = "https://api.openai.com"


class OpenAISkillsConfig(BaseSkillsAPIConfig):
    """OpenAI-specific Skills API configuration"""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def _get_api_key(
        self, litellm_params: Optional[GenericLiteLLMParams]
    ) -> Optional[str]:
        """Resolve OpenAI API key from params, globals, or environment."""
        api_key = None
        if litellm_params:
            api_key = litellm_params.api_key
        return (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )

    def _get_api_base(
        self, litellm_params: Optional[GenericLiteLLMParams]
    ) -> str:
        """Resolve OpenAI API base URL."""
        api_base = None
        if litellm_params:
            api_base = litellm_params.api_base
        return api_base or DEFAULT_OPENAI_API_BASE

    def get_api_base(
        self, litellm_params: Optional[GenericLiteLLMParams]
    ) -> str:
        """Resolve OpenAI API base URL (override for provider-agnostic dispatch)."""
        return self._get_api_base(litellm_params)

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """Add OpenAI-specific headers (Bearer token auth)."""
        api_key = self._get_api_key(litellm_params)

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI Skills API")

        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        endpoint: str,
        skill_id: Optional[str] = None,
    ) -> str:
        """Get complete URL for OpenAI Skills API."""
        if api_base is None:
            api_base = DEFAULT_OPENAI_API_BASE

        if skill_id:
            return f"{api_base}/v1/skills/{skill_id}"
        return f"{api_base}/v1/{endpoint}"

    def transform_create_skill_request(
        self,
        create_request: CreateSkillRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """Transform create skill request for OpenAI (multipart passthrough)."""
        verbose_logger.debug(
            "Transforming OpenAI create skill request: %s", create_request
        )
        return {k: v for k, v in create_request.items() if v is not None}

    def transform_create_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """Transform OpenAI response to canonical Skill object."""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming OpenAI create skill response: %s", response_json
        )
        return self._openai_skill_to_canonical(response_json)

    def transform_list_skills_request(
        self,
        list_params: ListSkillsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform list skills request for OpenAI."""
        api_base = self._get_api_base(litellm_params)
        url = self.get_complete_url(api_base=api_base, endpoint="skills")

        query_params: Dict[str, Any] = {}
        if "limit" in list_params and list_params["limit"]:
            query_params["limit"] = list_params["limit"]
        # Map Anthropic-style 'page' to OpenAI-style 'after' if present
        if "page" in list_params and list_params["page"]:
            query_params["after"] = list_params["page"]

        verbose_logger.debug(
            "OpenAI list skills request with params: %s", query_params
        )
        return url, query_params

    def transform_list_skills_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListSkillsResponse:
        """Transform OpenAI SkillList response to canonical ListSkillsResponse."""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming OpenAI list skills response: %s", response_json
        )

        skills = [
            self._openai_skill_to_canonical(s) for s in response_json.get("data", [])
        ]

        return ListSkillsResponse(
            data=skills,
            next_page=response_json.get("last_id"),
            has_more=response_json.get("has_more", False),
        )

    def transform_get_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform get skill request for OpenAI."""
        url = self.get_complete_url(
            api_base=api_base, endpoint="skills", skill_id=skill_id
        )
        verbose_logger.debug("OpenAI get skill request - URL: %s", url)
        return url, headers

    def transform_get_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """Transform OpenAI response to canonical Skill object."""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming OpenAI get skill response: %s", response_json
        )
        return self._openai_skill_to_canonical(response_json)

    def transform_delete_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """Transform delete skill request for OpenAI."""
        url = self.get_complete_url(
            api_base=api_base, endpoint="skills", skill_id=skill_id
        )
        verbose_logger.debug("OpenAI delete skill request - URL: %s", url)
        return url, headers

    def transform_delete_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteSkillResponse:
        """Transform OpenAI DeletedSkill response to canonical DeleteSkillResponse."""
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming OpenAI delete skill response: %s", response_json
        )
        return DeleteSkillResponse(
            id=response_json["id"],
            type="skill_deleted",
        )

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _openai_skill_to_canonical(data: dict) -> Skill:
        """Map OpenAI Skill JSON to canonical litellm Skill model.

        OpenAI fields → canonical fields:
          - name → display_title
          - created_at (unix int) → created_at (ISO string)
          - object → type
          - no 'source' field → default to 'custom'
        """
        import datetime

        created_ts = data.get("created_at", 0)
        if isinstance(created_ts, (int, float)):
            created_at_str = datetime.datetime.fromtimestamp(
                created_ts, tz=datetime.timezone.utc
            ).isoformat()
        else:
            created_at_str = str(created_ts)

        return Skill(
            id=data["id"],
            created_at=created_at_str,
            display_title=data.get("name"),
            latest_version=data.get("latest_version"),
            source="custom",
            type=data.get("object", "skill"),
            updated_at=created_at_str,  # OpenAI doesn't have updated_at; use created_at
        )
