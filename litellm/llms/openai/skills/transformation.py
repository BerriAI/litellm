"""
OpenAI Skills API configuration and transformations.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
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
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class OpenAISkillsConfig(BaseSkillsAPIConfig):
    """OpenAI-specific Skills API configuration."""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        api_key = (
            (litellm_params.api_key if litellm_params else None)
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )
        if api_key is None:
            raise ValueError("OpenAI API key is required for Skills API")

        headers["Authorization"] = f"Bearer {api_key}"
        # Skill creation may be multipart; let httpx set the content type.
        headers.pop("content-type", None)
        headers.pop("Content-Type", None)
        return headers

    def get_api_base(
        self,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> Optional[str]:
        return (
            (litellm_params.api_base if litellm_params else None)
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com"
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        endpoint: str,
        skill_id: Optional[str] = None,
    ) -> str:
        if api_base is None:
            api_base = self.get_api_base(None)
        if api_base is None:
            raise ValueError("api_base is required")

        api_base = api_base.rstrip("/")
        if api_base.endswith("/v1"):
            api_base = api_base[: -len("/v1")]

        if skill_id:
            encoded_skill_id = encode_url_path_segment(skill_id, field_name="skill_id")
            return f"{api_base}/v1/skills/{encoded_skill_id}"
        return f"{api_base}/v1/{endpoint}"

    def get_skill_file_field_name(self) -> str:
        return "file"

    def transform_create_skill_request(
        self,
        create_request: CreateSkillRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        verbose_logger.debug(
            "Transforming OpenAI create skill request: %s", create_request
        )
        return {k: v for k, v in create_request.items() if k == "files" and v}

    def transform_create_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming OpenAI create skill response: %s", response_json
        )
        return self._openai_skill_to_canonical_skill(response_json)

    def transform_list_skills_request(
        self,
        list_params: ListSkillsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        api_base = self.get_api_base(litellm_params)
        url = self.get_complete_url(api_base=api_base, endpoint="skills")

        query_params: Dict[str, Any] = {}
        raw_list_params: Dict[str, Any] = dict(list_params)
        if list_params.get("limit"):
            query_params["limit"] = list_params["limit"]
        if list_params.get("page"):
            query_params["after"] = list_params["page"]
        for openai_cursor_param in ("after", "before", "order"):
            if raw_list_params.get(openai_cursor_param):
                query_params[openai_cursor_param] = raw_list_params[openai_cursor_param]

        verbose_logger.debug(
            "OpenAI list skills request made with params: %s", query_params
        )
        return url, query_params

    def transform_list_skills_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListSkillsResponse:
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming OpenAI list skills response: %s", response_json
        )

        has_more = response_json.get("has_more", False)
        return ListSkillsResponse(
            data=[
                self._openai_skill_to_canonical_skill(skill)
                for skill in response_json.get("data", [])
            ],
            has_more=has_more,
            next_page=response_json.get("last_id") if has_more else None,
        )

    def transform_get_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
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
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming OpenAI get skill response: %s", response_json
        )
        return self._openai_skill_to_canonical_skill(response_json)

    def transform_delete_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
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
        response_json = raw_response.json()
        verbose_logger.debug(
            "Transforming OpenAI delete skill response: %s", response_json
        )
        return DeleteSkillResponse(
            id=response_json["id"],
            type=response_json.get("type", "skill_deleted"),
        )

    def _openai_skill_to_canonical_skill(self, skill: Dict[str, Any]) -> Skill:
        return Skill(
            id=skill["id"],
            created_at=self._format_openai_timestamp(skill.get("created_at")),
            updated_at=self._format_openai_timestamp(skill.get("updated_at")),
            display_title=skill.get("display_title") or skill.get("name"),
            latest_version=skill.get("latest_version") or skill.get("default_version"),
            source=skill.get("source") or "openai",
            type=skill.get("type") or skill.get("object") or "skill",
        )

    @staticmethod
    def _format_openai_timestamp(timestamp: Any) -> str:
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        if timestamp is None:
            return ""
        return str(timestamp)
