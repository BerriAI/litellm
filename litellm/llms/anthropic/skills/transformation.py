"""
Anthropic Skills API configuration and transformations
"""

from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
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

_RAW_JSON_PAYLOAD: Final = TypeAdapter(object)


class AnthropicSkillsConfig(BaseSkillsAPIConfig):
    """Anthropic-specific Skills API configuration"""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.ANTHROPIC

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        """Add Anthropic-specific headers"""
        from litellm.constants import ANTHROPIC_SKILLS_API_BETA_VERSION
        from litellm.llms.anthropic.common_utils import (
            AnthropicModelInfo,
            merge_anthropic_beta_headers,
            without_caller_credential_headers,
        )

        auth_header: Final = AnthropicModelInfo.get_auth_header(
            api_key=litellm_params.api_key if litellm_params is not None else None,
            api_base=litellm_params.api_base if litellm_params is not None else None,
            litellm_params=MappingProxyType(dict(litellm_params)) if litellm_params is not None else None,
            allow_workload_identity=True,
        )
        if auth_header is None:
            raise ValueError("ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN is required for Skills API")

        merged_beta: Final = merge_anthropic_beta_headers(
            merge_anthropic_beta_headers(headers.get("anthropic-beta"), auth_header.get("anthropic-beta")),
            ANTHROPIC_SKILLS_API_BETA_VERSION,
        )
        # The deployment's own credential is applied here, so a caller-supplied one must not ride
        # along upstream beside a minted federation Bearer.
        return {  # mutable-ok: validate_environment's contract returns a real dict, which httpx then consumes
            **without_caller_credential_headers(headers),
            **auth_header,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": merged_beta,
            "content-type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        endpoint: str,
        skill_id: str | None = None,
    ) -> str:
        """Get complete URL for Anthropic Skills API"""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        if api_base is None:
            api_base = AnthropicModelInfo.get_api_base()

        if skill_id:
            encoded_skill_id: Final = encode_url_path_segment(skill_id, field_name="skill_id")
            return f"{api_base}/v1/skills/{encoded_skill_id}"
        return f"{api_base}/v1/{endpoint}"

    def transform_create_skill_request(
        self,
        create_request: CreateSkillRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        """Transform create skill request for Anthropic"""
        verbose_logger.debug("Transforming create skill request: %s", create_request)

        # Anthropic expects the request body directly
        request_body: Final = {k: v for k, v in create_request.items() if v is not None}

        return request_body

    def transform_create_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """Transform Anthropic response to Skill object"""
        response_json: Final = _RAW_JSON_PAYLOAD.validate_python(raw_response.json())
        verbose_logger.debug("Transforming create skill response: %s", response_json)

        return Skill.model_validate(response_json)

    def transform_list_skills_request(
        self,
        list_params: ListSkillsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform list skills request for Anthropic"""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        api_base: Final = AnthropicModelInfo.get_api_base(litellm_params.api_base if litellm_params else None)
        url: Final = self.get_complete_url(api_base=api_base, endpoint="skills")

        # Build query parameters
        limit: Final = list_params.get("limit")
        page: Final = list_params.get("page")
        source: Final = list_params.get("source")
        query_params: Final[dict[str, int | str]] = {
            key: value for key, value in (("limit", limit), ("page", page), ("source", source)) if value
        }

        verbose_logger.debug(
            "List skills request made to Anthropic Skills endpoint with params: %s",
            query_params,
        )

        return url, query_params

    def transform_list_skills_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListSkillsResponse:
        """Transform Anthropic response to ListSkillsResponse"""
        response_json: Final = _RAW_JSON_PAYLOAD.validate_python(raw_response.json())
        verbose_logger.debug("Transforming list skills response: %s", response_json)

        return ListSkillsResponse.model_validate(response_json)

    def transform_get_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform get skill request for Anthropic"""
        url: Final = self.get_complete_url(api_base=api_base, endpoint="skills", skill_id=skill_id)

        verbose_logger.debug("Get skill request - URL: %s", url)

        return url, headers

    def transform_get_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Skill:
        """Transform Anthropic response to Skill object"""
        response_json: Final = _RAW_JSON_PAYLOAD.validate_python(raw_response.json())
        verbose_logger.debug("Transforming get skill response: %s", response_json)

        return Skill.model_validate(response_json)

    def transform_delete_skill_request(
        self,
        skill_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform delete skill request for Anthropic"""
        url: Final = self.get_complete_url(api_base=api_base, endpoint="skills", skill_id=skill_id)

        verbose_logger.debug("Delete skill request - URL: %s", url)

        return url, headers

    def transform_delete_skill_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteSkillResponse:
        """Transform Anthropic response to DeleteSkillResponse"""
        response_json: Final = _RAW_JSON_PAYLOAD.validate_python(raw_response.json())
        verbose_logger.debug("Transforming delete skill response: %s", response_json)

        return DeleteSkillResponse.model_validate(response_json)
