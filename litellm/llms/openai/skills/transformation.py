"""OpenAI Skills API configuration and transformations."""

from typing import Any  # noqa: RUF100, TID251  # provider query and response payloads are dynamic

import httpx

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.skills.transformation import BaseSkillsAPIConfig, LiteLLMLoggingObj
from litellm.llms.openai.common_utils import get_openai_credentials
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.llms.openai_skills import (
    CreateOpenAISkillRequest,
    ListOpenAISkillsParams,
    OpenAIDeletedSkill,
    OpenAIDeletedSkillVersion,
    OpenAISkill,
    OpenAISkillList,
    OpenAISkillVersion,
    OpenAISkillVersionList,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class OpenAISkillsConfig(BaseSkillsAPIConfig):
    """OpenAI native Skills API configuration."""

    is_openai_native = True

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        params = litellm_params or GenericLiteLLMParams()
        credentials = get_openai_credentials(api_base=params.api_base, api_key=params.api_key)
        if not credentials.api_key:
            raise ValueError("OPENAI_API_KEY is required for Skills API")
        headers["Authorization"] = f"Bearer {credentials.api_key}"
        headers.setdefault("Content-Type", "application/json")
        return headers

    def get_api_base(self, litellm_params: GenericLiteLLMParams) -> str | None:
        return get_openai_credentials(api_base=litellm_params.api_base).api_base

    def get_complete_url(
        self,
        api_base: str | None,
        endpoint: str,
        skill_id: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> str:
        base_url = get_openai_credentials(api_base=api_base).api_base.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        path = endpoint.strip("/")
        if skill_id is not None:
            path = f"{path}/{encode_url_path_segment(skill_id, field_name='skill_id')}"
        return f"{base_url}/{path}"

    def transform_create_skill_request(
        self,
        create_request: CreateOpenAISkillRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        return {"files": create_request.get("files", [])}

    def transform_create_skill_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj
    ) -> OpenAISkill:
        return OpenAISkill(**raw_response.json())

    def transform_list_skills_request(
        self,
        list_params: ListOpenAISkillsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        url = self.get_complete_url(litellm_params.api_base, "skills", litellm_params=litellm_params)
        query_params: dict[str, Any] = {}
        if list_params.get("limit") is not None:
            query_params["limit"] = list_params["limit"]
        if list_params.get("page"):
            query_params["after"] = list_params["page"]
        for key in ("after", "order"):
            if key in list_params and list_params[key]:
                query_params[key] = list_params[key]
        return url, query_params

    def transform_list_skills_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj
    ) -> OpenAISkillList:
        return OpenAISkillList(**raw_response.json())

    def transform_get_skill_request(
        self, skill_id: str, api_base: str, litellm_params: GenericLiteLLMParams, headers: dict
    ) -> tuple[str, dict]:
        return self.get_complete_url(api_base, "skills", skill_id, litellm_params), headers

    def transform_get_skill_response(self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj) -> OpenAISkill:
        return OpenAISkill(**raw_response.json())

    def transform_delete_skill_request(
        self, skill_id: str, api_base: str, litellm_params: GenericLiteLLMParams, headers: dict
    ) -> tuple[str, dict]:
        return self.get_complete_url(api_base, "skills", skill_id, litellm_params), headers

    def transform_delete_skill_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj
    ) -> OpenAIDeletedSkill:
        return OpenAIDeletedSkill(**raw_response.json())

    def get_skill_operation_url(
        self,
        operation: str,
        skill_id: str,
        version: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> str:
        params = litellm_params or GenericLiteLLMParams()
        endpoint = f"skills/{encode_url_path_segment(skill_id, field_name='skill_id')}"
        if operation in {"content", "versions"}:
            endpoint += f"/{operation}"
        elif operation in {"version_content", "version", "delete_version"}:
            if version is None:
                raise ValueError("version is required")
            encoded_version = encode_url_path_segment(version, field_name="version")
            endpoint += f"/versions/{encoded_version}"
            if operation == "version_content":
                endpoint += "/content"
        return self.get_complete_url(self.get_api_base(params), endpoint, litellm_params=params)

    def transform_skill_operation_response(
        self, operation: str, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj
    ) -> Any:
        if operation in {"content", "version_content"}:
            return HttpxBinaryResponseContent(response=raw_response)
        if operation == "update":
            return OpenAISkill(**raw_response.json())
        if operation == "create_version" or operation == "version":
            return OpenAISkillVersion(**raw_response.json())
        if operation == "list_versions":
            return OpenAISkillVersionList(**raw_response.json())
        if operation == "delete_version":
            return OpenAIDeletedSkillVersion(**raw_response.json())
        raise ValueError(f"Unsupported Skills operation: {operation}")
