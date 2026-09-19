"""Anthropic Managed Agents (beta managed-agents-2026-04-01). Anthropic assigns agent ids, archives
agents instead of deleting them, and on the proxy ``metadata`` carries LiteLLM's own request metadata,
so it is never forwarded as agent metadata.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.managed_agents import (
    invalid_request,
    managed_agents_api_base,
    managed_agents_headers,
    optional_str,
    raise_for_status,
)
from litellm.llms.base_llm.agents.transformation import BaseAgentsAPIConfig
from litellm.types.agents import (
    AgentCreateResponse,
    AgentDeleteResult,
    AgentListResponse,
    AgentVersionsResponse,
)


class _CreateAgentRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    model: str | Mapping[str, object] | None = None
    system: str | None = None
    description: str | None = None
    tools: Sequence[object] | None = None
    mcp_servers: Sequence[object] | None = None
    skills: Sequence[object] | None = None
    multiagent: Mapping[str, object] | None = None


class _PageQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    limit: int | None = None
    page: str | None = None
    include_archived: bool | None = None


class _GetQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int | None = None


class _Page(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[Mapping[str, object], ...] = ()
    next_page: str | None = None


class AnthropicAgentsConfig(BaseAgentsAPIConfig):
    def _agent_url(self, name: str, api_base: str | None) -> str:
        return f"{managed_agents_api_base(api_base)}/v1/agents/{encode_url_path_segment(name, field_name='agent id')}"

    def _page(self, raw_response: httpx.Response) -> _Page:
        raise_for_status(raw_response)
        try:
            return _Page.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise AnthropicError(
                status_code=raw_response.status_code,
                message=f"response does not match the Anthropic agents page schema: {e}",
                headers=raw_response.headers,
            )

    def _agent(self, raw_response: httpx.Response) -> AgentCreateResponse:
        raise_for_status(raw_response)
        try:
            return AgentCreateResponse.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise AnthropicError(
                status_code=raw_response.status_code,
                message=f"response does not match the Anthropic agent schema: {e}",
                headers=raw_response.headers,
            )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> Exception:
        return AnthropicError(status_code=status_code, message=error_message, headers=httpx.Headers(headers))

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        return f"{managed_agents_api_base(api_base)}/v1/agents"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        litellm_params: Mapping[str, object],
    ) -> dict[str, str]:  # mutable-ok: BaseAgentsAPIConfig.validate_environment signature
        return managed_agents_headers(
            headers,
            api_key=optional_str(litellm_params.get("api_key")),
            api_base=optional_str(litellm_params.get("api_base")),
        )

    def transform_create_request(
        self,
        name: str,
        litellm_params: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: BaseAgentsAPIConfig.transform_create_request signature
        if litellm_params.get("base_environment") is not None:
            raise invalid_request(
                "Anthropic environments are a separate resource and are not part of the agent: create one with "
                "POST /v1/environments and pass its id as `environment` when starting a session."
            )
        try:
            request: Final = _CreateAgentRequest.model_validate(
                MappingProxyType(
                    {
                        "name": name,
                        "model": litellm_params.get("base_agent"),
                        "system": litellm_params.get("instructions"),
                        "description": litellm_params.get("description"),
                        "tools": litellm_params.get("tools"),
                        "mcp_servers": litellm_params.get("mcp_servers"),
                        "skills": litellm_params.get("skills"),
                        "multiagent": litellm_params.get("multiagent"),
                    }
                )
            )
        except ValidationError as e:
            raise invalid_request(f"invalid Anthropic agent definition: {e}")
        return request.model_dump(mode="json", exclude_none=True)

    def transform_create_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentCreateResponse:
        return self._agent(raw_response)

    def transform_list_request(
        self,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseAgentsAPIConfig.transform_list_request signature
        return self.get_complete_url(api_base, litellm_params), _page_query(litellm_params)

    def transform_list_response(
        self,
        raw_response: httpx.Response,
    ) -> AgentListResponse:
        page: Final = self._page(raw_response)
        return AgentListResponse.model_validate(
            MappingProxyType({"agents": page.data, "next_page_token": page.next_page})
        )

    def transform_get_request(
        self,
        name: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseAgentsAPIConfig.transform_get_request signature
        try:
            query: Final = _GetQuery.model_validate(MappingProxyType({"version": litellm_params.get("version")}))
        except ValidationError as e:
            raise invalid_request(f"invalid agent version: {e}")
        return self._agent_url(name, api_base), query.model_dump(mode="json", exclude_none=True)

    def transform_get_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentCreateResponse:
        return self._agent(raw_response)

    def transform_delete_request(
        self,
        name: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        raise invalid_request(
            "Anthropic managed agents cannot be deleted, only archived: "
            f"POST {self._agent_url(name, api_base)}/archive."
        )

    def transform_delete_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentDeleteResult:
        return AgentDeleteResult(name=name, deleted=False)

    def transform_list_versions_request(
        self,
        name: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseAgentsAPIConfig.transform_list_versions_request signature
        return f"{self._agent_url(name, api_base)}/versions", _page_query(litellm_params)

    def transform_list_versions_response(
        self,
        raw_response: httpx.Response,
        name: str,
    ) -> AgentVersionsResponse:
        page: Final = self._page(raw_response)
        return AgentVersionsResponse.model_validate(
            MappingProxyType({"agent_versions": page.data, "next_page_token": page.next_page})
        )


def _page_query(
    litellm_params: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: the http handler passes this straight to httpx as query params
    try:
        query: Final = _PageQuery.model_validate(
            MappingProxyType(
                {
                    "limit": litellm_params.get("page_size"),
                    "page": litellm_params.get("page_token"),
                    "include_archived": litellm_params.get("include_archived"),
                }
            )
        )
    except ValidationError as e:
        raise invalid_request(f"invalid agents page query: {e}")
    return query.model_dump(mode="json", exclude_none=True)
