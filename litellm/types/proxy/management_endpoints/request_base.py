from collections.abc import Mapping, Sequence
from typing import (
    Any,  # noqa: TID251  # metadata and models stay Any so existing callers keep type-checking
    Final,
)

from pydantic import ConfigDict, SerializerFunctionWrapHandler, TypeAdapter, field_validator, model_serializer

from litellm.models.team import BudgetLimitEntry
from litellm.types.utils import LiteLLMPydanticObjectBase

_DUMPED: Final = TypeAdapter(Mapping[str, object])


class LiteLLM_ObjectPermissionBase(LiteLLMPydanticObjectBase):
    mcp_servers: list[str] | None = None
    mcp_access_groups: list[str] | None = None
    mcp_tool_permissions: dict[str, list[str]] | None = None
    mcp_toolsets: list[str] | None = None
    blocked_tools: list[str] | None = None
    vector_stores: list[str] | None = None
    agents: list[str] | None = None
    agent_access_groups: list[str] | None = None
    models: list[str] | None = None
    search_tools: list[str] | None = None
    mcp_tool_search_enabled: bool | None = None
    skills: list[str] | None = None
    apps: Sequence[str] | None = None
    mcp_auth_profiles: list[str] | None = None  # mutable-ok: request schema matches its sibling list fields

    @model_serializer(mode="wrap")
    def _omit_unset_auth_profile_grants(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202  # a return annotation collapses the OpenAPI response schema
        dumped: Final = _DUMPED.validate_python(handler(self))
        if dumped.get("mcp_auth_profiles", ()) is not None:
            return dumped
        return {key: value for key, value in dumped.items() if key != "mcp_auth_profiles"}


class GenerateRequestBase(LiteLLMPydanticObjectBase):
    """
    Overlapping schema between key and user generate/update requests
    """

    key_alias: str | None = None
    duration: str | None = None
    models: list[Any] | None = []
    spend: float | None = 0
    max_budget: float | None = None
    user_id: str | None = None
    team_id: str | None = None
    agent_id: str | None = None
    max_parallel_requests: int | None = None
    metadata: dict[Any, Any] | None = {}
    tpm_limit: int | None = None
    rpm_limit: int | None = None

    budget_duration: str | None = None
    budget_limits: list[BudgetLimitEntry] | None = None  # multiple concurrent budget windows
    allowed_cache_controls: list[object] | None = []
    config: dict[object, object] | None = {}
    permissions: dict[object, object] | None = {}
    model_max_budget: dict[object, object] | None = {}  # {"gpt-4": 5.0, "gpt-3.5-turbo": 5.0}, defaults to {}
    budget_fallbacks: dict[str, list[str]] | None = None

    model_config = ConfigDict(protected_namespaces=())
    model_rpm_limit: dict[object, object] | None = None
    model_tpm_limit: dict[object, object] | None = None
    mcp_rpm_limit: dict[str, int] | None = None
    tag_rpm_limit: dict[str, int] | None = None
    guardrails: list[str] | None = None
    policies: list[str] | None = None
    prompts: list[str] | None = None
    blocked: bool | None = None
    aliases: dict[object, object] | None = {}
    object_permission: LiteLLM_ObjectPermissionBase | None = None

    @field_validator("max_budget", mode="before")
    @classmethod
    def check_max_budget(cls, v: object) -> object:
        if v == "":
            return None
        return v
