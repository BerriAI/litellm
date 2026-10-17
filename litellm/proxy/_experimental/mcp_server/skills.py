from collections.abc import Mapping
from typing import Final, Literal

from mcp.types import CacheableResult, PaginatedRequest, PaginatedResult, Request, RequestParams, ResultType
from pydantic import BaseModel, ConfigDict, JsonValue
from pydantic.alias_generators import to_camel

SKILLS_EXTENSION: Final = "io.modelcontextprotocol/skills"


class SkillResource(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True, extra="allow")

    uri: str


class Skill(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True, extra="allow")

    uri: str
    frontmatter: Mapping[str, JsonValue]
    resources: tuple[SkillResource, ...] = ()


class ListSkillsRequest(PaginatedRequest[Literal["skills/list"]]):
    model_config = ConfigDict(frozen=True)

    method: Literal["skills/list"] = "skills/list"


class ListSkillsResult(PaginatedResult, CacheableResult):
    model_config = ConfigDict(frozen=True)

    skills: tuple[Skill, ...]
    result_type: ResultType = "complete"


class GetSkillRequestParams(RequestParams):
    model_config = ConfigDict(frozen=True)

    uri: str


class GetSkillRequest(Request[GetSkillRequestParams, Literal["skills/get"]]):
    model_config = ConfigDict(frozen=True)

    method: Literal["skills/get"] = "skills/get"
    params: GetSkillRequestParams


class GetSkillResult(CacheableResult):
    model_config = ConfigDict(frozen=True)

    skill: Skill
    result_type: ResultType = "complete"
