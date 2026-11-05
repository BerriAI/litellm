from collections.abc import Mapping
from typing import Literal

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel


class SkillHubSkill(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None = None
    version: str | None = None
    visibility: Literal["public", "private"]
    source: Mapping[str, str]
    download_url: str | None = None
    content_url: str | None = None


class SkillHubList(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    skills: tuple[SkillHubSkill, ...]
    count: int
    next_offset: int | None


class ImportSkillRequest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    url: str = Field(description="Public skills.sh skill URL or GitHub repository URL")
    skill: str | None = Field(None, description="Skill name to select when a repository contains several skills")
    ref: str = Field("HEAD", description="Git commit, tag or branch to import")
    path: str | None = Field(None, description="Repository-relative skill directory")
    version: str | None = Field(None, description="Version label for the imported snapshot")


class SkillHubMetadata(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str | None = Field(None, max_length=1024)
    version: str | None = None
    domain: str | None = None
    namespace: str | None = None
    keywords: tuple[str, ...] | None = None


class SkillHubPublication(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
