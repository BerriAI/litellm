from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel


class SkillHubSkill(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None = None
    version: str | None = None
    updated_at: datetime | None = None
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
    version: str | None = Field(None, description="Semantic version, or omit/null to start at 1.0.0")


class SkillHubMetadata(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str | None = Field(None, max_length=1024)
    version: str | None = Field(None, description="Explicit higher SemVer, or omit/null for an automatic patch bump")
    expected_updated_at: datetime | None = Field(
        None, description="Published updated_at read by the editor. A stale value returns 409", exclude=True
    )
    domain: str | None = None
    namespace: str | None = None
    keywords: tuple[str, ...] | None = None


class SkillHubPublication(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SkillHubVersion(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    version: str | None
    description: str | None
    manifest: Mapping[str, object]
    published_at: datetime | None
    published_by: str | None
    content_kind: Literal["stored_archive", "source_reference"]
    download_url: str | None


class SkillHubVersions(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    versions: tuple[SkillHubVersion, ...]
    next_offset: int | None
