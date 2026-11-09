from collections.abc import Mapping
from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field

from litellm.models.skill_submission import SubmissionOperation, SubmissionStatus
from litellm.types.llms.base import LiteLLMBaseModel

SkillSortField: TypeAlias = Literal[
    "name",
    "download_count",
    "updated_at",
    "created_at",
    "created_by",
    "published_by",
    "approved_by",
    "security_scan_passed",
]


class SkillHubSkill(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None = None
    version: str | None = None
    updated_at: datetime | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    approved_by: str | None = None
    published_by: str | None = None
    security_scan_passed: bool = False
    visibility: Literal["public", "private"]
    source: Mapping[str, str]
    download_url: str | None = None
    content_url: str | None = None
    domain: str | None = None
    namespace: str | None = None
    keywords: tuple[str, ...] | None = None
    download_count: int | None = Field(None, description="Gateway ZIP downloads, including repeat downloads")


class SkillHubList(LiteLLMBaseModel):
    tags: tuple[str, ...] = ()
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


class SkillHubAgentSkill(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    digest: str
    origin: str | None


class SkillHubProposal(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None
    version: str | None
    domain: str | None = None
    namespace: str | None = None
    keywords: tuple[str, ...] | None = None
    source: Mapping[str, str]
    agent_skill: SkillHubAgentSkill | None = None


class SkillHubPendingSkill(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["pending_review"] = "pending_review"
    action: Literal["submitted"] = "submitted"
    submission_id: str
    proposal: SkillHubProposal


SkillHubMutationResponse: TypeAlias = SkillHubSkill | SkillHubPendingSkill


class SkillHubSubmission(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    skill_name: str
    operation: SubmissionOperation
    status: SubmissionStatus
    proposal: SkillHubProposal
    has_archive: bool
    submitted_by: str
    created_at: datetime
    base_skill_id: str | None
    base_updated_at: datetime | None
    reviewed_by: str | None
    reviewed_at: datetime | None
    rejection_reason: str | None


class SkillHubSubmissionList(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    submissions: tuple[SkillHubSubmission, ...]
    count: int
    has_more: bool


class SkillHubSubmissionDetail(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    submission: SkillHubSubmission
    current_skill: SkillHubSkill | None


class SkillHubReviewResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["success"] = "success"
    action: Literal["approved", "rejected"]
    submission: SkillHubSubmission
    skill: SkillHubSkill | None


class SkillHubRejection(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str | None = Field(None, max_length=2000)
