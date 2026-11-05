from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypeAlias

from litellm.models.plugin_upload import (
    PluginPublication,
    PluginPublicationBase,
    PluginPublicationError,
    PublishedPlugin,
)

SubmissionStatus: TypeAlias = Literal["pending_review", "approved", "rejected"]
SubmissionOperation: TypeAlias = Literal["create", "update"]


@dataclass(frozen=True, slots=True)
class SkillProposal:
    name: str
    version: str | None
    description: str | None
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SkillSubmission:
    id: str
    plugin_name: str
    operation: SubmissionOperation
    status: SubmissionStatus
    proposal: SkillProposal
    has_archive: bool
    submitted_by: str
    created_at: datetime
    base_plugin_id: str | None
    base_updated_at: datetime | None
    reviewed_by: str | None
    reviewed_at: datetime | None
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class SubmissionPage:
    submissions: tuple[SkillSubmission, ...]
    has_more: bool


@dataclass(frozen=True, slots=True)
class SubmissionLimitExceeded:
    message: str


@dataclass(frozen=True, slots=True)
class SubmissionMissing:
    pass


@dataclass(frozen=True, slots=True)
class SubmissionConflict:
    message: str


@dataclass(frozen=True, slots=True)
class SubmissionReviewed:
    submission: SkillSubmission
    plugin: PublishedPlugin | None
    published_source: Mapping[str, str] | None = None


SubmissionReviewResult: TypeAlias = SubmissionReviewed | SubmissionMissing | SubmissionConflict | PluginPublicationError


class SkillSubmissionStore(Protocol):
    async def submit(
        self, upload: PluginPublication, submitted_by: str, base: PluginPublicationBase | None
    ) -> SkillSubmission | SubmissionLimitExceeded: ...

    async def list(
        self, submitted_by: str | None, status: SubmissionStatus | None, limit: int, offset: int
    ) -> SubmissionPage: ...

    async def get(self, submission_id: str, submitted_by: str | None) -> SkillSubmission | None: ...

    async def get_archive(self, submission_id: str, submitted_by: str | None) -> bytes | None: ...

    async def review(
        self,
        submission_id: str,
        reviewer: str,
        approve: bool,
        reason: str | None,
        *,
        publication_base_url: str | None = None,
    ) -> SubmissionReviewResult: ...
