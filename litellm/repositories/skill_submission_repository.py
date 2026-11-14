import hashlib
import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, Protocol, cast  # noqa: TID251  # Prisma's generated delegates are dynamic
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.models.plugin_upload import (
    PluginPublication,
    PluginPublicationBase,
    PluginPublicationError,
    PluginReview,
    PublishedPlugin,
    uploaded_plugin_source,
)
from litellm.models.skill_submission import (
    SkillProposal,
    SkillScanInput,
    SkillSubmission,
    SubmissionConflict,
    SubmissionLimitExceeded,
    SubmissionMissing,
    SubmissionOperation,
    SubmissionPage,
    SubmissionReviewed,
    SubmissionReviewResult,
    SubmissionStatus,
)
from litellm.repositories.plugin_publication import PluginPublicationTransaction, publish_plugin, verified_archive
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.skill_scanning import SkillScan

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient


class _StoredProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: Mapping[str, object]
    version: str | None
    revision: int = Field(default=0, ge=0)

    def proposal(self, name: str) -> SkillProposal:
        return SkillProposal(
            name, self.version, _DESCRIPTION.validate_python(self.manifest.get("description")), self.manifest
        )


class _PendingUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    storage_bytes: int


class _SubmissionRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    plugin_name: str
    operation: SubmissionOperation
    status: SubmissionStatus
    proposal_json: str
    has_archive: bool
    submitted_by: str
    created_at: datetime
    base_plugin_id: str | None
    base_updated_at: datetime | None
    reviewed_by: str | None
    reviewed_at: datetime | None
    rejection_reason: str | None

    def submission(self) -> SkillSubmission:
        stored: Final = _StoredProposal.model_validate_json(self.proposal_json)
        return SkillSubmission(
            id=self.id,
            plugin_name=self.plugin_name,
            operation=self.operation,
            status=self.status,
            proposal=stored.proposal(self.plugin_name),
            has_archive=self.has_archive,
            submitted_by=self.submitted_by,
            created_at=self.created_at,
            base_plugin_id=self.base_plugin_id,
            base_updated_at=self.base_updated_at,
            reviewed_by=self.reviewed_by,
            reviewed_at=self.reviewed_at,
            rejection_reason=self.rejection_reason,
            revision=stored.revision,
        )


_SCAN_ROWS: Final = TypeAdapter(tuple[Mapping[str, str | None], ...])
_ROWS: Final = TypeAdapter(tuple[_SubmissionRow, ...])
_DESCRIPTION: Final = TypeAdapter[str | None](str | None)
_SOURCE: Final = TypeAdapter(dict[str, str])
_USAGE: Final = TypeAdapter(tuple[_PendingUsage, ...])
_MAX_PENDING_SUBMISSIONS: Final = 20
_MAX_PENDING_BYTES: Final = 50 * 1024 * 1024
_SELECT: Final = """
SELECT id, plugin_name, operation, status, proposal_json,
       archive_content IS NOT NULL AS has_archive, submitted_by, created_at,
       base_plugin_id, base_updated_at, reviewed_by, reviewed_at, rejection_reason
FROM "LiteLLM_SkillSubmission"
"""


class SubmissionTransaction(PluginPublicationTransaction, Protocol):
    @property
    def litellm_skillsubmission(self) -> TableActions["prisma_models.LiteLLM_SkillSubmission"]: ...


class SubmissionDatabase(SubmissionTransaction, Protocol):
    def tx(self) -> AbstractAsyncContextManager[SubmissionTransaction]: ...


def _submission(row: "prisma_models.LiteLLM_SkillSubmission") -> SkillSubmission:
    return _SubmissionRow.model_validate(
        {**row.model_dump(), "has_archive": row.archive_content is not None}
    ).submission()


class SkillSubmissionRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._database = cast(  # cast-ok: Prisma dynamically exposes generated delegates and transactions
            SubmissionDatabase, prisma_client.writer_db
        )

    async def get_scan(self, submission_id: str) -> SkillScan | SubmissionMissing | None:
        rows: Final = _SCAN_ROWS.validate_python(
            await self._database.query_raw(
                'SELECT scan_json FROM "LiteLLM_SkillSubmission" WHERE id = $1', submission_id
            )
        )
        if not rows:
            return SubmissionMissing()
        value: Final = rows[0]["scan_json"]
        return None if value is None else SkillScan.model_validate_json(value)

    async def start_scan(
        self, submission_id: str, scan: SkillScan
    ) -> SkillScanInput | SubmissionMissing | SubmissionConflict:
        async with self._database.tx() as transaction:
            await transaction.query_raw(
                'SELECT id FROM "LiteLLM_SkillSubmission" WHERE id = $1 FOR UPDATE', submission_id
            )
            row: Final = await transaction.litellm_skillsubmission.find_unique(where={"id": submission_id})
            if row is None:
                return SubmissionMissing()
            if row.status != "pending_review":
                return SubmissionConflict("Only pending submissions can be scanned.")
            previous: Final = None if row.scan_json is None else SkillScan.model_validate_json(row.scan_json)
            if previous is not None and previous.at(scan.started_at).status == "running":
                return SubmissionConflict("A scan is already running for this submission.")
            stored: Final = _StoredProposal.model_validate_json(row.proposal_json)
            source: Final = _SOURCE.validate_python(stored.manifest.get("source", {}))
            inherited: Final = (
                await transaction.litellm_claudecodepluginupload.find_unique(where={"plugin_name": row.plugin_name})
                if row.archive_content is None and source.get("source") == "archive"
                else None
            )
            content: Final = (
                row.archive_content.decode()
                if row.archive_content is not None
                else verified_archive(
                    None if inherited is None else inherited.content.decode(), json.dumps(stored.manifest)
                )
            )
            managed_archive: Final = urlsplit(source.get("url", "")).path.endswith(
                f"/claude-code/plugins/{row.plugin_name}/archive/{source.get('sha256', '')}.zip"
            )
            if source.get("source") == "archive" and managed_archive and content is None:
                return SubmissionConflict("The proposal's archive is no longer available. Submit a new ZIP to scan it.")
            started: Final = scan.model_copy(
                update={"archive_sha256": None if content is None else hashlib.sha256(content).hexdigest()}
            )
            await transaction.litellm_skillsubmission.update(
                where={"id": submission_id}, data={"scan_json": started.model_dump_json()}
            )
            return SkillScanInput(started, source, content)

    async def finish_scan(self, submission_id: str, scan: SkillScan) -> None:
        async with self._database.tx() as transaction:
            rows: Final = _SCAN_ROWS.validate_python(
                await transaction.query_raw(
                    'SELECT scan_json FROM "LiteLLM_SkillSubmission" WHERE id = $1 FOR UPDATE', submission_id
                )
            )
            value: Final = rows[0]["scan_json"] if rows else None
            if value is None:
                return
            current: Final = SkillScan.model_validate_json(value)
            if current.id != scan.id or current.status != "running":
                return
            result: Final = (
                current.at(scan.completed_at)
                if scan.completed_at is not None and scan.completed_at >= current.expires_at
                else scan
            )
            await transaction.litellm_skillsubmission.update(
                where={"id": submission_id}, data={"scan_json": result.model_dump_json()}
            )

    async def submit(
        self, upload: PluginPublication, submitted_by: str, base: PluginPublicationBase | None
    ) -> SkillSubmission | SubmissionLimitExceeded:
        from prisma import Base64

        proposal_json: Final = _StoredProposal(manifest=dict(upload.manifest), version=upload.version).model_dump_json()
        incoming_bytes: Final = len(proposal_json.encode("utf-8")) + len(upload.content or b"")
        async with self._database.tx() as transaction:
            await transaction.query_raw(
                "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtext('litellm.skill_submissions'), hashtext($1))",
                submitted_by,
            )
            usage: Final = _USAGE.validate_python(
                await transaction.query_raw(
                    """SELECT COUNT(*) AS count,
                              COALESCE(SUM(octet_length(proposal_json)
                                  + COALESCE(octet_length(archive_content), 0)), 0) AS storage_bytes
                       FROM "LiteLLM_SkillSubmission" WHERE submitted_by = $1 AND status = 'pending_review'""",
                    submitted_by,
                )
            )[0]
            if usage.count >= _MAX_PENDING_SUBMISSIONS:
                return SubmissionLimitExceeded(
                    "You already have 20 pending skill submissions. "
                    "Wait for an admin to review one before submitting another."
                )
            if usage.storage_bytes + incoming_bytes > _MAX_PENDING_BYTES:
                return SubmissionLimitExceeded(
                    "Pending skill submissions cannot exceed 50 MiB of stored metadata and archives per user. "
                    "Wait for an admin review or submit less data."
                )
            row: Final = await transaction.litellm_skillsubmission.create(
                data={
                    "plugin_name": upload.name,
                    "operation": "create" if base is None else "update",
                    "proposal_json": proposal_json,
                    "archive_content": None if upload.content is None else Base64.encode(upload.content),
                    "submitted_by": submitted_by,
                    "base_plugin_id": None if base is None else base.id,
                    "base_updated_at": None if base is None else base.updated_at,
                }
            )
            return _submission(row)

    async def list(
        self, submitted_by: str | None, status: SubmissionStatus | None, limit: int, offset: int
    ) -> SubmissionPage:
        rows: Final = _ROWS.validate_python(
            await self._database.query_raw(
                _SELECT
                + """WHERE ($1::text IS NULL OR submitted_by = $1)
                     AND ($2::text IS NULL OR status = $2)
                     ORDER BY created_at DESC, id DESC LIMIT $3 OFFSET $4""",
                submitted_by,
                status,
                limit + 1,
                offset,
            )
        )
        return SubmissionPage(tuple(row.submission() for row in rows[:limit]), has_more=len(rows) > limit)

    async def get(self, submission_id: str, submitted_by: str | None) -> SkillSubmission | None:
        rows: Final = _ROWS.validate_python(
            await self._database.query_raw(
                _SELECT + "WHERE id = $1 AND ($2::text IS NULL OR submitted_by = $2)",
                submission_id,
                submitted_by,
            )
        )
        return None if not rows else rows[0].submission()

    async def get_archive(self, submission_id: str, submitted_by: str | None) -> bytes | None:
        row: Final = await self._database.litellm_skillsubmission.find_first(
            where={"id": submission_id, **({"submitted_by": submitted_by} if submitted_by is not None else {})}
        )
        return None if row is None or row.archive_content is None else row.archive_content.decode()

    async def edit(
        self, submission_id: str, owner: str, revision: int, upload: PluginPublication
    ) -> SkillSubmission | SubmissionMissing | SubmissionConflict | SubmissionLimitExceeded:
        from prisma import Base64

        async with self._database.tx() as transaction:
            await transaction.query_raw(
                "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtext('litellm.skill_submissions'), hashtext($1))",
                owner,
            )
            row: Final = await _owned_pending(transaction, submission_id, owner, revision)
            if isinstance(row, (SubmissionMissing, SubmissionConflict)):
                return row
            if upload.name != row.plugin_name:
                return SubmissionConflict("A submission's skill name cannot change.")
            proposal_json: Final = _StoredProposal(
                manifest=upload.manifest, version=upload.version, revision=revision + 1
            ).model_dump_json()
            archive: Final = row.archive_content if upload.content is None else Base64.encode(upload.content)
            usage: Final = _USAGE.validate_python(
                await transaction.query_raw(
                    """SELECT COUNT(*) AS count,
                              COALESCE(SUM(octet_length(proposal_json)
                                  + COALESCE(octet_length(archive_content), 0)), 0) AS storage_bytes
                       FROM "LiteLLM_SkillSubmission" WHERE submitted_by = $1 AND status = 'pending_review'""",
                    owner,
                )
            )[0]
            previous_bytes: Final = len(row.proposal_json.encode()) + (
                0 if row.archive_content is None else len(row.archive_content.decode())
            )
            incoming_bytes: Final = len(proposal_json.encode()) + (0 if archive is None else len(archive.decode()))
            if usage.storage_bytes - previous_bytes + incoming_bytes > _MAX_PENDING_BYTES:
                return SubmissionLimitExceeded(
                    "Pending skill submissions cannot exceed 50 MiB of stored metadata and archives per user."
                )
            updated: Final = await transaction.litellm_skillsubmission.update(
                where={"id": submission_id},
                data={"proposal_json": proposal_json, "archive_content": archive, "scan_json": None},
            )
            return SubmissionMissing() if updated is None else _submission(updated)

    async def withdraw(
        self, submission_id: str, owner: str, revision: int
    ) -> SkillSubmission | SubmissionMissing | SubmissionConflict:
        async with self._database.tx() as transaction:
            row: Final = await _owned_pending(transaction, submission_id, owner, revision)
            if isinstance(row, (SubmissionMissing, SubmissionConflict)):
                return row
            receipt: Final = _StoredProposal(
                manifest={"name": row.plugin_name, "source": {}},
                version=None,
                revision=revision,
            )
            updated: Final = await transaction.litellm_skillsubmission.update(
                where={"id": submission_id},
                data={
                    "status": "withdrawn",
                    "proposal_json": receipt.model_dump_json(),
                    "archive_content": None,
                    "scan_json": None,
                },
            )
            return SubmissionMissing() if updated is None else _submission(updated)

    async def review(
        self,
        submission_id: str,
        reviewer: str,
        approve: bool,
        reason: str | None,
        *,
        publication_base_url: str | None = None,
        expected_revision: int | None = None,
    ) -> SubmissionReviewResult:
        from prisma.errors import UniqueViolationError

        try:
            async with self._database.tx() as transaction:
                await transaction.query_raw(
                    'SELECT id FROM "LiteLLM_SkillSubmission" WHERE id = $1 FOR UPDATE', submission_id
                )
                row: Final = await transaction.litellm_skillsubmission.find_unique(where={"id": submission_id})
                if row is None:
                    return SubmissionMissing()
                submission: Final = _submission(row)
                if submission.status != "pending_review":
                    return SubmissionConflict("This submission is no longer pending review.")
                if submission.revision != (expected_revision if expected_revision is not None else 0):
                    return SubmissionConflict("This submission changed. Reload it before reviewing.")
                content: Final = None if row.archive_content is None else row.archive_content.decode()
                if approve and content is not None and not publication_base_url:
                    return SubmissionConflict("Uploaded skill approval requires a trusted publication URL.")
                source: Final = (
                    uploaded_plugin_source(publication_base_url, row.plugin_name, hashlib.sha256(content).hexdigest())
                    if approve and content is not None and publication_base_url
                    else _SOURCE.validate_python(submission.proposal.manifest.get("source"))
                )
                publication: Final = (
                    await _publish_submission(transaction, row, source, content, reviewer) if approve else None
                )
                if isinstance(publication, PluginPublicationError):
                    return publication
                reviewed_at: Final = datetime.now(timezone.utc)
                status: Final[SubmissionStatus] = "approved" if approve else "rejected"
                await transaction.litellm_skillsubmission.update(
                    where={"id": submission_id},
                    data={
                        "status": status,
                        "reviewed_by": reviewer,
                        "reviewed_at": reviewed_at,
                        "rejection_reason": None if approve else reason,
                    },
                )
                return SubmissionReviewed(
                    submission=replace(
                        submission,
                        status=status,
                        reviewed_by=reviewer,
                        reviewed_at=reviewed_at,
                        rejection_reason=None if approve else reason,
                    ),
                    plugin=publication,
                    published_source=source if approve else None,
                )
        except UniqueViolationError:
            return SubmissionConflict("A skill with this name is already published. Submit a new update proposal.")


async def _owned_pending(
    transaction: SubmissionTransaction, submission_id: str, owner: str, revision: int
) -> "prisma_models.LiteLLM_SkillSubmission | SubmissionMissing | SubmissionConflict":
    await transaction.query_raw('SELECT id FROM "LiteLLM_SkillSubmission" WHERE id = $1 FOR UPDATE', submission_id)
    row: Final = await transaction.litellm_skillsubmission.find_unique(where={"id": submission_id})
    if row is None or not owner or row.submitted_by != owner:
        return SubmissionMissing()
    if row.status != "pending_review":
        return SubmissionConflict("Only pending submissions can be edited or withdrawn.")
    if _StoredProposal.model_validate_json(row.proposal_json).revision != revision:
        return SubmissionConflict("This submission changed. Reload it before trying again.")
    return row


async def _publish_submission(
    transaction: PluginPublicationTransaction,
    row: "prisma_models.LiteLLM_SkillSubmission",
    source: Mapping[str, str],
    content: bytes | None,
    reviewer: str,
) -> PublishedPlugin | PluginPublicationError:
    stored: Final = _StoredProposal.model_validate_json(row.proposal_json)
    proposal: Final = stored.proposal(row.plugin_name)
    base: Final = (
        None
        if row.operation == "create"
        else PluginPublicationBase(id=row.base_plugin_id or "", updated_at=row.base_updated_at)
    )
    return await publish_plugin(
        transaction,
        PluginPublication(
            name=row.plugin_name,
            version=stored.version,
            description=proposal.description,
            manifest={**stored.manifest, "source": dict(source)},
            content=content,
            created_by=row.submitted_by,
            review=PluginReview(reviewer, scan_passed_for_archive(row.scan_json, source.get("sha256"))),
        ),
        base,
    )


def scan_passed_for_archive(scan_json: str | None, digest: str | None) -> bool:
    if not scan_json or not digest:
        return False
    try:
        scan: Final = SkillScan.model_validate_json(scan_json)
    except ValidationError:
        return False
    return (
        scan.status == "completed"
        and scan.archive_sha256 == digest
        and scan.report is not None
        and scan.report.complete
        and scan.report.risk_score == 0
        and not scan.report.findings
    )
