import asyncio
import hashlib
import io
import os
import uuid
import zipfile
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

import psycopg
import pytest
import pytest_asyncio
from pydantic import JsonValue, TypeAdapter

from litellm.models.plugin_upload import (
    PluginPublication,
    PluginPublicationBase,
    PluginPublicationError,
    PluginUploadStore,
    PublishedPlugin,
)
from litellm.models.skill_submission import (
    SkillSubmission,
    SkillSubmissionStore,
    SubmissionConflict,
    SubmissionLimitExceeded,
    SubmissionReviewed,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.plugin_upload_repository import PluginUploadRepository
from litellm.repositories.skill_submission_repository import SkillSubmissionRepository
from tests.integration._support.client import eventually
from tests.integration._support.database import read_rows, write_rows


@dataclass(frozen=True, slots=True)
class PendingUpdate:
    publications: PluginUploadStore
    submissions: SkillSubmissionStore
    original: PluginPublication
    proposal: PluginPublication
    base: PublishedPlugin
    submission: SkillSubmission


def _publication(name: str, description: str, base_url: str = "https://skills.example.test") -> PluginPublication:
    buffer: Final = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SKILL.md", f"---\nname: {name}\ndescription: {description}\n---\n{description}\n")
    content: Final = buffer.getvalue()
    digest: Final = hashlib.sha256(content).hexdigest()
    return PluginPublication(
        name=name,
        version=None,
        description=description,
        manifest={
            "name": name,
            "description": description,
            "extension": {"retained": True},
            "source": {
                "source": "archive",
                "url": f"{base_url}/claude-code/plugins/{name}/archive/{digest}.zip",
                "sha256": digest,
            },
        },
        content=content,
        created_by=f"{name}-owner",
    )


def _published_rows(name: str) -> Sequence[Mapping[str, JsonValue]]:
    return read_rows(
        """SELECT 'catalog' AS storage, to_jsonb(p) AS value
           FROM "LiteLLM_ClaudeCodePluginTable" p WHERE name=%s
           UNION ALL
           SELECT 'archive', to_jsonb(a) FROM "LiteLLM_ClaudeCodePluginUpload" a WHERE plugin_name=%s
           UNION ALL
           SELECT 'history', to_jsonb(v) FROM "LiteLLM_ClaudeCodePluginVersion" v
           JOIN "LiteLLM_ClaudeCodePluginTable" p ON p.id=v.plugin_id WHERE p.name=%s
           ORDER BY storage, value""",
        (name, name, name),
    )


@pytest_asyncio.fixture(loop_scope="function")
async def pending_update() -> AsyncIterator[PendingUpdate]:
    name: Final = f"integration-submission-{uuid.uuid4().hex}"
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    await client.connect()
    try:
        publications: Final = PluginUploadRepository(client)
        submissions: Final = SkillSubmissionRepository(client)
        original: Final = _publication(name, "Published instructions")
        base: Final = await publications.publish(original)
        assert isinstance(base, PublishedPlugin)
        proposal: Final = _publication(name, "Proposed instructions", "https://member.example.test")
        submission: Final = await submissions.submit(
            proposal, f"{name}-owner", PluginPublicationBase(base.id, base.updated_at)
        )
        assert isinstance(submission, SkillSubmission)
        yield PendingUpdate(publications, submissions, original, proposal, base, submission)
    finally:
        try:
            write_rows('DELETE FROM "LiteLLM_SkillSubmission" WHERE plugin_name=%s', (name,))
            write_rows('DELETE FROM "LiteLLM_ClaudeCodePluginTable" WHERE name=%s', (name,))
        finally:
            await client.disconnect()


async def test_approval_commits_version_archive_history_and_review(pending_update: PendingUpdate) -> None:
    case: Final = pending_update
    before: Final = await case.publications.get_metadata(case.original.name)
    assert before is not None
    assert await case.publications.get_content(before) == case.original.content
    assert case.submission.proposal.version is None
    reviewed: Final = await case.submissions.review(
        case.submission.id, "integration-reviewer", True, None, publication_base_url="https://skills.example.test"
    )
    assert isinstance(reviewed, SubmissionReviewed)
    assert reviewed.plugin is not None and reviewed.plugin.version == "1.0.1"
    stored: Final = await case.submissions.get(case.submission.id, None)
    assert stored is not None
    assert (stored.status, stored.reviewed_by, stored.rejection_reason) == (
        "approved",
        "integration-reviewer",
        None,
    )
    assert stored.reviewed_at is not None
    assert stored.proposal == case.submission.proposal
    assert await case.submissions.get_archive(stored.id, None) == case.proposal.content
    current: Final = await case.publications.get_plugin(case.original.name)
    assert current is not None
    published_manifest: Final = {
        **case.proposal.manifest,
        "_litellm_review": {"approved_by": "integration-reviewer", "security_scan_passed": False},
        "_litellm_published_by": "integration-reviewer",
        "source": {
            **TypeAdapter(dict[str, str]).validate_python(case.submission.proposal.manifest["source"]),
            "url": f"https://skills.example.test/claude-code/plugins/{case.original.name}/archive/"
            f"{hashlib.sha256(case.proposal.content or b'').hexdigest()}.zip",
        },
    }
    assert (current.id, current.version, current.description, current.manifest) == (
        case.base.id,
        reviewed.plugin.version,
        case.proposal.description,
        {**published_manifest, "version": reviewed.plugin.version},
    )
    metadata: Final = await case.publications.get_metadata(case.original.name)
    assert metadata is not None
    assert metadata.created_by == before.created_by == case.original.created_by
    assert metadata.created_at == before.created_at
    assert metadata.updated_at is not None and before.updated_at is not None
    assert metadata.updated_at > before.updated_at
    assert await case.publications.get_content(metadata) == case.proposal.content
    history: Final = await case.publications.versions(metadata, 10, 0)
    assert tuple((version.version, version.description, version.archive_available) for version in history) == (
        ("1.0.1", case.proposal.description, True),
        ("1.0.0", case.original.description, True),
    )
    assert history[0].published_by == "integration-reviewer"
    assert history[1].published_by == case.original.created_by
    for version, publication, manifest in zip(
        history, (case.proposal, case.original), (published_manifest, {**case.original.manifest, "_litellm_published_by": case.original.created_by}), strict=True
    ):
        assert version.manifest == {**manifest, "version": version.version}
        assert await case.publications.version_content(metadata, version.id) == publication.content


async def test_simultaneous_approve_reject_has_one_winner(pending_update: PendingUpdate) -> None:
    case: Final = pending_update
    before: Final = _published_rows(case.original.name)
    with psycopg.connect(os.environ["DATABASE_URL"]) as blocker:
        blocker.execute('SELECT id FROM "LiteLLM_SkillSubmission" WHERE id=%s FOR UPDATE', (case.submission.id,))
        blocker_pid: Final = str(blocker.info.backend_pid)
        reviews: Final = (
            asyncio.create_task(
                case.submissions.review(
                    case.submission.id,
                    "integration-approver",
                    True,
                    None,
                    publication_base_url="https://skills.example.test",
                )
            ),
            asyncio.create_task(
                case.submissions.review(case.submission.id, "integration-rejector", False, "Needs revision")
            ),
        )
        try:
            await asyncio.to_thread(
                eventually,
                lambda: read_rows(
                    """WITH RECURSIVE blocked AS (
                           SELECT %s::integer AS pid
                           UNION
                           SELECT a.pid FROM pg_stat_activity a
                           JOIN blocked b ON b.pid = ANY(pg_blocking_pids(a.pid))
                       ) SELECT pid FROM blocked WHERE pid<>%s::integer""",
                    (blocker_pid, blocker_pid),
                ),
                lambda rows: len(rows) == 2,
                seconds=3,
            )
        finally:
            blocker.rollback()
            results: Final = await asyncio.gather(*reviews)
    winners: Final = tuple(result for result in results if isinstance(result, SubmissionReviewed))
    assert len(winners) == 1
    assert sum(isinstance(result, SubmissionConflict) for result in results) == 1
    stored: Final = await case.submissions.get(case.submission.id, None)
    assert stored is not None and stored.reviewed_at is not None
    winner: Final = winners[0]
    assert (stored.status, stored.reviewed_by, stored.rejection_reason) == (
        winner.submission.status,
        winner.submission.reviewed_by,
        winner.submission.rejection_reason,
    )
    if isinstance(results[0], SubmissionConflict):
        assert (stored.status, stored.reviewed_by, stored.rejection_reason) == (
            "rejected",
            "integration-rejector",
            "Needs revision",
        )
        assert _published_rows(case.original.name) == before
        return
    assert (stored.status, stored.reviewed_by, stored.rejection_reason) == (
        "approved",
        "integration-approver",
        None,
    )
    metadata: Final = await case.publications.get_metadata(case.original.name)
    assert metadata is not None
    assert await case.publications.get_content(metadata) == case.proposal.content
    assert tuple(version.version for version in await case.publications.versions(metadata, 10, 0)) == (
        "1.0.1",
        "1.0.0",
    )


async def test_publication_failure_rolls_back_and_leaves_submission_pending(pending_update: PendingUpdate) -> None:
    case: Final = pending_update
    collision_id: Final = str(uuid.uuid4())
    write_rows(
        """INSERT INTO "LiteLLM_ClaudeCodePluginVersion" (id, plugin_id, version)
           VALUES (%s, %s, '1.0.1')""",
        (collision_id, case.base.id),
    )
    before: Final = _published_rows(case.original.name)
    failed: Final = await case.submissions.review(
        case.submission.id, "integration-reviewer", True, None, publication_base_url="https://skills.example.test"
    )
    assert isinstance(failed, SubmissionConflict)
    assert _published_rows(case.original.name) == before
    assert await case.submissions.get(case.submission.id, None) == case.submission
    assert await case.submissions.get_archive(case.submission.id, None) == case.proposal.content
    write_rows('DELETE FROM "LiteLLM_ClaudeCodePluginVersion" WHERE id=%s', (collision_id,))
    retried: Final = await case.submissions.review(
        case.submission.id, "integration-reviewer", True, None, publication_base_url="https://skills.example.test"
    )
    assert isinstance(retried, SubmissionReviewed)
    assert retried.plugin is not None and retried.plugin.version == "1.0.1"
    stored: Final = await case.submissions.get(case.submission.id, None)
    assert stored is not None and stored.status == "approved"


@pytest.mark.parametrize("change", ("update", "recreate"))
async def test_stale_target_cannot_publish(
    pending_update: PendingUpdate, change: Literal["update", "recreate"]
) -> None:
    case: Final = pending_update
    intervening: Final = _publication(case.original.name, "Intervening publication")
    if change == "recreate":
        write_rows('DELETE FROM "LiteLLM_ClaudeCodePluginTable" WHERE name=%s', (case.original.name,))
    replacement: Final = await case.publications.publish(
        intervening,
        PluginPublicationBase(case.base.id, case.base.updated_at) if change == "update" else None,
    )
    assert isinstance(replacement, PublishedPlugin)
    before: Final = _published_rows(case.original.name)
    reviewed: Final = await case.submissions.review(
        case.submission.id, "integration-reviewer", True, None, publication_base_url="https://skills.example.test"
    )
    assert isinstance(reviewed, PluginPublicationError) and reviewed.kind == "conflict"
    assert _published_rows(case.original.name) == before
    assert await case.submissions.get(case.submission.id, None) == case.submission


async def test_concurrent_submissions_share_user_limit_and_review_releases_capacity(
    pending_update: PendingUpdate,
) -> None:
    case: Final = pending_update
    owner: Final = case.submission.submitted_by
    base: Final = PluginPublicationBase(case.base.id, case.base.updated_at)
    for _ in range(18):
        assert isinstance(await case.submissions.submit(case.proposal, owner, base), SkillSubmission)
    peer: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    await peer.connect()
    try:
        with psycopg.connect(os.environ["DATABASE_URL"]) as blocker:
            blocker.execute(
                "SELECT pg_advisory_xact_lock(hashtext('litellm.skill_submissions'), hashtext(%s))", (owner,)
            )
            blocker_pid: Final = str(blocker.info.backend_pid)
            requests: Final = (
                asyncio.create_task(case.submissions.submit(case.proposal, owner, base)),
                asyncio.create_task(SkillSubmissionRepository(peer).submit(case.proposal, owner, base)),
            )
            try:
                await asyncio.to_thread(
                    eventually,
                    lambda: read_rows(
                        "SELECT pid FROM pg_stat_activity WHERE %s::integer = ANY(pg_blocking_pids(pid))",
                        (blocker_pid,),
                    ),
                    lambda rows: len(rows) == 2,
                    seconds=3,
                )
            finally:
                blocker.rollback()
                results: Final = await asyncio.gather(*requests)
    finally:
        await peer.disconnect()
    assert sum(isinstance(result, SkillSubmission) for result in results) == 1
    assert sum(isinstance(result, SubmissionLimitExceeded) for result in results) == 1
    full: Final = await case.submissions.list(owner, "pending_review", 20, 0)
    assert (len(full.submissions), full.has_more) == (20, False)
    other: Final = await case.submissions.submit(case.proposal, f"{owner}-other", base)
    assert isinstance(other, SkillSubmission)
    rejected: Final = await case.submissions.review(
        case.submission.id, "integration-reviewer", False, "Review complete"
    )
    assert isinstance(rejected, SubmissionReviewed)
    assert await case.submissions.get_archive(case.submission.id, owner) == case.proposal.content
    replacement: Final = await case.submissions.submit(case.proposal, owner, base)
    assert isinstance(replacement, SkillSubmission)
    after: Final = await case.submissions.list(owner, "pending_review", 20, 0)
    assert len(after.submissions) == 20
    assert all(row.id != case.submission.id and row.submitted_by == owner for row in after.submissions)


async def test_submission_pages_filter_owner_and_status_before_limit(pending_update: PendingUpdate) -> None:
    case: Final = pending_update
    owner: Final = case.submission.submitted_by
    other: Final = f"{owner}-other"
    generated: Final = await asyncio.gather(
        *(
            case.submissions.submit(case.proposal, identity, PluginPublicationBase(case.base.id, case.base.updated_at))
            for identity in (owner, other, owner, other)
        )
    )
    submitted: Final = tuple(result for result in generated if isinstance(result, SkillSubmission))
    assert len(submitted) == 4
    rejected: Final = await case.submissions.review(submitted[0].id, "integration-reviewer", False, None)
    assert isinstance(rejected, SubmissionReviewed)
    write_rows(
        """UPDATE "LiteLLM_SkillSubmission" SET created_at=TIMESTAMP '9999-01-01'
           WHERE plugin_name=%s""",
        (case.original.name,),
    )
    own_pending: Final = tuple(sorted((case.submission.id, submitted[2].id), reverse=True))
    first: Final = await case.submissions.list(owner, "pending_review", 1, 0)
    second: Final = await case.submissions.list(owner, "pending_review", 1, 1)
    empty: Final = await case.submissions.list(owner, "pending_review", 1, 2)
    assert (tuple(row.id for row in first.submissions), first.has_more) == (own_pending[:1], True)
    assert (tuple(row.id for row in second.submissions), second.has_more) == (own_pending[1:], False)
    assert (empty.submissions, empty.has_more) == ((), False)
    reviewed: Final = await case.submissions.list(owner, "rejected", 20, 0)
    assert (tuple(row.id for row in reviewed.submissions), reviewed.has_more) == ((submitted[0].id,), False)
    own_all: Final = await case.submissions.list(owner, None, 20, 0)
    assert tuple(row.id for row in own_all.submissions) == tuple(sorted((*own_pending, submitted[0].id), reverse=True))
    other_page: Final = await case.submissions.list(other, None, 20, 0)
    assert tuple(row.id for row in other_page.submissions) == tuple(
        sorted((submitted[1].id, submitted[3].id), reverse=True)
    )
    global_pending: Final = await case.submissions.list(None, "pending_review", 3, 0)
    all_pending: Final = tuple(sorted((*own_pending, submitted[1].id, submitted[3].id), reverse=True))
    assert (tuple(row.id for row in global_pending.submissions), global_pending.has_more) == (all_pending[:3], True)


@pytest.mark.parametrize("completion_seconds", (13, 21), ids=("before-deadline", "at-deadline"))
async def test_scan_claim_retry_and_late_result_preserve_proposal(
    pending_update: PendingUpdate, completion_seconds: int,
) -> None:
    from datetime import datetime, timedelta, timezone

    from litellm.models.skill_submission import SkillScanInput
    from litellm.types.proxy.skill_scanning import SkillScan, SkillScanReport

    case: Final = pending_update
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first: Final = SkillScan(
        id="first", status="running", requested_by="reviewer", started_at=now,
        expires_at=now + timedelta(seconds=10),
    )
    started: Final = await case.submissions.start_scan(case.submission.id, first)
    assert isinstance(started, SkillScanInput)
    assert started.content == case.proposal.content
    assert started.scan.archive_sha256 == hashlib.sha256(case.proposal.content or b"").hexdigest()
    duplicate: Final = await case.submissions.start_scan(case.submission.id, first.model_copy(update={"id": "duplicate"}))
    assert isinstance(duplicate, SubmissionConflict)
    second: Final = first.model_copy(update={
        "id": "second", "started_at": now + timedelta(seconds=11), "expires_at": now + timedelta(seconds=21),
    })
    retry: Final = await case.submissions.start_scan(case.submission.id, second)
    assert isinstance(retry, SkillScanInput)
    report: Final = SkillScanReport(
        scanner_version="test-version", complete=False, scan_mode="static-only", risk_score=90, findings=(),
    )
    await case.submissions.finish_scan(case.submission.id, started.scan.model_copy(update={
        "status": "completed", "report": report, "completed_at": now + timedelta(seconds=12),
    }))
    assert await case.submissions.get_scan(case.submission.id) == retry.scan
    completed: Final = retry.scan.model_copy(update={
        "status": "completed", "report": report, "completed_at": now + timedelta(seconds=completion_seconds),
    })
    expected: Final = completed if completion_seconds < 21 else retry.scan.at(now + timedelta(seconds=21))
    await case.submissions.finish_scan(case.submission.id, completed)
    assert await case.submissions.get_scan(case.submission.id) == expected
    assert await case.submissions.get(case.submission.id, None) == case.submission
    approved: Final = await case.submissions.review(case.submission.id, "reviewer", True, None, publication_base_url="https://gateway.test")
    assert isinstance(approved, SubmissionReviewed)
    assert await case.submissions.get_scan(case.submission.id) == expected
    denied: Final = await case.submissions.start_scan(case.submission.id, second)
    assert isinstance(denied, SubmissionConflict)


@pytest.mark.parametrize("available, remote", ((True, False), (False, False), (False, True)))
async def test_metadata_scan_requires_the_matching_managed_archive(
    pending_update: PendingUpdate, available: bool, remote: bool
) -> None:
    from datetime import datetime, timedelta, timezone

    from litellm.models.skill_submission import SkillScanInput
    from litellm.types.proxy.skill_scanning import SkillScan

    case: Final = pending_update
    submission: Final = await case.submissions.submit(
        replace(case.original, content=None, manifest={
            **case.original.manifest,
            **({"source": {"source": "archive", "url": "https://example.org/skill.zip"}} if remote else {}),
        }), "member", PluginPublicationBase(case.base.id, case.base.updated_at)
    )
    if not available:
        write_rows('DELETE FROM "LiteLLM_ClaudeCodePluginUpload" WHERE plugin_name=%s', (case.original.name,))
    now: Final = datetime.now(timezone.utc)
    result: Final = await case.submissions.start_scan(submission.id, SkillScan(
        id="metadata-scan", status="running", requested_by="admin", started_at=now,
        expires_at=now + timedelta(seconds=10),
    ))
    if available or remote:
        assert isinstance(result, SkillScanInput)
        assert result.content == (None if remote else case.original.content)
        if remote:
            assert result.source == {"source": "archive", "url": "https://example.org/skill.zip"}
            assert result.scan.archive_sha256 is None
    else:
        assert isinstance(result, SubmissionConflict)
        assert await case.submissions.get_scan(submission.id) is None
