import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma import Base64
from prisma.models import (
    LiteLLM_ClaudeCodePluginTable,
    LiteLLM_ClaudeCodePluginUpload,
)

from litellm.models.plugin_upload import (
    PluginPublication,
    PluginPublicationBase,
    PluginPublicationError,
    PluginReview,
    PublishedPlugin,
)
from litellm.repositories.plugin_publication import (
    PluginPublicationTransaction,
    publish_plugin,
    resolve_plugin_version,
)


@pytest.mark.parametrize("requested,current,updating,expected", [
    (None, None, False, "1.0.0"),
    (None, "1.0.0", True, "1.0.1"),
    (None, "1.2.3-rc.1+build", True, "1.2.4"),
    ("2.0.0", "1.9.9", True, "2.0.0"),
    ("1.0.0-rc.10", "1.0.0-rc.9", True, "1.0.0-rc.10"),
    ("1.0.0", "1.0.0-rc.1", True, "1.0.0"),
    ("1.0.0", None, True, "1.0.0"),
    ("2.0.0", "legacy", True, "2.0.0"),
])
def test_published_version_resolution(
    requested: str | None, current: str | None, updating: bool, expected: str,
) -> None:
    assert resolve_plugin_version(requested, current, updating=updating) == expected


@pytest.mark.parametrize("requested,current,kind", [
    ("1.0.0", "1.0.0", "invalid_version"),
    ("0.9.9", "1.0.0", "invalid_version"),
    ("1.0.0+new", "1.0.0+old", "invalid_version"),
    ("1.0.0-rc.1", "1.0.0", "invalid_version"),
    ("1.0.0-rc.9", "1.0.0-rc.10", "invalid_version"),
    ("1.2", "1.0.0", "invalid_version"),
    (None, None, "conflict"),
    (None, "legacy", "conflict"),
])
def test_invalid_or_non_increasing_versions_cannot_publish(
    requested: str | None, current: str | None, kind: str,
) -> None:
    result: Final = resolve_plugin_version(requested, current, updating=True)
    assert isinstance(result, PluginPublicationError)
    assert result.kind == kind


@pytest.fixture
def publication_tx() -> MagicMock:
    content: Final = b"original uploaded bytes"
    tx: Final = MagicMock()
    tx.query_raw = AsyncMock(return_value=())
    tx.litellm_claudecodeplugintable.find_unique = AsyncMock(return_value=LiteLLM_ClaudeCodePluginTable(
        id="skill-id", name="skill", version="1.0.0", enabled=False, anonymous_download_count=0,
        updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        manifest_json=json.dumps({"source": {"source": "archive", "sha256": hashlib.sha256(content).hexdigest()}}),
    ))
    tx.litellm_claudecodepluginupload.find_unique = AsyncMock(return_value=LiteLLM_ClaudeCodePluginUpload(
        plugin_name="skill", content=Base64.encode(content),
    ))
    tx.litellm_claudecodepluginversion.count = AsyncMock(return_value=0)
    for table in (tx.litellm_claudecodeplugintable, tx.litellm_claudecodepluginupload, tx.litellm_claudecodepluginversion):
        for operation in ("create", "update", "upsert", "delete_many"):
            setattr(table, operation, AsyncMock())
    return tx


@pytest.mark.parametrize("failure,kind", [
    ("duplicate", "conflict"), ("missing", "not_found"), ("invalid", "invalid_version"),
    ("digest", "conflict"), ("identity", "conflict"), ("timestamp", "conflict"),
])
@pytest.mark.asyncio
async def test_rejected_publication_never_writes_catalog_archive_or_history(
    publication_tx: MagicMock, failure: str, kind: str,
) -> None:
    current: Final = LiteLLM_ClaudeCodePluginTable.model_validate(
        publication_tx.litellm_claudecodeplugintable.find_unique.return_value,
    )
    publication_tx.litellm_claudecodeplugintable.find_unique.return_value = None if failure == "missing" else current
    expected: Final = None if failure == "duplicate" else PluginPublicationBase(
        "old-id" if failure == "identity" else current.id,
        current.updated_at - timedelta(seconds=1) if failure == "timestamp" else current.updated_at,
    )
    result: Final = await publish_plugin(
        cast(PluginPublicationTransaction, publication_tx),
        PluginPublication("skill", "0.0.1" if failure == "invalid" else None, "rejected",
                          json.loads(current.manifest_json), "admin", b"mismatched" if failure == "digest" else None),
        expected,
    )
    assert isinstance(result, PluginPublicationError) and result.kind == kind
    for table in (publication_tx.litellm_claudecodeplugintable, publication_tx.litellm_claudecodepluginupload,
                  publication_tx.litellm_claudecodepluginversion):
        for operation in ("create", "update", "upsert", "delete_many"):
            getattr(table, operation).assert_not_awaited()


@pytest.mark.parametrize("valid_archive", (True, False))
@pytest.mark.asyncio
async def test_first_legacy_update_preserves_its_original_snapshot(
    publication_tx: MagicMock, valid_archive: bool,
) -> None:
    current: Final = LiteLLM_ClaudeCodePluginTable.model_validate(
        publication_tx.litellm_claudecodeplugintable.find_unique.return_value,
    ).model_copy(update={"version": None})
    publication_tx.litellm_claudecodeplugintable.find_unique.return_value = current
    content: Final = b"original uploaded bytes" if valid_archive else b"corrupt legacy bytes"
    publication_tx.litellm_claudecodepluginupload.find_unique.return_value = LiteLLM_ClaudeCodePluginUpload(
        plugin_name="skill", content=Base64.encode(content),
    )

    async def update(where: Mapping[str, str], data: Mapping[str, object]) -> LiteLLM_ClaudeCodePluginTable:
        return current.model_copy(update=dict(data))

    publication_tx.litellm_claudecodeplugintable.update.side_effect = update
    source: Final = {
        "source": "archive", "url": "https://external.example.test/skill.zip",
        "sha256": hashlib.sha256(b"original uploaded bytes").hexdigest(),
    }
    result: Final = await publish_plugin(
        cast(PluginPublicationTransaction, publication_tx),
        PluginPublication("skill", "1.0.0", "replacement", {"source": source}, "admin"),
        PluginPublicationBase(current.id, current.updated_at),
    )
    assert isinstance(result, PublishedPlugin) and result.version == "1.0.0" and result.enabled is False
    original, replacement = publication_tx.litellm_claudecodepluginversion.create.call_args_list
    assert original.kwargs["data"] == {
        "plugin_id": current.id, "version": None, "description": current.description,
        "manifest_json": current.manifest_json, "published_at": current.updated_at, "published_by": None,
        "content": Base64.encode(content) if valid_archive else None, "archive_available": valid_archive,
    }
    assert replacement.kwargs["data"] == {
        "plugin_id": current.id, "version": result.version, "description": "replacement",
        "manifest_json": json.dumps({"source": source, "version": result.version, "_litellm_published_by": "admin"}),
        "published_at": result.updated_at, "published_by": "admin", "content": None, "archive_available": False,
    }


@pytest.mark.parametrize("same_source", (True, False))
@pytest.mark.parametrize("review", [None, PluginReview("reviewer@example.test", False)])
@pytest.mark.asyncio
async def test_publication_preserves_only_trusted_review_for_unchanged_source(
    publication_tx: MagicMock, review: PluginReview | None, same_source: bool,
) -> None:
    original = publication_tx.litellm_claudecodeplugintable.find_unique.return_value
    stored_review = {"approved_by": "original-reviewer", "security_scan_passed": True}
    stored_manifest = {**json.loads(original.manifest_json), "_litellm_review": stored_review}
    current = original.model_copy(update={"manifest_json": json.dumps(stored_manifest)})
    publication_tx.litellm_claudecodeplugintable.find_unique.return_value = current

    async def update(where: Mapping[str, str], data: Mapping[str, object]) -> LiteLLM_ClaudeCodePluginTable:
        return current.model_copy(update=dict(data))

    publication_tx.litellm_claudecodeplugintable.update.side_effect = update
    result = await publish_plugin(
        cast(PluginPublicationTransaction, publication_tx),
        PluginPublication("skill", None, "updated", {
            "_litellm_review": {"approved_by": "forged", "security_scan_passed": True},
            "source": stored_manifest["source"] if same_source else {"source": "github", "repo": "example/skills"},
        }, "author", review=review),
        PluginPublicationBase(current.id, current.updated_at),
    )
    assert isinstance(result, PublishedPlugin)
    manifest = json.loads(publication_tx.litellm_claudecodeplugintable.update.call_args.kwargs["data"]["manifest_json"])
    assert manifest["_litellm_published_by"] == (review.approved_by if review else "author")
    expected = (
        {"approved_by": review.approved_by, "security_scan_passed": review.security_scan_passed} if review
        else stored_review if same_source else None
    )
    assert manifest.get("_litellm_review") == expected
    assert (None if result.review is None else {
        "approved_by": result.review.approved_by, "security_scan_passed": result.review.security_scan_passed,
    }) == expected
    assert json.loads(publication_tx.litellm_claudecodepluginversion.create.call_args.kwargs["data"]["manifest_json"]).get("_litellm_review") == expected
