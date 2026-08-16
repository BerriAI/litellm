"""
Unit tests for passthrough managed IDs (Scope A).

Tests cover:
  - managed_id_codec: encode / decode / is_managed round-trip and rejection cases.
  - managed_id_rewriter._resolve_one: cross-route 404, access-check 403, unknown ID 404,
    raw pass-through.
  - managed_id_rewriter.rewrite_response_ids: file create swap, batch create swap,
    dedup reuse (no duplicate row), null field skip.
  - managed_id_rewriter.rewrite_path_ids / rewrite_query_ids / rewrite_body_ids:
    INPUT swap and raw pass-through.
  - Flag-off: feature flag disabled → no swap at all.
  - Cross-route: managed ID minted for 'openai' rejected on a different provider.
  - Forged: unknown base64 → 404.
"""

from __future__ import annotations

import base64
import json
import sys
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.base_llm.managed_resources.utils import (
    resolve_passthrough_managed_id_provider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.managed_id_codec import (
    decode,
    encode,
    is_managed,
    new_managed_id,
)
from litellm.proxy.pass_through_endpoints.managed_id_rewriter import (
    _MAX_RAW_ID_GUARD_LOOKUPS,
    _canonical_path,
    _passthrough_provider_marker,
    _resolve_one,
    is_passthrough_list_route,
    list_passthrough_ids_from_db,
    rewrite_body_ids,
    rewrite_path_ids,
    rewrite_query_ids,
    rewrite_response_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(user_id: str = "user-1", team_id: str = "team-1") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, team_id=team_id)


def _admin_user() -> UserAPIKeyAuth:
    u = UserAPIKeyAuth(user_id="admin", user_role="proxy_admin")
    return u


def _prisma_client() -> MagicMock:
    """Return a MagicMock prisma_client with async db methods."""
    pc = MagicMock()
    pc.db = MagicMock()
    pc.db.litellm_managedfiletable = MagicMock()
    pc.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
    pc.db.litellm_managedfiletable.find_many = AsyncMock(return_value=[])
    pc.db.litellm_managedfiletable.create = AsyncMock(return_value=None)
    pc.db.litellm_managedobjecttable = MagicMock()
    pc.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=None)
    pc.db.litellm_managedobjecttable.upsert = AsyncMock(return_value=None)
    pc.db.litellm_managedobjecttable.update = AsyncMock(return_value=None)
    return pc


def _managed_files_hook(store_side_effect: Any = None) -> MagicMock:
    hook = MagicMock()
    hook.get_unified_file_id = AsyncMock(return_value=None)
    hook.store_unified_file_id = AsyncMock(side_effect=store_side_effect)
    return hook


def _owner_scoped_file_find_many(row: Any):
    """Return a ``find_many`` that mimics Prisma owner-scoping for the managed
    file table: an owner-scoped query (one carrying ``created_by`` / ``team_id``
    / ``OR``) returns ``[]`` because the caller does not own *row*, while an
    unscoped (global) query returns ``[row]``. This reproduces the cross-tenant
    bypass that a caller-scoped dedup lookup allowed (the scoped query misses the
    other tenant's row, so a fresh managed ID gets minted for the attacker)."""

    async def _impl(*args: Any, where: Any = None, **kwargs: Any) -> Any:
        where = where or {}
        if "created_by" in where or "team_id" in where or "OR" in where:
            return []
        return [row]

    return _impl


# ---------------------------------------------------------------------------
# managed_id_codec — unit tests
# ---------------------------------------------------------------------------


class TestCodec:
    def test_encode_decode_roundtrip(self):
        managed_id = encode("openai", "uuid-abc", "file-xyz")
        payload = decode(managed_id)
        assert payload is not None
        assert payload.provider == "openai"
        assert payload.unified_uuid == "uuid-abc"
        assert payload.raw_provider_id == "file-xyz"

    def test_is_managed_true(self):
        assert is_managed(encode("openai", "u1", "file-abc")) is True

    def test_is_managed_false_for_raw_ids(self):
        assert is_managed("file-abc123") is False
        assert is_managed("batch_xyz") is False
        assert is_managed("resp_abc") is False

    def test_decode_returns_none_for_garbage(self):
        assert decode("not-base64!!!") is None
        assert decode("") is None
        assert decode("abc") is None

    def test_decode_returns_none_for_wrong_type(self):
        assert decode(None) is None  # type: ignore[arg-type]
        assert decode(42) is None  # type: ignore[arg-type]

    def test_decode_returns_none_for_unified_endpoint_id(self):
        # A unified-endpoint ID: starts with litellm_proxy: but lacks passthrough;
        plaintext = "litellm_proxy:application/octet-stream;unified_id,123;target_model_names,gpt-4"
        unified_id = base64.urlsafe_b64encode(plaintext.encode()).decode().rstrip("=")
        assert decode(unified_id) is None

    def test_new_managed_id_produces_valid_id(self):
        mid = new_managed_id("openai", "batch_abc")
        payload = decode(mid)
        assert payload is not None
        assert payload.provider == "openai"
        assert payload.raw_provider_id == "batch_abc"

    def test_encode_padding_insensitive(self):
        """Encoded IDs with varying lengths all decode correctly."""
        for raw in ("file-x", "file-ab", "file-abc", "file-abcd"):
            mid = encode("openai", "u", raw)
            p = decode(mid)
            assert p is not None and p.raw_provider_id == raw


# ---------------------------------------------------------------------------
# resolve_passthrough_managed_id_provider — provider scope mapping
# ---------------------------------------------------------------------------


class TestManagedIdProviderScope:
    """Managed-ID scoping is keyed on the explicit forwarded provider, and both
    azure and azure_ai must collapse to a single 'azure' scope so an ID minted
    while routing as one resolves while routing as the other."""

    def test_openai_scope(self):
        assert resolve_passthrough_managed_id_provider("openai") == "openai"
        assert (
            resolve_passthrough_managed_id_provider(litellm.LlmProviders.OPENAI)
            == "openai"
        )

    def test_azure_scope(self):
        assert resolve_passthrough_managed_id_provider("azure") == "azure"
        assert (
            resolve_passthrough_managed_id_provider(litellm.LlmProviders.AZURE)
            == "azure"
        )

    def test_azure_ai_collapses_to_azure(self):
        assert resolve_passthrough_managed_id_provider("azure_ai") == "azure"
        assert (
            resolve_passthrough_managed_id_provider(litellm.LlmProviders.AZURE_AI)
            == "azure"
        )

    def test_azure_ai_id_resolves_on_azure_route(self):
        """End-to-end consequence of the collapse: an ID whose scope was
        resolved from azure_ai shares the 'azure' namespace, so decoding +
        cross-route checks line up with an azure-scoped ID."""
        azure_ai_scope = resolve_passthrough_managed_id_provider("azure_ai")
        azure_scope = resolve_passthrough_managed_id_provider("azure")
        managed = new_managed_id(azure_ai_scope, "file-shared")
        assert decode(managed).provider == azure_scope

    def test_case_insensitive(self):
        assert resolve_passthrough_managed_id_provider("AZURE") == "azure"
        assert resolve_passthrough_managed_id_provider("OpenAI") == "openai"

    def test_namespaced_provider_suffix(self):
        assert resolve_passthrough_managed_id_provider("foo.azure") == "azure"
        assert resolve_passthrough_managed_id_provider("foo.azure_ai") == "azure"
        assert resolve_passthrough_managed_id_provider("foo.openai") == "openai"

    def test_non_openai_azure_providers_not_scoped(self):
        """Managed IDs only apply to explicit openai/azure pass-through; any
        other provider (or a missing one) must return None so a third-party
        OpenAI-compatible endpoint never triggers managed-ID minting."""
        for provider in (None, "", "cohere", "vllm", "anthropic", "gemini", "bedrock"):
            assert resolve_passthrough_managed_id_provider(provider) is None


# ---------------------------------------------------------------------------
# _canonical_path
# ---------------------------------------------------------------------------


class TestCanonicalPath:
    def test_strips_openai_prefix(self):
        assert _canonical_path("/openai/v1/batches/batch_x") == "/v1/batches/batch_x"

    def test_strips_openai_passthrough_prefix(self):
        assert _canonical_path("/openai_passthrough/v1/files") == "/v1/files"

    def test_leaves_bare_path_unchanged(self):
        assert _canonical_path("/v1/responses") == "/v1/responses"

    def test_strips_azure_openai_prefix(self):
        assert _canonical_path("/azure/openai/files") == "/v1/files"

    def test_strips_azure_openai_batch_with_id(self):
        assert (
            _canonical_path("/azure/openai/batches/batch_abc123")
            == "/v1/batches/batch_abc123"
        )

    def test_strips_azure_openai_responses(self):
        assert _canonical_path("/azure/openai/responses") == "/v1/responses"

    def test_strips_azure_ai_openai_prefix(self):
        assert _canonical_path("/azure_ai/openai/files") == "/v1/files"

    def test_strips_azure_ai_openai_batch_cancel(self):
        assert (
            _canonical_path("/azure_ai/openai/batches/batch_abc/cancel")
            == "/v1/batches/batch_abc/cancel"
        )

    def test_azure_path_already_carrying_v1_is_not_doubled(self):
        assert _canonical_path("/azure/openai/v1/files") == "/v1/files"
        assert (
            _canonical_path("/azure/openai/v1/batches/batch_abc")
            == "/v1/batches/batch_abc"
        )

    def test_strips_azure_openai_file_with_id(self):
        assert _canonical_path("/azure/openai/files/file-abc") == "/v1/files/file-abc"


# ---------------------------------------------------------------------------
# _resolve_one
# ---------------------------------------------------------------------------


class TestResolveOne:
    @pytest.mark.asyncio
    async def test_raw_id_passes_through(self):
        result = await _resolve_one("file-abc", "openai", _user(), None, None)
        assert result == "file-abc"

    @pytest.mark.asyncio
    async def test_cross_route_raises_404(self):
        mid = encode("anthropic", "u", "file-abc")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _resolve_one(mid, "openai", _user(), None, None)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_managed_id_raises_404(self):
        mid = encode("openai", "u", "file-abc")
        pc = _prisma_client()
        hook = _managed_files_hook()
        # Both lookups return None → 404
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _resolve_one(mid, "openai", _user(), pc, hook)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_access_denied_raises_403(self):
        mid = encode("openai", "u", "file-abc")
        hook = _managed_files_hook()
        file_row = MagicMock()
        file_row.created_by = "other-user"
        file_row.team_id = "other-team"
        hook.get_unified_file_id = AsyncMock(return_value=file_row)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _resolve_one(mid, "openai", _user("user-1", "team-1"), None, hook)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_file_id_resolves(self):
        mid = encode("openai", "u", "file-xyz")
        hook = _managed_files_hook()
        file_row = MagicMock()
        file_row.created_by = "user-1"
        file_row.team_id = "team-1"
        hook.get_unified_file_id = AsyncMock(return_value=file_row)
        result = await _resolve_one(mid, "openai", _user(), None, hook)
        assert result == "file-xyz"

    @pytest.mark.asyncio
    async def test_valid_batch_id_resolves_via_object_table(self):
        mid = encode("openai", "u", "batch_abc")
        pc = _prisma_client()
        obj_row = MagicMock()
        obj_row.created_by = "user-1"
        obj_row.team_id = "team-1"
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=obj_row)
        result = await _resolve_one(mid, "openai", _user(), pc, None)
        assert result == "batch_abc"

    @pytest.mark.asyncio
    async def test_admin_can_access_any_resource(self):
        mid = encode("openai", "u", "file-xyz")
        hook = _managed_files_hook()
        file_row = MagicMock()
        file_row.created_by = "other-user"
        file_row.team_id = "other-team"
        hook.get_unified_file_id = AsyncMock(return_value=file_row)
        result = await _resolve_one(mid, "openai", _admin_user(), None, hook)
        assert result == "file-xyz"


# ---------------------------------------------------------------------------
# rewrite_response_ids — OUTPUT
# ---------------------------------------------------------------------------


class TestRewriteResponseIds:
    @pytest.mark.asyncio
    async def test_file_create_mints_managed_id(self):
        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {"id": "file-abc123", "object": "file"}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/files",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert result is not body  # mutated copy
        assert result["id"] != "file-abc123"
        payload = decode(result["id"])
        assert payload is not None
        assert payload.raw_provider_id == "file-abc123"
        hook.store_unified_file_id.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_file_create_persist_failure_leaves_raw_id(self):
        """If the DB write fails, the response must keep the raw provider ID
        (which still resolves upstream) rather than swap in a managed ID that no
        DB row backs and that would 404 on every later resolve."""
        pc = _prisma_client()
        hook = _managed_files_hook(store_side_effect=Exception("db down"))
        body = {"id": "file-abc123", "object": "file"}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/files",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        hook.store_unified_file_id.assert_awaited_once()
        assert result["id"] == "file-abc123"
        assert decode(result["id"]) is None

    @pytest.mark.asyncio
    async def test_batch_create_mints_id_and_input_file_id(self):
        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {
            "id": "batch_xyz",
            "input_file_id": "file-abc",
            "output_file_id": None,
            "error_file_id": None,
        }
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/batches",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert decode(result["id"]).raw_provider_id == "batch_xyz"  # type: ignore[union-attr]
        assert decode(result["input_file_id"]).raw_provider_id == "file-abc"  # type: ignore[union-attr]
        # Null fields skipped
        assert result["output_file_id"] is None
        assert result["error_file_id"] is None

    @pytest.mark.asyncio
    async def test_response_create_mints_id(self):
        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {"id": "resp_abc", "object": "response"}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/responses",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert decode(result["id"]).raw_provider_id == "resp_abc"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_azure_response_create_mints_id(self):
        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {
            "id": "resp_0dce2668af072bdc006a195db1f96c8194b6217f8e0d0b3ccd",
            "object": "response",
            "status": "completed",
        }
        result = await rewrite_response_ids(
            provider="azure",
            method="POST",
            route="/azure/openai/responses",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert (
            decode(result["id"]).raw_provider_id  # type: ignore[union-attr]
            == "resp_0dce2668af072bdc006a195db1f96c8194b6217f8e0d0b3ccd"
        )

    @pytest.mark.asyncio
    async def test_no_map_entry_returns_body_unchanged(self):
        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {"id": "msg_xyz", "object": "message"}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/chat/completions",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert result is body  # same object, unchanged

    @pytest.mark.asyncio
    async def test_dedup_reuses_existing_file_row(self):
        """File uploaded via passthrough, then referenced in a batch — no new row."""
        existing_managed_id = new_managed_id("openai", "file-abc")
        existing_row = MagicMock()
        existing_row.unified_file_id = existing_managed_id
        existing_row.created_by = "user-1"
        existing_row.team_id = "team-1"

        pc = _prisma_client()
        # Dedup lookup finds existing row
        pc.db.litellm_managedfiletable.find_many = AsyncMock(
            return_value=[existing_row]
        )
        hook = _managed_files_hook()
        body = {
            "id": "batch_xyz",
            "input_file_id": "file-abc",
            "output_file_id": None,
            "error_file_id": None,
        }
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/batches",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        # input_file_id should be the SAME managed ID already in DB
        assert result["input_file_id"] == existing_managed_id
        # store_unified_file_id should NOT have been called (reused existing)
        hook.store_unified_file_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dedup_skips_cross_provider_file_row(self):
        """Same raw file ID for a different provider must mint a new managed ID."""
        azure_managed_id = new_managed_id("azure", "file-abc")
        existing_row = MagicMock()
        existing_row.unified_file_id = azure_managed_id

        pc = _prisma_client()
        pc.db.litellm_managedfiletable.find_many = AsyncMock(
            return_value=[existing_row]
        )
        hook = _managed_files_hook()
        body = {"id": "file-abc", "object": "file"}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/files",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert decode(result["id"]).provider == "openai"
        assert decode(result["id"]).raw_provider_id == "file-abc"
        assert result["id"] != azure_managed_id
        hook.store_unified_file_id.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dedup_reuses_same_provider_row_amid_collision(self):
        """When OpenAI and Azure both issued the same raw file ID, an Azure call
        must reuse the existing Azure managed row deterministically rather than
        mint a duplicate, even when the cross-provider OpenAI row is returned
        first by the DB."""
        raw_id = "file-collision"
        openai_row = MagicMock()
        openai_row.unified_file_id = new_managed_id("openai", raw_id)
        openai_row.created_by = "user-1"
        openai_row.team_id = "team-1"
        azure_managed_id = new_managed_id("azure", raw_id)
        azure_row = MagicMock()
        azure_row.unified_file_id = azure_managed_id
        azure_row.created_by = "user-1"
        azure_row.team_id = "team-1"

        pc = _prisma_client()
        # Cross-provider row listed first to expose any non-deterministic pick.
        pc.db.litellm_managedfiletable.find_many = AsyncMock(
            return_value=[openai_row, azure_row]
        )
        hook = _managed_files_hook()
        body = {"id": raw_id, "object": "file"}
        result = await rewrite_response_ids(
            provider="azure",
            method="GET",
            route=f"/azure/openai/files/{raw_id}",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert result["id"] == azure_managed_id
        hook.store_unified_file_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cross_owner_file_retrieve_raises_404(self):
        """
        A caller who fetches another tenant's raw ``file-...`` ID through
        GET /openai/v1/files/{file_id} (which bypasses the managed-ID input gate)
        must be denied with a 404 — the response path must NOT mint a fresh
        managed ID for that file under the attacker.
        """
        from fastapi import HTTPException

        pc = _prisma_client()
        other_owner_row = MagicMock()
        other_owner_row.created_by = "victim"
        other_owner_row.team_id = "victim-team"
        other_owner_row.unified_file_id = encode("openai", "victim", "file-victim")
        pc.db.litellm_managedfiletable.find_many = _owner_scoped_file_find_many(
            other_owner_row
        )
        hook = _managed_files_hook()

        body = {"id": "file-victim", "object": "file"}
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_response_ids(
                provider="openai",
                method="GET",
                route="/openai/v1/files/file-victim",
                body=body,
                user_api_key_dict=_user("attacker", "attacker-team"),
                prisma_client=pc,
                managed_files_hook=hook,
            )
        assert exc_info.value.status_code == 404
        # Must not mint / persist a managed ID for the attacker.
        hook.store_unified_file_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cross_owner_file_delete_raises_404(self):
        """DELETE is also a non-create route: cross-owner raw file IDs are denied."""
        from fastapi import HTTPException

        pc = _prisma_client()
        other_owner_row = MagicMock()
        other_owner_row.created_by = "victim"
        other_owner_row.team_id = "victim-team"
        other_owner_row.unified_file_id = encode("openai", "victim", "file-victim")
        pc.db.litellm_managedfiletable.find_many = _owner_scoped_file_find_many(
            other_owner_row
        )
        hook = _managed_files_hook()

        body = {"id": "file-victim", "object": "file", "deleted": True}
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_response_ids(
                provider="openai",
                method="DELETE",
                route="/openai/v1/files/file-victim",
                body=body,
                user_api_key_dict=_user("attacker", "attacker-team"),
                prisma_client=pc,
                managed_files_hook=hook,
            )
        assert exc_info.value.status_code == 404
        hook.store_unified_file_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cross_owner_file_create_leaves_raw_id(self):
        """
        On the create (POST /v1/files) path a cross-owner dedup hit must NOT 404
        the caller's own successful upload; leave the raw ID unmanaged instead
        (mirrors the batch/response create behaviour).
        """
        pc = _prisma_client()
        other_owner_row = MagicMock()
        other_owner_row.created_by = "victim"
        other_owner_row.team_id = "victim-team"
        other_owner_row.unified_file_id = encode("openai", "victim", "file-shared")
        pc.db.litellm_managedfiletable.find_many = _owner_scoped_file_find_many(
            other_owner_row
        )
        hook = _managed_files_hook()

        body = {"id": "file-shared", "object": "file"}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/files",
            body=body,
            user_api_key_dict=_user("uploader", "uploader-team"),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert result["id"] == "file-shared"
        hook.store_unified_file_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_team_member_reuses_shared_file_row(self):
        """A teammate of the file owner can reuse the existing managed file row
        (the cross-tenant guard scopes by team, not just the creating user)."""
        existing_managed_id = new_managed_id("openai", "file-team")
        existing_row = MagicMock()
        existing_row.unified_file_id = existing_managed_id
        existing_row.created_by = "owner-user"
        existing_row.team_id = "shared-team"

        pc = _prisma_client()
        pc.db.litellm_managedfiletable.find_many = AsyncMock(
            return_value=[existing_row]
        )
        hook = _managed_files_hook()

        body = {"id": "file-team", "object": "file"}
        result = await rewrite_response_ids(
            provider="openai",
            method="GET",
            route="/openai/v1/files/file-team",
            body=body,
            user_api_key_dict=_user("teammate", "shared-team"),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert result["id"] == existing_managed_id
        hook.store_unified_file_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_openai_passthrough_prefix_normalised(self):
        """Routes under /openai_passthrough/ work the same as /openai/."""
        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {"id": "file-abc", "object": "file"}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai_passthrough/v1/files",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert decode(result["id"]).raw_provider_id == "file-abc"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_batch_reuse_refreshes_stored_snapshot(self):
        """Retrieving a completed batch must refresh the stored snapshot so the
        DB-served list reflects fields (e.g. output_file_id) that were null at
        creation time. The dedup-reuse path must update file_object, not just
        return the existing id with a stale snapshot."""
        existing_managed_id = new_managed_id("openai", "batch_done")
        existing_row = MagicMock()
        existing_row.unified_object_id = existing_managed_id
        existing_row.created_by = "user-1"
        existing_row.team_id = "team-1"

        pc = _prisma_client()
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(
            return_value=existing_row
        )

        completed_body = {
            "id": "batch_done",
            "object": "batch",
            "status": "completed",
            "output_file_id": "file-out",
            "error_file_id": None,
        }
        result = await rewrite_response_ids(
            provider="openai",
            method="GET",
            route="/openai/v1/batches/batch_done",
            body=completed_body,
            user_api_key_dict=_user("user-1", "team-1"),
            prisma_client=pc,
            managed_files_hook=None,
        )

        # Reuses the existing managed id (no new row minted)
        assert result["id"] == existing_managed_id
        pc.db.litellm_managedobjecttable.upsert.assert_not_awaited()
        # The stored snapshot is refreshed with the completed batch body
        pc.db.litellm_managedobjecttable.update.assert_awaited_once()
        update_kwargs = pc.db.litellm_managedobjecttable.update.call_args.kwargs
        assert update_kwargs["where"] == {"unified_object_id": existing_managed_id}
        stored = json.loads(update_kwargs["data"]["file_object"])
        assert stored["status"] == "completed"
        # output_file_id is itself rewritten to a managed id wrapping the raw id
        assert decode(stored["output_file_id"]).raw_provider_id == "file-out"

    @pytest.mark.asyncio
    async def test_cross_provider_batch_collision_mints_new_id(self):
        """
        If OpenAI and Azure independently issue the same raw batch ID, the
        Azure call must mint its own row keyed by 'passthrough:azure:batch_shared'
        and must NOT raise 404.  The namespaced model_object_id prevents a
        UniqueConstraintViolation on the @unique column.
        """
        pc = _prisma_client()
        # Both providers return no existing row (different namespaced keys)
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=None)
        pc.db.litellm_managedobjecttable.upsert = AsyncMock(return_value=None)

        body = {"id": "batch_shared", "object": "batch", "input_file_id": None}
        result = await rewrite_response_ids(
            provider="azure",
            method="POST",
            route="/azure/openai/batches",
            body=body,
            user_api_key_dict=_user("user-azure", "team-azure"),
            prisma_client=pc,
            managed_files_hook=None,
        )
        # Must mint a fresh azure-scoped managed ID
        assert decode(result["id"]) is not None
        assert decode(result["id"]).provider == "azure"
        assert decode(result["id"]).raw_provider_id == "batch_shared"

        # Verify the upsert stored the namespaced model_object_id
        call_data = pc.db.litellm_managedobjecttable.upsert.call_args.kwargs["data"]
        assert (
            call_data["create"]["model_object_id"] == "passthrough:azure:batch_shared"
        )

    @pytest.mark.asyncio
    async def test_batch_create_persist_failure_leaves_raw_id(self):
        """If the object upsert fails, the batch response must keep the raw
        provider ID rather than return a managed ID with no backing DB row that
        would 404 on every subsequent resolve."""
        pc = _prisma_client()
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=None)
        pc.db.litellm_managedobjecttable.upsert = AsyncMock(
            side_effect=Exception("db down")
        )
        body = {"id": "batch_xyz", "object": "batch", "input_file_id": None}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/batches",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=None,
        )
        pc.db.litellm_managedobjecttable.upsert.assert_awaited_once()
        assert result["id"] == "batch_xyz"
        assert decode(result["id"]) is None

    @pytest.mark.asyncio
    async def test_concurrent_create_converges_on_winner_managed_id(self):
        """
        Two callers minting the same namespaced object row race: the dedup lookup
        finds nothing for both, but the @unique model_object_id lets only one
        insert win. The loser's upsert raises, and it must re-read the winner's
        row and return that managed ID rather than silently keeping the raw ID
        (which would leave the two callers divergent for the same upstream batch).
        """
        pc = _prisma_client()
        winner_managed_id = encode("openai", "winner-uuid", "batch_race")
        winner_row = MagicMock()
        winner_row.created_by = "user-1"
        winner_row.team_id = "team-1"
        winner_row.unified_object_id = winner_managed_id
        # First (dedup) lookup misses; post-collision re-read finds the winner.
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(
            side_effect=[None, winner_row]
        )
        pc.db.litellm_managedobjecttable.upsert = AsyncMock(
            side_effect=Exception("UniqueConstraintViolation: model_object_id")
        )

        body = {"id": "batch_race", "object": "batch", "input_file_id": None}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/batches",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=None,
        )
        # The loser converges on the winner's managed ID, not the raw batch ID.
        assert result["id"] == winner_managed_id
        assert decode(result["id"]).raw_provider_id == "batch_race"
        assert pc.db.litellm_managedobjecttable.find_first.await_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_create_race_with_cross_owner_winner_retrieve_404(self):
        """
        If the row that wins the insert race on a non-create (retrieve) route is
        owned by a different tenant, the loser must be denied with 404 rather
        than handed the raw ID — the post-collision re-read runs the same access
        check as the initial dedup hit.
        """
        from fastapi import HTTPException

        pc = _prisma_client()
        winner_row = MagicMock()
        winner_row.created_by = "other-user"
        winner_row.team_id = "other-team"
        winner_row.unified_object_id = encode("openai", "other-uuid", "batch_race")
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(
            side_effect=[None, winner_row]
        )
        pc.db.litellm_managedobjecttable.upsert = AsyncMock(
            side_effect=Exception("UniqueConstraintViolation: model_object_id")
        )

        body = {"id": "batch_race", "object": "batch", "input_file_id": None}
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_response_ids(
                provider="openai",
                method="GET",
                route="/openai/v1/batches/batch_race",
                body=body,
                user_api_key_dict=_user("attacker", "attacker-team"),
                prisma_client=pc,
                managed_files_hook=None,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_provider_batch_collision_dedup_uses_namespaced_key(self):
        """
        When OpenAI already has a row for batch_shared, an Azure request must
        look up 'passthrough:azure:batch_shared' (not 'batch_shared'), find
        nothing, and mint a new row — not raise 404 or reuse the OpenAI row.
        """
        pc = _prisma_client()
        # Simulate: OpenAI row exists under 'passthrough:openai:batch_shared',
        # but Azure lookup for 'passthrough:azure:batch_shared' returns None.
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=None)
        pc.db.litellm_managedobjecttable.upsert = AsyncMock(return_value=None)

        body = {"id": "batch_shared", "object": "batch", "input_file_id": None}
        result = await rewrite_response_ids(
            provider="azure",
            method="POST",
            route="/azure/openai/batches",
            body=body,
            user_api_key_dict=_user("user-azure", "team-azure"),
            prisma_client=pc,
            managed_files_hook=None,
        )
        # The dedup lookup must use the namespaced key
        lookup_where = pc.db.litellm_managedobjecttable.find_first.call_args.kwargs[
            "where"
        ]
        assert lookup_where["model_object_id"] == "passthrough:azure:batch_shared"
        # Result is a valid azure-scoped managed ID
        assert decode(result["id"]).provider == "azure"

    @pytest.mark.asyncio
    async def test_cross_owner_object_collision_returns_raw_id_not_404(self):
        """
        On the OUTPUT (mint) path, if the namespaced key is already owned by a
        different caller (e.g. two upstream accounts under one provider name
        issued the same raw batch ID), the caller's successful upstream create
        must NOT be turned into a 404. Leave their raw ID unmanaged instead.
        """
        pc = _prisma_client()
        other_owner_row = MagicMock()
        other_owner_row.created_by = "other-user"
        other_owner_row.team_id = "other-team"
        other_owner_row.unified_object_id = encode(
            "azure", "other-user", "batch_shared"
        )
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(
            return_value=other_owner_row
        )
        pc.db.litellm_managedobjecttable.upsert = AsyncMock(return_value=None)

        body = {"id": "batch_shared", "object": "batch", "input_file_id": None}
        result = await rewrite_response_ids(
            provider="azure",
            method="POST",
            route="/azure/openai/batches",
            body=body,
            user_api_key_dict=_user("user-azure", "team-azure"),
            prisma_client=pc,
            managed_files_hook=None,
        )
        # Caller gets their raw batch ID back, unmanaged; not a 404, and not
        # the other owner's managed ID.
        assert result["id"] == "batch_shared"
        # No new row is minted (would violate the @unique model_object_id).
        pc.db.litellm_managedobjecttable.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cross_owner_object_retrieve_raises_404(self):
        """
        On a retrieve route, a caller who supplies another owner's raw batch ID
        (which bypasses the managed-ID input gate) must be denied with a 404 —
        the upstream object must NOT be echoed back with its raw ID.
        """
        from fastapi import HTTPException

        pc = _prisma_client()
        other_owner_row = MagicMock()
        other_owner_row.created_by = "other-user"
        other_owner_row.team_id = "other-team"
        other_owner_row.unified_object_id = encode("openai", "other-user", "batch_xyz")
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(
            return_value=other_owner_row
        )

        body = {"id": "batch_xyz", "object": "batch", "input_file_id": None}
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_response_ids(
                provider="openai",
                method="GET",
                route="/openai/v1/batches/batch_xyz",
                body=body,
                user_api_key_dict=_user("attacker", "attacker-team"),
                prisma_client=pc,
                managed_files_hook=None,
            )
        assert exc_info.value.status_code == 404
        # Must not silently mint a row for the attacker either.
        pc.db.litellm_managedobjecttable.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cross_owner_response_delete_raises_404(self):
        """A delete route is also a non-create route: cross-owner access is denied."""
        from fastapi import HTTPException

        pc = _prisma_client()
        other_owner_row = MagicMock()
        other_owner_row.created_by = "other-user"
        other_owner_row.team_id = "other-team"
        other_owner_row.unified_object_id = encode("openai", "other-user", "resp_abc")
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(
            return_value=other_owner_row
        )

        body = {"id": "resp_abc", "object": "response"}
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_response_ids(
                provider="openai",
                method="DELETE",
                route="/openai/v1/responses/resp_abc",
                body=body,
                user_api_key_dict=_user("attacker", "attacker-team"),
                prisma_client=pc,
                managed_files_hook=None,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_batch_retrieve_swaps_output_file_id(self):
        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {
            "id": "batch_xyz",
            "input_file_id": "file-in",
            "output_file_id": "file-out",
            "error_file_id": "file-err",
        }
        result = await rewrite_response_ids(
            provider="openai",
            method="GET",
            route="/openai/v1/batches/batch_xyz",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        assert decode(result["output_file_id"]).raw_provider_id == "file-out"  # type: ignore[union-attr]
        assert decode(result["error_file_id"]).raw_provider_id == "file-err"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_file_create_persists_metadata_for_list(self):
        """The file's upstream metadata is stored so the DB-served list returns
        the same fields as a direct file GET (managed ID swapped in)."""
        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {
            "id": "file-abc123",
            "object": "file",
            "bytes": 120,
            "created_at": 1234567890,
            "filename": "train.jsonl",
            "purpose": "batch",
            "status": "processed",
        }
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/files",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        stored = hook.store_unified_file_id.call_args.kwargs["file_object"]
        assert stored is not None
        assert stored.filename == "train.jsonl"
        assert stored.bytes == 120
        assert stored.purpose == "batch"
        # Managed ID is swapped into the persisted metadata (never the raw one).
        assert stored.id == result["id"]
        assert decode(stored.id).raw_provider_id == "file-abc123"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_file_create_without_metadata_stores_no_file_object(self):
        """A minimal file response (no bytes/filename) falls back to storing the
        row without metadata rather than raising."""
        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {"id": "file-abc123", "object": "file"}
        await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/files",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        hook.store_unified_file_id.assert_awaited_once()
        assert hook.store_unified_file_id.call_args.kwargs["file_object"] is None

    @pytest.mark.asyncio
    async def test_file_create_persists_provider_marker_for_list_scope(self):
        """The minted file row must carry the provider marker (it flows into
        flat_model_file_ids), or the DB-pushed provider scope in
        list_passthrough_ids_from_db would never match it."""
        pc = _prisma_client()
        hook = _managed_files_hook()
        await rewrite_response_ids(
            provider="azure",
            method="POST",
            route="/azure/openai/files",
            body={"id": "file-abc123", "object": "file"},
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        mappings = hook.store_unified_file_id.call_args.kwargs["model_mappings"]
        assert _passthrough_provider_marker("azure") in mappings.values()
        assert _passthrough_provider_marker("openai") not in mappings.values()

    @pytest.mark.asyncio
    async def test_batch_snapshot_stores_managed_nested_file_ids(self):
        """The persisted batch snapshot must carry the managed nested file ID so
        the list response matches the rewritten direct GET response."""
        import json as _json

        pc = _prisma_client()
        hook = _managed_files_hook()
        body = {
            "id": "batch_xyz",
            "object": "batch",
            "input_file_id": "file-in",
            "output_file_id": None,
            "error_file_id": None,
        }
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/batches",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=pc,
            managed_files_hook=hook,
        )
        stored = pc.db.litellm_managedobjecttable.upsert.call_args.kwargs["data"][
            "create"
        ]["file_object"]
        snapshot = _json.loads(stored)
        assert snapshot["input_file_id"] == result["input_file_id"]
        assert decode(snapshot["input_file_id"]).raw_provider_id == "file-in"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# rewrite_path_ids — INPUT
# ---------------------------------------------------------------------------


class TestRewritePathIds:
    @pytest.mark.asyncio
    async def test_raw_segment_passes_through(self):
        result = await rewrite_path_ids(
            "/v1/batches/batch_abc", "openai", _user(), None, None
        )
        assert result == "/v1/batches/batch_abc"

    @pytest.mark.asyncio
    async def test_managed_segment_is_resolved(self):
        mid = encode("openai", "u", "batch_abc")
        hook = _managed_files_hook()
        pc = _prisma_client()
        obj_row = MagicMock()
        obj_row.created_by = "user-1"
        obj_row.team_id = "team-1"
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=obj_row)
        result = await rewrite_path_ids(
            f"/v1/batches/{mid}", "openai", _user(), pc, hook
        )
        assert result == "/v1/batches/batch_abc"

    @pytest.mark.asyncio
    async def test_cross_route_in_path_raises_404(self):
        mid = encode("anthropic", "u", "batch_abc")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await rewrite_path_ids(f"/v1/batches/{mid}", "openai", _user(), None, None)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# rewrite_query_ids — INPUT
# ---------------------------------------------------------------------------


class TestRewriteQueryIds:
    @pytest.mark.asyncio
    async def test_raw_params_pass_through(self):
        params = {"limit": "10", "after": "batch_xyz"}
        result = await rewrite_query_ids(params, "openai", _user(), None, None)
        assert result is params  # unchanged same object

    @pytest.mark.asyncio
    async def test_none_returns_none(self):
        result = await rewrite_query_ids(None, "openai", _user(), None, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_managed_param_is_resolved(self):
        mid = encode("openai", "u", "file-abc")
        hook = _managed_files_hook()
        file_row = MagicMock()
        file_row.created_by = "user-1"
        file_row.team_id = "team-1"
        hook.get_unified_file_id = AsyncMock(return_value=file_row)
        params = {"file_id": mid}
        result = await rewrite_query_ids(params, "openai", _user(), None, hook)
        assert result is not params
        assert result["file_id"] == "file-abc"  # type: ignore[index]


# ---------------------------------------------------------------------------
# rewrite_body_ids — INPUT
# ---------------------------------------------------------------------------


class TestRewriteBodyIds:
    @pytest.mark.asyncio
    async def test_raw_body_passes_through(self):
        body = {"input_file_id": "file-abc", "model": "gpt-4o"}
        result = await rewrite_body_ids(body, "openai", _user(), None, None)
        assert result is body

    @pytest.mark.asyncio
    async def test_none_returns_none(self):
        result = await rewrite_body_ids(None, "openai", _user(), None, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_managed_id_in_body_resolved(self):
        mid = encode("openai", "u", "file-xyz")
        hook = _managed_files_hook()
        file_row = MagicMock()
        file_row.created_by = "user-1"
        file_row.team_id = "team-1"
        hook.get_unified_file_id = AsyncMock(return_value=file_row)
        body = {"input_file_id": mid}
        result = await rewrite_body_ids(body, "openai", _user(), None, hook)
        assert result is not body
        assert result["input_file_id"] == "file-xyz"  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_litellm_internal_key_preserved(self):
        """litellm_logging_obj and similar keys are never walked."""
        logging_obj = object()
        body = {"litellm_logging_obj": logging_obj, "model": "gpt-4o"}
        result = await rewrite_body_ids(body, "openai", _user(), None, None)
        # Internal key preserved by reference
        assert result["litellm_logging_obj"] is logging_obj  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_nested_list_resolved(self):
        """Managed IDs inside nested lists are resolved."""
        mid = encode("openai", "u", "file-nested")
        hook = _managed_files_hook()
        file_row = MagicMock()
        file_row.created_by = "user-1"
        file_row.team_id = "team-1"
        hook.get_unified_file_id = AsyncMock(return_value=file_row)
        body = {"files": [mid, "raw-string"]}
        result = await rewrite_body_ids(body, "openai", _user(), None, hook)
        assert result["files"][0] == "file-nested"  # type: ignore[index]
        assert result["files"][1] == "raw-string"  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_forged_managed_id_raises_404(self):
        """An unknown managed ID in the body raises 404 (not passed to upstream)."""
        mid = encode("openai", "u", "file-forged")
        hook = _managed_files_hook()
        hook.get_unified_file_id = AsyncMock(return_value=None)
        pc = _prisma_client()
        body = {"input_file_id": mid}
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await rewrite_body_ids(body, "openai", _user(), pc, hook)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cross_user_access_denied_in_body(self):
        """A managed ID owned by a different user raises 403."""
        mid = encode("openai", "u", "file-other")
        hook = _managed_files_hook()
        file_row = MagicMock()
        file_row.created_by = "other-user"
        file_row.team_id = "other-team"
        hook.get_unified_file_id = AsyncMock(return_value=file_row)
        body = {"input_file_id": mid}
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await rewrite_body_ids(
                body, "openai", _user("user-1", "team-1"), None, hook
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_deeply_nested_body_does_not_overflow_stack(self):
        """A pathologically deep body must not blow the Python stack: rewriting
        stops at the depth cap and returns the body unchanged instead of raising
        RecursionError."""
        node: Any = {"leaf": "raw-value"}
        for _ in range(5000):
            node = {"nested": node}

        result = await rewrite_body_ids(node, "openai", _user(), None, None)
        assert result is node

    @pytest.mark.asyncio
    async def test_managed_id_resolved_within_depth_cap(self):
        """A managed ID nested well within the depth cap is still resolved, so
        the cap never truncates legitimately-shaped bodies."""
        mid = encode("openai", "u", "file-deep")
        hook = _managed_files_hook()
        file_row = MagicMock()
        file_row.created_by = "user-1"
        file_row.team_id = "team-1"
        hook.get_unified_file_id = AsyncMock(return_value=file_row)

        leaf = {"input_file_id": mid}
        node: Any = leaf
        for _ in range(20):
            node = {"nested": node}

        result = await rewrite_body_ids(node, "openai", _user(), None, hook)

        cursor = result
        for _ in range(20):
            cursor = cursor["nested"]  # type: ignore[index]
        assert cursor["input_file_id"] == "file-deep"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Raw-provider-ID input guard — a raw ID recovered by decoding another tenant's
# managed ID must NOT be forwarded upstream when it maps to a managed resource
# the caller does not own (otherwise a DELETE / cancel runs upstream before the
# response-side ownership check).
# ---------------------------------------------------------------------------


class TestRawProviderIdInputGuard:
    @staticmethod
    def _victim_file_row() -> MagicMock:
        row = MagicMock()
        row.created_by = "victim"
        row.team_id = "victim-team"
        row.unified_file_id = encode("openai", "victim", "file-victim")
        return row

    @staticmethod
    def _victim_object_row() -> MagicMock:
        row = MagicMock()
        row.created_by = "victim"
        row.team_id = "victim-team"
        row.unified_object_id = encode("openai", "victim", "batch_victim")
        return row

    @pytest.mark.asyncio
    async def test_raw_file_path_for_other_owner_denied(self):
        """DELETE /openai/v1/files/file-victim with a raw ID that belongs to
        another tenant's managed file is rejected (404) before forwarding."""
        from fastapi import HTTPException

        pc = _prisma_client()
        pc.db.litellm_managedfiletable.find_many = AsyncMock(
            return_value=[self._victim_file_row()]
        )
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_path_ids(
                "/openai/v1/files/file-victim",
                "openai",
                _user("attacker", "attacker-team"),
                pc,
                _managed_files_hook(),
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_raw_batch_cancel_path_for_other_owner_denied(self):
        """POST /openai/v1/batches/batch_victim/cancel with another tenant's raw
        batch ID is rejected (404) before the upstream cancel runs."""
        from fastapi import HTTPException

        pc = _prisma_client()
        pc.db.litellm_managedobjecttable.find_first = AsyncMock(
            return_value=self._victim_object_row()
        )
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_path_ids(
                "/openai/v1/batches/batch_victim/cancel",
                "openai",
                _user("attacker", "attacker-team"),
                pc,
                _managed_files_hook(),
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_raw_file_query_for_other_owner_denied(self):
        from fastapi import HTTPException

        pc = _prisma_client()
        pc.db.litellm_managedfiletable.find_many = AsyncMock(
            return_value=[self._victim_file_row()]
        )
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_query_ids(
                {"file_id": "file-victim"},
                "openai",
                _user("attacker", "attacker-team"),
                pc,
                _managed_files_hook(),
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_raw_file_body_for_other_owner_denied(self):
        from fastapi import HTTPException

        pc = _prisma_client()
        pc.db.litellm_managedfiletable.find_many = AsyncMock(
            return_value=[self._victim_file_row()]
        )
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_body_ids(
                {"input_file_id": "file-victim"},
                "openai",
                _user("attacker", "attacker-team"),
                pc,
                _managed_files_hook(),
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_raw_file_owned_by_caller_passes_through(self):
        """A raw ID the caller does own is left untouched and forwarded — the
        guard must not block legitimate raw-ID usage."""
        pc = _prisma_client()
        own_row = MagicMock()
        own_row.created_by = "user-1"
        own_row.team_id = "team-1"
        own_row.unified_file_id = encode("openai", "u", "file-mine")
        pc.db.litellm_managedfiletable.find_many = AsyncMock(return_value=[own_row])
        result = await rewrite_path_ids(
            "/openai/v1/files/file-mine",
            "openai",
            _user("user-1", "team-1"),
            pc,
            _managed_files_hook(),
        )
        assert result == "/openai/v1/files/file-mine"

    @pytest.mark.asyncio
    async def test_unmanaged_raw_id_passes_through(self):
        """A raw ID with no managed row at all is a genuine opt-out and is
        forwarded unchanged."""
        pc = _prisma_client()
        result = await rewrite_path_ids(
            "/openai/v1/files/file-never-managed",
            "openai",
            _user("attacker", "attacker-team"),
            pc,
            _managed_files_hook(),
        )
        assert result == "/openai/v1/files/file-never-managed"

    @pytest.mark.asyncio
    async def test_cross_provider_raw_file_not_blocked(self):
        """A raw ID whose only managed row belongs to a different provider is not
        this provider's resource, so the guard does not deny it."""
        pc = _prisma_client()
        azure_row = MagicMock()
        azure_row.created_by = "victim"
        azure_row.team_id = "victim-team"
        azure_row.unified_file_id = encode("azure", "victim", "file-victim")
        pc.db.litellm_managedfiletable.find_many = AsyncMock(return_value=[azure_row])
        result = await rewrite_path_ids(
            "/openai/v1/files/file-victim",
            "openai",
            _user("attacker", "attacker-team"),
            pc,
            _managed_files_hook(),
        )
        assert result == "/openai/v1/files/file-victim"


# ---------------------------------------------------------------------------
# Raw-provider-ID guard amplification — a body packed with id-shaped strings
# must not fan out into one (unindexed) DB scan per string. The guard de-dupes
# repeats and caps the distinct lookups per request, failing closed instead of
# skipping the guard.
# ---------------------------------------------------------------------------


class TestRawProviderIdGuardBudget:
    @pytest.mark.asyncio
    async def test_many_distinct_raw_ids_capped(self):
        """A body with more distinct raw file IDs than the per-request budget is
        rejected with 400, and the number of (unindexed) DB scans never exceeds
        the cap."""
        from fastapi import HTTPException

        pc = _prisma_client()
        body = {"ids": [f"file-{i}" for i in range(_MAX_RAW_ID_GUARD_LOOKUPS + 25)]}
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_body_ids(
                body, "openai", _user("attacker", "attacker-team"), pc, None
            )
        assert exc_info.value.status_code == 400
        assert (
            pc.db.litellm_managedfiletable.find_many.call_count
            == _MAX_RAW_ID_GUARD_LOOKUPS
        )

    @pytest.mark.asyncio
    async def test_repeated_raw_id_deduped(self):
        """The same raw ID repeated many times issues exactly one DB lookup."""
        pc = _prisma_client()
        body = {"ids": ["file-dup"] * (_MAX_RAW_ID_GUARD_LOOKUPS * 5)}
        result = await rewrite_body_ids(
            body, "openai", _user("attacker", "attacker-team"), pc, None
        )
        assert result is body
        assert pc.db.litellm_managedfiletable.find_many.call_count == 1

    @pytest.mark.asyncio
    async def test_distinct_ids_under_cap_not_rejected(self):
        """A realistically-sized body (few distinct raw IDs) is never rejected and
        each distinct ID is guarded once."""
        pc = _prisma_client()
        body = {"ids": [f"file-{i}" for i in range(5)]}
        result = await rewrite_body_ids(
            body, "openai", _user("user-1", "team-1"), pc, None
        )
        assert result is body
        assert pc.db.litellm_managedfiletable.find_many.call_count == 5

    @pytest.mark.asyncio
    async def test_budget_is_per_input_surface(self):
        """Each input surface (path / query / body) gets its own budget, so a
        request distributing IDs across them is still bounded per surface."""
        from fastapi import HTTPException

        pc = _prisma_client()
        params = {f"k{i}": f"file-{i}" for i in range(_MAX_RAW_ID_GUARD_LOOKUPS + 5)}
        with pytest.raises(HTTPException) as exc_info:
            await rewrite_query_ids(
                params, "openai", _user("attacker", "attacker-team"), pc, None
            )
        assert exc_info.value.status_code == 400
        assert (
            pc.db.litellm_managedfiletable.find_many.call_count
            == _MAX_RAW_ID_GUARD_LOOKUPS
        )


# ---------------------------------------------------------------------------
# Flag-off: behaviour unchanged when passthrough_managed_object_ids is False
# ---------------------------------------------------------------------------


class TestFlagOff:
    """
    When the feature flag is off the pass_through_request code paths skip both
    hooks entirely.  Here we verify the rewriter modules themselves are pure
    no-ops when called with no DB / hook: raw IDs pass through.
    """

    @pytest.mark.asyncio
    async def test_raw_file_in_response_not_swapped_without_hook(self):
        body = {"id": "file-abc", "object": "file"}
        result = await rewrite_response_ids(
            provider="openai",
            method="POST",
            route="/openai/v1/files",
            body=body,
            user_api_key_dict=_user(),
            prisma_client=None,
            managed_files_hook=None,
        )
        # Without DB/hook, _mint_or_reuse_file returns raw_id unchanged
        assert result is body or result["id"] == "file-abc"

    @pytest.mark.asyncio
    async def test_decode_failure_body_untouched(self):
        body = {"id": "file-abc123"}
        result = await rewrite_body_ids(body, "openai", _user(), None, None)
        assert result is body


# ---------------------------------------------------------------------------
# list_passthrough_ids_from_db — unit tests
# ---------------------------------------------------------------------------


def _prisma_with_list(file_rows=None, batch_rows=None) -> MagicMock:
    """Return a prisma_client whose find_many honors the provider scope pushed
    into the ``where`` clause, mirroring how Postgres would filter rows.

    File rows are scoped via ``flat_model_file_ids: {has: <marker>}`` and object
    rows via ``model_object_id: {startswith: passthrough:<provider>:}``; the mock
    applies the same predicate so a test feeding mixed-provider rows exercises
    the real DB-pushdown contract instead of an unscoped passthrough."""
    pc = _prisma_client()

    def _file_filter(*args, where=None, take=None, **kwargs):
        rows = list(file_rows or [])
        marker = (where or {}).get("flat_model_file_ids", {}) or {}
        marker = marker.get("has")
        if marker is not None:
            rows = [
                r
                for r in rows
                if marker in (getattr(r, "flat_model_file_ids", None) or [])
            ]
        return rows if take is None else rows[:take]

    def _batch_filter(*args, where=None, take=None, **kwargs):
        rows = list(batch_rows or [])
        prefix = (where or {}).get("model_object_id", {}) or {}
        prefix = prefix.get("startswith")
        if prefix is not None:
            rows = [
                r
                for r in rows
                if str(getattr(r, "model_object_id", "") or "").startswith(prefix)
            ]
        return rows if take is None else rows[:take]

    if file_rows is not None:
        pc.db.litellm_managedfiletable.find_many = AsyncMock(side_effect=_file_filter)
    if batch_rows is not None:
        pc.db.litellm_managedobjecttable.find_many = AsyncMock(
            side_effect=_batch_filter
        )
    return pc


def _fake_file_row(
    unified_id: str, created_by: str = "user-1", team_id: str = "team-1"
):
    row = MagicMock()
    row.unified_file_id = unified_id
    row.created_by = created_by
    row.team_id = team_id
    row.file_object = {"filename": "test.jsonl", "bytes": 42, "purpose": "batch"}
    payload = decode(unified_id)
    row.flat_model_file_ids = (
        [payload.raw_provider_id, _passthrough_provider_marker(payload.provider)]
        if payload is not None
        else []
    )

    import datetime

    row.created_at = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    return row


def _fake_batch_row(
    unified_id: str, created_by: str = "user-1", team_id: str = "team-1"
):
    row = MagicMock()
    row.unified_object_id = unified_id
    row.created_by = created_by
    row.team_id = team_id
    row.file_object = {"status": "completed", "input_file_id": "file-managed-1"}
    row.file_purpose = "batch"
    payload = decode(unified_id)
    row.model_object_id = (
        f"passthrough:{payload.provider}:{payload.raw_provider_id}"
        if payload is not None
        else None
    )

    import datetime

    row.created_at = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    return row


class TestListPassthroughIdsFromDb:
    """Tests for list_passthrough_ids_from_db and is_passthrough_list_route."""

    def test_is_passthrough_list_route_files(self):
        assert is_passthrough_list_route("openai", "GET", "/openai/v1/files") is True

    def test_is_passthrough_list_route_batches(self):
        assert (
            is_passthrough_list_route("azure", "GET", "/azure/openai/batches") is True
        )

    def test_is_passthrough_list_route_not_for_post(self):
        assert is_passthrough_list_route("openai", "POST", "/openai/v1/files") is False

    def test_is_passthrough_list_route_not_for_single_resource(self):
        # GET /v1/files/{file_id} is not a list route
        assert (
            is_passthrough_list_route("openai", "GET", "/openai/v1/files/file-abc")
            is False
        )

    def test_is_passthrough_list_route_azure_ai_prefix(self):
        assert (
            is_passthrough_list_route("azure", "GET", "/azure_ai/openai/files") is True
        )

    def test_is_passthrough_list_route_azure_path_already_carrying_v1(self):
        assert (
            is_passthrough_list_route("azure", "GET", "/azure/openai/v1/files") is True
        )
        assert (
            is_passthrough_list_route("azure", "GET", "/azure/openai/v1/batches")
            is True
        )

    def test_is_passthrough_list_route_not_for_azure_single_resource(self):
        assert (
            is_passthrough_list_route("azure", "GET", "/azure/openai/files/file-abc")
            is False
        )

    @pytest.mark.asyncio
    async def test_list_files_returns_owned_rows(self):
        managed_id = new_managed_id("openai", "file-abc")
        fake_row = _fake_file_row(managed_id)
        pc = _prisma_with_list(file_rows=[fake_row])

        result = await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/files",
            user_api_key_dict=_user("user-1", "team-1"),
            prisma_client=pc,
        )

        assert result is not None
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == managed_id
        assert result["data"][0]["object"] == "file"
        assert result["first_id"] == managed_id

    @pytest.mark.asyncio
    async def test_list_batches_returns_owned_rows(self):
        managed_id = new_managed_id("openai", "batch_abc")
        fake_row = _fake_batch_row(managed_id)
        pc = _prisma_with_list(batch_rows=[fake_row])

        result = await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/batches",
            user_api_key_dict=_user("user-1", "team-1"),
            prisma_client=pc,
        )

        assert result is not None
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == managed_id
        assert result["data"][0]["object"] == "batch"

    @pytest.mark.asyncio
    async def test_list_files_admin_gets_all_rows(self):
        """Admin should receive all rows; the where filter passed to DB is {}."""
        rows = [
            _fake_file_row(new_managed_id("openai", "file-1")),
            _fake_file_row(new_managed_id("openai", "file-2")),
        ]
        pc = _prisma_with_list(file_rows=rows)

        result = await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/files",
            user_api_key_dict=_admin_user(),
            prisma_client=pc,
        )

        assert result is not None
        assert len(result["data"]) == 2
        # Admin adds no owner scoping, but the provider scope is always pushed
        # to the DB; the only where clause is the provider marker filter.
        call_kwargs = pc.db.litellm_managedfiletable.find_many.call_args.kwargs
        assert call_kwargs["where"] == {
            "flat_model_file_ids": {"has": _passthrough_provider_marker("openai")}
        }

    @pytest.mark.asyncio
    async def test_list_files_user_scoped_where(self):
        """Regular user should get a where clause scoped to their user_id / team_id."""
        pc = _prisma_with_list(file_rows=[])

        await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/files",
            user_api_key_dict=_user("user-2", "team-2"),
            prisma_client=pc,
        )

        call_kwargs = pc.db.litellm_managedfiletable.find_many.call_args.kwargs
        where = call_kwargs["where"]
        # The OR clause should scope to user-2 or team-2
        assert "OR" in where
        entries = where["OR"]
        assert {"created_by": "user-2"} in entries
        assert {"team_id": "team-2"} in entries

    @pytest.mark.asyncio
    async def test_list_has_more_flag(self):
        """has_more is True when DB returns limit+1 rows."""
        rows = [
            _fake_file_row(new_managed_id("openai", f"file-{i}")) for i in range(21)
        ]  # limit=20, fetch 21
        pc = _prisma_with_list(file_rows=rows)

        result = await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/files",
            user_api_key_dict=_admin_user(),
            prisma_client=pc,
            query_params={"limit": "20"},
        )

        assert result is not None
        assert result["has_more"] is True
        assert len(result["data"]) == 20  # extra row trimmed

    @pytest.mark.asyncio
    async def test_list_returns_none_for_non_list_route(self):
        pc = _prisma_with_list()

        result = await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/files/file-abc",  # single-resource, not a list
            user_api_key_dict=_user(),
            prisma_client=pc,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_list_db_error_returns_empty_not_none(self):
        """DB failure must return an empty list, not None (which would fall through
        to the upstream provider and leak the provider-wide listing)."""
        pc = _prisma_with_list()
        pc.db.litellm_managedfiletable.find_many = AsyncMock(
            side_effect=Exception("db down")
        )

        result = await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/files",
            user_api_key_dict=_admin_user(),
            prisma_client=pc,
        )

        # Must not return None (which would fall through to upstream)
        assert result is not None
        assert result["data"] == []
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_list_returns_empty_for_caller_without_identity(self):
        """Caller with neither user_id nor team_id should get an empty list."""
        pc = _prisma_with_list(
            file_rows=[_fake_file_row(new_managed_id("openai", "file-1"))]
        )
        anon = UserAPIKeyAuth()  # no user_id, no team_id, not admin

        result = await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/files",
            user_api_key_dict=anon,
            prisma_client=pc,
        )

        assert result is not None
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_list_files_pushes_provider_scope_to_db(self):
        """File listing scopes by provider at the DB level via the provider
        marker in flat_model_file_ids, so a single query serves the page and a
        mixed-provider pool can never truncate or leak the other provider.

        A large azure-only pool must return an empty openai page with
        has_more=False in exactly one DB round-trip.
        """
        azure_rows = [
            _fake_file_row(new_managed_id("azure", f"file-{i}")) for i in range(50)
        ]
        pc = _prisma_with_list(file_rows=azure_rows)

        result = await list_passthrough_ids_from_db(
            provider="openai",  # asking for openai but DB only has azure rows
            route="/openai/v1/files",
            user_api_key_dict=_admin_user(),
            prisma_client=pc,
            query_params={"limit": "20"},
        )

        assert result is not None
        assert result["data"] == []
        assert result["has_more"] is False
        where = pc.db.litellm_managedfiletable.find_many.call_args.kwargs["where"]
        assert where["flat_model_file_ids"] == {
            "has": _passthrough_provider_marker("openai")
        }
        assert pc.db.litellm_managedfiletable.find_many.await_count == 1

    @pytest.mark.asyncio
    async def test_list_ignores_cross_provider_cursor(self):
        """An ``after`` cursor minted for a different provider must not shift the
        created_at boundary: it would skip/repeat this provider's rows.  The
        cursor is ignored and the unscoped first page is served."""
        import datetime

        azure_row = _fake_file_row(new_managed_id("azure", "file-azure"))
        pc = _prisma_with_list(file_rows=[azure_row])

        cursor_row = MagicMock()
        cursor_row.created_at = datetime.datetime(
            2025, 6, 1, tzinfo=datetime.timezone.utc
        )
        pc.db.litellm_managedfiletable.find_first = AsyncMock(return_value=cursor_row)

        result = await list_passthrough_ids_from_db(
            provider="azure",
            route="/azure/openai/files",
            user_api_key_dict=_admin_user(),
            prisma_client=pc,
            query_params={"after": new_managed_id("openai", "file-openai")},
        )

        assert result is not None
        where = pc.db.litellm_managedfiletable.find_many.call_args.kwargs["where"]
        assert "created_at" not in where
        assert "OR" not in where and "AND" not in where

    @pytest.mark.asyncio
    async def test_list_applies_same_provider_cursor(self):
        """An ``after`` cursor minted for the same provider advances pagination
        past the cursor row using a compound (created_at, id) boundary so rows
        sharing the cursor row's timestamp are not skipped."""
        import datetime

        azure_row = _fake_file_row(new_managed_id("azure", "file-azure"))
        pc = _prisma_with_list(file_rows=[azure_row])

        cursor_row = MagicMock()
        cursor_row.created_at = datetime.datetime(
            2025, 6, 1, tzinfo=datetime.timezone.utc
        )
        pc.db.litellm_managedfiletable.find_first = AsyncMock(return_value=cursor_row)

        cursor_id = new_managed_id("azure", "file-cursor")
        result = await list_passthrough_ids_from_db(
            provider="azure",
            route="/azure/openai/files",
            user_api_key_dict=_admin_user(),
            prisma_client=pc,
            query_params={"after": cursor_id},
        )

        assert result is not None
        where = pc.db.litellm_managedfiletable.find_many.call_args.kwargs["where"]
        assert "created_at" not in where
        assert where["OR"] == [
            {"created_at": {"lt": cursor_row.created_at}},
            {
                "AND": [
                    {"created_at": cursor_row.created_at},
                    {"unified_file_id": {"lt": cursor_id}},
                ]
            },
        ]

    @pytest.mark.asyncio
    async def test_list_cursor_does_not_drop_created_at_ties(self):
        """Regression: paginating a pool whose rows all share one created_at must
        return every row exactly once. A timestamp-only ``lt`` cursor boundary
        would skip every tied row after the first page; the compound
        (created_at, id) boundary keeps the walk complete."""
        import datetime

        shared_ts = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        rows = [_fake_file_row(new_managed_id("azure", f"file-{i}")) for i in range(5)]
        for row in rows:
            row.created_at = shared_ts
        all_ids = {row.unified_file_id for row in rows}

        def _matches(row, where):
            for key, cond in where.items():
                if key == "AND":
                    if not all(_matches(row, c) for c in cond):
                        return False
                elif key == "OR":
                    if not any(_matches(row, c) for c in cond):
                        return False
                elif key == "flat_model_file_ids":
                    marker = (cond or {}).get("has")
                    if marker not in (getattr(row, "flat_model_file_ids", None) or []):
                        return False
                else:
                    actual = getattr(row, key, None)
                    if isinstance(cond, dict):
                        for op, val in cond.items():
                            if op == "lt" and not (actual is not None and actual < val):
                                return False
                            if op == "gt" and not (actual is not None and actual > val):
                                return False
                            if op == "startswith" and not str(actual or "").startswith(
                                val
                            ):
                                return False
                    elif actual != cond:
                        return False
            return True

        def _find_many(*_a, where=None, order=None, take=None, **_k):
            matched = [r for r in rows if _matches(r, where or {})]
            for spec in reversed(order or []):
                ((field, direction),) = spec.items()
                matched.sort(
                    key=lambda r: getattr(r, field), reverse=(direction == "desc")
                )
            return matched if take is None else matched[:take]

        def _find_first(*_a, where=None, **_k):
            return next((r for r in rows if _matches(r, where or {})), None)

        pc = _prisma_client()
        pc.db.litellm_managedfiletable.find_many = AsyncMock(side_effect=_find_many)
        pc.db.litellm_managedfiletable.find_first = AsyncMock(side_effect=_find_first)

        collected: list = []
        after = None
        for _ in range(len(rows) + 2):
            params = {"limit": "2"}
            if after is not None:
                params["after"] = after
            result = await list_passthrough_ids_from_db(
                provider="azure",
                route="/azure/openai/files",
                user_api_key_dict=_admin_user(),
                prisma_client=pc,
                query_params=params,
            )
            assert result is not None
            collected.extend(item["id"] for item in result["data"])
            if not result["has_more"]:
                break
            after = result["last_id"]

        assert sorted(collected) == sorted(all_ids)
        assert len(collected) == len(set(collected))

    @pytest.mark.asyncio
    async def test_list_files_filters_by_provider(self):
        openai_row = _fake_file_row(new_managed_id("openai", "file-openai"))
        azure_row = _fake_file_row(new_managed_id("azure", "file-azure"))
        pc = _prisma_with_list(file_rows=[azure_row, openai_row])

        result = await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/files",
            user_api_key_dict=_admin_user(),
            prisma_client=pc,
        )

        assert result is not None
        assert len(result["data"]) == 1
        assert decode(result["data"][0]["id"]).provider == "openai"

    @pytest.mark.asyncio
    async def test_list_batches_pushes_provider_scope_to_db(self):
        """Batch listing scopes by provider at the DB level via the namespaced
        model_object_id, so a single query serves the page instead of scanning."""
        batch_row = _fake_batch_row(new_managed_id("azure", "batch_abc"))
        pc = _prisma_with_list(batch_rows=[batch_row])

        result = await list_passthrough_ids_from_db(
            provider="azure",
            route="/azure/openai/batches",
            user_api_key_dict=_admin_user(),
            prisma_client=pc,
        )

        assert result is not None
        assert len(result["data"]) == 1
        where = pc.db.litellm_managedobjecttable.find_many.call_args.kwargs["where"]
        assert where["model_object_id"] == {"startswith": "passthrough:azure:"}
        assert pc.db.litellm_managedobjecttable.find_many.await_count == 1
