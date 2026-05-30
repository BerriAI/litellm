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
import sys
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.managed_id_codec import (
    decode,
    encode,
    is_managed,
    new_managed_id,
)
from litellm.proxy.pass_through_endpoints.managed_id_rewriter import (
    _canonical_path,
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
    pc.db.litellm_managedfiletable.create = AsyncMock(return_value=None)
    pc.db.litellm_managedobjecttable = MagicMock()
    pc.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=None)
    pc.db.litellm_managedobjecttable.upsert = AsyncMock(return_value=None)
    return pc


def _managed_files_hook(store_side_effect: Any = None) -> MagicMock:
    hook = MagicMock()
    hook.get_unified_file_id = AsyncMock(return_value=None)
    hook.store_unified_file_id = AsyncMock(side_effect=store_side_effect)
    return hook


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

        pc = _prisma_client()
        # Dedup lookup finds existing row
        pc.db.litellm_managedfiletable.find_first = AsyncMock(return_value=existing_row)
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
        pc.db.litellm_managedfiletable.find_first = AsyncMock(return_value=existing_row)
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
    """Return a prisma_client whose find_many returns the given fake rows."""
    pc = _prisma_client()

    if file_rows is not None:
        pc.db.litellm_managedfiletable.find_many = AsyncMock(return_value=file_rows)
    if batch_rows is not None:
        pc.db.litellm_managedobjecttable.find_many = AsyncMock(return_value=batch_rows)
    return pc


def _fake_file_row(
    unified_id: str, created_by: str = "user-1", team_id: str = "team-1"
):
    row = MagicMock()
    row.unified_file_id = unified_id
    row.created_by = created_by
    row.team_id = team_id
    row.file_object = {"filename": "test.jsonl", "bytes": 42, "purpose": "batch"}

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
        # Verify the DB was called with empty where (unscoped = admin)
        call_kwargs = pc.db.litellm_managedfiletable.find_many.call_args.kwargs
        assert call_kwargs["where"] == {}

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
    async def test_list_has_more_true_when_scan_cap_hit_with_full_batch(self):
        """
        When the 20-iteration scan cap is exhausted and the last DB batch was
        full-sized (indicating more rows exist), has_more must be True even if
        we haven't accumulated fetch_limit matched rows yet.  Without this fix,
        a high-mixed-provider pool returns has_more=False on a truncated page.
        """
        import datetime
        from unittest.mock import AsyncMock

        # Simulate a pool of rows where every DB page is full (fetch_limit = 21)
        # but ALL rows belong to "azure" — so an "openai" scan never matches any.
        # After max_scans the last batch was still full, so has_more must be True.
        fetch_limit = 21  # raw_limit=20 → fetch_limit=21
        azure_rows = [
            _fake_file_row(new_managed_id("azure", f"file-{i}"))
            for i in range(fetch_limit)
        ]
        # Shift created_at so repeated calls return "different" pages
        for i, row in enumerate(azure_rows):
            row.created_at = datetime.datetime(
                2025, 1, 1, tzinfo=datetime.timezone.utc
            ) - datetime.timedelta(seconds=i)

        call_count = 0

        async def always_full_azure_pages(**kwargs):
            nonlocal call_count
            call_count += 1
            return azure_rows  # always returns a full page of azure rows

        pc = _prisma_with_list()
        pc.db.litellm_managedfiletable.find_many = always_full_azure_pages

        result = await list_passthrough_ids_from_db(
            provider="openai",  # asking for openai but DB only has azure rows
            route="/openai/v1/files",
            user_api_key_dict=_admin_user(),
            prisma_client=pc,
            query_params={"limit": "20"},
        )

        assert result is not None
        # No openai-matched rows, but scan cap was hit with a full last batch
        assert result["has_more"] is True
        assert result["data"] == []

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
