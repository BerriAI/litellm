"""
Tests for enterprise/litellm_enterprise/proxy/hooks/managed_files.py

Regression test for afile_retrieve called without credentials in
async_post_call_success_hook when processing completed batch responses.
"""

import asyncio
import base64
import json
import logging

import pytest
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.types.llms.openai import FileListPage, OpenAIFileObject
from litellm.types.utils import LiteLLMBatch


def _make_file_object(file_id: str = "file-output-abc") -> OpenAIFileObject:
    return OpenAIFileObject(
        id=file_id,
        bytes=100,
        created_at=1700000000,
        filename="output.jsonl",
        object="file",
        purpose="batch_output",
        status="processed",
    )


def _make_batch_response(
    batch_id: str = "batch-123",
    output_file_id: Optional[str] = "file-output-abc",
    status: str = "completed",
    model_id: str = "model-deploy-xyz",
    model_name: str = "azure/gpt-4",
) -> LiteLLMBatch:
    """Create a LiteLLMBatch response with hidden params set as the router would."""
    batch = LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id="file-input-abc",
        object="batch",
        status=status,
        output_file_id=output_file_id,
    )
    batch._hidden_params = {
        "unified_file_id": "some-unified-id",
        "unified_batch_id": "some-unified-batch-id",
        "model_id": model_id,
        "model_name": model_name,
    }
    return batch


def _make_user_api_key_dict() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id="test-user",
        parent_otel_span=None,
    )


def _make_team_member_api_key_dict() -> UserAPIKeyAuth:
    """The shape most real virtual keys carry: a user_id and a team_id."""
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id="test-user",
        team_id="test-team",
        parent_otel_span=None,
    )


def _make_service_account_api_key_dict() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-service",
        team_id="test-team",
        parent_otel_span=None,
    )


def _make_admin_api_key_dict() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-admin",
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        parent_otel_span=None,
    )


def _make_managed_file_row(
    unified_file_id: str,
    purpose: str = "batch_output",
    created_by: str = "test-user",
    team_id: Optional[str] = None,
) -> MagicMock:
    file_object = _make_file_object(f"file-provider-{unified_file_id}").model_copy(
        update={"purpose": purpose}
    )
    return MagicMock(
        unified_file_id=unified_file_id,
        file_object=file_object.model_dump(),
        created_by=created_by,
        team_id=team_id,
    )


def _make_unparseable_managed_file_row(
    unified_file_id: str,
    created_by: str = "test-user",
    team_id: Optional[str] = None,
) -> MagicMock:
    """A row whose stored blob cannot be parsed back into a file object."""
    return MagicMock(
        unified_file_id=unified_file_id,
        file_object=None,
        created_by=created_by,
        team_id=team_id,
    )


def _row_matches_where(row, where) -> bool:
    """Apply the Prisma ``where`` shapes build_owner_filter actually emits:
    ``{}``, a single equality, and the ``OR`` of equalities a key carrying
    both a user_id and a team_id produces."""
    for field, expected in where.items():
        if field == "OR":
            if not any(_row_matches_where(row, clause) for clause in expected):
                return False
        elif getattr(row, field) != expected:
            return False
    return True


class _FakeManagedFileTable:
    """In-memory stand-in for the managed file table, newest row first."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.find_many_calls = []
        self.find_first_calls = []

    def _owned_rows(self, where):
        return [row for row in self.rows if _row_matches_where(row, where)]

    async def find_first(self, where):
        self.find_first_calls.append(where)
        return next(iter(self._owned_rows(where)), None)

    async def find_many(self, where, take=None, order=None, cursor=None, skip=0):
        self.find_many_calls.append(
            {"where": where, "take": take, "order": order, "cursor": cursor, "skip": skip}
        )
        rows = self._owned_rows(where)
        if cursor is not None:
            start = next(
                index
                for index, row in enumerate(rows)
                if row.unified_file_id == cursor["unified_file_id"]
            )
            rows = rows[start + skip :]
        return rows if take is None else rows[:take]


def _make_managed_files_over_rows(rows):
    managed_files = _make_managed_files_instance()
    table = _FakeManagedFileTable(rows)
    managed_files.prisma_client.db.litellm_managedfiletable = table
    return managed_files, table


def _make_managed_files_instance():
    """Create a _PROXY_LiteLLMManagedFiles with storage methods mocked out."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    mock_cache = MagicMock()
    mock_prisma = MagicMock()

    instance = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=mock_cache,
        prisma_client=mock_prisma,
    )
    instance.store_unified_file_id = AsyncMock()
    instance.store_unified_object_id = AsyncMock()
    return instance


@pytest.mark.asyncio
async def test_should_pass_credentials_to_afile_retrieve():
    """
    When async_post_call_success_hook processes a completed batch with an output_file_id,
    it calls afile_retrieve to fetch file metadata. It must pass credentials from the
    router deployment, not just custom_llm_provider and file_id.

    Regression test for: managed_files.py:919 calling afile_retrieve without api_key/api_base.
    """
    managed_files = _make_managed_files_instance()
    batch_response = _make_batch_response(
        model_id="model-deploy-xyz",
        model_name="azure/gpt-4",
        output_file_id="file-output-abc",
    )
    user_api_key_dict = _make_user_api_key_dict()

    mock_credentials = {
        "api_key": "test-azure-key",
        "api_base": "https://my-azure.openai.azure.com/",
        "api_version": "2025-03-01-preview",
        "custom_llm_provider": "azure",
    }

    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(
        return_value=mock_credentials
    )

    mock_afile_retrieve = AsyncMock(return_value=_make_file_object("file-output-abc"))

    with (
        patch("litellm.afile_retrieve", mock_afile_retrieve),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch_response,
        )

        mock_afile_retrieve.assert_called()
        call_kwargs = mock_afile_retrieve.call_args

        assert call_kwargs.kwargs.get("api_key") == "test-azure-key", (
            f"afile_retrieve must receive api_key from router credentials. "
            f"Got kwargs: {call_kwargs.kwargs}"
        )
        assert (
            call_kwargs.kwargs.get("api_base") == "https://my-azure.openai.azure.com/"
        ), (
            f"afile_retrieve must receive api_base from router credentials. "
            f"Got kwargs: {call_kwargs.kwargs}"
        )


@pytest.mark.asyncio
async def test_get_user_created_file_ids_skips_rows_without_file_object():
    managed_files = _make_managed_files_instance()
    managed_files.prisma_client.db.litellm_managedfiletable.find_many = AsyncMock(
        return_value=[
            MagicMock(
                file_object=_make_file_object().model_dump(),
                unified_file_id="unified-id-1",
            ),
            MagicMock(file_object=None, unified_file_id="unified-id-2"),
        ]
    )

    files = await managed_files.get_user_created_file_ids(
        _make_user_api_key_dict(), ["file-output-abc"]
    )

    assert [file.id for file in files] == ["unified-id-1"]


@pytest.mark.asyncio
async def test_get_user_created_file_ids_remaps_stored_raw_provider_id_to_unified_id():
    """
    Rows registered from batch outputs store the provider's file object, whose
    id is the raw provider id (e.g. file-abc). Listing must return the row's
    unified_file_id so callers get ids that work on the managed routes.

    Regression test for https://github.com/BerriAI/litellm/issues/35362.
    """
    unified_id = "bGl0ZWxsbV9wcm94eTt1bmlmaWVkX2lkLGRlYWRiZWVm"
    raw_provider_object = _make_file_object("file-raw-provider-123")
    managed_files = _make_managed_files_instance()
    managed_files.prisma_client.db.litellm_managedfiletable.find_many = AsyncMock(
        return_value=[
            MagicMock(
                file_object=raw_provider_object.model_dump(),
                unified_file_id=unified_id,
            ),
        ]
    )

    files = await managed_files.get_user_created_file_ids(
        _make_user_api_key_dict(), ["file-raw-provider-123"]
    )

    assert [file.id for file in files] == [unified_id]
    assert files[0].filename == raw_provider_object.filename
    assert files[0].purpose == raw_provider_object.purpose


@pytest.mark.asyncio
async def test_afile_list_returns_owner_scoped_managed_files():
    managed_files = _make_managed_files_instance()
    managed_files.prisma_client.db.litellm_managedfiletable.find_many = AsyncMock(
        return_value=[
            MagicMock(
                file_object=_make_file_object("file-provider-id").model_dump(),
                unified_file_id="unified-file-id",
            ),
            MagicMock(
                file_object=_make_file_object("file-other-purpose").model_copy(
                    update={"purpose": "batch"}
                ).model_dump(),
                unified_file_id="unified-other-purpose",
            ),
        ]
    )

    response = await managed_files.afile_list(
        purpose="batch_output",
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
    )

    managed_files.prisma_client.db.litellm_managedfiletable.find_many.assert_awaited_once_with(
        where={"created_by": "test-user"},
        take=10001,
        order=[{"created_at": "desc"}, {"unified_file_id": "desc"}],
    )
    assert [file.id for file in response.data] == ["unified-file-id"]
    assert response.first_id == "unified-file-id"
    assert response.last_id == "unified-file-id"
    assert response.has_more is False


@pytest.mark.asyncio
async def test_afile_list_returns_a_page_object_callbacks_can_read():
    """Post-call hooks receive the listing and read ``.data`` off it, the way the
    provider SDK's page lets them. The body on the wire stays a plain list page."""
    from fastapi.encoders import jsonable_encoder

    managed_files, _ = _make_managed_files_over_rows([_make_managed_file_row("unified-file-id")])

    page = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
    )

    assert isinstance(page, FileListPage)
    assert [file.id for file in page.data] == ["unified-file-id"]

    body = jsonable_encoder(page)
    assert list(body) == ["object", "data", "first_id", "last_id", "has_more"]
    assert body["object"] == "list"
    assert [file["id"] for file in body["data"]] == ["unified-file-id"]
    assert body["first_id"] == "unified-file-id"
    assert body["last_id"] == "unified-file-id"
    assert body["has_more"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", ["nonexistent_purpose", "EVALS", "batch "])
async def test_afile_list_rejects_a_purpose_the_files_api_never_accepts(purpose):
    """No stored file can carry an undocumented purpose, so filtering on one is a
    bad request rather than a legitimately empty page."""
    managed_files, table = _make_managed_files_over_rows([_make_managed_file_row("unified-file-id")])

    with pytest.raises(ProxyException) as exc_info:
        await managed_files.afile_list(
            purpose=purpose,
            litellm_parent_otel_span=None,
            user_api_key_dict=_make_user_api_key_dict(),
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.param == "purpose"
    assert table.find_many_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", ["batch", "assistants", "fine-tune", "evals", None])
async def test_afile_list_accepts_every_documented_purpose(purpose):
    managed_files, _ = _make_managed_files_over_rows([_make_managed_file_row("unified-file-id")])

    page = await managed_files.afile_list(
        purpose=purpose,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
    )

    assert isinstance(page, FileListPage)


@pytest.mark.asyncio
async def test_afile_list_does_not_leak_another_callers_files():
    managed_files, table = _make_managed_files_over_rows(
        [
            _make_managed_file_row("unified-mine-2"),
            _make_managed_file_row("unified-theirs", created_by="other-user"),
            _make_managed_file_row("unified-mine-1"),
        ]
    )

    response = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
    )

    assert [file.id for file in response.data] == ["unified-mine-2", "unified-mine-1"]
    assert table.find_many_calls[0]["where"] == {"created_by": "test-user"}


@pytest.mark.asyncio
async def test_afile_list_returns_own_and_team_files_for_a_key_carrying_both_ids():
    managed_files, table = _make_managed_files_over_rows(
        [
            _make_managed_file_row("unified-mine"),
            _make_managed_file_row("unified-teammates", created_by="other-user", team_id="test-team"),
            _make_managed_file_row("unified-outsiders", created_by="outsider", team_id="other-team"),
        ]
    )

    response = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_team_member_api_key_dict(),
    )

    assert [file.id for file in response.data] == ["unified-mine", "unified-teammates"]
    assert table.find_many_calls[0]["where"] == {
        "OR": [{"created_by": "test-user"}, {"team_id": "test-team"}]
    }


@pytest.mark.asyncio
async def test_afile_list_scopes_a_service_account_key_to_its_team():
    managed_files, table = _make_managed_files_over_rows(
        [
            _make_managed_file_row("unified-teams", created_by="other-user", team_id="test-team"),
            _make_managed_file_row("unified-outsiders", created_by="outsider", team_id="other-team"),
        ]
    )

    response = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_service_account_api_key_dict(),
    )

    assert [file.id for file in response.data] == ["unified-teams"]
    assert table.find_many_calls[0]["where"] == {"team_id": "test-team"}


@pytest.mark.asyncio
async def test_afile_list_returns_every_callers_files_for_a_proxy_admin():
    managed_files, table = _make_managed_files_over_rows(
        [
            _make_managed_file_row("unified-mine"),
            _make_managed_file_row("unified-theirs", created_by="other-user", team_id="other-team"),
        ]
    )

    response = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_admin_api_key_dict(),
    )

    assert [file.id for file in response.data] == ["unified-mine", "unified-theirs"]
    assert table.find_many_calls[0]["where"] == {}


@pytest.mark.asyncio
async def test_afile_list_pages_a_team_key_across_both_halves_of_its_filter():
    """Keyset pagination has to walk an OR filter as one ordered set, without
    repeating a row across pages or dropping one between them."""
    managed_files, table = _make_managed_files_over_rows(
        [
            _make_managed_file_row("unified-0"),
            _make_managed_file_row("unified-1", created_by="other-user", team_id="test-team"),
            _make_managed_file_row("unified-2"),
            _make_managed_file_row("unified-3", created_by="outsider", team_id="other-team"),
            _make_managed_file_row("unified-4", created_by="other-user", team_id="test-team"),
        ]
    )
    user_api_key_dict = _make_team_member_api_key_dict()

    seen = []
    cursor = None
    for _ in range(4):
        response = await managed_files.afile_list(
            purpose=None,
            litellm_parent_otel_span=None,
            user_api_key_dict=user_api_key_dict,
            limit=2,
            after=cursor,
        )
        seen.extend(file.id for file in response.data)
        if not response.has_more:
            break
        cursor = response.last_id

    assert seen == ["unified-0", "unified-1", "unified-2", "unified-4"]
    assert all(
        call["where"] == {"OR": [{"created_by": "test-user"}, {"team_id": "test-team"}]}
        for call in table.find_many_calls
    )


@pytest.mark.asyncio
async def test_afile_list_orders_newest_first_and_breaks_ties_on_the_cursor_column():
    managed_files, table = _make_managed_files_over_rows([_make_managed_file_row("unified-mine")])

    await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
    )

    assert table.find_many_calls[0]["order"] == [
        {"created_at": "desc"},
        {"unified_file_id": "desc"},
    ]


@pytest.mark.asyncio
async def test_afile_list_denies_a_caller_without_a_user_or_team():
    managed_files, table = _make_managed_files_over_rows([_make_managed_file_row("unified-mine")])

    response = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", parent_otel_span=None),
    )

    assert response.data == []
    assert response.has_more is False
    assert table.find_many_calls == []


@pytest.mark.asyncio
async def test_afile_list_filters_by_purpose():
    managed_files, _ = _make_managed_files_over_rows(
        [
            _make_managed_file_row("unified-batch-output"),
            _make_managed_file_row("unified-batch", purpose="batch"),
        ]
    )

    response = await managed_files.afile_list(
        purpose="batch",
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
    )

    assert [file.id for file in response.data] == ["unified-batch"]


async def _walk_afile_list(managed_files, user_api_key_dict, purpose, limit):
    """Page through the listing the way the official SDK does, off ``data[-1].id``."""
    seen = []
    after = None
    while True:
        page = await managed_files.afile_list(
            purpose=purpose,
            litellm_parent_otel_span=None,
            user_api_key_dict=user_api_key_dict,
            limit=limit,
            after=after,
        )
        page_ids = [file.id for file in page.data]
        assert not set(page_ids) & set(seen)
        seen.extend(page_ids)
        if not page.has_more:
            return seen
        assert page_ids, "an SDK stops paging on an empty page, so has_more must never ride one"
        after = page_ids[-1]


@pytest.mark.asyncio
async def test_afile_list_fills_a_page_past_rows_the_purpose_filter_drops():
    """The newest rows do not match, so the page must reach past them rather than come back empty."""
    managed_files, _ = _make_managed_files_over_rows(
        [
            _make_managed_file_row("unified-0"),
            _make_managed_file_row("unified-1"),
            _make_managed_file_row("unified-2", purpose="batch"),
            _make_managed_file_row("unified-3"),
            _make_managed_file_row("unified-4", purpose="batch"),
        ]
    )
    user_api_key_dict = _make_user_api_key_dict()

    first_page = await managed_files.afile_list(
        purpose="batch",
        litellm_parent_otel_span=None,
        user_api_key_dict=user_api_key_dict,
        limit=1,
    )

    assert [file.id for file in first_page.data] == ["unified-2"]
    assert first_page.has_more is True
    assert first_page.last_id == "unified-2"

    second_page = await managed_files.afile_list(
        purpose="batch",
        litellm_parent_otel_span=None,
        user_api_key_dict=user_api_key_dict,
        limit=1,
        after=first_page.last_id,
    )

    assert [file.id for file in second_page.data] == ["unified-4"]
    assert second_page.has_more is False


@pytest.mark.parametrize("limit", [1, 2, 3])
@pytest.mark.asyncio
async def test_afile_list_walks_every_purpose_match_at_any_limit(limit):
    managed_files, _ = _make_managed_files_over_rows(
        [
            _make_managed_file_row("unified-0"),
            _make_managed_file_row("unified-1"),
            _make_managed_file_row("unified-2", purpose="batch"),
            _make_managed_file_row("unified-3"),
            _make_managed_file_row("unified-4", purpose="batch"),
            _make_managed_file_row("unified-5", purpose="batch"),
            _make_managed_file_row("unified-6"),
        ]
    )

    seen = await _walk_afile_list(managed_files, _make_user_api_key_dict(), "batch", limit)

    assert seen == ["unified-2", "unified-4", "unified-5"]


@pytest.mark.asyncio
async def test_afile_list_fills_a_page_past_rows_that_do_not_parse():
    managed_files, _ = _make_managed_files_over_rows(
        [
            _make_unparseable_managed_file_row("unified-0"),
            _make_unparseable_managed_file_row("unified-1"),
            _make_managed_file_row("unified-2"),
        ]
    )

    page = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
        limit=1,
    )

    assert [file.id for file in page.data] == ["unified-2"]
    assert page.has_more is False


_DEEP_SCAN_ROW_COUNT = 2000
_DEEP_SCAN_QUERY_BUDGET = 10


@pytest.mark.asyncio
async def test_afile_list_bounds_the_queries_a_deep_purpose_match_costs():
    """A tiny limit over rows the filter drops must not turn one request into thousands of queries."""
    managed_files, table = _make_managed_files_over_rows(
        [_make_managed_file_row(f"unified-{index:05d}") for index in range(_DEEP_SCAN_ROW_COUNT)]
        + [_make_managed_file_row("unified-match", purpose="batch")]
    )

    page = await managed_files.afile_list(
        purpose="batch",
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
        limit=1,
    )

    assert [file.id for file in page.data] == ["unified-match"]
    assert page.has_more is False
    assert len(table.find_many_calls) <= _DEEP_SCAN_QUERY_BUDGET


@pytest.mark.asyncio
async def test_afile_list_bounds_the_queries_a_deep_unparseable_run_costs():
    """Rows that will not parse drop out like a filter does, so they get the same bound."""
    managed_files, table = _make_managed_files_over_rows(
        [_make_unparseable_managed_file_row(f"unified-{index:05d}") for index in range(_DEEP_SCAN_ROW_COUNT)]
        + [_make_managed_file_row("unified-parses")]
    )

    page = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
        limit=1,
    )

    assert [file.id for file in page.data] == ["unified-parses"]
    assert page.has_more is False
    assert len(table.find_many_calls) <= _DEEP_SCAN_QUERY_BUDGET


@pytest.mark.asyncio
async def test_afile_list_reads_one_chunk_when_the_first_one_fills_the_page():
    """The widened chunk must stay off the common path, where the newest rows already fill the page."""
    managed_files, table = _make_managed_files_over_rows(
        [_make_managed_file_row(f"unified-{index:05d}") for index in range(_DEEP_SCAN_ROW_COUNT)]
    )

    page = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
        limit=2,
    )

    assert [file.id for file in page.data] == ["unified-00000", "unified-00001"]
    assert page.has_more is True
    assert [call["take"] for call in table.find_many_calls] == [3]


@pytest.mark.asyncio
async def test_afile_list_reports_no_more_pages_when_nothing_matches():
    managed_files, _ = _make_managed_files_over_rows(
        [_make_managed_file_row(f"unified-{index}") for index in range(5)]
    )

    page = await managed_files.afile_list(
        purpose="batch",
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
        limit=2,
    )

    assert page.data == []
    assert page.has_more is False
    assert page.first_id is None
    assert page.last_id is None


@pytest.mark.asyncio
async def test_afile_list_honors_limit_and_reports_more_pages():
    managed_files, table = _make_managed_files_over_rows(
        [_make_managed_file_row(f"unified-{index}") for index in range(5)]
    )

    response = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
        limit=2,
    )

    assert [file.id for file in response.data] == ["unified-0", "unified-1"]
    assert response.has_more is True
    assert table.find_many_calls[0]["take"] == 3


@pytest.mark.asyncio
async def test_afile_list_pages_through_every_file_without_overlap():
    managed_files, table = _make_managed_files_over_rows(
        [_make_managed_file_row(f"unified-{index}") for index in range(5)]
    )
    user_api_key_dict = _make_user_api_key_dict()

    seen = []
    after = None
    while True:
        page = await managed_files.afile_list(
            purpose=None,
            litellm_parent_otel_span=None,
            user_api_key_dict=user_api_key_dict,
            limit=2,
            after=after,
        )
        page_ids = [file.id for file in page.data]
        assert not set(page_ids) & set(seen)
        seen.extend(page_ids)
        if not page.has_more:
            break
        after = page.last_id

    assert seen == [f"unified-{index}" for index in range(5)]
    assert table.find_many_calls[1]["cursor"] == {"unified_file_id": "unified-1"}
    assert table.find_many_calls[1]["skip"] == 1


@pytest.mark.parametrize(
    "unknown_cursor",
    ["unified-theirs", "unified-nowhere"],
    ids=["another-users-file", "no-such-file"],
)
@pytest.mark.asyncio
async def test_afile_list_rejects_an_after_cursor_outside_the_callers_files(unknown_cursor):
    from litellm.proxy._types import ProxyException

    managed_files, table = _make_managed_files_over_rows(
        [
            _make_managed_file_row("unified-mine"),
            _make_managed_file_row("unified-theirs", created_by="other-user"),
        ]
    )

    with pytest.raises(ProxyException) as exc_info:
        await managed_files.afile_list(
            purpose=None,
            litellm_parent_otel_span=None,
            user_api_key_dict=_make_user_api_key_dict(),
            after=unknown_cursor,
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.param == "after"
    assert exc_info.value.message == f"Invalid 'after' cursor: no file found with id '{unknown_cursor}'."
    assert table.find_first_calls[0] == {
        "created_by": "test-user",
        "unified_file_id": unknown_cursor,
    }
    assert table.find_many_calls == []


@pytest.mark.parametrize(
    "limit, bound, expected_range",
    [
        (0, "below minimum", ">= 1"),
        (-1, "below minimum", ">= 1"),
        (10001, "above maximum", "<= 10000"),
    ],
)
@pytest.mark.asyncio
async def test_afile_list_rejects_a_limit_outside_the_openai_range(limit, bound, expected_range):
    from litellm.proxy._types import ProxyException

    managed_files, table = _make_managed_files_over_rows([_make_managed_file_row("unified-mine")])

    with pytest.raises(ProxyException) as exc_info:
        await managed_files.afile_list(
            purpose=None,
            litellm_parent_otel_span=None,
            user_api_key_dict=_make_user_api_key_dict(),
            limit=limit,
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.param == "limit"
    assert exc_info.value.message == (
        f"Invalid 'limit': integer {bound} value. Expected a value {expected_range}, but got {limit} instead."
    )
    assert table.find_many_calls == []


@pytest.mark.parametrize("limit", [1, 10000])
@pytest.mark.asyncio
async def test_afile_list_accepts_the_ends_of_the_openai_limit_range(limit):
    managed_files, table = _make_managed_files_over_rows([_make_managed_file_row("unified-mine")])

    response = await managed_files.afile_list(
        purpose=None,
        litellm_parent_otel_span=None,
        user_api_key_dict=_make_user_api_key_dict(),
        limit=limit,
    )

    assert [file.id for file in response.data] == ["unified-mine"]
    assert response.has_more is False
    assert table.find_many_calls[0]["take"] == limit + 1


@pytest.mark.asyncio
async def test_parse_managed_file_object_warning_omits_rejected_values(caplog):
    from litellm_enterprise.proxy.hooks.managed_files import (
        _parse_managed_file_object,
    )

    with caplog.at_level(logging.WARNING):
        parsed = _parse_managed_file_object(
            {"id": "file-corrupt", "object": "file", "filename": "confidential.jsonl"},
            "unified-corrupt",
        )

    assert parsed is None
    assert "unified-corrupt" in caplog.text
    assert "bytes" in caplog.text
    assert "confidential.jsonl" not in caplog.text


@pytest.mark.asyncio
async def test_get_user_created_file_ids_skips_unparseable_rows():
    managed_files = _make_managed_files_instance()
    managed_files.prisma_client.db.litellm_managedfiletable.find_many = AsyncMock(
        return_value=[
            MagicMock(
                file_object={"id": "file-corrupt", "object": "file"},
                unified_file_id="unified-corrupt",
            ),
            MagicMock(
                file_object=_make_file_object().model_dump(),
                unified_file_id="unified-valid",
            ),
        ]
    )

    files = await managed_files.get_user_created_file_ids(
        _make_user_api_key_dict(), ["file-output-abc"]
    )

    assert [file.id for file in files] == ["unified-valid"]


@pytest.mark.asyncio
async def test_should_fallback_when_no_router():
    """
    When llm_router is not available, afile_retrieve should still be called
    with the fallback behavior (custom_llm_provider extracted from model_name).
    """
    managed_files = _make_managed_files_instance()
    batch_response = _make_batch_response(
        model_id="model-deploy-xyz",
        model_name="azure/gpt-4",
        output_file_id="file-output-abc",
    )
    user_api_key_dict = _make_user_api_key_dict()

    mock_afile_retrieve = AsyncMock(return_value=_make_file_object("file-output-abc"))

    with (
        patch("litellm.afile_retrieve", mock_afile_retrieve),
        patch("litellm.proxy.proxy_server.llm_router", None),
    ):
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch_response,
        )

        mock_afile_retrieve.assert_called()
        call_kwargs = mock_afile_retrieve.call_args
        assert call_kwargs.kwargs.get("custom_llm_provider") == "azure"
        assert call_kwargs.kwargs.get("file_id") == "file-output-abc"


@pytest.mark.asyncio
async def test_should_not_double_wrap_already_unified_output_file_id():
    """After ensure_batch_response_managed_file_ids, retrieve must not re-wrap
    output_file_id or store a nested unified id as the provider mapping."""
    import base64

    managed_files = _make_managed_files_instance()
    provider_file_id = "file-WXWt9R4LzmU5WpeKzjCfLR"
    model_id = "openai/openai/gpt-5.5-batch"
    already_unified = managed_files.get_unified_output_file_id(
        output_file_id=provider_file_id,
        model_id=model_id,
        model_name="openai/openai/gpt-5.5-batch",
    )

    batch_response = _make_batch_response(
        model_id=model_id,
        model_name="openai/openai/gpt-5.5-batch",
        output_file_id=already_unified,
    )
    user_api_key_dict = _make_user_api_key_dict()

    mock_credentials = {
        "api_key": "test-key",
        "api_base": "https://api.openai.com/v1",
        "custom_llm_provider": "openai",
    }
    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(
        return_value=mock_credentials
    )
    mock_afile_retrieve = AsyncMock(return_value=_make_file_object(provider_file_id))

    with (
        patch("litellm.afile_retrieve", mock_afile_retrieve),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch_response,
        )

    assert batch_response.output_file_id == already_unified
    mock_afile_retrieve.assert_called_once()
    assert mock_afile_retrieve.call_args.kwargs["file_id"] == provider_file_id
    managed_files.store_unified_file_id.assert_awaited_once()
    assert managed_files.store_unified_file_id.await_args.kwargs["model_mappings"] == {
        model_id: provider_file_id
    }

    decoded = base64.urlsafe_b64decode(
        already_unified + "=" * (-len(already_unified) % 4)
    ).decode()
    assert decoded.count(f"llm_output_file_id,{provider_file_id}") == 1


@pytest.mark.asyncio
async def test_should_skip_non_file_unified_id_on_output_file_id():
    """Batch-style unified ids lack llm_output_file_id; must not IndexError or re-wrap."""
    import base64

    managed_files = _make_managed_files_instance()
    batch_unified = (
        base64.urlsafe_b64encode(
            b"litellm_proxy;model_id:openai/openai/gpt-5.5-batch;llm_batch_id:batch_abc"
        )
        .decode()
        .rstrip("=")
    )

    batch_response = _make_batch_response(
        model_id="openai/openai/gpt-5.5-batch",
        model_name="openai/openai/gpt-5.5-batch",
        output_file_id=batch_unified,
    )
    user_api_key_dict = _make_user_api_key_dict()

    with patch("litellm.afile_retrieve", AsyncMock()) as mock_afile_retrieve:
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch_response,
        )

    assert batch_response.output_file_id == batch_unified
    mock_afile_retrieve.assert_not_called()
    managed_files.store_unified_file_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_afile_content_passes_trusted_model_credentials_to_router():
    """
    afile_content must hand the deployment's credential snapshot to the router
    call as an immutable server-side mapping. Cloud-storage providers (Bedrock
    S3) validate file ids against the bucket in that snapshot, so without it
    unified-id content retrieval only works when AWS_S3_BUCKET_NAME is set.
    """
    from types import MappingProxyType

    managed_files = _make_managed_files_instance()
    unified_file_id = "unified-file-id"
    s3_uri = "s3://my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    managed_files.get_model_file_id_mapping = AsyncMock(
        return_value={unified_file_id: {"model-123": s3_uri}}
    )

    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(
        return_value={
            "custom_llm_provider": "bedrock",
            "s3_bucket_name": "my-bucket",
            "aws_region_name": "us-west-2",
        }
    )
    mock_router.afile_content = AsyncMock(return_value=MagicMock())

    await managed_files.afile_content(
        file_id=unified_file_id,
        litellm_parent_otel_span=None,
        llm_router=mock_router,
    )

    call_kwargs = mock_router.afile_content.call_args.kwargs
    assert call_kwargs["model"] == "model-123"
    assert call_kwargs["file_id"] == s3_uri
    trusted_credentials = call_kwargs["_litellm_internal_model_credentials"]
    assert isinstance(trusted_credentials, MappingProxyType)
    assert trusted_credentials["s3_bucket_name"] == "my-bucket"


@pytest.mark.asyncio
async def test_afile_content_bedrock_unified_id_end_to_end(monkeypatch):
    """
    Proxy repro for Bedrock batch output retrieval: a unified file id that
    resolves to an s3:// output object must be fetched via a SigV4-signed S3
    GET using the deployment's s3_bucket_name (no AWS_S3_BUCKET_NAME env).

    Regression test for "BedrockFilesConfig does not support file content
    retrieval" raised on this path.
    """
    import httpx
    import respx

    import litellm
    from litellm import Router

    monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()

    router = Router(
        model_list=[
            {
                "model_name": "bedrock-claude",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "aws_access_key_id": "AKIAEXAMPLE",
                    "aws_secret_access_key": "secret",
                    "aws_region_name": "us-west-2",
                    "s3_bucket_name": "my-bucket",
                },
                "model_info": {"id": "model-123"},
            }
        ]
    )

    managed_files = _make_managed_files_instance()
    unified_file_id = "unified-file-id"
    s3_uri = "s3://my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    managed_files.get_model_file_id_mapping = AsyncMock(
        return_value={unified_file_id: {"model-123": s3_uri}}
    )

    expected_url = "https://s3.us-west-2.amazonaws.com/my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    with respx.mock:
        route = respx.get(expected_url).mock(
            return_value=httpx.Response(200, content=b'{"recordId": "x"}')
        )

        response = await managed_files.afile_content(
            file_id=unified_file_id,
            litellm_parent_otel_span=None,
            llm_router=router,
        )

    assert route.called
    assert (
        route.calls[0].request.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
    )
    assert response.content == b'{"recordId": "x"}'


@pytest.mark.asyncio
async def test_afile_content_error_reports_unified_id_not_provider_uri():
    """When every model attempt fails, the error must name the caller's unified
    file id, never the resolved internal s3:// URI (no internal-path leak)."""
    managed_files = _make_managed_files_instance()
    unified_file_id = "litellm_proxy_unified_id_abc"
    s3_uri = "s3://my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    managed_files.get_model_file_id_mapping = AsyncMock(
        return_value={unified_file_id: {"model-123": s3_uri}}
    )

    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(return_value=None)
    mock_router.afile_content = AsyncMock(side_effect=Exception("deployment failed"))

    with pytest.raises(Exception, match='LiteLLM Managed File object with') as exc_info:
        await managed_files.afile_content(
            file_id=unified_file_id,
            litellm_parent_otel_span=None,
            llm_router=mock_router,
        )

    message = str(exc_info.value)
    assert unified_file_id in message
    assert s3_uri not in message


def _make_real_managed_files_instance():
    """Create a _PROXY_LiteLLMManagedFiles with a real store_unified_file_id but
    an AsyncMock prisma client, so the DB write path itself can be asserted."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    mock_cache = MagicMock()
    mock_cache.async_set_cache = AsyncMock()

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedfiletable.upsert = AsyncMock()
    mock_prisma.db.litellm_managedfiletable.create = AsyncMock(
        side_effect=AssertionError(
            "store_unified_file_id must upsert, not create, on the retrieve path"
        )
    )

    return (
        _PROXY_LiteLLMManagedFiles(
            internal_usage_cache=mock_cache,
            prisma_client=mock_prisma,
        ),
        mock_prisma,
    )


def _make_object_store_instance():
    """A real store_unified_object_id over an AsyncMock prisma client, so both the
    upsert and the update-only write path can be asserted."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    mock_cache = MagicMock()
    mock_cache.async_set_cache = AsyncMock()

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedobjecttable.upsert = AsyncMock()
    mock_prisma.db.litellm_managedobjecttable.update_many = AsyncMock()

    return (
        _PROXY_LiteLLMManagedFiles(
            internal_usage_cache=mock_cache,
            prisma_client=mock_prisma,
        ),
        mock_prisma,
    )


@pytest.mark.asyncio
async def test_poll_refreshes_batch_state_without_claiming_the_row():
    """Regression (stale batch state): a poll observes a batch it did not create, so it
    must still refresh status and file_object -- otherwise GET /v1/batches serves the
    create-time snapshot forever -- while writing none of the attribution columns and
    never creating a row it would then own."""
    managed_files, mock_prisma = _make_object_store_instance()
    poller = UserAPIKeyAuth(
        api_key="sk-the-poller", user_id="bob", team_id="team-bravo", parent_otel_span=None
    )

    await managed_files.store_unified_object_id(
        unified_object_id="uoi-1",
        file_object=_make_batch_response(status="completed"),
        litellm_parent_otel_span=None,
        model_object_id="batch-123",
        file_purpose="batch",
        user_api_key_dict=poller,
        request_tags=("poller:tag",),
        persist_attribution=False,
        create_if_missing=False,
    )

    # the row is refreshed in place, and cannot be conjured by a poll
    mock_prisma.db.litellm_managedobjecttable.upsert.assert_not_awaited()
    update_many = mock_prisma.db.litellm_managedobjecttable.update_many
    update_many.assert_awaited_once()
    call = update_many.await_args
    assert call.kwargs["where"] == {"unified_object_id": "uoi-1"}

    written = call.kwargs["data"]
    assert written["status"] == "completed"
    assert json.loads(written["file_object"])["output_file_id"] == "file-output-abc"
    # nothing the poller could be billed for
    for owned in ("api_key", "request_tags", "created_by", "team_id"):
        assert owned not in written


@pytest.mark.asyncio
async def test_create_still_upserts_and_claims_attribution():
    """The create is the one caller that can speak for the batch, so it keeps the upsert
    (creating the row when absent) and writes the attribution columns."""
    managed_files, mock_prisma = _make_object_store_instance()
    creator = UserAPIKeyAuth(
        api_key="sk-the-creator", user_id="alice", team_id="team-alpha", parent_otel_span=None
    )

    await managed_files.store_unified_object_id(
        unified_object_id="uoi-2",
        file_object=_make_batch_response(status="validating"),
        litellm_parent_otel_span=None,
        model_object_id="batch-456",
        file_purpose="batch",
        user_api_key_dict=creator,
        request_tags=("env:prod",),
        persist_attribution=True,
    )

    mock_prisma.db.litellm_managedobjecttable.update_many.assert_not_awaited()
    upsert = mock_prisma.db.litellm_managedobjecttable.upsert
    upsert.assert_awaited_once()
    created = upsert.await_args.kwargs["data"]["create"]
    # UserAPIKeyAuth hashes an sk- token on construction; the hash is what is billed
    assert created["api_key"] == creator.api_key
    assert created["api_key"] != "sk-the-creator"
    assert created["created_by"] == "alice"
    assert created["team_id"] == "team-alpha"


@pytest.mark.asyncio
async def test_default_callers_still_create_their_rows():
    """create_if_missing defaults to True, so the fine-tune, Responses and managed
    /v1/batches callers, none of which passes it, keep upserting exactly as before."""
    managed_files, mock_prisma = _make_object_store_instance()

    await managed_files.store_unified_object_id(
        unified_object_id="uoi-3",
        file_object=_make_batch_response(),
        litellm_parent_otel_span=None,
        model_object_id="ft-789",
        file_purpose="fine-tune",
        user_api_key_dict=_make_user_api_key_dict(),
    )

    mock_prisma.db.litellm_managedobjecttable.upsert.assert_awaited_once()
    mock_prisma.db.litellm_managedobjecttable.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_unified_file_id_is_idempotent_via_upsert():
    """Regression test for the managed-batch retrieve 500 (UniqueViolationError on
    unified_file_id): re-registering an already-stored output file id must upsert on
    unified_file_id, never do an unconditional create that raises on conflict."""
    managed_files, mock_prisma = _make_real_managed_files_instance()
    file_id = "litellm_proxy_unified_output_id_abc"
    model_mappings = {"model-deploy-xyz": "file-output-abc"}

    for _ in range(2):
        await managed_files.store_unified_file_id(
            file_id=file_id,
            file_object=_make_file_object(),
            litellm_parent_otel_span=None,
            model_mappings=model_mappings,
            user_api_key_dict=_make_user_api_key_dict(),
        )

    mock_prisma.db.litellm_managedfiletable.create.assert_not_awaited()
    upsert_mock = mock_prisma.db.litellm_managedfiletable.upsert
    assert upsert_mock.await_count == 2
    for upsert_call in upsert_mock.await_args_list:
        assert upsert_call.kwargs["where"] == {"unified_file_id": file_id}
        upsert_data = upsert_call.kwargs["data"]
        assert upsert_data["create"]["unified_file_id"] == file_id
        assert json.loads(upsert_data["create"]["model_mappings"]) == model_mappings
        assert json.loads(upsert_data["update"]["model_mappings"]) == model_mappings


def test_get_unified_output_file_id_is_deterministic_per_output_file():
    managed_files, _ = _make_real_managed_files_instance()

    first = managed_files.get_unified_output_file_id(
        output_file_id="file-output-abc",
        model_id="model-deploy-xyz",
        model_name="azure/gpt-4",
    )
    repeat = managed_files.get_unified_output_file_id(
        output_file_id="file-output-abc",
        model_id="model-deploy-xyz",
        model_name="azure/gpt-4",
    )
    other_file = managed_files.get_unified_output_file_id(
        output_file_id="file-output-def",
        model_id="model-deploy-xyz",
        model_name="azure/gpt-4",
    )
    other_model = managed_files.get_unified_output_file_id(
        output_file_id="file-output-abc",
        model_id="model-deploy-other",
        model_name="azure/gpt-4",
    )

    assert first == repeat
    assert len({first, other_file, other_model}) == 3


@pytest.mark.asyncio
async def test_concurrent_first_registrations_converge_on_one_row():
    managed_files, mock_prisma = _make_real_managed_files_instance()

    minted_ids = tuple(
        managed_files.get_unified_output_file_id(
            output_file_id="file-output-abc",
            model_id="model-deploy-xyz",
            model_name=None,
        )
        for _ in range(2)
    )
    await asyncio.gather(
        *(
            managed_files.store_unified_file_id(
                file_id=unified_id,
                file_object=None,
                litellm_parent_otel_span=None,
                model_mappings={"model-deploy-xyz": "file-output-abc"},
                user_api_key_dict=_make_user_api_key_dict(),
            )
            for unified_id in minted_ids
        )
    )

    upserted_row_keys = {
        upsert_call.kwargs["where"]["unified_file_id"]
        for upsert_call in mock_prisma.db.litellm_managedfiletable.upsert.await_args_list
    }
    assert minted_ids[0] == minted_ids[1]
    assert upserted_row_keys == {minted_ids[0]}


def _b64_unified_input_file_id(target_model_names: str) -> str:
    unified_input_file_id = (
        "litellm_proxy:application/octet-stream;unified_id,input-uuid;"
        f"target_model_names,{target_model_names}"
    )
    return base64.urlsafe_b64encode(unified_input_file_id.encode()).decode().rstrip("=")


@pytest.mark.asyncio
async def test_hook_mint_prefers_input_file_target_model_names():
    managed_files = _make_managed_files_instance()
    batch_response = _make_batch_response(model_name="model-a")
    batch_response._hidden_params["unified_file_id"] = _b64_unified_input_file_id(
        "model-a,model-b"
    )

    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(return_value={})

    with (
        patch("litellm.afile_retrieve", AsyncMock(return_value=_make_file_object())),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=_make_user_api_key_dict(),
            response=batch_response,
        )

    assert batch_response.output_file_id == managed_files.get_unified_output_file_id(
        output_file_id="file-output-abc",
        model_id="model-deploy-xyz",
        model_name="model-a,model-b",
    )


@pytest.mark.asyncio
async def test_hook_mint_falls_back_to_response_input_file_id_target_models():
    managed_files = _make_managed_files_instance()
    batch_response = _make_batch_response()
    batch_response.input_file_id = _b64_unified_input_file_id("model-a,model-b")
    batch_response._hidden_params = {
        "unified_batch_id": "some-unified-batch-id",
        "model_id": "model-deploy-xyz",
    }

    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(return_value={})

    with (
        patch("litellm.afile_retrieve", AsyncMock(return_value=_make_file_object())),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=_make_user_api_key_dict(),
            response=batch_response,
        )

    assert batch_response.output_file_id == managed_files.get_unified_output_file_id(
        output_file_id="file-output-abc",
        model_id="model-deploy-xyz",
        model_name="model-a,model-b",
    )


@pytest.mark.asyncio
async def test_cost_job_and_retrieve_paths_mint_identical_unified_output_file_ids():
    from litellm.proxy.openai_files_endpoints.common_utils import (
        ensure_batch_response_managed_file_ids,
    )
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    managed_files, mock_prisma = _make_real_managed_files_instance()
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
    unified_input_file_id = _b64_unified_input_file_id("model-a")

    retrieve_response = LiteLLMBatch(
        id="batch-123",
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id=unified_input_file_id,
        object="batch",
        status="completed",
        output_file_id="file-output-abc",
    )
    retrieve_response._hidden_params = {"model_id": "model-deploy-xyz"}

    await ensure_batch_response_managed_file_ids(
        response=retrieve_response,
        managed_files_obj=managed_files,
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
        user_api_key_dict=_make_user_api_key_dict(),
    )

    job = MagicMock()
    job.file_object = {
        "id": "batch-123",
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/chat/completions",
        "input_file_id": unified_input_file_id,
        "object": "batch",
        "status": "completed",
    }
    cost_job_model_name = CheckBatchCost._get_managed_file_model_name(
        job=job, deployment_info=MagicMock(model_name="vertex_ai/gemini-3-pro")
    )

    assert cost_job_model_name == "model-a"
    assert retrieve_response.output_file_id == managed_files.get_unified_output_file_id(
        output_file_id="file-output-abc",
        model_id="model-deploy-xyz",
        model_name=cost_job_model_name,
    )


@pytest.mark.asyncio
async def test_batch_create_hook_persists_creating_key_and_tags():
    """Regression: the /v1/batches create hook must persist the creating key and the
    request's tags on the managed object row. CheckBatchCost, which owns the batch's
    accounting once the retrieve path defers to it, bills whatever the row carries, and
    without these columns the cost lands on the user alone and the key's spend and
    budget never see it."""
    managed_files = _make_managed_files_instance()
    creator = UserAPIKeyAuth(api_key="sk-the-creator", user_id="alice", parent_otel_span=None)
    create_response = _make_batch_response(status="validating", output_file_id=None)

    await managed_files.async_post_call_success_hook(
        data={"litellm_metadata": {"tags": ["env:prod", "team:ml"], "user_api_key": creator.api_key}},
        user_api_key_dict=creator,
        response=create_response,
    )

    managed_files.store_unified_object_id.assert_awaited_once()
    stored = managed_files.store_unified_object_id.await_args.kwargs
    assert stored["persist_attribution"] is True
    assert stored["request_tags"] == ("env:prod", "team:ml")
    assert stored["user_api_key_dict"] is creator


@pytest.mark.asyncio
async def test_batch_retrieve_hook_does_not_claim_attribution():
    """A retrieve carries unified_batch_id but no unified_file_id, so it must not rewrite
    the row's paying key to whoever happens to poll the batch."""
    managed_files = _make_managed_files_instance()
    retrieve_response = _make_batch_response(status="in_progress", output_file_id=None)
    retrieve_response._hidden_params = {
        "unified_batch_id": "some-unified-batch-id",
        "model_id": "model-deploy-xyz",
        "model_name": "azure/gpt-4",
    }

    await managed_files.async_post_call_success_hook(
        data={"litellm_metadata": {"tags": ["poller:tag"]}},
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-the-poller", user_id="bob", parent_otel_span=None),
        response=retrieve_response,
    )

    managed_files.store_unified_object_id.assert_awaited_once()
    assert managed_files.store_unified_object_id.await_args.kwargs["persist_attribution"] is False
