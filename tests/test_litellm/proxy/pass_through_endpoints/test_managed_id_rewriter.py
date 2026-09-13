import datetime
import json
from collections.abc import AsyncIterator, Iterable
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.managed_id_codec import decode, new_managed_id
from litellm.proxy.pass_through_endpoints.managed_id_rewriter import (
    list_passthrough_ids_from_db,
    rewrite_streamed_response_ids,
)


def _user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="user-1", team_id="team-1")


def _prisma_client(file_rows=None, batch_rows=None) -> MagicMock:
    pc = MagicMock()
    pc.db = MagicMock()
    pc.db.litellm_managedfiletable = MagicMock()
    pc.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
    pc.db.litellm_managedfiletable.find_many = AsyncMock(
        side_effect=lambda *args, take=None, **kwargs: list(file_rows or [])[:take]
    )
    pc.db.litellm_managedobjecttable = MagicMock()
    pc.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=None)
    pc.db.litellm_managedobjecttable.find_many = AsyncMock(
        side_effect=lambda *args, take=None, **kwargs: list(batch_rows or [])[:take]
    )
    pc.db.litellm_managedobjecttable.upsert = AsyncMock(return_value=None)
    return pc


RAW_RESPONSE_ID = "resp_0123456789abcdef"


def _response_stream_bytes(raw_id: str = RAW_RESPONSE_ID) -> bytes:
    events = (
        ("response.created", {"type": "response.created", "response": {"id": raw_id, "status": "in_progress"}}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "mango"}),
        ("response.completed", {"type": "response.completed", "response": {"id": raw_id, "status": "completed"}}),
    )
    return b"".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events)


async def _chunks(payload: bytes, size: int) -> AsyncIterator[bytes]:
    for start in range(0, len(payload), size):
        yield payload[start : start + size]


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in stream])


def _response_ids(sse: bytes) -> Iterable[str]:
    for line in sse.decode().splitlines():
        if line.startswith("data:"):
            event = json.loads(line[len("data:") :])
            if "response" in event:
                yield event["response"]["id"]


def _file_row(unified_id: str) -> MagicMock:
    row = MagicMock()
    row.unified_file_id = unified_id
    row.created_by = "user-1"
    row.team_id = "team-1"
    row.file_object = {"filename": "test.jsonl", "bytes": 42, "purpose": "batch"}
    row.created_at = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    return row


def _batch_row(unified_id: str) -> MagicMock:
    row = MagicMock()
    row.unified_object_id = unified_id
    row.created_by = "user-1"
    row.team_id = "team-1"
    row.file_object = {"status": "completed", "input_file_id": "file-managed-1"}
    row.file_purpose = "batch"
    row.created_at = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit,expected_message,expected_openai_code",
    [
        (
            "-1",
            "Invalid 'limit': integer below minimum value. Expected a value >= 0, but got -1 instead.",
            "integer_below_min_value",
        ),
        (
            "101",
            "Invalid 'limit': integer above maximum value. Expected a value <= 100, but got 101 instead.",
            "integer_above_max_value",
        ),
    ],
)
async def test_list_batches_out_of_range_limit_raises_400(limit, expected_message, expected_openai_code):
    pc = _prisma_client(batch_rows=[_batch_row(new_managed_id("openai", "batch_abc"))])

    with pytest.raises(ProxyException) as exc:
        await list_passthrough_ids_from_db(
            provider="openai",
            route="/openai/v1/batches",
            user_api_key_dict=_user(),
            prisma_client=pc,
            query_params={"limit": limit},
        )

    assert exc.value.code == "400"
    assert exc.value.param == "limit"
    assert exc.value.type == "invalid_request_error"
    assert exc.value.openai_code == expected_openai_code
    assert exc.value.message == expected_message
    pc.db.litellm_managedobjecttable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_list_batches_limit_zero_returns_empty_page_without_db_query():
    pc = _prisma_client(batch_rows=[_batch_row(new_managed_id("openai", "batch_abc"))])

    result = await list_passthrough_ids_from_db(
        provider="openai",
        route="/openai/v1/batches",
        user_api_key_dict=_user(),
        prisma_client=pc,
        query_params={"limit": "0"},
    )

    assert result == {
        "object": "list",
        "data": [],
        "first_id": None,
        "last_id": None,
        "has_more": False,
    }
    pc.db.litellm_managedobjecttable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_list_files_limit_above_batch_cap_still_served():
    managed_id = new_managed_id("openai", "file-abc")
    pc = _prisma_client(file_rows=[_file_row(managed_id)])

    result = await list_passthrough_ids_from_db(
        provider="openai",
        route="/openai/v1/files",
        user_api_key_dict=_user(),
        prisma_client=pc,
        query_params={"limit": "101"},
    )

    assert result is not None
    assert [item["id"] for item in result["data"]] == [managed_id]


@pytest.mark.asyncio
async def test_list_files_drops_batch_guardrail_key_persisted_by_an_older_proxy():
    """Rows written before the response serializer dropped the key still carry an explicit null."""
    managed_id = new_managed_id("openai", "file-abc")
    row = _file_row(managed_id)
    row.file_object = {**row.file_object, "litellm_batch_guardrail": None}
    pc = _prisma_client(file_rows=[row])

    result = await list_passthrough_ids_from_db(
        provider="openai",
        route="/openai/v1/files",
        user_api_key_dict=_user(),
        prisma_client=pc,
        query_params={},
    )

    assert result is not None
    assert "litellm_batch_guardrail" not in result["data"][0]
    assert result["data"][0]["filename"] == "test.jsonl"


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
async def test_streamed_response_is_owned_and_rewritten_across_chunk_boundaries(chunk_size: int):
    """A streamed POST /v1/responses records the caller as owner once and returns
    the minted id in every event, no matter how the transport splits the SSE bytes."""
    pc = _prisma_client()

    output = await _collect(
        rewrite_streamed_response_ids(
            stream=_chunks(_response_stream_bytes(), chunk_size),
            provider="openai",
            method="POST",
            route="/openai_passthrough/v1/responses",
            user_api_key_dict=_user(),
            prisma_client=pc,
        )
    )

    pc.db.litellm_managedobjecttable.upsert.assert_awaited_once()
    created = pc.db.litellm_managedobjecttable.upsert.await_args.kwargs["data"]["create"]
    assert created["created_by"] == "user-1"
    assert created["team_id"] == "team-1"
    assert created["file_purpose"] == "response"
    assert created["model_object_id"] == f"passthrough:openai:{RAW_RESPONSE_ID}"
    managed_id = created["unified_object_id"]
    assert decode(managed_id).raw_provider_id == RAW_RESPONSE_ID
    assert list(_response_ids(output)) == [managed_id, managed_id]
    assert RAW_RESPONSE_ID.encode() not in output
    assert output == _response_stream_bytes(managed_id)


@pytest.mark.asyncio
async def test_streamed_response_with_cr_only_frame_delimiters_is_still_owned_and_rewritten():
    """SSE also terminates lines with a lone CR; those frames must mint and rewrite too."""
    pc = _prisma_client()
    payload = _response_stream_bytes().replace(b"\n", b"\r")

    output = await _collect(
        rewrite_streamed_response_ids(
            stream=_chunks(payload, 7),
            provider="openai",
            method="POST",
            route="/openai_passthrough/v1/responses",
            user_api_key_dict=_user(),
            prisma_client=pc,
        )
    )

    pc.db.litellm_managedobjecttable.upsert.assert_awaited_once()
    managed_id = pc.db.litellm_managedobjecttable.upsert.await_args.kwargs["data"]["create"]["unified_object_id"]
    assert RAW_RESPONSE_ID.encode() not in output
    assert output == _response_stream_bytes(managed_id).replace(b"\n", b"\r")


@pytest.mark.asyncio
async def test_streamed_bytes_untouched_on_routes_without_a_response_id():
    pc = _prisma_client()
    payload = _response_stream_bytes()

    output = await _collect(
        rewrite_streamed_response_ids(
            stream=_chunks(payload, 5),
            provider="openai",
            method="POST",
            route="/openai_passthrough/v1/chat/completions",
            user_api_key_dict=_user(),
            prisma_client=pc,
        )
    )

    assert output == payload
    pc.db.litellm_managedobjecttable.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_streamed_response_stays_raw_and_intact_when_the_row_cannot_be_persisted():
    pc = _prisma_client()
    pc.db.litellm_managedobjecttable.upsert = AsyncMock(side_effect=RuntimeError("db down"))
    payload = _response_stream_bytes()

    output = await _collect(
        rewrite_streamed_response_ids(
            stream=_chunks(payload, 3),
            provider="openai",
            method="POST",
            route="/openai_passthrough/v1/responses",
            user_api_key_dict=_user(),
            prisma_client=pc,
        )
    )

    assert output == payload
    pc.db.litellm_managedobjecttable.upsert.assert_awaited_once()
