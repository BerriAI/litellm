"""Pin ``PrismaClient`` read-side data operations.

Symbols pinned here:
  - ``PrismaClient.hash_token``
  - ``PrismaClient.jsonify_object``
  - ``PrismaClient.jsonify_team_object``
  - ``PrismaClient.check_view_exists``
  - ``PrismaClient.get_request_status``
  - ``PrismaClient.get_generic_data``
  - ``PrismaClient._query_first_with_cached_plan_fallback``
  - ``PrismaClient.get_data``
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.utils import PrismaClient


def test_hash_token_method_returns_sha256(prisma_client: PrismaClient) -> None:
    token = "sk-token-xyz"
    actual = {
        "result": prisma_client.hash_token(token),
        "len": len(prisma_client.hash_token(token)),
        "expected": hashlib.sha256(token.encode()).hexdigest(),
        "deterministic": prisma_client.hash_token(token)
        == prisma_client.hash_token(token),
    }
    assert actual == {
        "result": hashlib.sha256(token.encode()).hexdigest(),
        "len": 64,
        "expected": hashlib.sha256(token.encode()).hexdigest(),
        "deterministic": True,
    }


def test_hash_token_method_error_on_non_string(prisma_client: PrismaClient) -> None:
    with pytest.raises(AttributeError):
        prisma_client.hash_token(None)  # type: ignore[arg-type]


def test_jsonify_object_serializes_nested_dicts(prisma_client: PrismaClient) -> None:
    data = {
        "metadata": {"a": 1, "b": [2, 3]},
        "models": ["gpt-4o", "gpt-4o-mini"],
        "token": "abc",
        "spend": 1.23,
    }
    result = prisma_client.jsonify_object(data)
    parsed_meta = json.loads(result["metadata"])
    assert result == {
        "metadata": json.dumps(data["metadata"]),
        "models": ["gpt-4o", "gpt-4o-mini"],
        "token": "abc",
        "spend": 1.23,
    }
    assert parsed_meta == {"a": 1, "b": [2, 3]}


def test_jsonify_object_fallback_for_unserializable_dict(
    prisma_client: PrismaClient,
) -> None:
    class _Bad:
        pass

    data = {"metadata": {"x": _Bad()}, "label": "ok", "n": 1}
    result = prisma_client.jsonify_object(data)
    assert result == {
        "metadata": "failed-to-serialize-json",
        "label": "ok",
        "n": 1,
    }


def test_jsonify_object_error_on_non_dict(prisma_client: PrismaClient) -> None:
    with pytest.raises(AttributeError):
        prisma_client.jsonify_object(None)  # type: ignore[arg-type]


def test_jsonify_team_object_converts_members_to_json_string(
    prisma_client: PrismaClient,
) -> None:
    data = {
        "team_id": "t1",
        "members_with_roles": [{"role": "admin", "user_id": "u1"}],
        "metadata": {"foo": "bar"},
        "models": ["gpt-4"],
    }
    result = prisma_client.jsonify_team_object(data)
    assert result == {
        "team_id": "t1",
        "members_with_roles": json.dumps(data["members_with_roles"]),
        "metadata": json.dumps(data["metadata"]),
        "models": ["gpt-4"],
    }


def test_jsonify_team_object_error_on_non_dict(prisma_client: PrismaClient) -> None:
    with pytest.raises(AttributeError):
        prisma_client.jsonify_team_object(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"status": "failure"}, "failure"),
        ({"status": "success"}, "success"),
        ({}, "success"),
        ("not-json", "success"),
        (json.dumps({"status": "failure"}), "failure"),
    ],
)
def test_get_request_status_pins_status_resolution(
    prisma_client: PrismaClient, metadata: Any, expected: str
) -> None:
    assert prisma_client.get_request_status({"metadata": metadata}) == expected


def test_get_request_status_error_returns_success_default(
    prisma_client: PrismaClient,
) -> None:
    """``get_request_status`` swallows AttributeError / JSONDecodeError and
    defaults to ``success`` to avoid blocking the request pipeline.
    """

    class _Broken:
        def get(self, *_: Any, **__: Any) -> Any:
            raise AttributeError("broken metadata")

    actual = prisma_client.get_request_status({"metadata": _Broken()})
    assert actual == "success"


@pytest.mark.asyncio
async def test_get_generic_data_dispatches_by_table(
    prisma_client: PrismaClient,
) -> None:
    row = SimpleNamespace(user_id="u1", spend=0.5, name="Alice")
    prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=row)
    result = await prisma_client.get_generic_data(
        key="user_id", value="u1", table_name="users"
    )
    actual = {
        "result_is_row": result is row,
        "find_first_count": prisma_client.db.litellm_usertable.find_first.await_count,
        "where_kwarg": prisma_client.db.litellm_usertable.find_first.await_args.kwargs[
            "where"
        ],
        "user_attr": result.user_id,
    }
    assert actual == {
        "result_is_row": True,
        "find_first_count": 1,
        "where_kwarg": {"user_id": "u1"},
        "user_attr": "u1",
    }


@pytest.mark.asyncio
async def test_get_generic_data_unknown_table_returns_none(
    prisma_client: PrismaClient,
) -> None:
    result = await prisma_client.get_generic_data(
        key="x", value="y", table_name="bogus"  # type: ignore[arg-type]
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_generic_data_logs_failure_handler_and_raises_on_error(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_usertable.find_first = AsyncMock(
        side_effect=RuntimeError("db boom")
    )
    with pytest.raises(RuntimeError, match="db boom"):
        await prisma_client.get_generic_data(
            key="user_id", value="x", table_name="users"
        )


@pytest.mark.asyncio
async def test_query_first_with_cached_plan_fallback_happy_returns_row(
    prisma_client: PrismaClient,
) -> None:
    expected = {"token": "abc", "team_spend": 1.0, "team_max_budget": 5.0}
    prisma_client.db.query_first = AsyncMock(return_value=expected)
    result = await prisma_client._query_first_with_cached_plan_fallback(
        "SELECT * FROM x WHERE token = $1", "abc"
    )
    actual = {
        "result": result,
        "call_count": prisma_client.db.query_first.await_count,
        "args": prisma_client.db.query_first.await_args.args,
        "matches": result == expected,
    }
    assert actual == {
        "result": expected,
        "call_count": 1,
        "args": ("SELECT * FROM x WHERE token = $1", "abc"),
        "matches": True,
    }


@pytest.mark.asyncio
async def test_query_first_with_cached_plan_fallback_retries_on_cached_plan_error(
    prisma_client: PrismaClient,
) -> None:
    expected = {"token": "abc", "team_spend": 1.0, "team_max_budget": 5.0}
    prisma_client.db.query_first = AsyncMock(
        side_effect=[
            RuntimeError("cached plan must not change result type"),
            expected,
        ]
    )
    result = await prisma_client._query_first_with_cached_plan_fallback(
        "SELECT * FROM x WHERE token = $1", "abc"
    )
    assert result == expected
    assert prisma_client.db.query_first.await_count == 2
    second_call_sql = prisma_client.db.query_first.await_args_list[1].args[0]
    assert "cache_invalidated_" in second_call_sql


@pytest.mark.asyncio
async def test_query_first_with_cached_plan_fallback_reraises_non_plan_errors(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.query_first = AsyncMock(side_effect=RuntimeError("totally unrelated"))
    with pytest.raises(RuntimeError, match="totally unrelated"):
        await prisma_client._query_first_with_cached_plan_fallback("SELECT 1")


@pytest.mark.asyncio
async def test_check_view_exists_noop_when_all_views_present(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.query_raw = AsyncMock(
        return_value=[
            {
                "view_count": 8,
                "view_names": [
                    "LiteLLM_VerificationTokenView",
                    "MonthlyGlobalSpend",
                    "Last30dKeysBySpend",
                    "Last30dModelsBySpend",
                    "MonthlyGlobalSpendPerKey",
                    "MonthlyGlobalSpendPerUserPerKey",
                    "Last30dTopEndUsersSpend",
                    "DailyTagSpend",
                ],
            }
        ]
    )
    prisma_client.db.execute_raw = AsyncMock()
    result = await prisma_client.check_view_exists()
    actual = {
        "result": result,
        "query_raw_calls": prisma_client.db.query_raw.await_count,
        "execute_raw_calls": prisma_client.db.execute_raw.await_count,
        "view_query_contains_token_view": "LiteLLM_VerificationTokenView"
        in prisma_client.db.query_raw.await_args.args[0],
    }
    assert actual == {
        "result": None,
        "query_raw_calls": 1,
        "execute_raw_calls": 0,
        "view_query_contains_token_view": True,
    }


@pytest.mark.asyncio
async def test_check_view_exists_creates_token_view_when_missing(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.query_raw = AsyncMock(
        return_value=[
            {
                "view_count": 1,
                "view_names": ["DailyTagSpend"],
            }
        ]
    )
    prisma_client.db.execute_raw = AsyncMock()
    prisma_client.health_check = AsyncMock(return_value=[{"?column?": 1}])
    result = await prisma_client.check_view_exists()
    actual = {
        "result": result,
        "create_called": prisma_client.db.execute_raw.await_count,
        "create_sql_starts_with_create_view": prisma_client.db.execute_raw.await_args.args[
            0
        ]
        .strip()
        .startswith('CREATE VIEW "LiteLLM_VerificationTokenView"'),
    }
    assert actual == {
        "result": None,
        "create_called": 1,
        "create_sql_starts_with_create_view": True,
    }


@pytest.mark.asyncio
async def test_check_view_exists_raises_when_query_raw_fails(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.query_raw = AsyncMock(side_effect=RuntimeError("db down"))
    with pytest.raises(RuntimeError, match="db down"):
        await prisma_client.check_view_exists()


@pytest.mark.asyncio
async def test_get_data_token_find_unique_returns_record(
    prisma_client: PrismaClient,
) -> None:
    token = "sk-key-1"
    hashed = hashlib.sha256(token.encode()).hexdigest()
    record = SimpleNamespace(token=hashed, user_id="u1", expires=None, spend=0.5)
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=record
    )

    result = await prisma_client.get_data(token=token, table_name="key")
    actual = {
        "result_is_record": result is record,
        "where_arg": prisma_client.db.litellm_verificationtoken.find_unique.await_args.kwargs[
            "where"
        ],
        "include_arg": prisma_client.db.litellm_verificationtoken.find_unique.await_args.kwargs[
            "include"
        ],
        "token_field_matches": result.token == hashed,
    }
    assert actual == {
        "result_is_record": True,
        "where_arg": {"token": hashed},
        "include_arg": {"litellm_budget_table": True},
        "token_field_matches": True,
    }


@pytest.mark.asyncio
async def test_get_data_token_find_unique_missing_token_raises_401(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    with pytest.raises(HTTPException) as excinfo:
        await prisma_client.get_data(token="sk-missing", table_name="key")
    err = excinfo.value
    assert "invalid user key" in err.detail
    assert err.status_code == 401


@pytest.mark.asyncio
async def test_get_data_user_find_unique_returns_user_row(
    prisma_client: PrismaClient,
) -> None:
    row = SimpleNamespace(
        user_id="u-7",
        spend=1.5,
        max_budget=10.0,
        organization_memberships=[],
    )
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=row)
    result = await prisma_client.get_data(user_id="u-7", table_name="user")
    actual = {
        "result_is_row": result is row,
        "where_arg": prisma_client.db.litellm_usertable.find_unique.await_args.kwargs[
            "where"
        ],
        "include_arg": prisma_client.db.litellm_usertable.find_unique.await_args.kwargs[
            "include"
        ],
        "spend": row.spend,
    }
    assert actual == {
        "result_is_row": True,
        "where_arg": {"user_id": "u-7"},
        "include_arg": {"organization_memberships": True},
        "spend": 1.5,
    }


@pytest.mark.asyncio
async def test_get_data_logs_and_raises_on_db_error(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        side_effect=RuntimeError("network split")
    )
    with pytest.raises(RuntimeError, match="network split"):
        await prisma_client.get_data(token="sk-broken", table_name="key")
