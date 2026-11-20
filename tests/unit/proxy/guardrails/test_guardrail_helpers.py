from collections.abc import Iterator, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_helpers import validate_guardrail_names

DB_ROWS: Final = MappingProxyType({"valid": "active", "also-valid": "active", "submitted": "pending_review"})


def _filter(where: Mapping[str, object]) -> Mapping[str, object]:
    clauses: Final = where.get("AND", (where,))
    return {key: value for clause in clauses for key, value in clause.items()}  # comprehension-ok: flattens prisma AND clauses


def _prisma(rows: Mapping[str, str] = DB_ROWS) -> MagicMock:
    async def find_many(*, where: Mapping[str, object]) -> list[SimpleNamespace]:
        flat: Final = _filter(where)
        names: Final = flat["guardrail_name"]["in"]
        return [SimpleNamespace(guardrail_name=name) for name in names if rows.get(name) == flat.get("status")]

    client: Final = MagicMock()
    client.writer_db.litellm_guardrailstable.find_many = AsyncMock(side_effect=find_many)
    client.db.litellm_guardrailstable.find_many = AsyncMock(return_value=[])
    return client


def _looked_up(client: MagicMock) -> list[frozenset[str]]:
    calls: Final = client.writer_db.litellm_guardrailstable.find_many.await_args_list
    return [frozenset(_filter(call.kwargs["where"])["guardrail_name"]["in"]) for call in calls]


async def _validate(
    guardrails: list[str] | None,
    metadata: Mapping[str, object] | None,
    client: MagicMock,
    existing_metadata: Mapping[str, object] | None = None,
) -> None:
    await validate_guardrail_names(guardrails, metadata, existing_metadata, client, license_first=False)


@pytest.fixture
def guardrail_loaded_here() -> Iterator[str]:
    lists: Final = [
        litellm.callbacks,
        litellm.success_callback,
        litellm.failure_callback,
        litellm._async_success_callback,
        litellm._async_failure_callback,
    ]
    snapshots: Final = [list(cb_list) for cb_list in lists]
    litellm.logging_callback_manager.add_litellm_callback(CustomGuardrail(guardrail_name="loaded-here"))
    try:
        yield "loaded-here"
    finally:
        for cb_list, snapshot in zip(lists, snapshots):
            cb_list[:] = snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("guardrails", "metadata"),
    (
        (["valid", "typo"], None),
        (None, {"guardrails": ["valid", "typo"]}),
        (None, {"guardrails": [{"valid": {}}, {"typo": {"extra_body": {"k": "v"}}}]}),
    ),
    ids=("top_level", "metadata_names", "metadata_dynamic_params"),
)
async def test_rejects_only_the_unregistered_name(
    guardrails: list[str] | None, metadata: dict[str, object] | None
) -> None:
    with pytest.raises(HTTPException) as exc:
        await _validate(guardrails, metadata, _prisma())

    assert exc.value.status_code == 400
    message: Final = exc.value.detail["error"]
    assert "typo" in message
    assert "valid" not in message


@pytest.mark.asyncio
async def test_active_db_rows_and_loaded_guardrails_pass(guardrail_loaded_here: str) -> None:
    client: Final = _prisma()

    await _validate(None, {"guardrails": ["valid", {"also-valid": {}}, guardrail_loaded_here]}, client)

    assert _looked_up(client) == [frozenset({"valid", "also-valid"})]


@pytest.mark.asyncio
async def test_a_guardrail_pending_review_is_unknown() -> None:
    with pytest.raises(HTTPException) as exc:
        await _validate(["submitted"], None, _prisma())

    assert "submitted" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_the_lookup_reads_the_writer_so_a_lagging_replica_cannot_hide_a_new_guardrail() -> None:
    client: Final = _prisma()

    await _validate(["valid"], None, client)

    client.db.litellm_guardrailstable.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_db_checks_only_loaded_guardrails(guardrail_loaded_here: str) -> None:
    await validate_guardrail_names([guardrail_loaded_here], None, None, None, license_first=False)
    with pytest.raises(HTTPException):
        await validate_guardrail_names(["valid"], None, None, None, license_first=False)


@pytest.mark.asyncio
async def test_top_level_guardrails_replace_the_metadata_list() -> None:
    client: Final = _prisma()

    await _validate(["valid"], {"guardrails": ["typo"]}, client)

    assert _looked_up(client) == [frozenset({"valid"})]


@pytest.mark.asyncio
async def test_names_already_on_the_object_are_not_rechecked() -> None:
    client: Final = _prisma()
    stored: Final = {"guardrails": ["stale"]}

    await _validate(["stale"], None, client, stored)
    await _validate(["stale", "valid"], None, client, stored)
    with pytest.raises(HTTPException):
        await _validate(["stale", "typo"], None, client, stored)

    assert _looked_up(client) == [frozenset({"valid"}), frozenset({"typo"})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("guardrails", "metadata"),
    ((None, None), (None, {"tags": ["t"]}), (None, {"guardrails": None}), ([], None), (None, {"guardrails": []})),
    ids=("nothing", "other_metadata", "metadata_null", "empty_top_level", "empty_metadata"),
)
async def test_no_requested_names_skips_the_lookup(
    guardrails: list[str] | None, metadata: dict[str, object] | None
) -> None:
    client: Final = _prisma()

    await _validate(guardrails, metadata, client)

    assert _looked_up(client) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entries", "shown"),
    (("pii", '"pii"'), (5, "5"), ([1], "[1]"), ([None, "stale"], '[null, "stale"]')),
    ids=("string", "number", "non_name_entry", "null_entry"),
)
async def test_malformed_metadata_guardrails_fail_and_name_the_value(entries: object, shown: str) -> None:
    with pytest.raises(HTTPException) as exc:
        await _validate(None, {"guardrails": entries}, _prisma())

    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == (
        f"Invalid guardrail detected - {shown}. Please provide a valid list of guardrail names."
    )


@pytest.mark.asyncio
async def test_resending_a_malformed_stored_value_still_fails() -> None:
    with pytest.raises(HTTPException) as exc:
        await _validate(None, {"guardrails": "pii", "note": "unrelated edit"}, _prisma(), {"guardrails": "pii"})

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_malformed_message_quotes_a_long_value_only_in_part() -> None:
    with pytest.raises(HTTPException) as exc:
        await _validate(None, {"guardrails": "g" * 5000}, _prisma())

    message: Final = exc.value.detail["error"]
    assert '"' + "g" * 99 + "..." in message
    assert len(message) < 200


@pytest.mark.asyncio
@pytest.mark.parametrize(("license_first", "status_code"), ((True, 403), (False, 400)))
async def test_license_first_puts_the_license_error_ahead_of_an_unknown_name(
    monkeypatch: pytest.MonkeyPatch, license_first: bool, status_code: int
) -> None:
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", False)

    with pytest.raises(HTTPException) as exc:
        await validate_guardrail_names(["typo"], None, None, _prisma(), license_first=license_first)

    assert exc.value.status_code == status_code


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", (False, True))
async def test_the_team_modify_guardrails_permission_is_left_alone(allowed: bool) -> None:
    client: Final = _prisma()
    permission: Final = {"modify_guardrails": allowed}

    await _validate(None, {"guardrails": permission}, client, {"guardrails": permission})
    await _validate(None, {"guardrails": permission}, client)

    assert _looked_up(client) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entries",
    ({"valid": True}, {"modify_guardrails": False, "valid": True}, {"modify_guardrails": "no"}),
    ids=("toggle_map", "permission_plus_attachment", "non_bool_permission"),
)
async def test_other_stored_dicts_fail_loudly_since_they_never_run(entries: dict[str, object]) -> None:
    with pytest.raises(HTTPException) as exc:
        await _validate(None, {"guardrails": entries}, _prisma())

    assert exc.value.status_code == 400
    assert exc.value.detail["error"].startswith("Invalid guardrail detected - {")


async def _pre_call(
    data: dict[str, object], reject_unregistered: bool, skip_guardrails: bool = False
) -> dict[str, object] | None:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import CallTypes
    from litellm.proxy.utils import ProxyLogging

    ProxyLogging._callback_capabilities_cache.clear()
    with patch(
        "litellm.proxy.proxy_server.general_settings", {"reject_unregistered_guardrails": reject_unregistered}
    ):
        return await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            data=data,
            call_type=CallTypes.acompletion.value,
            skip_guardrails=skip_guardrails,
        )


def _request(metadata: dict[str, object]) -> dict[str, object]:
    return {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "metadata": metadata}


@pytest.mark.asyncio
async def test_pre_call_rejects_a_named_guardrail_when_none_are_registered() -> None:
    with patch("litellm.callbacks", []), pytest.raises(HTTPException) as exc:
        await _pre_call(_request({"guardrails": ["deleted-guardrail"]}), reject_unregistered=True)

    assert exc.value.status_code == 400
    assert exc.value.detail["error"].startswith("Unknown guardrail(s): deleted-guardrail.")


@pytest.mark.asyncio
async def test_pre_call_names_only_the_unregistered_guardrails(guardrail_loaded_here: str) -> None:
    data: Final = _request({"guardrails": [guardrail_loaded_here, {"typo": {}}]})
    data["litellm_metadata"] = {"guardrails": ["other-typo"]}

    with pytest.raises(HTTPException) as exc:
        await _pre_call(data, reject_unregistered=True)

    assert exc.value.status_code == 400
    assert exc.value.detail["error"].startswith("Unknown guardrail(s): other-typo, typo.")


@pytest.mark.asyncio
async def test_pre_call_serves_a_request_whose_guardrails_are_all_registered(guardrail_loaded_here: str) -> None:
    data: Final = _request({"guardrails": [guardrail_loaded_here]})

    assert await _pre_call(data, reject_unregistered=True) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reject_unregistered", "skip_guardrails"), ((False, False), (True, True)), ids=("flag_off", "skip_guardrails")
)
async def test_pre_call_serves_unregistered_guardrails_without_the_flag_or_when_skipping(
    reject_unregistered: bool, skip_guardrails: bool
) -> None:
    data: Final = _request({"guardrails": ["deleted-guardrail"]})

    with patch("litellm.callbacks", []):
        assert await _pre_call(data, reject_unregistered, skip_guardrails) is not None
