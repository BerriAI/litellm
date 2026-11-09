from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm._auth.failure import NativeAuthRuntimeError
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy._types import LiteLLM_EndUserTable, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth import auth_checks
from litellm.proxy.auth import end_user_context_adapter as adapter
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from tests.unit.proxy.auth import phase1_budget_auth_reference as auth_reference
from tests.unit.proxy.auth import phase1_budget_context_reference as legacy


@pytest.fixture(params=(False, True), ids=("reference", "native"))
def native(request: pytest.FixtureRequest) -> bool:
    return request.param is True  # pyright: ignore[reportAny]  # boolean fixture parameter


@pytest.mark.parametrize(
    "value,expected", ((None, None), ("", None), (" ", " "), ("budget", "budget"), (0, None), (False, None), ({}, None))
)
def test_metadata_id_keeps_only_nonempty_strings(native: bool, value: object, expected: str | None) -> None:
    function: Final = adapter.get_key_end_user_budget_id if native else auth_reference.get_key_end_user_budget_id
    assert function({"end_user_budget_id": value}) == expected
    assert function(None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", (None, "", "key"))
@pytest.mark.parametrize("global_id", (None, "", "global"))
@pytest.mark.parametrize("db_present", (False, True))
async def test_default_budget_id_and_cache_selection(
    native: bool, explicit: str | None, global_id: str | None, db_present: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "max_end_user_budget_id", global_id)
    cache: Final = UserApiKeyCache()
    selected: Final = explicit if explicit is not None else global_id
    budget: Final = LiteLLM_BudgetTable(budget_id=selected, max_budget=0)
    await cache.async_set_cache(key=f"default_end_user_budget:{selected}", value=budget, model_type=LiteLLM_BudgetTable)
    function: Final = adapter.get_default_end_user_budget if native else auth_reference.get_default_end_user_budget
    result: Final = await function(Mock() if db_present else None, cache, budget_id=explicit)
    assert result == (budget if db_present and selected is not None else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("own_id", (None, "", "own"))
@pytest.mark.parametrize("own_present", (False, True))
@pytest.mark.parametrize("key_id", (None, "", "key"))
@pytest.mark.parametrize("global_id", (None, "", "global"))
@pytest.mark.parametrize("found", (False, True))
async def test_default_application_preserves_own_budget_and_copies_only_resolved_defaults(
    native: bool,
    own_id: str | None,
    own_present: bool,
    key_id: str | None,
    global_id: str | None,
    found: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "max_end_user_budget_id", global_id)
    own: Final = LiteLLM_BudgetTable(budget_id="own", max_budget=2)
    default: Final = LiteLLM_BudgetTable(budget_id="default", max_budget=3) if found else None
    row: Final = LiteLLM_EndUserTable(
        user_id="eu", blocked=False, budget_id=own_id, litellm_budget_table=own if own_present else None
    )
    load: Final = AsyncMock(return_value=default)
    monkeypatch.setattr(auth_checks, "get_default_end_user_budget", load)
    monkeypatch.setattr(auth_reference, "get_default_end_user_budget", load)
    function: Final = (
        adapter.apply_default_budget_to_end_user if native else auth_reference._apply_default_budget_to_end_user  # pyright: ignore[reportPrivateUsage]  # Phase 1 parity reference
    )  # pyright: ignore[reportPrivateUsage]  # Phase 1 parity reference
    result: Final = await function(row, Mock(), UserApiKeyCache(), key_end_user_budget_id=key_id)
    unchanged: Final = (own_id is not None and own_present) or (key_id is None and global_id is None) or not found
    assert (result is row) is unchanged
    assert result.litellm_budget_table is (row.litellm_budget_table if unchanged else default)
    assert row.litellm_budget_table is (own if own_present else None)
    expected_ids: Final = (
        ()
        if own_id is not None and own_present
        else (
            ((key_id,) if key_id is not None else ())
            + ((global_id,) if global_id is not None and (key_id is None or not found) else ())
        )
    )
    assert tuple(call.kwargs.get("budget_id", global_id) for call in load.await_args_list) == expected_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("global_id", (None, "", "global"))
@pytest.mark.parametrize("key_id", (None, "", "key"))
@pytest.mark.parametrize("validate", (False, True))
@pytest.mark.parametrize("max_budget", (None, 0))
@pytest.mark.parametrize("ids", (None, frozenset[str](), frozenset(("eu",))))
async def test_unrestricted_registry_requires_absence_of_all_budget_and_validation_constraints(
    native: bool,
    global_id: str | None,
    key_id: str | None,
    validate: bool,
    max_budget: int | None,
    ids: frozenset[str] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "max_end_user_budget_id", global_id)
    monkeypatch.setattr(litellm, "validate_end_user_id_in_db", validate)
    load: Final = AsyncMock(return_value=ids)
    monkeypatch.setattr(auth_checks, "_load_end_user_restricted_registry", load)
    monkeypatch.setattr(auth_reference, "_load_end_user_restricted_registry", load)
    function: Final = adapter.end_user_is_known_unrestricted if native else auth_reference._end_user_is_known_unrestricted  # pyright: ignore[reportPrivateUsage]  # Phase 1 parity reference
    result: Final = await function("eu", Mock(), UserApiKeyCache(), max_budget, key_id)
    blocked: Final = global_id is not None or key_id is not None or validate or max_budget is not None
    assert result is (not blocked and ids is not None and "eu" not in ids)
    assert load.await_count == (0 if blocked else 1)


@pytest.mark.parametrize("value", (None, 0, 1))
@pytest.mark.parametrize("models", (None, {}, {"model": {"budget_limit": 0}}))
def test_budget_parameter_projection_keeps_zero_and_empty_model_limits(
    native: bool,
    value: int | None,
    models: Mapping[str, object] | None,
) -> None:
    budget: Final = LiteLLM_BudgetTable(
        tpm_limit=value,
        rpm_limit=value,
        tpd_limit=value,
        max_budget=value,
        model_max_budget=dict(models) if models is not None else None,
    )
    params: Final[dict[str, object]] = {"existing": True}  # mutable-ok: legacy output argument
    if native:
        adapter.apply_budget_limits_to_end_user_params(params, budget, "eu")
    else:
        legacy._apply_budget_limits_to_end_user_params(params, budget, "eu")  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # Phase 1 unparameterized reference dictionary
    expected: Final = {
        "existing": True,
        **(
            {
                name: value
                for name in ("end_user_tpm_limit", "end_user_rpm_limit", "end_user_tpd_limit", "end_user_max_budget")
            }
            if value is not None
            else {}
        ),
        **({"end_user_model_max_budget": models} if models is not None else {}),
    }
    assert params == expected
    assert tuple(params) == tuple(expected)
    if models is not None:
        assert params["end_user_model_max_budget"] is getattr(budget, "model_max_budget")


@pytest.mark.parametrize("value", (None, 0, "opaque"))
def test_token_projection_preserves_unset_limits_and_never_copies_max_budget(native: bool, value: object) -> None:
    token: Final = UserAPIKeyAuth(
        end_user_id="old",
        end_user_tpm_limit=9,
        end_user_rpm_limit=9,
        end_user_tpd_limit=9,
        end_user_max_budget=9,
        end_user_model_max_budget={"old": {}},
        allowed_model_region="eu",
    )
    params: Final = {
        name: value
        for name in (
            "end_user_tpm_limit",
            "end_user_rpm_limit",
            "end_user_tpd_limit",
            "end_user_max_budget",
            "end_user_model_max_budget",
            "allowed_model_region",
        )
    }
    result: Final = (
        adapter.update_valid_token_with_end_user_params(token, params)
        if native
        else legacy.update_valid_token_with_end_user_params(token, params)  # pyright: ignore[reportUnknownMemberType]  # Phase 1 unparameterized reference dictionary
    )
    assert result is token
    assert token.end_user_id is None
    assert token.end_user_tpm_limit == (9 if value is None else value)
    assert token.end_user_rpm_limit == (9 if value is None else value)
    assert token.end_user_tpd_limit == (9 if value is None else value)
    assert token.allowed_model_region == ("eu" if value is None else value)
    assert token.end_user_model_max_budget == ({"old": {}} if value is None else value)
    assert token.end_user_max_budget == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", ("valid", "invalid", None, 0))
@pytest.mark.parametrize("global_id,key_id", ((None, None), ("", None), ("global", None), (None, "")))
@pytest.mark.parametrize("end_found,user_found", ((False, False), (True, False), (False, True)))
@pytest.mark.parametrize("identity", ("eu", "email@example.test"))
async def test_validation_uses_raw_row_presence_and_preserves_unknown_ids_with_defaults(
    native: bool,
    cached: object,
    global_id: str | None,
    key_id: str | None,
    end_found: bool,
    user_found: bool,
    identity: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "validate_end_user_id_in_db", True)
    monkeypatch.setattr(litellm, "max_end_user_budget_id", global_id)
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(key=f"end_user_validation:{identity}", value=cached)
    end: Final = AsyncMock(return_value=LiteLLM_EndUserTable(user_id=identity, blocked=False) if end_found else None)
    user: Final = AsyncMock(return_value=LiteLLM_UserTable(user_id=identity) if user_found else None)
    monkeypatch.setattr(auth_checks, "get_end_user_object", end)
    monkeypatch.setattr(auth_reference, "get_end_user_object", end)
    monkeypatch.setattr(auth_checks, "get_user_object", user)
    monkeypatch.setattr(auth_reference, "get_user_object", user)
    function: Final = (
        adapter.resolve_and_validate_end_user_id if native else auth_reference.resolve_and_validate_end_user_id
    )
    result: Final = await function(identity, Mock(), cache, key_end_user_budget_id=key_id)
    valid: Final = cached == "valid" or (cached != "invalid" and (end_found or user_found))
    assert result == (identity if valid or bool(global_id) or key_id is not None else None)
    assert end.await_count == (0 if cached in ("valid", "invalid") else 1)
    assert user.await_count == (0 if cached in ("valid", "invalid") or end_found else 1)
    if user.await_count:
        assert user.await_args is not None
        assert user.await_args.kwargs["user_email"] == (identity if "@" in identity else None)
    assert await cache.async_get_cache(key=f"end_user_validation:{identity}") == ("valid" if valid else "invalid")


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", (None, "", "eu"))
@pytest.mark.parametrize("enabled,db", ((False, False), (False, True), (True, False)))
async def test_validation_disabled_or_without_db_skips_all_lookups(
    native: bool,
    identity: str | None,
    enabled: bool,
    db: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "validate_end_user_id_in_db", enabled)
    cache: Final = Mock(spec=UserApiKeyCache)
    read: Final = AsyncMock(side_effect=AssertionError("cache must not be read"))
    cache.async_get_cache = read
    function: Final = (
        adapter.resolve_and_validate_end_user_id if native else auth_reference.resolve_and_validate_end_user_id
    )
    assert await function(identity, Mock() if db else None, cache) == identity
    read.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("row_mode", ("missing", "empty", "budget"))
@pytest.mark.parametrize("initial_max", (None, 0, 7))
@pytest.mark.parametrize(
    "key_id,global_id,found",
    ((None, None, False), ("key", None, True), (None, "global", True), ("key", "global", False)),
)
async def test_lookup_uses_row_before_default_and_only_fills_missing_max_budget(
    native: bool,
    row_mode: str,
    initial_max: int | None,
    key_id: str | None,
    global_id: str | None,
    found: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "max_end_user_budget_id", global_id)
    budget: Final = LiteLLM_BudgetTable(budget_id="budget", max_budget=5, tpm_limit=0, model_max_budget={})
    row: Final = (
        LiteLLM_EndUserTable(user_id="eu", blocked=False, litellm_budget_table=budget if row_mode == "budget" else None)
        if row_mode != "missing"
        else None
    )
    end: Final = AsyncMock(return_value=row)
    default: Final = AsyncMock(return_value=budget if found else None)
    monkeypatch.setattr(auth_checks, "get_end_user_object", end)
    monkeypatch.setattr(auth_reference, "get_end_user_object", end)
    monkeypatch.setattr(legacy, "get_end_user_object", end)
    monkeypatch.setattr(auth_checks, "get_default_end_user_budget", default)
    monkeypatch.setattr(auth_reference, "get_default_end_user_budget", default)
    token: Final = UserAPIKeyAuth(
        end_user_id="eu", end_user_max_budget=initial_max, end_user_tpm_limit=9, metadata={"end_user_budget_id": key_id}
    )
    result: Final = (
        await adapter.lookup_end_user_and_apply_budget(
            token, "/chat/completions", None, Mock(), UserApiKeyCache(), None
        )
        if native
        else await legacy._lookup_end_user_and_apply_budget(  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # Phase 1 untyped reference boundary
            token, "/chat/completions", None, Mock(), UserApiKeyCache(), None
        )  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # Phase 1 untyped reference boundary
    )
    actual_token, actual_row = result
    assert actual_token is token and actual_row is row
    default_used: Final = row is None and (key_id is not None or global_id is not None) and found
    assert token.end_user_max_budget == (5 if default_used and initial_max is None else initial_max)
    assert token.end_user_tpm_limit == (0 if row_mode == "budget" or default_used else 9)
    assert token.end_user_model_max_budget == ({} if row_mode == "budget" or default_used else None)
    assert end.await_count == 1
    assert default.await_count == (
        0
        if row is not None
        else (int(key_id is not None) + int(global_id is not None and (key_id is None or not found)))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("native_failure", (False, True))
async def test_lookup_io_failure_does_not_select_defaults_and_native_failure_is_terminal(
    native_failure: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure: Final = NativeAuthRuntimeError("native") if native_failure else RuntimeError("io")
    load: Final = AsyncMock(side_effect=failure)
    default: Final = AsyncMock(side_effect=AssertionError("no defaults after lookup error"))
    monkeypatch.setattr(auth_checks, "get_end_user_object", load)
    monkeypatch.setattr(auth_reference, "get_end_user_object", load)
    monkeypatch.setattr(auth_checks, "get_default_end_user_budget", default)
    monkeypatch.setattr(auth_reference, "get_default_end_user_budget", default)
    token: Final = UserAPIKeyAuth(end_user_id="eu", metadata={"end_user_budget_id": "key"})
    if native_failure:
        with pytest.raises(NativeAuthRuntimeError) as raised:
            await adapter.lookup_end_user_and_apply_budget(token, "", None, Mock(), UserApiKeyCache(), None)
        assert raised.value is failure
    else:
        assert await adapter.lookup_end_user_and_apply_budget(token, "", None, Mock(), UserApiKeyCache(), None) == (
            token,
            None,
        )
    default.assert_not_called()


@pytest.mark.asyncio
async def test_native_validation_failure_stops_before_user_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "validate_end_user_id_in_db", True)
    failure: Final = NativeAuthRuntimeError("native")
    monkeypatch.setattr(auth_checks, "get_end_user_object", AsyncMock(side_effect=failure))
    monkeypatch.setattr(auth_reference, "get_end_user_object", AsyncMock(side_effect=failure))
    user: Final = AsyncMock()
    monkeypatch.setattr(auth_checks, "get_user_object", user)
    monkeypatch.setattr(auth_reference, "get_user_object", user)
    with pytest.raises(NativeAuthRuntimeError) as raised:
        await adapter.resolve_and_validate_end_user_id("eu", Mock(), UserApiKeyCache())
    assert raised.value is failure
    user.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("found", (False, True))
async def test_default_budget_cache_miss_reads_database_and_caches_only_found_rows(
    native: bool,
    found: bool,
) -> None:
    cache: Final = UserApiKeyCache()
    row: Final = LiteLLM_BudgetTable(budget_id="budget", max_budget=0) if found else None
    read: Final = AsyncMock(return_value=row)
    db: Final = Mock(db=Mock(litellm_budgettable=Mock(find_unique=read)))
    function: Final = adapter.get_default_end_user_budget if native else auth_reference.get_default_end_user_budget
    assert await function(db, cache, budget_id="budget") == row
    read.assert_awaited_once_with(where={"budget_id": "budget"})
    assert await cache.async_get_cache(key="default_end_user_budget:budget", model_type=LiteLLM_BudgetTable) == row
    if found:
        assert await function(db, cache, budget_id="budget") == row
        read.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("native_failure", (False, True))
async def test_default_budget_database_failure_keeps_native_fault_terminal(native_failure: bool) -> None:
    failure: Final = NativeAuthRuntimeError("native") if native_failure else RuntimeError("io")
    read: Final = AsyncMock(side_effect=failure)
    db: Final = Mock(db=Mock(litellm_budgettable=Mock(find_unique=read)))
    if native_failure:
        with pytest.raises(NativeAuthRuntimeError) as raised:
            await adapter.get_default_end_user_budget(db, UserApiKeyCache(), budget_id="budget")
        assert raised.value is failure
    else:
        assert await adapter.get_default_end_user_budget(db, UserApiKeyCache(), budget_id="budget") is None
    read.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_budget_cache_failure_is_not_swallowed(native: bool) -> None:
    failure: Final = RuntimeError("cache unavailable")
    cache: Final = Mock(spec=UserApiKeyCache)
    read: Final = AsyncMock(side_effect=failure)
    cache.async_get_cache = read
    function: Final = adapter.get_default_end_user_budget if native else auth_reference.get_default_end_user_budget
    with pytest.raises(RuntimeError) as raised:
        await function(Mock(), cache, budget_id="budget")
    assert raised.value is failure
    read.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("own_failure", (False, True))
@pytest.mark.parametrize("user_found", (False, True))
async def test_validation_lookup_errors_select_secondary_identity_read(
    native: bool,
    own_failure: bool,
    user_found: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own: Final = AsyncMock(side_effect=RuntimeError("end-user read")) if own_failure else AsyncMock(return_value=None)
    user: Final = (
        AsyncMock(return_value=LiteLLM_UserTable(user_id="eu"))
        if user_found
        else AsyncMock(side_effect=RuntimeError("user read"))
    )
    monkeypatch.setattr(auth_checks, "get_end_user_object", own)
    monkeypatch.setattr(auth_reference, "get_end_user_object", own)
    monkeypatch.setattr(auth_checks, "get_user_object", user)
    monkeypatch.setattr(auth_reference, "get_user_object", user)
    function: Final = adapter.end_user_id_exists_in_db if native else auth_reference._end_user_id_exists_in_db  # pyright: ignore[reportPrivateUsage]  # Phase 1 parity reference
    assert await function("eu", Mock(), UserApiKeyCache()) is user_found
    own.assert_awaited_once()
    user.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_budget_denial_preserves_exception_identity(native: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = litellm.BudgetExceededError(current_cost=2, max_budget=1)
    read: Final = AsyncMock(side_effect=failure)
    monkeypatch.setattr(auth_checks, "get_end_user_object", read)
    monkeypatch.setattr(auth_reference, "get_end_user_object", read)
    monkeypatch.setattr(legacy, "get_end_user_object", read)
    invocation: Final = (
        adapter.lookup_end_user_and_apply_budget(UserAPIKeyAuth(), "", None, Mock(), UserApiKeyCache(), None)
        if native
        else legacy._lookup_end_user_and_apply_budget(UserAPIKeyAuth(), "", None, Mock(), UserApiKeyCache(), None)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # Phase 1 untyped reference boundary
    )
    with pytest.raises(litellm.BudgetExceededError) as raised:
        await invocation
    assert raised.value is failure
    read.assert_awaited_once()
