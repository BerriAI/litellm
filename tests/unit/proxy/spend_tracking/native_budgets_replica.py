from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import pytest
from fastapi import Request

import litellm
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enforcement import FieldView
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, SourcedFields
from litellm.integrations.custom_logger_v2.legacy import PROXY_REQUEST_KEYS, KwargsFields
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import _authorize_authenticated_request
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking import budget_reservation, native_budgets
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.proxy.v2_proxy_fields import PRE_CALL_SOURCES
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.router import Router


def native_budgets_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "callbacks", [])
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # each test installs its owner over its own store
    yield
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # later tests run without a native owner


def install_native_owner(store: SpendCounterStore) -> native_budgets.NativeBudgetsLogger:
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # the owner binds the store the test installed
    for stale in [
        callback for callback in litellm.callbacks if isinstance(callback, native_budgets.NativeBudgetsLogger)
    ]:
        litellm.callbacks.remove(stale)
    native_budgets.install_native_budgets(store, litellm.callbacks)
    return native_owner()


def native_owner() -> native_budgets.NativeBudgetsLogger:
    if native_budgets.native_budgets() is None:
        return install_native_owner(proxy_server.spend_counter_store)
    owner: Final = next(
        (callback for callback in litellm.callbacks if isinstance(callback, native_budgets.NativeBudgetsLogger)), None
    )
    assert owner is not None, "the native budgets owner is not in litellm.callbacks"
    return owner


async def native_pre_call(auth: UserAPIKeyAuth, data: Mapping[str, object]) -> Patch | Abort | None:
    request: Final = {**data, "litellm_call_id": uuid.uuid4().hex}
    fields: Final = SourcedFields(
        KwargsFields(request, PROXY_REQUEST_KEYS), PRE_CALL_SOURCES, HookArgs(auth=auth, request=request)
    )
    outcome: Final = await native_owner().run_rust(
        Hook.pre_call,
        frozenset({Field.budget_plan}),
        FieldView(fields),
        CallContext[object](hook=Hook.pre_call, call=None, state=None),
    )
    if isinstance(outcome, Patch):
        _ = fields.apply(outcome)
    return outcome


async def admit_native_plan(auth: UserAPIKeyAuth, data: Mapping[str, object]) -> None:
    if auth.budget_plan is None:
        return
    outcome: Final = await native_pre_call(auth, data)
    if isinstance(outcome, Abort) and outcome.exception is not None:
        raise outcome.exception


async def authorize_and_admit(
    token: UserAPIKeyAuth,
    request: Request,
    request_data: dict[str, object],
    route: str,
    api_key: str,
    *,
    force_virtual_key_checks: bool = False,
) -> UserAPIKeyAuth | None:
    billed: Final = await _authorize_authenticated_request(
        token, request, request_data, route, api_key, force_virtual_key_checks=force_virtual_key_checks
    )
    await admit_native_plan(billed or token, request_data)
    return billed


UNLIMITED_BUDGET: Final = 1e12


async def native_reservation(
    store: SpendCounterStore,
    auth: UserAPIKeyAuth,
    entries: Sequence[Mapping[str, object]],
    reserved_cost: float,
    input_cost: float = 0.0,
    route: str = "/v1/chat/completions",
) -> dict[str, object]:
    _ = install_native_owner(store)
    plan: Final = {
        "counters": [
            {
                "counter_key": entry["counter_key"],
                "max_budget": entry.get("max_budget", UNLIMITED_BUDGET),
                "fallback_spend": entry.get("fallback_spend", 0.0),
                "entity_type": entry.get("entity_type"),
                "entity_id": entry.get("entity_id"),
                "spend_log_entity_id": entry.get("spend_log_entity_id", entry.get("entity_id")),
            }
            for entry in entries
        ],
        "reservation_cost": float(reserved_cost),
        "input_cost": float(input_cost),
        "input_tokens": None,
        "throttle_counter": None,
        "fail_closed": False,
        "status_code": int(litellm.budget_exceeded_status_code),
        "reservation_id": native_budgets.new_reservation_id(),
    }
    _ = native_budgets.pending_native_reservation(plan, auth, route, {}, {})
    reservation: Final = await native_budgets.admit_native_reservation_now(auth)
    assert reservation is not None and native_budgets.is_native_reservation(reservation)
    return reservation


async def reserve_budget_for_request(
    request_body: dict,
    route: str,
    llm_router: Router | None,
    valid_token: UserAPIKeyAuth | None,
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    end_user_id: str | None = None,
    end_user_object: object = None,
    apply_user_budget_to_team_keys: bool = False,
    fail_closed_budget_enforcement: bool = False,
    raw_body: bytes | None = None,
    path_params: Mapping[str, object] | None = None,
) -> dict | None:
    _ = native_owner()
    plan: Final = await budget_reservation.reserve_budget_for_request(
        request_body=request_body,
        route=route,
        llm_router=llm_router,
        valid_token=valid_token,
        team_object=team_object,
        user_object=user_object,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        end_user_id=end_user_id,
        end_user_object=end_user_object,
        apply_user_budget_to_team_keys=apply_user_budget_to_team_keys,
        fail_closed_budget_enforcement=fail_closed_budget_enforcement,
        raw_body=raw_body,
        path_params=path_params,
    )
    if plan is None or valid_token is None:
        return plan
    assert valid_token.budget_plan is not None, f"{route} reserved through the Python engine, not the native owner"
    outcome: Final = await native_pre_call(valid_token, request_body)
    if isinstance(outcome, Abort) and outcome.exception is not None:
        raise outcome.exception
    return plan if native_budgets.is_native_reservation(plan) else None
