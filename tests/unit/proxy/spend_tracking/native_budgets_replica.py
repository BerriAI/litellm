from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enforcement import FieldView
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, SourcedFields
from litellm.integrations.custom_logger_v2.legacy import PROXY_REQUEST_KEYS, KwargsFields
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, UserAPIKeyAuth
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
