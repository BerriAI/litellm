from types import MappingProxyType
from typing import Final

import pytest
from starlette.types import Message, Receive, Scope, Send

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LitellmUserRoles,
    ModelAccessDeniedProxyException,
    ProxyErrorTypes,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import can_customer_access_model, can_key_call_model
from litellm.proxy.offering_router import (
    OfferingAccessGuard,
    OfferingRouterView,
    OfferingServingSnapshot,
    OfferingSnapshotMiddleware,
)
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params
from litellm.types.utils import CallTypes


async def test_asgi_stream_pins_authority_until_final_chunk_and_passes_lifespan_through() -> None:
    first: Final = OfferingServingSnapshot(
        Router(model_list=[]), frozenset({"first"}), MappingProxyType({}), frozenset()
    )
    second: Final = OfferingServingSnapshot(
        Router(model_list=[]), frozenset({"second"}), MappingProxyType({}), frozenset()
    )
    view: Final = OfferingRouterView(first)
    sent: Final[list[Message]] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    async def stream(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            assert view.serving_snapshot() is second
            await send({"type": "lifespan.startup.complete"})
            return
        assert view.serving_snapshot() is first
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        view.publish_snapshot(second)
        assert view.serving_snapshot() is first
        assert view.latest_snapshot() is second
        await send({"type": "http.response.body", "body": b"last", "more_body": False})

    middleware: Final = OfferingSnapshotMiddleware(stream, lambda: view)
    await middleware({"type": "http"}, receive, send)
    assert view.serving_snapshot() is second
    assert tuple(message["body"] for message in sent if message["type"] == "http.response.body") == (b"first", b"last")
    await middleware({"type": "lifespan"}, receive, send)
    assert sent[-1] == {"type": "lifespan.startup.complete"}


def test_native_mutations_cannot_publish_unmanaged_routes_in_external_mode() -> None:
    native: Final = Router(model_list=[])
    view: Final = OfferingRouterView(OfferingServingSnapshot(native, frozenset(), MappingProxyType({}), frozenset()))
    with pytest.raises(litellm.BadRequestError, match="external offering configuration"):
        view.add_deployment(Deployment(model_name="unmanaged", litellm_params=LiteLLM_Params(model="openai/unmanaged")))
    assert not native.get_model_list()
    view.num_retries = 2
    assert native.num_retries == 2


async def test_native_batches_cannot_bypass_offering_selection_with_uploaded_record_models() -> None:
    view: Final = OfferingRouterView(
        OfferingServingSnapshot(Router(model_list=[]), frozenset({"selected"}), MappingProxyType({}), frozenset())
    )
    guard: Final = OfferingAccessGuard(view)
    auth: Final = UserAPIKeyAuth()
    cache: Final = DualCache()
    for data in ({"model": "selected"}, {"input_file_id": "file-already-uploaded"}):
        rejected: Final = await guard.async_pre_call_hook(auth, cache, data, "acreate_batch")
        assert isinstance(rejected, litellm.BadRequestError)
        assert "batch records" in rejected.message
    assert isinstance(
        await guard.async_pre_call_hook(auth, cache, {"model": "selected", "purpose": "batch"}, "acreate_file"),
        litellm.BadRequestError,
    )
    assert (
        await guard.async_pre_call_hook(auth, cache, {"model": "selected", "purpose": "assistants"}, "acreate_file")
        is None
    )


async def test_unavailable_and_unselected_offerings_cannot_forward_or_override_connection() -> None:
    router: Final = OfferingRouterView(
        OfferingServingSnapshot(
            Router(model_list=[]),
            frozenset({"selected"}),
            MappingProxyType({"missing": "absent from supplier"}),
            frozenset({"openai/backend"}),
            frozenset({"selected-deployment"}),
        )
    )
    guard: Final = OfferingAccessGuard(router)
    auth: Final = UserAPIKeyAuth()
    cache: Final = DualCache()
    assert await guard.async_filter_listed_models(auth, ("selected", "missing", "unselected")) == ("selected",)
    assert await guard.async_pre_call_hook(auth, cache, {"model": "selected"}, "acompletion") is None
    assert isinstance(
        await guard.async_pre_call_hook(auth, cache, {"model": "missing"}, "acompletion"),
        litellm.ServiceUnavailableError,
    )
    assert isinstance(
        await guard.async_pre_call_hook(auth, cache, {"model": "unselected"}, "acompletion"), litellm.NotFoundError
    )
    assert isinstance(
        await guard.async_pre_call_hook(
            auth, cache, {"model": "selected", "api_base": "https://other.test"}, "acompletion"
        ),
        litellm.BadRequestError,
    )
    await guard.async_pre_call_deployment_hook(
        {"model": "openai/backend", "metadata": {"model_info": {"id": "selected-deployment"}}},
        CallTypes.acompletion,
    )
    try:
        await guard.async_pre_call_deployment_hook({"model": "openai/unselected"}, CallTypes.acompletion)
    except litellm.NotFoundError:
        pass
    else:
        raise AssertionError("An unselected concrete deployment was allowed")
    try:
        await guard.async_pre_call_deployment_hook(
            {"model": "openai/backend", "metadata": {"model_info": {"id": "unavailable-other-connection"}}},
            CallTypes.acompletion,
        )
    except litellm.NotFoundError:
        pass
    else:
        raise AssertionError("An unavailable deployment sharing the backend model was allowed")


async def test_globally_available_offerings_preserve_customer_and_key_model_permissions() -> None:
    native: Final = Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": f"openai/{name}", "api_key": "fixture-key"}}
            for name in ("allowed", "restricted")
        ]
    )
    router: Final = OfferingRouterView(
        OfferingServingSnapshot(
            native,
            frozenset({"allowed", "restricted"}),
            MappingProxyType({}),
            frozenset({"openai/allowed", "openai/restricted"}),
        )
    )
    guard: Final = OfferingAccessGuard(router)
    key: Final = UserAPIKeyAuth(models=["allowed", "restricted"])
    customer: Final = LiteLLM_EndUserTable(user_id="fixture-customer", blocked=False, models=["allowed"])
    with router.pin_snapshot():
        assert await guard.async_pre_call_hook(key, DualCache(), {"model": "restricted"}, "acompletion") is None
        assert await can_key_call_model("restricted", None, key, router) is True
        with pytest.raises(ModelAccessDeniedProxyException) as denied_customer:
            can_customer_access_model("restricted", customer, router, key)
        assert denied_customer.value.code == "403"
        assert denied_customer.value.type == ProxyErrorTypes.customer_model_access_denied
        assert can_customer_access_model("allowed", customer, router, key) is True
        assert await can_key_call_model("allowed", None, key, router) is True
        with pytest.raises(ModelAccessDeniedProxyException) as denied_key:
            await can_key_call_model("restricted", None, UserAPIKeyAuth(models=["allowed"]), router)
        assert denied_key.value.code == "403"
        assert denied_key.value.type == ProxyErrorTypes.key_model_access_denied


async def test_public_and_router_aliases_follow_offering_availability_without_crossing_teams() -> None:
    active: Final = "model_name_team1_active"
    missing: Final = "model_name_team1_missing"
    other_team: Final = "model_name_team2_active"
    native: Final = Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": f"openai/{name}", "api_key": "fixture-key"},
                "model_info": {"team_id": team, "team_public_model_name": public},
            }
            for name, team, public in (
                (active, "team1", "fast"),
                (missing, "team1", "gone"),
                (other_team, "team2", "gone"),
            )
        ],
        model_group_alias={"active-alias": active, "missing-alias": missing},
    )
    view: Final = OfferingRouterView(
        OfferingServingSnapshot(
            native, frozenset({active, other_team}), MappingProxyType({missing: "absent"}), frozenset()
        )
    )
    auth: Final = UserAPIKeyAuth(team_id="team1", aliases={"my-fast": active, "my-gone": missing})
    guard: Final = OfferingAccessGuard(view)
    with view.pin_snapshot():
        assert await guard.async_filter_listed_models(
            auth, ("fast", "gone", "active-alias", "missing-alias", "my-fast", "my-gone")
        ) == (
            "fast",
            "active-alias",
            "my-fast",
        )
        assert await guard.async_pre_call_hook(auth, DualCache(), {"model": "fast"}, "acompletion") is None
        assert isinstance(
            await guard.async_pre_call_hook(auth, DualCache(), {"model": "gone"}, "acompletion"),
            litellm.ServiceUnavailableError,
        )
        assert isinstance(
            await guard.async_pre_call_hook(auth, DualCache(), {"model": "missing-alias"}, "acompletion"),
            litellm.ServiceUnavailableError,
        )
        admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
        assert await guard.async_filter_listed_models(admin, ("fast",)) == ("fast",)
        legacy: Final = OfferingAccessGuard(view, {"use_team_public_model_name": False})
        assert await legacy.async_filter_listed_models(auth, ("fast", active)) == (active,)


async def test_body_model_override_cannot_bypass_offering_authority() -> None:
    view: Final = OfferingRouterView(
        OfferingServingSnapshot(Router(model_list=[]), frozenset({"selected"}), MappingProxyType({}), frozenset())
    )
    guard: Final = OfferingAccessGuard(view)
    assert guard.message_logging is True
    assert guard.turn_off_message_logging is False
    auth: Final = UserAPIKeyAuth()
    for call_type in ("acompletion", "aresponses"):
        assert isinstance(
            await guard.async_pre_call_hook(
                auth, DualCache(), {"model": "selected", "extra_body": {"model": "unselected"}}, call_type
            ),
            litellm.BadRequestError,
        )
        assert (
            await guard.async_pre_call_hook(
                auth, DualCache(), {"model": "selected", "extra_body": {"metadata": {"fixture": "allowed"}}}, call_type
            )
            is None
        )
