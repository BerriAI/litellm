from types import MappingProxyType
from typing import Final

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.offering_router import OfferingAccessGuard, OfferingRouterView, OfferingServingSnapshot
from litellm.router import Router
from litellm.types.utils import CallTypes


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
