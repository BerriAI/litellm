from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, NoReturn

import pytest

from litellm import Router
from litellm.proxy._types import ModelAccessDeniedProxyException, ProxyErrorTypes, UserAPIKeyAuth
from litellm.proxy.auth.candidate_model_access import (
    RouterCandidateAccessCheck,
    key_may_call_model,
    refuse_ungranted_deployment_selector,
    router_candidate_access_check,
)
from litellm.router_api import RoutingContext

SHARED: Final = "openai/shared"
GROUP_A_ID: Final = "group-a-deployment"
GROUP_B_ID: Final = "group-b-deployment"


def _deployment(name: str, model: str, model_id: str, access_group: str) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_key": "k", "mock_response": model_id},
        "model_info": {"id": model_id, "access_groups": [access_group]},
    }


def _router(check: object = router_candidate_access_check) -> Router:
    return Router(
        model_list=[
            _deployment("name-a", SHARED, GROUP_A_ID, "group-a"),
            _deployment("name-b", SHARED, GROUP_B_ID, "group-b"),
        ],
        candidate_access_check=check,  # pyright: ignore[reportArgumentType]  # a test double of the check
    )


def _request(*models: str, team_models: Sequence[str] = ()) -> dict[str, object]:
    token: Final = UserAPIKeyAuth(
        api_key="hashed", models=list(models), team_id="a-team" if team_models else None, team_models=list(team_models)
    )
    return {"metadata": {"user_api_key_auth": token}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_kwargs", "kept"),
    [
        pytest.param(_request("group-a"), (GROUP_A_ID,), id="access group of one deployment"),
        pytest.param(_request("name-b"), (GROUP_B_ID,), id="public name of one deployment"),
        pytest.param(_request("group-a", "name-b"), (GROUP_A_ID, GROUP_B_ID), id="a grant for each deployment"),
        pytest.param(_request("group-c"), (), id="no grant for either deployment"),
        pytest.param(_request("group-a", team_models=["group-b"]), (), id="key and team grant different deployments"),
        pytest.param(_request(SHARED), None, id="the litellm model itself"),
        pytest.param(_request(), None, id="an unrestricted key"),
        pytest.param({}, None, id="no key"),
    ],
)
async def test_the_check_keeps_the_candidates_whose_public_name_the_key_could_request(
    request_kwargs: Mapping[str, object], kept: tuple[str, ...] | None
) -> None:
    answer: Final = await router_candidate_access_check(
        model=SHARED, candidate_ids=(GROUP_A_ID, GROUP_B_ID), request_kwargs=request_kwargs, ctx=_router()._core_ctx
    )

    assert answer == kept


def _broken_access_group_lookup(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("access group store unavailable")


@pytest.mark.asyncio
async def test_the_check_keeps_nothing_when_an_authorization_lookup_breaks(monkeypatch: pytest.MonkeyPatch) -> None:
    router: Final = _router()
    monkeypatch.setattr(type(router.registry()), "access_groups", _broken_access_group_lookup)

    answer: Final = await router_candidate_access_check(
        model=SHARED, candidate_ids=(GROUP_A_ID,), request_kwargs=_request("group-a"), ctx=router._core_ctx
    )

    assert answer == ()


@pytest.mark.asyncio
async def test_the_health_bundle_does_not_admit_a_litellm_model_through_a_granted_deployment() -> None:
    token: Final = UserAPIKeyAuth(api_key="hashed", models=["group-a"])

    assert await key_may_call_model("name-a", token, None, _router()) is True
    assert await key_may_call_model(SHARED, token, None, _router()) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "request_kwargs", "refused"),
    [
        pytest.param(SHARED, _request("group-a"), True, id="litellm model held through a deployment"),
        pytest.param(SHARED, _request(SHARED), False, id="litellm model the key lists"),
        pytest.param("name-b", _request("group-a"), False, id="a model group is left to the model access checks"),
        pytest.param(SHARED, {}, False, id="no key"),
    ],
)
async def test_a_direct_dispatch_refuses_a_litellm_model_the_key_holds_only_through_a_deployment(
    model: str, request_kwargs: Mapping[str, object], refused: bool
) -> None:
    dispatch: Final = refuse_ungranted_deployment_selector({**request_kwargs, "model": model}, None, _router())

    if not refused:
        await dispatch
        return
    with pytest.raises(ModelAccessDeniedProxyException) as denied:
        await dispatch
    assert (denied.value.type, denied.value.code) == (ProxyErrorTypes.key_model_access_denied, "403")


def _served_by(response: object) -> str:
    return str(response.choices[0].message.content)  # pyright: ignore[reportAttributeAccessIssue]  # a ModelResponse


@pytest.mark.asyncio
async def test_a_request_for_a_litellm_model_is_load_balanced_across_the_granted_deployments_only() -> None:
    router: Final = _router()
    messages: Final = [{"role": "user", "content": "hi"}]

    restricted: Final = {
        _served_by(
            await router.acompletion(model=SHARED, messages=messages, specific_deployment=True, **_request("group-a"))
        )
        for _ in range(12)
    }
    listed: Final = {
        _served_by(
            await router.acompletion(model=SHARED, messages=messages, specific_deployment=True, **_request(SHARED))
        )
        for _ in range(24)
    }

    assert restricted == {GROUP_A_ID}
    assert listed == {GROUP_A_ID, GROUP_B_ID}


@pytest.mark.asyncio
@pytest.mark.parametrize("specific_deployment", [True, False])
async def test_a_request_left_with_no_deployment_is_refused_as_a_denied_key(specific_deployment: bool) -> None:
    with pytest.raises(ModelAccessDeniedProxyException) as denied:
        await _router().acompletion(
            model=SHARED,
            messages=[{"role": "user", "content": "hi"}],
            specific_deployment=specific_deployment,
            **_request("group-a", team_models=["group-b"]),
        )

    assert (denied.value.type, denied.value.code) == (ProxyErrorTypes.key_model_access_denied, "403")


def test_a_sync_request_for_a_litellm_model_is_served_by_the_granted_deployments_only() -> None:
    router: Final = _router()
    messages: Final = [{"role": "user", "content": "hi"}]

    served: Final = {
        _served_by(router.completion(model=SHARED, messages=messages, specific_deployment=True, **_request("group-a")))
        for _ in range(12)
    }

    assert served == {GROUP_A_ID}
    with pytest.raises(ModelAccessDeniedProxyException):
        router.completion(model=SHARED, messages=messages, specific_deployment=True, **_request("group-c"))


@pytest.mark.asyncio
async def test_a_selection_for_a_litellm_model_picks_among_the_granted_deployments_only() -> None:
    router: Final = _router()

    picked: Final = {
        (
            await router.async_get_available_deployment(
                model=SHARED, request_kwargs=_request("group-a"), specific_deployment=True
            )
        )["model_info"]["id"]
        for _ in range(12)
    }

    assert picked == {GROUP_A_ID}


@dataclass(slots=True)
class _RecordingCheck:
    asked: list[str] = field(default_factory=list)  # mutable-ok: records the calls the router makes

    async def __call__(
        self, *, model: str, candidate_ids: Sequence[str], request_kwargs: Mapping[str, object], ctx: RoutingContext
    ) -> Sequence[str] | None:
        self.asked.append(model)
        return await RouterCandidateAccessCheck()(
            model=model, candidate_ids=candidate_ids, request_kwargs=request_kwargs, ctx=ctx
        )

    def denied(self, model: str) -> Exception:
        return RouterCandidateAccessCheck().denied(model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "request_kwargs", "asked"),
    [
        pytest.param(SHARED, _request("group-a"), [SHARED], id="a key naming a litellm model"),
        pytest.param("name-a", _request("group-a"), [], id="a key naming a model group"),
        pytest.param(SHARED, {}, [], id="no key"),
    ],
)
async def test_the_router_asks_the_check_only_for_a_keyed_request_naming_a_litellm_model(
    model: str, request_kwargs: Mapping[str, object], asked: list[str]
) -> None:
    check: Final = _RecordingCheck()

    await _router(check).acompletion(model=model, messages=[{"role": "user", "content": "hi"}], **request_kwargs)

    assert check.asked == asked
