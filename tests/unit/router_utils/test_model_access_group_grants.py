from typing import Final

import pytest

import litellm
from litellm.router_utils.model_access_group_grants import explicit_model_grants
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


def _deployment(model_name: str, reply: str, model_id: str, access_groups: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {"model": f"openai/{model_id}-upstream", "api_key": "sk-fake", "mock_response": reply},
        "model_info": {"id": model_id, "access_groups": list(access_groups)},
    }


def _router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            _deployment("private-model", "private", "private-id", ("group-x",)),
            _deployment("public-model", "public", "public-id"),
            _deployment("shared-model", "shared-x", "shared-x-id", ("group-x",)),
            _deployment("shared-model", "shared-y", "shared-y-id", ("group-y",)),
        ],
        model_group_alias={"alias-to-private": "private-model"},
        routing_groups=[
            {
                "group_name": "mixed-group",
                "models": ["private-model", "public-model"],
                "routing_strategy": "simple-shuffle",
            }
        ],
    )


def _metadata(grants: frozenset[str] | None) -> dict[str, object]:
    caller: Final = UserAPIKeyAuth(api_key="sk-caller")
    caller.model_access_group_grants = grants
    return {"user_api_key_auth": caller}


async def _reply(router: litellm.Router, model: str, grants: frozenset[str] | None) -> str | None:
    response: Final = await router.acompletion(
        model=model, messages=[{"role": "user", "content": "hi"}], metadata=_metadata(grants)
    )
    assert isinstance(response, litellm.ModelResponse)
    return response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]  # non-streaming choices


@pytest.mark.parametrize(
    ("model", "grants", "reply"),
    [
        pytest.param("private-model", frozenset({"group-x"}), "private", id="group-grant"),
        pytest.param("private-model", frozenset({"private-model"}), "private", id="model-name-grant"),
        pytest.param("alias-to-private", frozenset({"private-model"}), "private", id="alias-of-granted-model"),
        pytest.param("private-model", None, "private", id="setting-off-or-exempt"),
        pytest.param("public-model", frozenset(), "public", id="untagged-model"),
    ],
)
async def test_granted_callers_reach_the_deployment(model: str, grants: frozenset[str] | None, reply: str) -> None:
    assert await _reply(_router(), model, grants) == reply


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("private-model", id="model-name"),
        pytest.param("alias-to-private", id="alias"),
        pytest.param("private-id", id="deployment-id"),
        pytest.param("openai/private-id-upstream", id="provider-model-name"),
    ],
)
async def test_ungranted_callers_get_403_on_every_name_for_a_tagged_deployment(model: str) -> None:
    with pytest.raises(litellm.PermissionDeniedError) as denied:
        await _reply(_router(), model, explicit_model_grants(["*", "openai/*", "all-proxy-models", "group-y"]))
    assert denied.value.status_code == 403
    assert f"The requested model '{model}' is not available for this API key" in denied.value.message


@pytest.mark.parametrize(
    ("grants", "replies"),
    [
        pytest.param(frozenset(), {"public"}, id="ungranted-caller-gets-only-untagged-members"),
        pytest.param(frozenset({"mixed-group"}), {"private", "public"}, id="group-named-explicitly-reaches-all"),
    ],
)
async def test_a_routing_group_serves_the_members_the_caller_is_granted(
    grants: frozenset[str], replies: set[str]
) -> None:
    router: Final = _router()
    assert {await _reply(router, "mixed-group", grants) for _ in range(40)} == replies


async def test_a_model_split_across_groups_serves_the_copy_in_the_callers_group() -> None:
    router: Final = _router()
    replies: Final = {await _reply(router, "shared-model", frozenset({"group-x"})) for _ in range(20)}
    assert replies == {"shared-x"}


async def test_a_catch_all_deployment_in_a_group_is_not_granted_by_a_catch_all_grant() -> None:
    router: Final = litellm.Router(model_list=[_deployment("*", "catch-all", "catch-all-id", ("group-x",))])
    with pytest.raises(litellm.PermissionDeniedError):
        await _reply(router, "*", explicit_model_grants(["*"]))
    with pytest.raises(litellm.PermissionDeniedError):
        await _reply(router, "openai/gpt-4o", explicit_model_grants(["*"]))
    assert await _reply(router, "openai/gpt-4o", explicit_model_grants(["group-x"])) == "catch-all"


async def test_the_default_fallback_does_not_reach_a_tagged_deployment() -> None:
    router: Final = litellm.Router(
        model_list=[_deployment("private-model", "private", "private-id", ("group-x",))],
        default_fallbacks=["private-model"],
    )
    with pytest.raises(litellm.PermissionDeniedError):
        await _reply(router, "unknown-model", frozenset())
    assert await _reply(router, "unknown-model", frozenset({"group-x"})) == "private"


@pytest.mark.parametrize(
    ("models", "grants"),
    [
        pytest.param(["*", "all-proxy-models", "all-team-models"], frozenset(), id="blanket"),
        pytest.param(["openai/*", "group-x", "private-model"], frozenset({"group-x", "private-model"}), id="mixed"),
        pytest.param([], frozenset(), id="empty"),
    ],
)
def test_only_names_that_grant_one_thing_count_as_explicit(models: list[str], grants: frozenset[str]) -> None:
    assert explicit_model_grants(models) == grants
