from typing import Final

import pytest

import litellm

from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_model
from litellm.proxy.auth.key_disallowed_models import (
    disallowed_models_removed_by,
    ensure_key_may_call_model,
    first_disallowed_model,
    without_key_disallowed_models,
)
from litellm.proxy.auth.user_api_key_auth import enforce_key_and_fallback_model_access
from litellm.types.proxy.management_endpoints.key_requests import UpdateKeyRequest


def _token(**fields: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth.model_validate({"token": "hashed-token", **fields})


def _denied_model(error: pytest.ExceptionInfo[ProxyException]) -> str:
    assert error.value.type == ProxyErrorTypes.key_model_access_denied
    assert str(error.value.code) == "403"
    return error.value.message


@pytest.mark.parametrize(
    "fields,requested",
    [
        pytest.param({}, "gpt-denied", id="exact-name"),
        pytest.param({"aliases": {"short": "gpt-denied"}}, "short", id="key-alias"),
        pytest.param({"team_model_aliases": {"team-short": "gpt-denied"}}, "team-short", id="team-alias"),
        pytest.param({}, ["gpt-allowed", "gpt-denied"], id="denied-not-first-in-list"),
    ],
)
def test_a_disallowed_model_is_refused_however_it_is_named(fields: dict[str, object], requested: object) -> None:
    token: Final = _token(disallowed_models=("gpt-denied",), **fields)
    with pytest.raises(ProxyException) as error:
        ensure_key_may_call_model(requested, token, None)  # pyright: ignore[reportArgumentType]  # parametrized str or list
    assert "is not available for this API key" in _denied_model(error)


def test_a_key_alias_that_a_global_alias_redirects_to_a_disallowed_model_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {"public-name": "gpt-denied"})
    token: Final = _token(disallowed_models=("gpt-denied",), aliases={"short": "public-name"})
    with pytest.raises(ProxyException) as error:
        ensure_key_may_call_model("short", token, None)
    _denied_model(error)


def test_a_deployment_id_of_a_disallowed_model_group_is_refused_and_unlisted() -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-denied",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fixture"},
                "model_info": {"id": "deployment-of-denied"},
            },
            {
                "model_name": "gpt-allowed",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fixture"},
                "model_info": {"id": "deployment-of-allowed"},
            },
        ]
    )
    token: Final = _token(disallowed_models=("gpt-denied",))
    with pytest.raises(ProxyException) as error:
        ensure_key_may_call_model("deployment-of-denied", token, router)
    _denied_model(error)
    assert first_disallowed_model(["deployment-of-allowed"], token, router) is None
    assert without_key_disallowed_models(["deployment-of-denied", "deployment-of-allowed"], token, router) == (
        "deployment-of-allowed",
    )


@pytest.mark.parametrize(
    "requested,refused",
    [
        pytest.param("openai/gpt-4o", True, id="matches-provider-wildcard"),
        pytest.param("anthropic/claude", False, id="other-provider"),
        pytest.param("openai", False, id="bare-provider-name"),
    ],
)
def test_a_wildcard_entry_refuses_every_model_it_matches(requested: str, refused: bool) -> None:
    token: Final = _token(disallowed_models=("openai/*",))
    assert (first_disallowed_model([requested], token, None) == requested) is refused


@pytest.mark.parametrize("disallowed", [None, (), ("gpt-other",)])
def test_a_model_outside_disallowed_models_is_not_refused(disallowed: tuple[str, ...] | None) -> None:
    assert first_disallowed_model(["gpt-allowed"], _token(disallowed_models=disallowed), None) is None
    ensure_key_may_call_model("gpt-allowed", _token(disallowed_models=disallowed), None)


def test_listing_drops_every_disallowed_model_and_every_alias_pointing_at_one() -> None:
    token: Final = _token(disallowed_models=("b", "d"), aliases={"alias-b": "b"})
    assert without_key_disallowed_models(["a", "b", "alias-b", "c", "d"], token, None) == ("a", "c")


@pytest.mark.asyncio
async def test_disallowed_models_win_over_the_keys_own_allow_list() -> None:
    token: Final = _token(models=["gpt-allowed", "gpt-denied"], disallowed_models=("gpt-denied",))
    assert await can_key_call_model("gpt-allowed", None, token, None) is True
    with pytest.raises(ProxyException) as error:
        await can_key_call_model("gpt-denied", None, token, None)
    _denied_model(error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"models": ["all-team-models"], "team_id": "team-1"}, id="all-team-models"),
        pytest.param({"config": {"model_list": []}}, id="config-key"),
    ],
)
@pytest.mark.parametrize(
    "request_data",
    [
        pytest.param({"model": "gpt-denied"}, id="model"),
        pytest.param({"model": "gpt-allowed", "fallbacks": ["gpt-denied"]}, id="fallback"),
    ],
)
async def test_keys_that_skip_the_allow_list_still_honor_disallowed_models(
    fields: dict[str, object], request_data: dict[str, object]
) -> None:
    token: Final = _token(disallowed_models=("gpt-denied",), **fields)
    with pytest.raises(ProxyException) as error:
        await enforce_key_and_fallback_model_access(
            valid_token=token,
            request_data=request_data,
            route="/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )
    _denied_model(error)


@pytest.mark.asyncio
async def test_keys_that_skip_the_allow_list_are_served_models_outside_disallowed_models() -> None:
    request_data: Final[dict[str, object]] = {"model": "gpt-allowed"}
    assert (
        await enforce_key_and_fallback_model_access(
            valid_token=_token(models=["all-team-models"], team_id="team-1", disallowed_models=("gpt-denied",)),
            request_data=request_data,
            route="/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )
        is None
    )
    assert request_data == {"model": "gpt-allowed"}


@pytest.mark.parametrize(
    "update,stored,removed",
    [
        pytest.param({}, ["a", "b"], frozenset[str](), id="field-not-sent"),
        pytest.param({"disallowed_models": []}, ["a", "b"], frozenset({"a", "b"}), id="cleared"),
        pytest.param({"disallowed_models": None}, ["a"], frozenset({"a"}), id="null"),
        pytest.param({"disallowed_models": ["b"]}, ["a", "b"], frozenset({"a"}), id="narrowed"),
        pytest.param({"disallowed_models": ["a", "b", "c"]}, ["a", "b"], frozenset[str](), id="added"),
        pytest.param({"disallowed_models": []}, None, frozenset[str](), id="nothing-stored"),
    ],
)
def test_removed_disallowed_models_are_the_stored_entries_the_update_drops(
    update: dict[str, object], stored: list[str] | None, removed: frozenset[str]
) -> None:
    assert disallowed_models_removed_by(UpdateKeyRequest.model_validate({"key": "sk-1", **update}), stored) == removed
