import json
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from fastapi import HTTPException

import litellm
from litellm import Router
from litellm.constants import SVG_AUTHOR_BUDGET_MODELS_METADATA_KEY, SVG_AUTHOR_DEPLOYMENT_METADATA_KEY
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.image_endpoints.svg_author_model import svg_author_model_updates

AUTHOR_DEPLOYMENT_ID: Final = "my-nano-deployment"
AUTHOR_STAMP: Final = {
    "deployment_id": AUTHOR_DEPLOYMENT_ID,
    "provider_model": "gpt-5-nano",
    "image_deployment_ids": ("image-deployment",),
}


@pytest.fixture
def router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt-image-2.6",
                "litellm_params": {"model": "openai/gpt-image-2.6-flare", "api_key": "sk-test"},
                "model_info": {"id": "image-deployment", "mode": "image_generation"},
            },
            {
                "model_name": "my-nano",
                "litellm_params": {"model": "openai/gpt-5-nano", "api_key": "sk-test"},
                "model_info": {"id": AUTHOR_DEPLOYMENT_ID, "mode": "chat"},
            },
            {
                "model_name": "restricted-chat",
                "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-test"},
                "model_info": {"id": "restricted-deployment", "mode": "chat"},
            },
        ]
    )


def _svg_top_level(model: str) -> Mapping[str, object]:
    return {"svg": {"model": model}}


def _svg_json_string(model: str) -> Mapping[str, object]:
    return {"svg": json.dumps({"model": model})}


def _svg_form_field(model: str) -> Mapping[str, object]:
    return {"svg[model]": model}


def _svg_in_extra_body(model: str) -> Mapping[str, object]:
    return {"extra_body": {"svg": {"model": model}}}


def _svg_in_extra_body_string(model: str) -> Mapping[str, object]:
    return {"extra_body": json.dumps({"svg": {"model": model}})}


def _svg_form_field_in_extra_body(model: str) -> Mapping[str, object]:
    return {"extra_body": {"svg[model]": model}}


def _svg_in_extra_body_pairs(model: str) -> Mapping[str, object]:
    return {"extra_body": [["svg", {"model": model}]]}


SHAPES: Final = (
    _svg_top_level,
    _svg_json_string,
    _svg_form_field,
    _svg_in_extra_body,
    _svg_in_extra_body_string,
    _svg_form_field_in_extra_body,
    _svg_in_extra_body_pairs,
)


def _author_in(container: Mapping[str, object]) -> str | None:
    if "svg[model]" in container:
        return str(container["svg[model]"])
    svg: Final = container.get("svg")
    if svg is None:
        return None
    return (json.loads(svg) if isinstance(svg, str) else svg)["model"]


def _author_sent(body: Mapping[str, object]) -> str | None:
    extra_body: Final = body.get("extra_body")
    if extra_body is None:
        return _author_in(body)
    return _author_in(json.loads(extra_body) if isinstance(extra_body, str) else dict(extra_body))


def _request(shape: Callable[[str], Mapping[str, object]], author: str) -> Mapping[str, object]:
    return {"model": "gpt-image-2.6", "prompt": "a red circle", "output_format": "svg", "metadata": {}, **shape(author)}


def _price_stamp(authorized: Mapping[str, object]) -> object:
    metadata: Final = authorized["metadata"]
    assert isinstance(metadata, Mapping)
    return metadata.get(SVG_AUTHOR_DEPLOYMENT_METADATA_KEY)


async def _authorize(body: Mapping[str, object], key: UserAPIKeyAuth, router: Router) -> Mapping[str, object]:
    updates: Final = await svg_author_model_updates(
        body=body, user_api_key_dict=key, llm_router=router, metadata_variable_name="metadata"
    )
    return {**body, **updates}


@pytest.mark.asyncio
@pytest.mark.parametrize("author", ["restricted-chat", "gpt-5.5"])
@pytest.mark.parametrize("shape", SHAPES)
async def test_key_without_access_to_the_svg_author_is_rejected(
    shape: Callable[[str], Mapping[str, object]], author: str, router: Router
) -> None:
    with pytest.raises(ProxyException) as denied:
        await _authorize(_request(shape, author), UserAPIKeyAuth(models=["gpt-image-2.6"]), router)
    assert denied.value.code == "403"


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", SHAPES)
async def test_allowed_proxy_author_is_sent_as_its_provider_model_and_recorded(
    shape: Callable[[str], Mapping[str, object]], router: Router
) -> None:
    request: Final = _request(shape, "my-nano")
    authorized: Final = await _authorize(request, UserAPIKeyAuth(models=["gpt-image-2.6", "my-nano"]), router)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert _price_stamp(authorized) == AUTHOR_STAMP


@pytest.mark.asyncio
async def test_key_alias_for_the_author_is_sent_as_its_provider_model(router: Router) -> None:
    key: Final = UserAPIKeyAuth(models=["gpt-image-2.6", "my-nano"], aliases={"nano": "my-nano"})
    authorized: Final = await _authorize(_request(_svg_top_level, "nano"), key, router)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert _price_stamp(authorized) == AUTHOR_STAMP


@pytest.mark.asyncio
async def test_wildcard_author_is_sent_without_its_provider_prefix() -> None:
    wildcard: Final = Router(
        model_list=[{"model_name": "openai/*", "litellm_params": {"model": "openai/*", "api_key": "sk-test"}}]
    )
    authorized: Final = await _authorize(_request(_svg_top_level, "openai/gpt-5-mini"), UserAPIKeyAuth(), wildcard)
    assert _author_sent(authorized) == "gpt-5-mini"


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", SHAPES)
async def test_unrestricted_key_keeps_a_raw_provider_author_unchanged(
    shape: Callable[[str], Mapping[str, object]], router: Router
) -> None:
    request: Final = _request(shape, "gpt-5-mini")
    authorized: Final = await _authorize(request, UserAPIKeyAuth(), router)
    assert {**authorized, "metadata": {}} == request
    assert authorized["metadata"] == {SVG_AUTHOR_BUDGET_MODELS_METADATA_KEY: ["gpt-5-mini"]}


@pytest.mark.asyncio
async def test_request_without_svg_is_untouched(router: Router) -> None:
    request: Final = {"model": "gpt-image-2.6", "prompt": "a red circle", "metadata": {}}
    assert await _authorize(request, UserAPIKeyAuth(models=["gpt-image-2.6"]), router) == request


@pytest.mark.asyncio
@pytest.mark.parametrize("second_shape", [_svg_in_extra_body, _svg_form_field_in_extra_body])
async def test_conflicting_svg_authors_are_rejected(
    second_shape: Callable[[str], Mapping[str, object]], router: Router
) -> None:
    request: Final = {**_request(_svg_top_level, "my-nano"), **second_shape("restricted-chat")}
    with pytest.raises(HTTPException) as rejected:
        await _authorize(request, UserAPIKeyAuth(), router)
    assert rejected.value.status_code == 400


@pytest.mark.asyncio
async def test_key_or_team_model_group_alias_for_the_author_is_sent_as_its_provider_model(router: Router) -> None:
    request: Final = {
        **_request(_svg_top_level, "nano-alias"),
        "router_settings_override": {"model_group_alias": {"nano-alias": "my-nano"}},
    }
    authorized: Final = await _authorize(request, UserAPIKeyAuth(models=["gpt-image-2.6", "my-nano"]), router)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert _price_stamp(authorized) == AUTHOR_STAMP


@pytest.mark.asyncio
async def test_model_group_alias_target_the_key_cannot_call_is_rejected(router: Router) -> None:
    request: Final = {
        **_request(_svg_top_level, "chat-alias"),
        "router_settings_override": {"model_group_alias": {"chat-alias": "restricted-chat"}},
    }
    with pytest.raises(ProxyException) as denied:
        await _authorize(request, UserAPIKeyAuth(models=["gpt-image-2.6", "chat-alias"]), router)
    assert denied.value.code == "403"


@pytest.mark.asyncio
async def test_customer_model_limits_apply_to_the_svg_author(router: Router) -> None:
    customer_scoped: Final = UserAPIKeyAuth().with_customer_model_scopes((("gpt-image-2.6", "my-nano"),))
    request: Final = {**_request(_svg_top_level, "restricted-chat"), "metadata": {"user_api_key_auth": customer_scoped}}
    with pytest.raises(ProxyException) as denied:
        await _authorize(request, UserAPIKeyAuth(), router)
    assert denied.value.code == "403"


@pytest.mark.asyncio
async def test_chained_key_aliases_send_the_model_they_approve(router: Router) -> None:
    key: Final = UserAPIKeyAuth(
        models=["gpt-image-2.6", "my-nano"],
        aliases={"nano-alias": "restricted-chat", "restricted-chat": "my-nano"},
    )
    authorized: Final = await _authorize(_request(_svg_top_level, "nano-alias"), key, router)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert _price_stamp(authorized) == AUTHOR_STAMP


@pytest.mark.asyncio
async def test_key_alias_through_the_global_alias_map_sends_the_model_it_approves(
    router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {"restricted-chat": "my-nano"})
    key: Final = UserAPIKeyAuth(models=["gpt-image-2.6", "my-nano"], aliases={"nano": "restricted-chat"})
    authorized: Final = await _authorize(_request(_svg_top_level, "nano"), key, router)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert _price_stamp(authorized) == AUTHOR_STAMP


@pytest.mark.asyncio
async def test_saved_model_group_alias_is_applied_once_to_the_author(router: Router) -> None:
    request: Final = {
        **_request(_svg_top_level, "nano-alias"),
        "router_settings_override": {"model_group_alias": {"nano-alias": "my-nano", "my-nano": "restricted-chat"}},
    }
    authorized: Final = await _authorize(request, UserAPIKeyAuth(), router)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert _price_stamp(authorized) == AUTHOR_STAMP


@pytest.mark.asyncio
async def test_team_alias_for_the_author_is_sent_as_its_team_deployment_model() -> None:
    team_router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-image-2.6",
                "litellm_params": {"model": "openai/gpt-image-2.6-flare", "api_key": "sk-test"},
                "model_info": {"id": "image-deployment"},
            },
            {
                "model_name": "model_name_team1_nano",
                "litellm_params": {"model": "openai/gpt-5-nano", "api_key": "sk-test"},
                "model_info": {"id": "team-nano-deployment"},
            },
        ]
    )
    key: Final = UserAPIKeyAuth(
        team_id="team1",
        team_models=["gpt-image-2.6", "team-nano"],
        team_model_aliases={"team-nano": "model_name_team1_nano"},
    )
    authorized: Final = await _authorize(_request(_svg_top_level, "team-nano"), key, team_router)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert _price_stamp(authorized) == {
        "deployment_id": "team-nano-deployment",
        "provider_model": "gpt-5-nano",
        "image_deployment_ids": ("image-deployment",),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "author_params",
    [
        {"model": "azure/gpt-5-nano", "api_key": "sk-test", "api_base": "https://example.openai.azure.com"},
        {"model": "hosted_vllm/gpt-5-nano", "api_base": "http://localhost:8000"},
        {"model": "openai/gpt-5-nano", "api_key": "sk-other-account"},
    ],
)
async def test_author_on_another_upstream_account_is_sent_but_not_priced_by_its_deployment(
    author_params: Mapping[str, object],
) -> None:
    mixed: Final = Router(
        model_list=[
            {
                "model_name": "gpt-image-2.6",
                "litellm_params": {"model": "openai/gpt-image-2.6-flare", "api_key": "sk-test"},
                "model_info": {"id": "image-deployment"},
            },
            {"model_name": "cheap-nano", "litellm_params": author_params, "model_info": {"id": "cheap-nano"}},
        ]
    )
    authorized: Final = await _authorize(_request(_svg_top_level, "cheap-nano"), UserAPIKeyAuth(), mixed)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert _price_stamp(authorized) is None


@pytest.mark.asyncio
async def test_author_price_covers_only_image_deployments_on_the_authors_account() -> None:
    two_accounts: Final = Router(
        model_list=[
            {
                "model_name": "gpt-image-2.6",
                "litellm_params": {"model": "openai/gpt-image-2.6-flare", "api_key": "sk-test"},
                "model_info": {"id": "image-same-account"},
            },
            {
                "model_name": "gpt-image-2.6",
                "litellm_params": {"model": "openai/gpt-image-2.6-flare", "api_key": "sk-other-account"},
                "model_info": {"id": "image-other-account"},
            },
            {
                "model_name": "my-nano",
                "litellm_params": {"model": "openai/gpt-5-nano", "api_key": "sk-test"},
                "model_info": {"id": AUTHOR_DEPLOYMENT_ID},
            },
        ]
    )
    authorized: Final = await _authorize(_request(_svg_top_level, "my-nano"), UserAPIKeyAuth(), two_accounts)
    stamp: Final = authorized["metadata"][SVG_AUTHOR_DEPLOYMENT_METADATA_KEY]
    assert stamp["image_deployment_ids"] == ("image-same-account",)


@pytest.mark.asyncio
@pytest.mark.parametrize(("spend", "denied"), [(0.5, False), (1.0, True)], ids=["under-budget", "at-budget"])
async def test_svg_author_over_its_model_budget_is_refused(spend: float, denied: bool, router: Router) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy._types import Litellm_EntityType
    from litellm.proxy.hooks.model_max_budget_limiter import model_budget_spend_cache_key

    key: Final = UserAPIKeyAuth(
        token="svg-budget-key",
        models=["gpt-image-2.6", "my-nano"],
        model_max_budget={"my-nano": {"budget_limit": 1.0, "time_period": "1d"}},
    )
    spend_key: Final = model_budget_spend_cache_key(
        entity_type=Litellm_EntityType.KEY, entity_id="svg-budget-key", budget_model="my-nano", budget_duration="1d"
    )
    await proxy_server.model_max_budget_limiter.dual_cache.async_set_cache(spend_key, spend)
    try:
        if denied:
            with pytest.raises(ProxyException) as refused:
                await _authorize(_request(_svg_top_level, "my-nano"), key, router)
            assert refused.value.type == "budget_exceeded"
        else:
            assert _author_sent(await _authorize(_request(_svg_top_level, "my-nano"), key, router)) == "gpt-5-nano"
    finally:
        await proxy_server.model_max_budget_limiter.dual_cache.async_delete_cache(spend_key)


@pytest.mark.asyncio
async def test_svg_author_deployment_guardrails_join_the_requests_own() -> None:
    guarded: Final = Router(
        model_list=[
            {
                "model_name": "gpt-image-2.6",
                "litellm_params": {"model": "openai/gpt-image-2.6-flare", "api_key": "sk-test"},
                "model_info": {"id": "image-deployment"},
            },
            {
                "model_name": "my-nano",
                "litellm_params": {"model": "openai/gpt-5-nano", "api_key": "sk-test", "guardrails": ["pii-mask"]},
                "model_info": {"id": AUTHOR_DEPLOYMENT_ID},
            },
        ]
    )
    request: Final = {**_request(_svg_top_level, "my-nano"), "metadata": {"guardrails": ["caller-guard"]}}
    authorized: Final = await _authorize(request, UserAPIKeyAuth(), guarded)
    assert authorized["metadata"]["guardrails"] == ["caller-guard", "pii-mask"]
    assert authorized["metadata"][SVG_AUTHOR_DEPLOYMENT_METADATA_KEY] == AUTHOR_STAMP


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "author_fields",
    [
        {"svg[model]": ["restricted-chat"]},
        {"svg": {"model": ["restricted-chat"]}},
        {"svg[model][]": ["restricted-chat"]},
        {"extra_body": {"svg[model]": ["restricted-chat"]}},
        {"svg": {"model": {"name": "restricted-chat"}}},
    ],
)
async def test_svg_author_that_is_not_a_single_name_is_rejected(
    author_fields: Mapping[str, object], router: Router
) -> None:
    request: Final = {"model": "gpt-image-2.6", "prompt": "a red circle", "metadata": {}, **author_fields}
    with pytest.raises(HTTPException) as rejected:
        await _authorize(request, UserAPIKeyAuth(models=["gpt-image-2.6"]), router)
    assert rejected.value.status_code == 400


@pytest.mark.asyncio
async def test_svg_author_reached_through_the_global_alias_map_is_held_to_its_own_budget(
    router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy._types import Litellm_EntityType
    from litellm.proxy.hooks.model_max_budget_limiter import model_budget_spend_cache_key

    monkeypatch.setattr(litellm, "model_alias_map", {"restricted-chat": "my-nano"})
    key: Final = UserAPIKeyAuth(
        token="svg-alias-budget-key",
        models=["gpt-image-2.6", "my-nano"],
        model_max_budget={"my-nano": {"budget_limit": 1.0, "time_period": "1d"}},
    )
    spend_key: Final = model_budget_spend_cache_key(
        entity_type=Litellm_EntityType.KEY,
        entity_id="svg-alias-budget-key",
        budget_model="my-nano",
        budget_duration="1d",
    )
    await proxy_server.model_max_budget_limiter.dual_cache.async_set_cache(spend_key, 1.0)
    try:
        with pytest.raises(ProxyException) as refused:
            await _authorize(_request(_svg_top_level, "restricted-chat"), key, router)
    finally:
        await proxy_server.model_max_budget_limiter.dual_cache.async_delete_cache(spend_key)
    assert refused.value.type == "budget_exceeded"


@pytest.mark.asyncio
async def test_svg_author_is_not_held_to_another_groups_budget_on_the_same_provider_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy._types import Litellm_EntityType
    from litellm.proxy.hooks.model_max_budget_limiter import model_budget_spend_cache_key

    shared_model: Final = Router(
        model_list=[
            {
                "model_name": "gpt-image-2.6",
                "litellm_params": {"model": "openai/gpt-image-2.6-flare", "api_key": "sk-test"},
                "model_info": {"id": "image-deployment"},
            },
            {
                "model_name": "my-nano",
                "litellm_params": {"model": "openai/gpt-5-nano", "api_key": "sk-test"},
                "model_info": {"id": AUTHOR_DEPLOYMENT_ID},
            },
            {
                "model_name": "other-nano",
                "litellm_params": {"model": "openai/gpt-5-nano", "api_key": "sk-test"},
                "model_info": {"id": "other-nano-deployment"},
            },
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", shared_model)
    key: Final = UserAPIKeyAuth(
        token="svg-other-group-key",
        models=["gpt-image-2.6", "my-nano", "other-nano"],
        model_max_budget={"other-nano": {"budget_limit": 1.0, "time_period": "1d"}},
    )
    spend_key: Final = model_budget_spend_cache_key(
        entity_type=Litellm_EntityType.KEY,
        entity_id="svg-other-group-key",
        budget_model="other-nano",
        budget_duration="1d",
    )
    await proxy_server.model_max_budget_limiter.dual_cache.async_set_cache(spend_key, 1.0)
    try:
        authorized: Final = await _authorize(_request(_svg_top_level, "my-nano"), key, shared_model)
    finally:
        await proxy_server.model_max_budget_limiter.dual_cache.async_delete_cache(spend_key)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert authorized["metadata"][SVG_AUTHOR_BUDGET_MODELS_METADATA_KEY] == ["my-nano"]


@pytest.mark.asyncio
async def test_already_resolved_image_model_is_not_aliased_again_for_the_price_stamp(router: Router) -> None:
    request: Final = {
        **_request(_svg_top_level, "my-nano"),
        "router_settings_override": {"model_group_alias": {"gpt-image-2.6": "restricted-chat"}},
    }
    authorized: Final = await _authorize(request, UserAPIKeyAuth(), router)
    assert _price_stamp(authorized) == AUTHOR_STAMP
