import json
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from fastapi import HTTPException

from litellm import Router
from litellm.constants import SVG_AUTHOR_DEPLOYMENT_METADATA_KEY
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.image_endpoints.svg_author_model import svg_author_model_updates

AUTHOR_DEPLOYMENT_ID: Final = "my-nano-deployment"
AUTHOR_STAMP: Final = {"deployment_id": AUTHOR_DEPLOYMENT_ID, "provider_model": "gpt-5-nano"}


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


SHAPES: Final = (
    _svg_top_level,
    _svg_json_string,
    _svg_form_field,
    _svg_in_extra_body,
    _svg_in_extra_body_string,
    _svg_form_field_in_extra_body,
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
    return _author_in(json.loads(extra_body) if isinstance(extra_body, str) else extra_body)


def _request(shape: Callable[[str], Mapping[str, object]], author: str) -> Mapping[str, object]:
    return {"model": "gpt-image-2.6", "prompt": "a red circle", "output_format": "svg", "metadata": {}, **shape(author)}


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
    carriers: Final = sorted(request.keys() & {"svg", "extra_body"})
    assert _author_sent(authorized) == "gpt-5-nano"
    assert authorized["metadata"] == {SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: AUTHOR_STAMP}
    assert [type(authorized[field]) for field in carriers] == [type(request[field]) for field in carriers]


@pytest.mark.asyncio
async def test_key_alias_for_the_author_is_sent_as_its_provider_model(router: Router) -> None:
    key: Final = UserAPIKeyAuth(models=["gpt-image-2.6", "my-nano"], aliases={"nano": "my-nano"})
    authorized: Final = await _authorize(_request(_svg_top_level, "nano"), key, router)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert authorized["metadata"] == {SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: AUTHOR_STAMP}


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
    assert await _authorize(request, UserAPIKeyAuth(), router) == request


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
    assert authorized["metadata"] == {SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: AUTHOR_STAMP}


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
async def test_chained_key_aliases_cannot_approve_a_different_model_than_the_one_sent(router: Router) -> None:
    key: Final = UserAPIKeyAuth(
        models=["gpt-image-2.6", "my-nano"],
        aliases={"nano-alias": "restricted-chat", "restricted-chat": "my-nano"},
    )
    with pytest.raises(ProxyException) as denied:
        await _authorize(_request(_svg_top_level, "nano-alias"), key, router)
    assert denied.value.code == "403"


@pytest.mark.asyncio
async def test_saved_model_group_alias_is_applied_once_to_the_author(router: Router) -> None:
    request: Final = {
        **_request(_svg_top_level, "nano-alias"),
        "router_settings_override": {"model_group_alias": {"nano-alias": "my-nano", "my-nano": "restricted-chat"}},
    }
    authorized: Final = await _authorize(request, UserAPIKeyAuth(), router)
    assert _author_sent(authorized) == "gpt-5-nano"
    assert authorized["metadata"] == {SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: AUTHOR_STAMP}


@pytest.mark.asyncio
async def test_team_alias_for_the_author_is_sent_as_its_team_deployment_model() -> None:
    team_router: Final = Router(
        model_list=[
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
    assert authorized["metadata"] == {
        SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: {"deployment_id": "team-nano-deployment", "provider_model": "gpt-5-nano"}
    }
