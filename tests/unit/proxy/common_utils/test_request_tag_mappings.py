from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from pydantic import ValidationError

from litellm.proxy._types import ConfigGeneralSettings
from litellm.proxy.common_utils.http_parsing_utils import populate_request_with_path_params, read_request_body
from litellm.proxy.common_utils.request_tag_mappings import resolve_baggage_tag_mappings
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup, add_litellm_data_to_request
from litellm.types.proxy.request_tag_mappings import RequestTagMapping


@pytest.mark.parametrize(
    "header,value", [("entrypoint=a", "a"), ("region=sg, entrypoint=customer%20support;version=1", "customer support")]
)
def test_variable_mapper_uses_decoded_baggage_value(header, value):
    rule = RequestTagMapping(key="entrypoint", tag_prefix="usecase:")
    assert resolve_baggage_tag_mappings(header, [rule]) == ((f"usecase:{value}", False),)


@pytest.mark.parametrize(
    "header,expected",
    [("entrypoint=a", "department:support"), ("entrypoint=b", "department:sales"), ("entrypoint=x", "usecase:x")],
)
def test_exact_rules_override_variable_fallback_independent_of_order(header, expected):
    rules = [
        RequestTagMapping(key="entrypoint", tag_prefix="usecase:"),
        RequestTagMapping(key="entrypoint", value="a", tag="department:support"),
        RequestTagMapping(key="entrypoint", value="b", tag="department:sales"),
    ]
    assert resolve_baggage_tag_mappings(header, rules) == ((expected, False),)
    assert resolve_baggage_tag_mappings(header, list(reversed(rules))) == ((expected, False),)


def test_exact_rules_work_without_a_variable_rule():
    rule = RequestTagMapping(key="entrypoint", value="a", tag="usecase:manual")
    assert resolve_baggage_tag_mappings("entrypoint=a", [rule]) == (("usecase:manual", False),)
    assert resolve_baggage_tag_mappings("entrypoint=x", [rule]) == ()


@pytest.mark.parametrize("header", ["", "region=sg", "entrypoint=", "entrypoint=a,entrypoint=b", "entrypoint=x"])
def test_required_mapping_rejects_missing_ambiguous_or_unmapped_values(header):
    rule = RequestTagMapping(key="entrypoint", value="a", tag="usecase:a", required=True)
    with pytest.raises(HTTPException) as error:
        resolve_baggage_tag_mappings(header, [rule])
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    "rule",
    [
        {"key": "entrypoint", "tag": "usecase:a"},
        {"key": "entrypoint", "value": "a", "tag_prefix": "usecase:"},
        {"key": "entrypoint", "value": "a", "tag": "usecase:a", "tag_prefix": "usecase:"},
        {"key": "", "tag_prefix": "usecase:"},
    ],
)
def test_invalid_rule_is_rejected_during_config_validation(rule):
    with pytest.raises(ValidationError):
        ConfigGeneralSettings.model_validate({"request_tag_mappings": [rule]})


def test_duplicate_fallbacks_are_rejected():
    with pytest.raises(ValidationError, match="Only one mapping"):
        ConfigGeneralSettings.model_validate(
            {
                "request_tag_mappings": [
                    {"key": "entrypoint", "tag_prefix": "one:"},
                    {"key": "entrypoint", "tag_prefix": "two:"},
                ]
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,slot",
    [("/chat/completions", "metadata"), ("/v1/messages", "litellm_metadata"), ("/v1/responses", "litellm_metadata")],
)
async def test_mapped_tag_merges_with_existing_tags_in_the_budget_metadata_slot(route, slot):
    data = {slot: {"tags": ["existing", "usecase:a"]}}
    request = Request({"type": "http", "headers": [(b"baggage", b"entrypoint=a")]})
    LiteLLMProxyRequestSetup.pre_seed_litellm_metadata_for_route(data, route)
    await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
        request,
        data,
        {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:"}]},
        None,
        MagicMock(),
        MagicMock(),
    )
    assert data[slot]["tags"] == ["existing", "usecase:a"]


@pytest.mark.asyncio
async def test_budgeted_mapper_requires_a_database():
    request = Request({"type": "http", "headers": [(b"baggage", b"entrypoint=a")]})
    config = {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:", "require_budgeted_tag": True}]}
    with pytest.raises(HTTPException) as error:
        await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
            request, {}, config, None, MagicMock(), MagicMock()
        )
    assert error.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,slot",
    [("/chat/completions", "metadata"), ("/v1/messages", "litellm_metadata"), ("/v1/responses", "litellm_metadata")],
)
@pytest.mark.parametrize("query", [b"api-version=qa-version", b"key=qa-virtual-key"])
async def test_mapping_survives_cached_body_read_without_leaking_auth_query_fields(route, slot, query):
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import ProxyConfig

    async def receive():
        return {"type": "http.request", "body": b'{"model":"qa-model"}'}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": route,
            "scheme": "http",
            "server": ("localhost", 4000),
            "query_string": query,
            "headers": [(b"content-type", b"application/json"), (b"baggage", b"entrypoint=a")],
        },
        receive,
    )
    data = populate_request_with_path_params(await read_request_body(request), request)
    LiteLLMProxyRequestSetup.pre_seed_litellm_metadata_for_route(data, route)
    await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
        request,
        data,
        {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:"}]},
        None,
        MagicMock(),
        MagicMock(),
    )
    downstream = await read_request_body(request)
    assert downstream == {"model": "qa-model"}
    enriched = await add_litellm_data_to_request(
        data=downstream,
        request=request,
        user_api_key_dict=UserAPIKeyAuth(api_key="qa-key"),
        proxy_config=ProxyConfig(),
        general_settings={},
    )
    assert enriched[slot]["tags"] == ["usecase:a"]
    assert "key" not in enriched
    assert "api-version" not in enriched
    assert "key" not in enriched["proxy_server_request"]["body"]
    assert "api-version" not in enriched["proxy_server_request"]["body"]
