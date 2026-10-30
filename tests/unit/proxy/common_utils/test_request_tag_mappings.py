import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from pydantic import ValidationError

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.constants import BAGGAGE_MAPPED_TAGS_METADATA_KEY
from litellm.models.tag import LiteLLM_TagTable
from litellm.proxy._types import ConfigGeneralSettings, ProxyException
from litellm.proxy.common_utils.http_parsing_utils import read_request_body
from litellm.proxy.common_utils.request_tag_mappings import is_baggage_mapped_route, resolve_baggage_tag_mappings
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, tag_cache_key
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup, add_litellm_data_to_request
from litellm.types.proxy.request_tag_mappings import RequestTagMapping


def test_accounting_enrichment_without_request_state_preserves_legacy_metadata():
    request = cast(Request, SimpleNamespace(headers={}))
    data = {"metadata": {"tags": ["existing"]}}
    LiteLLMProxyRequestSetup.apply_authenticated_baggage_tags(request, data)
    assert data == {"metadata": {"tags": ["existing"]}}


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [b"", b"entrypoint=a,entrypoint=b"])
@pytest.mark.parametrize("client_metadata", [None, '{"requester_metadata":{"guardrails":["qa-guard"]}}'])
async def test_mapping_rejection_keeps_string_metadata_safe_for_auth_failure_logging(header, client_metadata):
    from litellm.utils import get_request_guardrails

    request = Request({"type": "http", "headers": [(b"baggage", header)]})
    data = {"metadata": client_metadata}
    with pytest.raises(ProxyException) as error:
        await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
            request,
            data,
            {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:", "required": True}]},
            None,
            MagicMock(),
        )
    assert error.value.code == "400"
    assert get_request_guardrails(data) == (["qa-guard"] if client_metadata is not None else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("invalid_metadata", [[], 42, "not-json", '"not-an-object"'])
async def test_invalid_mapper_metadata_maps_like_empty_metadata_and_keeps_the_client_cache(slot, invalid_metadata):
    from litellm.proxy.auth.user_api_key_auth import _read_request_body_deferring_parse_failure
    from litellm.utils import get_request_guardrails

    body = {"metadata": '{"tags":["client"]}', "litellm_metadata": '{"tags":["alternate"]}'}
    body[slot] = invalid_metadata

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/responses", "headers": [(b"baggage", b"entrypoint=a")]},
        receive,
    )
    data, parse_error = await _read_request_body_deferring_parse_failure(request)
    assert parse_error is None
    await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
        request,
        data,
        {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:"}]},
        None,
        MagicMock(),
    )
    assert "usecase:a" in data["litellm_metadata"]["tags"]
    assert "tags" not in data[slot] or slot == "litellm_metadata"
    assert get_request_guardrails(data) == []
    assert await read_request_body(request) == body


@pytest.mark.parametrize(
    "header,value",
    [
        ("entrypoint=a", "a"),
        ("entrypoint = a", "a"),
        ("region=sg, entrypoint=customer%20support;version=1", "customer support"),
    ],
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


@pytest.mark.parametrize("header", ["entrypoint=%00", "entrypoint=a%0Ab", "entrypoint=b,entry%70oint=a"])
def test_variable_mapper_skips_control_characters_and_percent_encoded_duplicate_keys(header):
    rule = RequestTagMapping(key="entrypoint", tag_prefix="usecase:")
    assert resolve_baggage_tag_mappings(header, [rule]) == ()


@pytest.mark.parametrize(
    "header,expected",
    [
        ("entry+point=a", (("usecase:a", False),)),
        ("entry%2Bpoint=a", (("usecase:a", False),)),
        ("entry+point=a,entry%2Bpoint=b", ()),
    ],
)
def test_key_with_a_plus_matches_both_wire_spellings_and_rejects_both_at_once(header, expected):
    rule = RequestTagMapping(key="entry+point", tag_prefix="usecase:")
    assert resolve_baggage_tag_mappings(header, [rule]) == expected


def test_exact_rules_work_without_a_variable_rule():
    rule = RequestTagMapping(key="entrypoint", value="a", tag="usecase:manual")
    assert resolve_baggage_tag_mappings("entrypoint=a", [rule]) == (("usecase:manual", False),)
    assert resolve_baggage_tag_mappings("entrypoint=x", [rule]) == ()


@pytest.mark.parametrize(
    "header", ["", "region=sg", "entrypoint=", "entrypoint=a,entrypoint=b", "entrypoint=b,entry%70oint=a", "entrypoint=x"]
)
def test_required_mapping_rejects_missing_ambiguous_or_unmapped_values(header):
    rule = RequestTagMapping(key="entrypoint", value="a", tag="usecase:a", required=True)
    with pytest.raises(ProxyException) as error:
        resolve_baggage_tag_mappings(header, [rule])
    assert error.value.code == "400"


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
    )
    assert data[slot]["tags"] == ["existing", "usecase:a"]


@pytest.mark.asyncio
async def test_budgeted_mapper_requires_a_database():
    request = Request({"type": "http", "headers": [(b"baggage", b"entrypoint=a")]})
    config = {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:", "require_budgeted_tag": True}]}
    with pytest.raises(ProxyException) as error:
        await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
            request, {}, config, None, MagicMock()
        )
    assert error.value.code == "503"


@pytest.mark.asyncio
async def test_budgeted_mapper_answers_503_when_the_tag_budget_lookup_fails():
    request = Request({"type": "http", "headers": [(b"baggage", b"entrypoint=a")]})
    prisma_client = MagicMock()
    prisma_client.db.litellm_tagtable.find_many = AsyncMock(side_effect=Exception("database unreachable"))
    config = {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:", "require_budgeted_tag": True}]}
    with pytest.raises(ProxyException, match="could not read tag budgets") as error:
        await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
            request, {"metadata": {}}, config, prisma_client, UserApiKeyCache()
        )
    assert error.value.code == "503"


@pytest.mark.asyncio
async def test_budgeted_mapper_serves_through_a_db_outage_when_requests_are_allowed_on_db_unavailable():
    request = Request({"type": "http", "headers": [(b"baggage", b"entrypoint=a")]})
    prisma_client = MagicMock()
    prisma_client.db.litellm_tagtable.find_many = AsyncMock(side_effect=Exception("database unreachable"))
    data = {"metadata": {}}
    config = {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:", "require_budgeted_tag": True}]}
    with patch.dict("litellm.proxy.proxy_server.general_settings", {"allow_requests_on_db_unavailable": True}):
        await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
            request, data, config, prisma_client, UserApiKeyCache()
        )
    assert data["metadata"]["tags"] == ["usecase:a"]


async def _cache_with_tag(tag: LiteLLM_TagTable) -> UserApiKeyCache:
    cache = UserApiKeyCache()
    await cache.async_set_cache(key=tag_cache_key(tag.tag_name), value=tag, model_type=LiteLLM_TagTable)
    return cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget_table",
    [None, LiteLLM_BudgetTable(budget_id="qa-budget", max_budget=None, tpm_limit=10)],
)
async def test_budgeted_mapper_rejects_a_tag_without_a_max_budget(budget_table):
    request = Request({"type": "http", "headers": [(b"baggage", b"entrypoint=a")]})
    data = {"metadata": {}}
    cache = await _cache_with_tag(LiteLLM_TagTable(tag_name="usecase:a", litellm_budget_table=budget_table))
    config = {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:", "require_budgeted_tag": True}]}
    with pytest.raises(ProxyException, match="usecase:a requires an existing tag budget") as error:
        await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
            request, data, config, MagicMock(), cache
        )
    assert error.value.code == "400"
    assert data == {"metadata": {}}


@pytest.mark.asyncio
async def test_budgeted_mapper_admits_a_tag_with_a_max_budget_and_unbudgeted_rules_need_none():
    request = Request({"type": "http", "headers": [(b"baggage", b"entrypoint=a,region=sg")]})
    data = {"metadata": {}}
    cache = await _cache_with_tag(
        LiteLLM_TagTable(
            tag_name="usecase:a", litellm_budget_table=LiteLLM_BudgetTable(budget_id="qa-budget", max_budget=1.0)
        )
    )
    config = {
        "request_tag_mappings": [
            {"key": "entrypoint", "tag_prefix": "usecase:", "require_budgeted_tag": True},
            {"key": "region", "tag_prefix": "region:"},
        ]
    }
    await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
        request, data, config, MagicMock(), cache
    )
    assert data["metadata"]["tags"] == ["usecase:a", "region:sg"]


@pytest.mark.parametrize(
    "route",
    ["/v1/chat/completions", "/chat/completions", "/v1/messages", "/v1/responses", "/a2a/qa-agent/message/send"],
)
def test_inference_routes_take_baggage_mappings(route):
    assert is_baggage_mapped_route(route)


@pytest.mark.parametrize(
    "route",
    [
        "/utils/token_counter",
        "/utils/transform_request",
        "/v1/messages/count_tokens",
        "/v1/responses/input_tokens",
        "/v1beta/models/gemini-pro:countTokens",
        "/mcp",
    ],
)
def test_unbilled_and_mcp_routes_skip_baggage_mappings(route):
    assert not is_baggage_mapped_route(route)


@pytest.mark.parametrize("registered", [True, False])
def test_custom_pass_through_routes_take_baggage_mappings_once_registered(registered):
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import _registered_pass_through_routes

    endpoints = {"qa-endpoint:exact:/billing-api": {"target": "http://example.com", "headers": {}}}
    with patch.dict(_registered_pass_through_routes, endpoints if registered else {}, clear=True):
        assert is_baggage_mapped_route("/billing-api") is registered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,slot",
    [("/chat/completions", "metadata"), ("/v1/messages", "litellm_metadata"), ("/v1/responses", "litellm_metadata")],
)
@pytest.mark.parametrize("query", [b"api-version=qa-version", b"key=qa-virtual-key"])
@pytest.mark.parametrize("client_metadata", [None, '{"requester_metadata":{"guardrails":[]}}'])
async def test_mapping_survives_cached_body_read_without_leaking_auth_query_fields(route, slot, query, client_metadata):
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import ProxyConfig

    body = {"model": "qa-model"}
    if client_metadata is not None:
        body["metadata"] = client_metadata

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

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
    from litellm.proxy.auth.user_api_key_auth import _read_request_body_deferring_parse_failure

    data, parse_error = await _read_request_body_deferring_parse_failure(request)
    assert parse_error is None
    LiteLLMProxyRequestSetup.pre_seed_litellm_metadata_for_route(data, route)
    await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
        request,
        data,
        {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:"}]},
        None,
        MagicMock(),
    )
    downstream = await read_request_body(request)
    assert downstream == body
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_metadata",
    [{"trace_name": "qa"}, {"trace_name": "qa", "tags": ["client:x"]}, {"trace_name": "qa", "tags": ["usecase:a"]}],
)
async def test_mapped_tags_stay_out_of_requester_metadata(client_metadata):
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import _read_request_body_deferring_parse_failure
    from litellm.proxy.proxy_server import ProxyConfig

    body = {"model": "qa-model", "metadata": client_metadata}

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "scheme": "http",
            "server": ("localhost", 4000),
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"baggage", b"entrypoint=a")],
        },
        receive,
    )
    data, parse_error = await _read_request_body_deferring_parse_failure(request)
    assert parse_error is None
    await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
        request, data, {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:"}]}, None, MagicMock()
    )
    enriched = await add_litellm_data_to_request(
        data=await read_request_body(request),
        request=request,
        user_api_key_dict=UserAPIKeyAuth(api_key="qa-key"),
        proxy_config=ProxyConfig(),
        general_settings={},
    )
    requester_metadata = enriched["metadata"]["requester_metadata"]
    assert "usecase:a" in enriched["metadata"]["tags"]
    assert requester_metadata["trace_name"] == "qa"
    assert requester_metadata.get("tags") == client_metadata.get("tags")
    assert BAGGAGE_MAPPED_TAGS_METADATA_KEY not in requester_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,body_fields",
    [
        ([(b"x-litellm-tags", b"usecase:a")], {}),
        ([], {"tags": ["usecase:a"]}),
        ([], {"litellm_metadata": {"tags": ["usecase:a"]}}),
    ],
    ids=["header", "body", "litellm_metadata"],
)
async def test_client_sent_tag_still_routes_when_baggage_maps_to_it(headers, body_fields):
    import litellm
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import _read_request_body_deferring_parse_failure
    from litellm.proxy.proxy_server import ProxyConfig

    body = {"model": "gpt-4", **body_fields}

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "scheme": "http",
            "server": ("localhost", 4000),
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"baggage", b"entrypoint=a"), *headers],
        },
        receive,
    )
    data, parse_error = await _read_request_body_deferring_parse_failure(request)
    assert parse_error is None
    await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
        request, data, {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:"}]}, None, MagicMock()
    )
    enriched = await add_litellm_data_to_request(
        data=await read_request_body(request),
        request=request,
        user_api_key_dict=UserAPIKeyAuth(api_key="qa-key"),
        proxy_config=ProxyConfig(),
        general_settings={},
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4o", "tags": ["usecase:a"]},
                "model_info": {"id": "usecase-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4o-mini", "tags": ["default"]},
                "model_info": {"id": "default-deployment"},
            },
        ],
        enable_tag_filtering=True,
    )
    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata=dict(enriched["metadata"]),
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "usecase-deployment"
