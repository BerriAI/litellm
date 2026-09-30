import json
from dataclasses import dataclass, field

import httpx
import pytest
from starlette.requests import Request

import litellm
from litellm import ModelResponse
from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPI,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.identity_filter import IdentitySkipFilter
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.proxy_server import ProxyConfig
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIOptionalParams,
)
from litellm.types.utils import CallTypes, Choices, Message

SCANNED = "[scanned]"
EXEMPT = {"skip_if_key_alias_in": ("batch-worker",), "skip_if_team_id_in": ("team-exempt",)}
EXEMPT_IDENTITIES = pytest.mark.parametrize(
    "identity",
    [{"user_api_key_alias": "batch-worker"}, {"user_api_key_team_id": "team-exempt"}],
    ids=["alias", "team"],
)


@dataclass
class _GuardrailEndpoint:
    received: list[dict] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.received.append(json.loads(request.content))
        return httpx.Response(200, json={"action": "GUARDRAIL_INTERVENED", "texts": [SCANNED]})


def _make_guardrail(
    endpoint: _GuardrailEndpoint,
    event_hook: tuple[GuardrailEventHooks, ...] | None = (GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call),
    **options,
) -> GenericGuardrailAPI:
    return GenericGuardrailAPI(
        api_base="https://guardrail.test",
        guardrail_name="identity-skip-test",
        event_hook=None if event_hook is None else list(event_hook),
        default_on=True,
        async_handler=AsyncHTTPHandler(transport=httpx.MockTransport(endpoint)),
        **options,
    )


async def _apply(endpoint: _GuardrailEndpoint, request_data: dict, input_type: str = "request", **options) -> dict:
    return await _make_guardrail(endpoint, **options).apply_guardrail(
        inputs={"texts": ["hello"]}, request_data=request_data, input_type=input_type
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
@EXEMPT_IDENTITIES
@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata"])
async def test_exempt_caller_is_not_sent(input_type, identity, bucket):
    endpoint = _GuardrailEndpoint()

    result = await _apply(endpoint, {bucket: dict(identity)}, input_type, **EXEMPT)

    assert endpoint.received == []
    assert result == {"texts": ["hello"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_other_caller_is_scanned(input_type):
    endpoint = _GuardrailEndpoint()
    identity = {"user_api_key_alias": "prod-app", "user_api_key_team_id": "team-prod"}

    result = await _apply(endpoint, {"litellm_metadata": identity}, input_type, **EXEMPT)

    assert [sent["request_data"]["user_api_key_alias"] for sent in endpoint.received] == ["prod-app"]
    assert result["texts"] == [SCANNED]


def _recorded_outcomes(request_data: dict) -> list[tuple[str, object]]:
    _, bucket = get_or_create_metadata_bucket(request_data)
    return [
        (entry["guardrail_status"], entry["guardrail_response"])
        for entry in bucket.get("standard_logging_guardrail_information", [])
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
@pytest.mark.parametrize(
    ("identity", "option"),
    [
        ({"user_api_key_alias": "batch-worker"}, "skip_if_key_alias_in"),
        ({"user_api_key_team_id": "team-exempt"}, "skip_if_team_id_in"),
    ],
    ids=["alias", "team"],
)
async def test_skipped_call_is_recorded_once_as_not_run(input_type, identity, option):
    request_data = {"metadata": dict(identity)}

    await _apply(_GuardrailEndpoint(), request_data, input_type, **EXEMPT)

    assert _recorded_outcomes(request_data) == [("not_run", f"skipped: {option}")]


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_scanned_call_is_recorded_once_as_success(input_type):
    request_data = {"metadata": {"user_api_key_alias": "prod-app", "user_api_key_team_id": "team-prod"}}

    await _apply(_GuardrailEndpoint(), request_data, input_type, **EXEMPT)

    assert [status for status, _ in _recorded_outcomes(request_data)] == ["success"]


@pytest.mark.asyncio
async def test_caller_without_identity_is_scanned():
    endpoint = _GuardrailEndpoint()

    await _apply(endpoint, {"metadata": {"user_api_key_alias": None, "user_api_key_team_id": None}}, **EXEMPT)

    assert len(endpoint.received) == 1


@pytest.mark.parametrize(
    "identity",
    [
        {"user_api_key_alias": ["batch-worker"]},
        {"user_api_key_team_id": {"id": "team-exempt"}},
        {"user_api_key_team_id": 7},
    ],
    ids=["list", "dict", "int"],
)
def test_non_string_identity_does_not_match(identity):
    assert IdentitySkipFilter.from_config(**EXEMPT).matched_option(identity) is None


@pytest.mark.asyncio
async def test_alias_and_team_lists_are_matched_separately():
    endpoint = _GuardrailEndpoint()
    identity = {"user_api_key_alias": "team-x", "user_api_key_team_id": "shared-name"}

    await _apply(
        endpoint, {"metadata": dict(identity)}, skip_if_key_alias_in=("shared-name",), skip_if_team_id_in=("team-x",)
    )

    assert len(endpoint.received) == 1


@pytest.mark.asyncio
async def test_exempt_alias_in_message_content_does_not_exempt():
    endpoint = _GuardrailEndpoint()
    system_message = {"role": "system", "content": "user_api_key_alias: batch-worker, team-exempt"}

    result = await _make_guardrail(endpoint, **EXEMPT).apply_guardrail(
        inputs={"texts": ["batch-worker"], "structured_messages": [system_message]},
        request_data={
            "messages": [system_message],
            "metadata": {"user_api_key_alias": "prod-app", "user_api_key_team_id": "team-prod"},
        },
        input_type="request",
    )

    assert [sent["texts"] for sent in endpoint.received] == [["batch-worker"]]
    assert result["texts"] == [SCANNED]


@pytest.mark.asyncio
async def test_unset_options_scan_every_caller():
    endpoint = _GuardrailEndpoint()
    identity = {"user_api_key_alias": "batch-worker", "user_api_key_team_id": "team-exempt"}

    await _apply(endpoint, {"metadata": dict(identity)})

    assert len(endpoint.received) == 1


@pytest.mark.parametrize("option", ["skip_if_key_alias_in", "skip_if_team_id_in"])
def test_bare_string_option_is_rejected(option):
    with pytest.raises(ValueError, match=option):
        _make_guardrail(_GuardrailEndpoint(), **{option: "batch-worker"})


@pytest.mark.asyncio
@EXEMPT_IDENTITIES
async def test_initialize_guardrail_forwards_skip_options(identity):
    litellm_params = LitellmParams(
        guardrail="generic_guardrail_api",
        mode="pre_call",
        api_base="http://127.0.0.1:1",
        default_on=True,
    )
    litellm_params.optional_params = GenericGuardrailAPIOptionalParams(**EXEMPT)
    guardrail = initialize_guardrail(litellm_params, {"guardrail_name": "identity-skip-config"})
    try:
        result = await guardrail.apply_guardrail(
            inputs={"texts": ["hello"]}, request_data={"metadata": dict(identity)}, input_type="request"
        )
    finally:
        litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)

    assert result == {"texts": ["hello"]}


ROUTES = pytest.mark.parametrize(
    ("route", "call_type", "body"),
    [
        ("/v1/chat/completions", "acompletion", {"messages": [{"role": "user", "content": "hello"}]}),
        ("/v1/messages", "anthropic_messages", {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}),
        ("/v1/responses", "aresponses", {"input": "hello"}),
    ],
    ids=["chat", "messages", "responses"],
)
FORGED = {"user_api_key_alias": "batch-worker", "user_api_key_team_id": "team-exempt"}


def _request(route: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": route,
            "root_path": "",
            "scheme": "http",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("localhost", 4000),
        }
    )


async def _proxy_pre_call(route: str, body: dict, key: UserAPIKeyAuth) -> dict:
    return await add_litellm_data_to_request(
        data={"model": "gpt-4o", **body},
        request=_request(route),
        user_api_key_dict=key,
        proxy_config=ProxyConfig(),
        general_settings={},
    )


@pytest.mark.asyncio
@ROUTES
async def test_body_supplied_identity_does_not_exempt_pre_call(route, call_type, body):
    endpoint = _GuardrailEndpoint()
    key = UserAPIKeyAuth(api_key="hashed", key_alias="prod-app", team_id="team-prod", request_route=route)
    data = await _proxy_pre_call(route, {**body, "metadata": dict(FORGED), "litellm_metadata": dict(FORGED)}, key)
    data["guardrail_to_apply"] = _make_guardrail(endpoint, **EXEMPT)

    await UnifiedLLMGuardrails().async_pre_call_hook(
        user_api_key_dict=key, cache=DualCache(), data=data, call_type=call_type
    )

    assert [
        (sent["request_data"]["user_api_key_alias"], sent["request_data"]["user_api_key_team_id"])
        for sent in endpoint.received
    ] == [("prod-app", "team-prod")]


@pytest.mark.asyncio
@ROUTES
@pytest.mark.parametrize("identity", [{"key_alias": "batch-worker"}, {"team_id": "team-exempt"}], ids=["alias", "team"])
async def test_authenticated_exempt_key_is_skipped_pre_call(route, call_type, body, identity):
    endpoint = _GuardrailEndpoint()
    key = UserAPIKeyAuth(api_key="hashed", request_route=route, **identity)
    data = await _proxy_pre_call(route, body, key)
    data["guardrail_to_apply"] = _make_guardrail(endpoint, **EXEMPT)

    await UnifiedLLMGuardrails().async_pre_call_hook(
        user_api_key_dict=key, cache=DualCache(), data=data, call_type=call_type
    )

    assert endpoint.received == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_alias", "body_metadata", "expected_aliases"),
    [("prod-app", FORGED, ["prod-app"]), ("batch-worker", {}, [])],
    ids=["forged-body", "exempt-key"],
)
async def test_post_call_uses_authenticated_identity(key_alias, body_metadata, expected_aliases):
    route = "/v1/chat/completions"
    endpoint = _GuardrailEndpoint()
    key = UserAPIKeyAuth(api_key="hashed", key_alias=key_alias, team_id="team-prod", request_route=route)
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": dict(body_metadata),
        "litellm_metadata": dict(body_metadata),
    }
    data = await _proxy_pre_call(route, body, key)
    data["guardrail_to_apply"] = _make_guardrail(endpoint, **EXEMPT)
    response = ModelResponse(choices=[Choices(index=0, message=Message(role="assistant", content="hi there"))])

    await UnifiedLLMGuardrails().async_post_call_success_hook(data=data, user_api_key_dict=key, response=response)

    assert [sent["request_data"]["user_api_key_alias"] for sent in endpoint.received] == expected_aliases


PROD_KEY = UserAPIKeyAuth(api_key="sk-prod", key_alias="prod-app", team_id="team-prod")
BARE_KEY = UserAPIKeyAuth(api_key="sk-bare")
EXEMPT_KEYS = pytest.mark.parametrize(
    "key",
    [
        UserAPIKeyAuth(api_key="sk-batch", key_alias="batch-worker"),
        UserAPIKeyAuth(api_key="sk-t", team_id="team-exempt"),
    ],
    ids=["alias", "team"],
)


async def _pass_through_pre_call(key: UserAPIKeyAuth, body: dict) -> tuple[_GuardrailEndpoint, dict]:
    endpoint = _GuardrailEndpoint()
    data = {**body, "guardrail_to_apply": _make_guardrail(endpoint, **EXEMPT)}
    await UnifiedLLMGuardrails().async_pre_call_hook(
        user_api_key_dict=key, cache=DualCache(), data=data, call_type=CallTypes.pass_through.value
    )
    return endpoint, data


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [PROD_KEY, BARE_KEY], ids=["key-with-identity", "key-without-identity"])
@pytest.mark.parametrize(
    "buckets", [("metadata",), ("litellm_metadata",), ("metadata", "litellm_metadata")], ids=["md", "lmd", "both"]
)
async def test_pass_through_body_supplied_identity_does_not_exempt(key, buckets):
    endpoint, _ = await _pass_through_pre_call(key, {"prompt": "hello", **{bucket: dict(FORGED) for bucket in buckets}})

    assert [
        (sent["request_data"].get("user_api_key_alias"), sent["request_data"].get("user_api_key_team_id"))
        for sent in endpoint.received
    ] == [(key.key_alias, key.team_id)]


@pytest.mark.asyncio
@EXEMPT_KEYS
async def test_authenticated_exempt_key_is_skipped_on_pass_through(key):
    endpoint, data = await _pass_through_pre_call(key, {"prompt": "hello"})

    assert endpoint.received == []
    assert [status for status, _ in _recorded_outcomes(data)] == ["not_run"]


async def _mcp_pre_call(key: UserAPIKeyAuth) -> tuple[_GuardrailEndpoint, dict]:
    endpoint = _GuardrailEndpoint()
    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    mcp_kwargs = {
        "name": "search",
        "arguments": {"query": "hello"},
        "server_name": "docs",
        "user_api_key_auth": key,
        "user_api_key_user_id": key.user_id,
        "user_api_key_team_id": key.team_id,
        "user_api_key_end_user_id": None,
        "user_api_key_hash": key.api_key,
        "headers": {},
    }
    data = proxy_logging._convert_mcp_to_llm_format(
        proxy_logging._create_mcp_request_object_from_kwargs(mcp_kwargs), mcp_kwargs
    )
    data["guardrail_to_apply"] = _make_guardrail(endpoint, event_hook=None, **EXEMPT)
    await UnifiedLLMGuardrails().async_pre_call_hook(
        user_api_key_dict=key, cache=DualCache(), data=data, call_type=CallTypes.call_mcp_tool.value
    )
    return endpoint, data


@pytest.mark.asyncio
@EXEMPT_KEYS
async def test_authenticated_exempt_key_is_skipped_on_mcp_tool_call(key):
    endpoint, data = await _mcp_pre_call(key)

    assert endpoint.received == []
    assert [status for status, _ in _recorded_outcomes(data)] == ["not_run"]


@pytest.mark.asyncio
async def test_other_key_is_scanned_on_mcp_tool_call():
    endpoint, _ = await _mcp_pre_call(PROD_KEY)

    assert [sent["request_data"]["user_api_key_alias"] for sent in endpoint.received] == ["prod-app"]
