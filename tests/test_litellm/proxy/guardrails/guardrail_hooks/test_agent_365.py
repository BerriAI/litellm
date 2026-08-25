from types import SimpleNamespace
from typing import Any, Final

import httpx
import time

import pytest
from fastapi import HTTPException

from litellm.exceptions import Timeout as LitellmTimeout
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.agent_365 import (
    Agent365Guardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from litellm.types.guardrails import (
    GuardrailEventHooks,
    LitellmParams,
    SupportedGuardrailIntegrations,
)
from litellm.types.proxy.guardrails.guardrail_hooks.agent_365 import (
    AGENT_365_PROD_API_BASE,
    AGENT_365_PROD_RESOURCE_APP_ID,
    Agent365GuardrailConfigModel,
)

FAKE_ASSERTION: Final = "eyJhbGciOi.eyJhdWQiOi.c2lnbmF0dXJl"
TOKEN_URL: Final = "https://login.microsoftonline.com/tenant-abc/oauth2/v2.0/token"
EVALUATE_URL: Final = f"{AGENT_365_PROD_API_BASE}/agents/tool-evaluation/evaluate"


def _response(status_code: int, payload: Any = None, text: str | None = None) -> httpx.Response:
    request: Final = httpx.Request("POST", "https://example.test")
    if payload is not None:
        return httpx.Response(status_code=status_code, json=payload, request=request)
    return httpx.Response(status_code=status_code, text=text or "", request=request)


def _token_response(access_token: str = "obo-access-token", expires_in: int = 3599) -> httpx.Response:
    return _response(200, {"access_token": access_token, "expires_in": expires_in})


def _allow_response(correlation_id: str = "corr-1") -> httpx.Response:
    return _response(
        200,
        {
            "allowed": True,
            "defender": {"status": "Evaluated", "verdict": "Allow", "message": None},
            "observability": {"status": "Recorded"},
            "correlationId": correlation_id,
        },
    )


def _block_response(message: str = "Blocked by policy", correlation_id: str = "corr-2") -> httpx.Response:
    return _response(
        200,
        {
            "allowed": False,
            "defender": {"status": "Evaluated", "verdict": "Block", "message": message},
            "correlationId": correlation_id,
        },
    )


class FakeHandler:
    def __init__(self, items: list[Any]):
        self._items = list(items)
        self.calls: list[SimpleNamespace] = []

    async def post(self, *, url, headers=None, data=None, json=None, timeout=None):
        self.calls.append(SimpleNamespace(url=url, headers=headers, data=data, json=json, timeout=timeout))
        if not self._items:
            raise AssertionError("FakeHandler ran out of programmed responses")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        if item.status_code >= 400:
            raise httpx.HTTPStatusError("error status", request=item.request, response=item)
        return item


def _make_guardrail(
    handler: FakeHandler,
    *,
    unreachable_fallback: str = "fail_closed",
    agent_id: str | None = None,
    api_base: str = AGENT_365_PROD_API_BASE,
) -> Agent365Guardrail:
    return Agent365Guardrail(
        guardrail_name="agent-365-guard",
        tenant_id="tenant-abc",
        client_id="client-xyz",
        client_secret="secret-123",
        api_base=api_base,
        agent_id=agent_id,
        unreachable_fallback=unreachable_fallback,
        async_handler=handler,
        event_hook="pre_mcp_call",
        default_on=True,
    )


def _mcp_data(**overrides: Any) -> dict:
    data: Final[dict] = {
        "mcp_tool_name": "send_email",
        "mcp_arguments": {"to": "user@example.com", "body": "hello"},
        "mcp_server_name": "outlook_mcp",
        "incoming_bearer_token": FAKE_ASSERTION,
        "metadata": {"headers": {"mcp-session-id": "sess-123"}},
    }
    data.update(overrides)
    return data


def _user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="hashed-key", key_alias="my-agent-key")


async def _run(guardrail: Agent365Guardrail, data: dict, call_type: str = "call_mcp_tool"):
    return await guardrail.async_pre_call_hook(
        user_api_key_dict=_user(),
        cache=None,
        data=data,
        call_type=call_type,
    )


class TestRegistryWiring:
    def test_enum_member_exists(self):
        assert SupportedGuardrailIntegrations.AGENT_365.value == "agent_365"

    def test_initializer_registry(self):
        assert guardrail_initializer_registry["agent_365"] is initialize_guardrail

    def test_class_registry(self):
        assert guardrail_class_registry["agent_365"] is Agent365Guardrail

    def test_config_model_wired(self):
        assert Agent365Guardrail.get_config_model() is Agent365GuardrailConfigModel
        assert Agent365GuardrailConfigModel.ui_friendly_name() == "Microsoft Agent 365"

    def test_supported_event_hooks(self):
        assert Agent365Guardrail.get_supported_event_hooks() == [GuardrailEventHooks.pre_mcp_call]


class TestInitializeGuardrail:
    def test_requires_tenant_id(self, monkeypatch):
        monkeypatch.delenv("AGENT365_TENANT_ID", raising=False)
        params: Final = LitellmParams(
            guardrail="agent_365",
            mode="pre_mcp_call",
            client_id="client-xyz",
            api_key="secret-123",
        )
        with pytest.raises(ValueError, match="tenant_id is required"):
            initialize_guardrail(params, {"guardrail_name": "a365"})

    def test_requires_client_secret(self, monkeypatch):
        monkeypatch.delenv("AGENT365_CLIENT_SECRET", raising=False)
        params: Final = LitellmParams(
            guardrail="agent_365",
            mode="pre_mcp_call",
            tenant_id="tenant-abc",
            client_id="client-xyz",
        )
        with pytest.raises(ValueError, match="client_secret"):
            initialize_guardrail(params, {"guardrail_name": "a365"})

    def test_env_var_fallbacks(self, monkeypatch):
        monkeypatch.delenv("AGENT365_RESOURCE_APP_ID", raising=False)
        monkeypatch.setenv("AGENT365_TENANT_ID", "env-tenant")
        monkeypatch.setenv("AGENT365_CLIENT_ID", "env-client")
        monkeypatch.setenv("AGENT365_CLIENT_SECRET", "env-secret")
        monkeypatch.setenv("AGENT365_API_BASE", "https://env.example.test")
        params: Final = LitellmParams(guardrail="agent_365", mode="pre_mcp_call")
        guardrail: Final = initialize_guardrail(params, {"guardrail_name": "a365-env"})
        assert guardrail.tenant_id == "env-tenant"
        assert guardrail.client_id == "env-client"
        assert guardrail.client_secret == "env-secret"
        assert guardrail.api_base == "https://env.example.test"
        assert guardrail.resource_app_id == AGENT_365_PROD_RESOURCE_APP_ID
        assert guardrail.unreachable_fallback == "fail_closed"

    def test_explicit_params_win(self, monkeypatch):
        monkeypatch.setenv("AGENT365_TENANT_ID", "env-tenant")
        params: Final = LitellmParams(
            guardrail="agent_365",
            mode="pre_mcp_call",
            tenant_id="param-tenant",
            client_id="client-xyz",
            client_secret="param-secret",
            agent_id="agent-007",
            unreachable_fallback="fail_open",
            timeout=5,
        )
        guardrail: Final = initialize_guardrail(params, {"guardrail_name": "a365-params"})
        assert guardrail.tenant_id == "param-tenant"
        assert guardrail.client_secret == "param-secret"
        assert guardrail.agent_id == "agent-007"
        assert guardrail.unreachable_fallback == "fail_open"
        assert guardrail.request_timeout == 5.0

    def test_wrong_mode_rejected(self):
        params: Final = LitellmParams(
            guardrail="agent_365",
            mode="post_call",
            tenant_id="tenant-abc",
            client_id="client-xyz",
            api_key="secret-123",
        )
        with pytest.raises(Exception, match="post_call"):
            initialize_guardrail(params, {"guardrail_name": "a365-badmode"})


def _guardrail_info(data: dict) -> dict:
    entries: Final = data["metadata"]["standard_logging_guardrail_information"]
    return entries[-1]


class TestAllowFlow:
    @pytest.mark.asyncio
    async def test_allowed_call_passes_through(self):
        handler: Final = FakeHandler([_token_response(), _allow_response()])
        guardrail: Final = _make_guardrail(handler)
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data)
        assert result is data
        info: Final = _guardrail_info(data)
        assert info["guardrail_status"] == "success"
        assert info["guardrail_provider"] == "agent_365"
        assert info["guardrail_response"]["verdict"] == "Allow"
        assert info["guardrail_response"]["defender_status"] == "Evaluated"
        assert info["guardrail_response"]["correlation_id"] == "corr-1"
        assert info["guardrail_response"]["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_obo_exchange_form(self):
        handler: Final = FakeHandler([_token_response(), _allow_response()])
        guardrail: Final = _make_guardrail(handler)
        await _run(guardrail, _mcp_data())
        token_call: Final = handler.calls[0]
        assert token_call.url == TOKEN_URL
        assert token_call.data["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
        assert token_call.data["requested_token_use"] == "on_behalf_of"
        assert token_call.data["assertion"] == FAKE_ASSERTION
        assert token_call.data["client_id"] == "client-xyz"
        assert token_call.data["client_secret"] == "secret-123"
        assert token_call.data["scope"] == f"{AGENT_365_PROD_RESOURCE_APP_ID}/ThreatProtection.Evaluate.All"

    @pytest.mark.asyncio
    async def test_evaluate_payload(self):
        handler: Final = FakeHandler([_token_response(), _allow_response()])
        guardrail: Final = _make_guardrail(handler, agent_id="agent-007")
        await _run(guardrail, _mcp_data())
        evaluate_call: Final = handler.calls[1]
        assert evaluate_call.url == EVALUATE_URL
        assert evaluate_call.headers["Authorization"] == "Bearer obo-access-token"
        assert evaluate_call.json["tool"] == {"name": "send_email"}
        assert evaluate_call.json["serverName"] == "outlook_mcp"
        assert evaluate_call.json["arguments"] == {"to": "user@example.com", "body": "hello"}
        assert evaluate_call.json["conversationId"] == "sess-123"
        assert evaluate_call.json["agentId"] == "agent-007"

    @pytest.mark.asyncio
    async def test_agent_id_falls_back_to_key_alias(self):
        handler: Final = FakeHandler([_token_response(), _allow_response()])
        guardrail: Final = _make_guardrail(handler)
        await _run(guardrail, _mcp_data())
        assert handler.calls[1].json["agentId"] == "my-agent-key"

    @pytest.mark.asyncio
    async def test_non_mcp_call_type_skipped(self):
        handler: Final = FakeHandler([])
        guardrail: Final = _make_guardrail(handler)
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data, call_type="completion")
        assert result is data
        assert handler.calls == []


class TestConversationId:
    @pytest.mark.asyncio
    async def test_session_id_header_case_insensitive(self):
        handler: Final = FakeHandler([_token_response(), _allow_response()])
        guardrail: Final = _make_guardrail(handler)
        data: Final = _mcp_data(metadata={"headers": {"Mcp-Session-Id": "sess-CASED"}})
        await _run(guardrail, data)
        assert handler.calls[1].json["conversationId"] == "sess-CASED"

    @pytest.mark.asyncio
    async def test_falls_back_to_logging_obj_session_id(self):
        handler: Final = FakeHandler([_token_response(), _allow_response()])
        guardrail: Final = _make_guardrail(handler)
        logging_obj: Final = SimpleNamespace(
            model_call_details={"mcp_tool_call_metadata": {"mcp_session_id": "sess-from-logging"}},
            litellm_call_id="call-id-1",
        )
        data: Final = _mcp_data(metadata={"headers": {}}, litellm_logging_obj=logging_obj)
        await _run(guardrail, data)
        assert handler.calls[1].json["conversationId"] == "sess-from-logging"

    @pytest.mark.asyncio
    async def test_falls_back_to_litellm_call_id(self):
        handler: Final = FakeHandler([_token_response(), _allow_response()])
        guardrail: Final = _make_guardrail(handler)
        data: Final = _mcp_data(metadata={"headers": {}}, litellm_call_id="call-id-2")
        await _run(guardrail, data)
        assert handler.calls[1].json["conversationId"] == "call-id-2"


class TestBlockFlow:
    @pytest.mark.asyncio
    async def test_blocked_call_raises_400(self):
        handler: Final = FakeHandler([_token_response(), _block_response(message="Injection detected")])
        guardrail: Final = _make_guardrail(handler)
        data: Final = _mcp_data()
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, data)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "Blocked by Microsoft Defender"
        assert exc_info.value.detail["message"] == "Injection detected"
        assert exc_info.value.detail["tool"] == "send_email"
        assert exc_info.value.detail["correlation_id"] == "corr-2"
        info: Final = _guardrail_info(data)
        assert info["guardrail_status"] == "guardrail_intervened"
        assert info["guardrail_response"]["verdict"] == "Block"

    @pytest.mark.asyncio
    async def test_blocked_even_with_fail_open(self):
        handler: Final = FakeHandler([_token_response(), _block_response()])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_http_400_always_blocks_even_fail_open(self):
        handler: Final = FakeHandler([_token_response(), _response(400, text="Bad request: serverName missing")])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 400
        assert "rejected" in exc_info.value.detail["error"]


class TestUnreachableFallback:
    @pytest.mark.asyncio
    async def test_evaluate_litellm_timeout_fail_closed(self):
        handler: Final = FakeHandler(
            [
                _token_response(),
                LitellmTimeout(message="Connection timed out", model="default-model-name", llm_provider="httpx"),
            ]
        )
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_evaluate_timeout_fail_closed(self):
        handler: Final = FakeHandler([_token_response(), httpx.ReadTimeout("timed out")])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503
        assert "fail_closed" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_evaluate_timeout_fail_open(self):
        handler: Final = FakeHandler([_token_response(), httpx.ReadTimeout("timed out")])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data)
        assert result is data
        info: Final = _guardrail_info(data)
        assert info["guardrail_status"] == "guardrail_failed_to_respond"
        assert info["guardrail_response"]["verdict"] == "Unscanned"

    @pytest.mark.asyncio
    async def test_evaluate_5xx_fail_closed(self):
        handler: Final = FakeHandler([_token_response(), _response(502, text="bad gateway")])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503
        assert "502" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_missing_bearer_token_fail_closed(self):
        handler: Final = FakeHandler([])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data(incoming_bearer_token=None))
        assert exc_info.value.status_code == 401
        assert handler.calls == []

    @pytest.mark.asyncio
    async def test_non_jwt_bearer_token_fail_closed(self):
        handler: Final = FakeHandler([])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data(incoming_bearer_token="sk-litellm-virtual-key"))
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_bearer_token_blocks_even_fail_open(self):
        handler: Final = FakeHandler([])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data(incoming_bearer_token=None)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, data)
        assert exc_info.value.status_code == 401
        assert handler.calls == []
        info: Final = _guardrail_info(data)
        assert info["guardrail_status"] == "guardrail_intervened"
        assert info["guardrail_response"]["verdict"] == "Rejected"

    @pytest.mark.asyncio
    async def test_obo_rejected_blocks_even_fail_open(self):
        handler: Final = FakeHandler(
            [_response(400, {"error": "invalid_grant", "error_description": "AADSTS50013: bad assertion"})]
        )
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 401
        assert "invalid_grant" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_evaluate_4xx_blocks_even_fail_open(self):
        handler: Final = FakeHandler([_token_response(), _response(403, text="obo token lacks the scope")])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, data)
        assert exc_info.value.status_code == 400
        assert "403" in exc_info.value.detail["message"]
        assert "lacks the scope" not in exc_info.value.detail["message"]
        info: Final = _guardrail_info(data)
        assert info["guardrail_status"] == "guardrail_intervened"
        assert info["guardrail_response"]["reason"] == "HTTP 403: obo token lacks the scope"

    @pytest.mark.asyncio
    async def test_obo_rejected_fail_closed(self):
        handler: Final = FakeHandler(
            [_response(400, {"error": "invalid_grant", "error_description": "AADSTS50013: bad assertion"})]
        )
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 401
        assert "invalid_grant" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_obo_endpoint_5xx_fail_open(self):
        handler: Final = FakeHandler([_response(503, text="entra down")])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data)
        assert result is data
        info: Final = _guardrail_info(data)
        assert info["guardrail_status"] == "guardrail_failed_to_respond"
        assert info["guardrail_response"]["verdict"] == "Unscanned"


class TestOboTokenCache:
    @pytest.mark.asyncio
    async def test_same_assertion_reuses_token(self):
        handler: Final = FakeHandler([_token_response(), _allow_response(), _allow_response()])
        guardrail: Final = _make_guardrail(handler)
        await _run(guardrail, _mcp_data())
        await _run(guardrail, _mcp_data())
        token_calls: Final = [c for c in handler.calls if c.url == TOKEN_URL]
        assert len(token_calls) == 1

    @pytest.mark.asyncio
    async def test_different_assertions_get_distinct_tokens(self):
        other_assertion: Final = "eyJhbGciOi.eyJvdGhlciI.b3RoZXJzaWc"
        handler: Final = FakeHandler(
            [
                _token_response(access_token="token-a"),
                _allow_response(),
                _token_response(access_token="token-b"),
                _allow_response(),
            ]
        )
        guardrail: Final = _make_guardrail(handler)
        await _run(guardrail, _mcp_data())
        await _run(guardrail, _mcp_data(incoming_bearer_token=other_assertion))
        token_calls: Final = [c for c in handler.calls if c.url == TOKEN_URL]
        assert len(token_calls) == 2
        assert handler.calls[3].headers["Authorization"] == "Bearer token-b"

    @pytest.mark.asyncio
    async def test_expired_token_refreshed(self):
        handler: Final = FakeHandler(
            [
                _token_response(access_token="short-lived", expires_in=1),
                _allow_response(),
                _token_response(access_token="fresh"),
                _allow_response(),
            ]
        )
        guardrail: Final = _make_guardrail(handler)
        await _run(guardrail, _mcp_data())
        await _run(guardrail, _mcp_data())
        token_calls: Final = [c for c in handler.calls if c.url == TOKEN_URL]
        assert len(token_calls) == 2
        assert handler.calls[3].headers["Authorization"] == "Bearer fresh"


class TestEarlyPhasePassthrough:
    @pytest.mark.asyncio
    async def test_rest_body_shape_without_mcp_fields_skipped(self):
        handler: Final = FakeHandler([])
        guardrail: Final = _make_guardrail(handler)
        data: Final = {
            "server_id": "266024044f9612bf481c78f6cfef1ff0",
            "name": "deepwiki-read_wiki_structure",
            "arguments": {"repoName": "BerriAI/litellm"},
            "metadata": {"headers": {"mcp-session-id": "sess-123"}},
        }
        result: Final = await _run(guardrail, data)
        assert result is data
        assert handler.calls == []
        assert "standard_logging_guardrail_information" not in data["metadata"]


class TestRegistryDiscovery:
    def test_auto_discovery_finds_agent_365(self):
        from litellm.proxy.guardrails.guardrail_registry import (
            get_guardrail_class_from_hooks,
            get_guardrail_initializer_from_hooks,
        )

        assert "agent_365" in get_guardrail_initializer_from_hooks()
        assert get_guardrail_class_from_hooks()["agent_365"] is Agent365Guardrail


class TestMalformedResponses:
    @pytest.mark.asyncio
    async def test_obo_html_body_fail_open(self):
        handler: Final = FakeHandler([_response(200, text="<html>blocked by egress proxy</html>")])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data)
        assert result is data
        assert _guardrail_info(data)["guardrail_response"]["verdict"] == "Unscanned"

    @pytest.mark.asyncio
    async def test_obo_html_body_fail_closed(self):
        handler: Final = FakeHandler([_response(200, text="<html>outage</html>")])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503
        assert "non-JSON" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_obo_non_object_json_fail_closed(self):
        handler: Final = FakeHandler([_response(200, ["not", "a", "dict"])])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_evaluate_html_body_fail_open(self):
        handler: Final = FakeHandler([_token_response(), _response(200, text="<html>waf page</html>")])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data)
        assert result is data
        assert _guardrail_info(data)["guardrail_response"]["verdict"] == "Unscanned"

    @pytest.mark.asyncio
    async def test_evaluate_html_body_fail_closed(self):
        handler: Final = FakeHandler([_token_response(), _response(200, text="<html>waf page</html>")])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_evaluate_non_object_json_fail_closed(self):
        handler: Final = FakeHandler([_token_response(), _response(200, "allowed")])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_bad_expires_in_still_allows(self):
        handler: Final = FakeHandler([_response(200, {"access_token": "tok-1", "expires_in": "soon"}), _allow_response()])
        guardrail: Final = _make_guardrail(handler)
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data)
        assert result is data

    @pytest.mark.asyncio
    async def test_obo_litellm_timeout_fail_open(self):
        handler: Final = FakeHandler(
            [LitellmTimeout(message="Connection timed out", model="default-model-name", llm_provider="httpx")]
        )
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data)
        assert result is data
        assert _guardrail_info(data)["guardrail_status"] == "guardrail_failed_to_respond"


class TestDeltaHardening:
    @pytest.mark.asyncio
    async def test_non_string_access_token_fail_closed(self):
        handler: Final = FakeHandler([_response(200, {"access_token": None, "expires_in": 3599})])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503
        assert "access_token" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_numeric_string_expires_in_honored(self):
        handler: Final = FakeHandler([_response(200, {"access_token": "tok-9", "expires_in": "120"}), _allow_response()])
        guardrail: Final = _make_guardrail(handler)
        await _run(guardrail, _mcp_data())
        entries: Final = list(guardrail._obo_token_cache.values())
        assert len(entries) == 1
        assert entries[0][1] - time.time() < 200

    @pytest.mark.asyncio
    async def test_evaluate_400_records_intervention(self):
        handler: Final = FakeHandler([_token_response(), _response(400, text="bad request shape")])
        guardrail: Final = _make_guardrail(handler)
        data: Final = _mcp_data()
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, data)
        assert exc_info.value.status_code == 400
        info: Final = _guardrail_info(data)
        assert info["guardrail_status"] == "guardrail_intervened"
        assert info["guardrail_response"]["verdict"] == "Rejected"


class TestVeriaHardening:
    @pytest.mark.asyncio
    async def test_evaluate_401_evicts_cached_obo_token(self):
        handler: Final = FakeHandler(
            [
                _token_response(),
                _response(401, text="token expired"),
                _token_response(access_token="tok-2"),
                _allow_response(),
            ]
        )
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException):
            await _run(guardrail, _mcp_data())
        result: Final = await _run(guardrail, _mcp_data())
        assert result is not None
        token_calls: Final = [c for c in handler.calls if c.url == TOKEN_URL]
        assert len(token_calls) == 2

    @pytest.mark.asyncio
    async def test_evaluate_429_blocks_even_fail_open_as_throttled(self):
        handler: Final = FakeHandler([_token_response(), _response(429, text="slow down")])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, data)
        assert exc_info.value.status_code == 503
        assert "429" in exc_info.value.detail["message"]
        info: Final = _guardrail_info(data)
        assert info["guardrail_status"] == "guardrail_failed_to_respond"
        assert info["guardrail_response"]["verdict"] == "Throttled"

    @pytest.mark.asyncio
    async def test_evaluate_500_is_unavailable(self):
        handler: Final = FakeHandler([_token_response(), _response(500, text="oops")])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503
        assert "500" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_token_endpoint_429_blocks_even_fail_open_as_throttled(self):
        handler: Final = FakeHandler(
            [_response(429, {"error": "temporarily_throttled", "error_description": "AADSTS90056"})]
        )
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, data)
        assert exc_info.value.status_code == 503
        assert "429" in exc_info.value.detail["message"]
        info: Final = _guardrail_info(data)
        assert info["guardrail_status"] == "guardrail_failed_to_respond"
        assert info["guardrail_response"]["verdict"] == "Throttled"

    @pytest.mark.asyncio
    async def test_token_endpoint_408_non_json_blocks_as_throttled(self):
        handler: Final = FakeHandler([_response(408, text="Request Timeout")])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, data)
        assert exc_info.value.status_code == 503
        assert _guardrail_info(data)["guardrail_response"]["verdict"] == "Throttled"

    @pytest.mark.asyncio
    async def test_token_endpoint_4xx_html_stays_infra_fail_open(self):
        handler: Final = FakeHandler([_response(403, text="<html>waf block page</html>")])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data)
        assert result is data
        assert _guardrail_info(data)["guardrail_response"]["verdict"] == "Unscanned"

    @pytest.mark.asyncio
    async def test_entra_200_missing_access_token_is_malformed(self):
        handler: Final = FakeHandler([_response(200, {"token_type": "Bearer"})])
        guardrail: Final = _make_guardrail(handler)
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, _mcp_data())
        assert exc_info.value.status_code == 503
        assert "access_token" in exc_info.value.detail["message"]

    @pytest.mark.asyncio
    async def test_evaluate_5xx_fail_open_allows_unscanned_once(self):
        handler: Final = FakeHandler([_token_response(), _response(502, text='{"error": "bad gateway"}')])
        guardrail: Final = _make_guardrail(handler, unreachable_fallback="fail_open")
        data: Final = _mcp_data()
        result: Final = await _run(guardrail, data)
        assert result is data
        records: Final = data["metadata"]["standard_logging_guardrail_information"]
        assert len(records) == 1
        assert records[0]["guardrail_response"]["verdict"] == "Unscanned"
        assert records[0]["guardrail_status"] == "guardrail_failed_to_respond"
