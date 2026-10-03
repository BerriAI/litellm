import base64
import json

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import MCPGuardrailTranslationHandler
from litellm.proxy.guardrails.guardrail_hooks.ismalicious import initialize_guardrail
from litellm.proxy.guardrails.guardrail_hooks.ismalicious.ismalicious import IsMaliciousGuardrail
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams, Mode

URL = "https://example.com/path?q=a,b&x=1#part"
KEY = base64.b64encode(b"test-key:test-secret").decode()


def service_response(request, verdict="allow", truncated=False):
    if request.url.path == "/gate/url":
        body = {
            "url": request.url.params["u"],
            "entity": "example.com",
            "verdict": verdict,
            "sources": 0,
            "latency_ms": 1,
        }
    else:
        body = {
            "verdict": verdict,
            "injection": {"score": 0.0, "families": [], "spans": []},
            "links": [{"url": URL, "entity": "example.com", "verdict": "unknown", "sources": 0}],
            "links_truncated": truncated,
            "mode": "fast",
            "latency_ms": 1,
        }
    return httpx.Response(200, json=body)


def guardrail(handler=service_response):
    return IsMaliciousGuardrail(
        api_key=KEY,
        guardrail_name="ismalicious",
        event_hook=[GuardrailEventHooks.pre_mcp_call, GuardrailEventHooks.post_mcp_call],
        default_on=True,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_native_pre_handler_preserves_original_url_and_arguments():
    requests = []

    def handler(request):
        requests.append(request)
        return service_response(request)

    data = {
        "mcp_tool_name": "fetch",
        "mcp_arguments": {"url": URL, "other": "unchanged"},
        "headers": {"Authorization": "Incoming private token"},
    }
    result = await MCPGuardrailTranslationHandler().process_input_messages(data, guardrail(handler))
    assert result is data
    assert requests[0].url.params["u"] == URL
    assert json.loads(json.loads(requests[1].content)["content"]) == [URL, "unchanged"]
    assert data["mcp_arguments"]["url"] == URL
    assert all(request.headers["X-API-KEY"] == KEY for request in requests)
    assert all("Authorization" not in request.headers for request in requests)
    assert all(request.extensions["timeout"]["read"] == 15 for request in requests)


@pytest.mark.asyncio
async def test_native_post_handler_preserves_text_and_structured_content_identity():
    payload = CallToolResult(
        content=[TextContent(type="text", text="Allowed å🚀 content")],
        structuredContent={"context": "Additional untrusted context", "count": 3},
    )
    requests = []

    def handler(request):
        requests.append(request)
        return service_response(request)

    original_content = payload.content
    original_structured = payload.structured_content
    result = await MCPGuardrailTranslationHandler().process_output_response(payload, guardrail(handler))
    assert result is payload
    assert result.content is original_content
    assert result.structured_content is original_structured
    inspected = json.loads(json.loads(requests[0].content)["content"])
    assert "Allowed å🚀 content" in inspected
    assert "Additional untrusted context" in inspected
    assert "context" in inspected and "count" in inspected and "3" in inspected


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["url", "scan"])
@pytest.mark.parametrize("verdict", ["warn", "block", "unknown"])
async def test_native_pre_handler_does_not_return_a_blocked_request(phase, verdict):
    def handler(request):
        return service_response(request, verdict if request.url.path.endswith(phase) else "allow")

    with pytest.raises(GuardrailRaisedException) as exc:
        await MCPGuardrailTranslationHandler().process_input_messages(
            {"mcp_tool_name": "fetch", "mcp_arguments": {"url": URL}}, guardrail(handler)
        )
    assert URL not in str(exc.value) and KEY not in str(exc.value)
    assert exc.value.blocked_content is (verdict in {"warn", "block"})


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["warn", "block", "unknown"])
async def test_native_post_handler_does_not_return_blocked_text(verdict):
    payload = CallToolResult(content=[TextContent(type="text", text="Private untrusted result")])

    def handler(request):
        return service_response(request, verdict)

    with pytest.raises(GuardrailRaisedException) as exc:
        await MCPGuardrailTranslationHandler().process_output_response(payload, guardrail(handler))
    assert "Private untrusted result" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["429", "302", "500", "bad-json", "missing", "timeout", "truncated"])
async def test_native_post_handler_fails_closed_without_retry(failure):
    calls = []

    def handler(request):
        calls.append(request)
        if failure.isdigit():
            return httpx.Response(int(failure), headers={"Location": "https://attacker.example"})
        if failure == "bad-json":
            return httpx.Response(200, content=b"not-json")
        if failure == "missing":
            return httpx.Response(200, json={"verdict": "allow"})
        if failure == "timeout":
            raise httpx.ReadTimeout("Sensitive detail", request=request)
        return service_response(request, truncated=True)

    with pytest.raises(GuardrailRaisedException) as exc:
        await MCPGuardrailTranslationHandler().process_output_response(
            CallToolResult(content=[TextContent(type="text", text="Untrusted result")]), guardrail(handler)
        )
    assert len(calls) == 1
    assert not exc.value.blocked_content
    assert "Sensitive detail" not in str(exc.value)


@pytest.mark.asyncio
async def test_serialized_utf8_request_body_limit_refuses_without_network():
    def handler(request):
        pytest.fail("An oversized request must not be sent")

    with pytest.raises(GuardrailRaisedException):
        await guardrail(handler).apply_guardrail(
            inputs={"texts": ["🚀" * (1024 * 1024 // 4)]}, request_data={}, input_type="response"
        )


@pytest.mark.asyncio
async def test_native_handler_scans_structured_only_untrusted_content():
    def handler(request):
        assert "Hidden instruction" in json.loads(request.content)["content"]
        return service_response(request, "block")

    with pytest.raises(GuardrailRaisedException):
        await MCPGuardrailTranslationHandler().process_output_response(
            CallToolResult(content=[], structuredContent={"instruction": "Hidden instruction"}), guardrail(handler)
        )


@pytest.mark.asyncio
async def test_return_inputs_are_identity_without_rewriting():
    inputs = {"texts": ["Allowed text"]}
    assert await guardrail().apply_guardrail(inputs, {}, "response") is inputs


@pytest.mark.parametrize("mode", [GuardrailEventHooks.pre_call, GuardrailEventHooks.during_mcp_call, None, []])
def test_unsupported_modes_fail_at_startup(mode):
    with pytest.raises(ValueError, match="requires pre_mcp_call"):
        IsMaliciousGuardrail(api_key=KEY, event_hook=mode)


@pytest.mark.parametrize(
    "api_key", ["not-base64", "dXNlcg==", "OnNlY3JldA==", base64.b64encode(b"\xff:secret").decode()]
)
def test_invalid_credentials_are_not_echoed(api_key):
    with pytest.raises(ValueError, match="requires a Base64") as exc:
        IsMaliciousGuardrail(api_key=api_key, event_hook=GuardrailEventHooks.pre_mcp_call)
    assert api_key not in str(exc.value)


def test_credentials_never_redirect_to_a_custom_endpoint():
    with pytest.raises(ValueError, match="HTTPS API"):
        IsMaliciousGuardrail(
            api_key=KEY, api_base="https://attacker.example", event_hook=GuardrailEventHooks.pre_mcp_call
        )


def test_missing_explicit_credentials_fail_at_startup():
    with pytest.raises(ValueError, match="requires a Base64"):
        IsMaliciousGuardrail(event_hook=GuardrailEventHooks.pre_mcp_call)


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["allow", "block"])
async def test_native_decision_logging_excludes_content_and_credentials(verdict):
    def handler(request):
        return service_response(request, verdict)

    data = {"request_id": "test"}
    inputs = {"texts": ["Private untrusted result"]}
    if verdict == "block":
        with pytest.raises(GuardrailRaisedException):
            await guardrail(handler).apply_guardrail(inputs=inputs, request_data=data, input_type="response")
    else:
        await guardrail(handler).apply_guardrail(inputs=inputs, request_data=data, input_type="response")
    recorded = json.dumps(data)
    assert "standard_logging_guardrail_information" in recorded
    assert "Private untrusted result" not in recorded
    assert URL not in recorded and KEY not in recorded


def test_mcp_subcalls_without_guardrail_metadata_use_explicit_default_on():
    gate = guardrail()
    assert gate.should_run_guardrail(data={}, event_type=GuardrailEventHooks.pre_mcp_call)
    assert gate.should_run_guardrail(data={}, event_type=GuardrailEventHooks.post_mcp_call)


@pytest.mark.asyncio
async def test_images_passed_to_the_provider_are_not_silently_allowed():
    with pytest.raises(GuardrailRaisedException):
        await guardrail().apply_guardrail({"texts": ["caption"], "images": ["image"]}, {}, "response")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["backward-span", "latency-overflow"])
async def test_malformed_response_semantics_fail_closed(invalid):
    def handler(request):
        body = service_response(request).json()
        if invalid == "backward-span":
            body["injection"]["spans"] = [{"start": 9, "end": 2, "family": "instruction"}]
        else:
            body["latency_ms"] = 2**63
        return httpx.Response(200, json=body)

    with pytest.raises(GuardrailRaisedException) as exc:
        await guardrail(handler).apply_guardrail({"texts": ["Untrusted"]}, {}, "response")
    assert not exc.value.blocked_content


@pytest.mark.parametrize("mode", ["pre_mcp_call", ["pre_mcp_call", "post_mcp_call"]])
@pytest.mark.parametrize("default_on", [False, True])
def test_native_config_roundtrip_registers_requested_policy(mode, default_on):
    config_model = IsMaliciousGuardrail.get_config_model()
    config = config_model.model_validate({"api_key": KEY})
    params = LitellmParams(
        guardrail="ismalicious", mode=mode, default_on=default_on, **config.model_dump(exclude_none=True)
    )
    callback = initialize_guardrail(params, {"guardrail_name": "selected-gate", "litellm_params": params})
    assert any(item is callback for item in litellm.callbacks)
    assert callback.should_run_guardrail(data={}, event_type=GuardrailEventHooks.pre_mcp_call) is default_on
    assert callback.should_run_guardrail(data={}, event_type=GuardrailEventHooks.post_mcp_call) is (
        default_on and isinstance(mode, list)
    )


def test_native_initializer_refuses_conditional_mcp_modes():
    params = LitellmParams(guardrail="ismalicious", api_key=KEY, mode=Mode(tags={}, default="pre_mcp_call"))
    with pytest.raises(ValueError, match="explicit MCP modes"):
        initialize_guardrail(params, {"guardrail_name": "selected-gate", "litellm_params": params})


@pytest.mark.asyncio
async def test_valid_ordered_span_allows_original_content():
    def handler(request):
        body = service_response(request).json()
        body["injection"]["spans"] = [{"start": 0, "end": 1, "family": "fixture"}]
        return httpx.Response(200, json=body)

    payload = CallToolResult(content=[TextContent(type="text", text="Allowed result")])
    assert await MCPGuardrailTranslationHandler().process_output_response(payload, guardrail(handler)) is payload


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["https://[invalid", "https://user:password@example.com/", "https://example.com/\n"])
async def test_native_pre_handler_refuses_malformed_urls_before_network(url):
    def handler(request):
        pytest.fail("A malformed URL must not be sent")

    with pytest.raises(GuardrailRaisedException) as exc:
        await MCPGuardrailTranslationHandler().process_input_messages(
            {"mcp_tool_name": "fetch", "mcp_arguments": {"url": url}}, guardrail(handler)
        )
    assert url not in str(exc.value)


@pytest.mark.asyncio
async def test_native_pre_handler_refuses_a_service_response_for_a_different_url():
    def handler(request):
        body = service_response(request).json()
        body["url"] = "https://example.com/path"
        return httpx.Response(200, json=body)

    with pytest.raises(GuardrailRaisedException):
        await MCPGuardrailTranslationHandler().process_input_messages(
            {"mcp_tool_name": "fetch", "mcp_arguments": {"url": URL}}, guardrail(handler)
        )


@pytest.mark.asyncio
async def test_non_utf8_content_is_refused_before_network():
    def handler(request):
        pytest.fail("Invalid UTF-8 must not be sent")

    with pytest.raises(GuardrailRaisedException):
        await guardrail(handler).apply_guardrail({"texts": ["\ud800"]}, {}, "response")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.RemoteProtocolError])
async def test_native_transport_connection_errors_are_not_retried(error):
    calls = []

    def handler(request):
        calls.append(request)
        raise error("Private upstream details", request=request)

    with pytest.raises(GuardrailRaisedException) as exc:
        await MCPGuardrailTranslationHandler().process_output_response(
            CallToolResult(content=[TextContent(type="text", text="Untrusted result")]), guardrail(handler)
        )
    assert len(calls) == 1
    assert "Private upstream details" not in str(exc.value)


class ClosureAwareTransport(httpx.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)
        self.closed = False

    async def handle_async_request(self, request):
        if self.closed:
            raise httpx.ConnectError("Transport already closed", request=request)
        return await super().handle_async_request(request)

    async def aclose(self):
        self.closed = True
        await super().aclose()


@pytest.mark.asyncio
async def test_native_pool_survives_calls_without_sharing_credentials():
    requests = []

    def handler(request):
        requests.append(request)
        return service_response(request)

    transport = ClosureAwareTransport(handler)
    other_key = base64.b64encode(b"other-key:other-secret").decode()
    for key in (KEY, other_key):
        callback = IsMaliciousGuardrail(api_key=key, event_hook=GuardrailEventHooks.post_mcp_call, transport=transport)
        payload = CallToolResult(content=[TextContent(type="text", text="Allowed result")])
        assert await MCPGuardrailTranslationHandler().process_output_response(payload, callback) is payload
    assert [request.headers["X-API-KEY"] for request in requests] == [KEY, other_key]
    assert not transport.closed
