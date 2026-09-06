from tests.test_litellm.proxy.guardrails.guardrail_hooks._cisco_ai_defense_test_utils import (
    Any,
    AsyncMock,
    CiscoAIDefenseGuardrail,
    Dict,
    DualCache,
    HTTPException,
    MCP_URL,
    Response,
    SimpleNamespace,
    UserAPIKeyAuth,
    _make_guardrail,
    _make_model_response_with_content,
    _mcp_request,
    _mcp_response,
    _mcp_result_text,
    _mock_inspect_response,
    _patch_inspection_post,
    _redact_response,
    _safe_response,
    _violation_response,
    datetime,
    init_guardrails_v2,
    json,
    litellm,
    pytest,
)


def test_cisco_ai_defense_config_via_init_v2_mcp(monkeypatch):
    monkeypatch.setenv("CISCO_AI_DEFENSE_API_KEY", "test-key")
    litellm.guardrail_name_config_map = {}

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "cisco-mcp",
                "litellm_params": {
                    "guardrail": "cisco_ai_defense",
                    "mode": "pre_mcp_call",
                    "default_on": True,
                    "optional_params": {"inspection_type": "mcp"},
                },
            }
        ],
        config_file_path="",
    )


class TestCiscoAIDefenseMCPMode:
    @pytest.mark.asyncio
    async def test_mcp_mode_inspects_mcp_request(self):
        g = _make_guardrail(inspection_type="mcp", event_hook="pre_mcp_call")
        data = _mcp_request(
            name="send_email", args={"to": "x@y.com"}, litellm_call_id="call-1"
        )
        post_mock = AsyncMock(return_value=_safe_response(url=MCP_URL))
        with _patch_inspection_post(g, post_mock):
            result = await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="mcp_call",
            )
        assert result == data
        assert post_mock.call_args.kwargs["url"] == MCP_URL
        assert post_mock.call_args.kwargs["follow_redirects"] is False
        sent_payload = post_mock.call_args.kwargs["json"]
        assert sent_payload["jsonrpc"] == "2.0"
        assert sent_payload["method"] == "tools/call"
        assert sent_payload["params"]["name"] == "send_email"
        assert sent_payload["params"]["arguments"] == {"to": "x@y.com"}
        assert "request" not in sent_payload
        assert "metadata" not in sent_payload
        assert "config" not in sent_payload

    @pytest.mark.asyncio
    async def test_mcp_mode_blocks_violation(self):
        g = _make_guardrail(inspection_type="mcp", event_hook="pre_mcp_call")
        data = _mcp_request(name="leak_secrets", args={"target": "evil"})
        with _patch_inspection_post(
            g, AsyncMock(return_value=_violation_response(url=MCP_URL))
        ):
            with pytest.raises(HTTPException) as exc:
                await g.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(),
                    cache=DualCache(),
                    data=data,
                    call_type="mcp_call",
                )
        assert exc.value.detail["surface"] == "mcp"

    @pytest.mark.asyncio
    async def test_mcp_mode_skips_chat_traffic(self):
        g = _make_guardrail(inspection_type="mcp", event_hook="pre_mcp_call")
        data = {"messages": [{"role": "user", "content": "hello"}]}
        post_mock = AsyncMock()
        with _patch_inspection_post(g, post_mock):
            result = await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="completion",
            )
        assert result == data
        post_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_mode_inspects_jsonrpc_envelope(self):
        g = _make_guardrail(inspection_type="mcp", event_hook="pre_mcp_call")
        data = _mcp_request(name="do_thing", args={"x": 1}, jsonrpc=True, id="abc")
        post_mock = AsyncMock(return_value=_safe_response(url=MCP_URL))
        with _patch_inspection_post(g, post_mock):
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="mcp_call",
            )
        sent_payload = post_mock.call_args.kwargs["json"]
        assert sent_payload["jsonrpc"] == "2.0"
        assert sent_payload["id"] == "abc"
        assert sent_payload["params"]["name"] == "do_thing"
        assert sent_payload["params"]["arguments"] == {"x": 1}

    @pytest.mark.parametrize(
        "verdict_extra",
        [
            {"sanitized_payload": {"params": {"arguments": {"note": "ssn [REDACTED]"}}}},
            {"sanitized_text": "ssn [REDACTED]"},
        ],
        ids=["structured_arguments", "sanitized_text_fallback"],
    )
    @pytest.mark.asyncio
    async def test_mcp_input_redaction_reaches_tool_call(self, verdict_extra):
        from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
        from litellm.proxy.utils import ProxyLogging

        original_args = {"note": "ssn 123-45-6789"}
        sanitized_args = {"note": "ssn [REDACTED]"}

        g = _make_guardrail(
            inspection_type="mcp",
            event_hook="pre_mcp_call",
            on_flagged_action="monitor",
        )
        data = _mcp_request(name="send_email", args=dict(original_args))
        cisco_resp = _mock_inspect_response(
            {
                "is_safe": False,
                "classifications": ["PRIVACY_VIOLATION"],
                "severity": "HIGH",
                "rules": [{"rule_name": "PII"}],
                "action": "redact",
                **verdict_extra,
            },
            url=MCP_URL,
        )

        with _patch_inspection_post(g, AsyncMock(return_value=cisco_resp)):
            result = await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="mcp_call",
            )

        forwarded = ProxyLogging(
            user_api_key_cache=UserApiKeyCache()
        )._convert_mcp_hook_response_to_kwargs(
            response_data=result, original_kwargs={"arguments": dict(original_args)}
        )
        assert forwarded["arguments"] == sanitized_args, (
            "Sanitized MCP arguments did not reach the tool call. The proxy "
            "bridge forwards redactions only via ``modified_arguments``, so a "
            "redact verdict proceeded while the original unsanitized arguments "
            f"still hit the MCP server. Got: {forwarded['arguments']!r}"
        )

    @pytest.mark.asyncio
    async def test_mcp_response_hook_inspects_tool_output(self):
        g = _make_guardrail(
            inspection_type="mcp", event_hook=["pre_mcp_call", "during_mcp_call"]
        )

        response_obj = _mcp_response(
            SimpleNamespace(
                content=[{"type": "text", "text": "Here is the secret API key abc123"}]
            )
        )

        post_mock = AsyncMock(return_value=_safe_response(url=MCP_URL))
        kwargs = {
            "name": "lookup_secret",
            "arguments": {"key": "production"},
            "mcp_server_name": "vault",
            "litellm_call_id": "call-42",
        }
        with _patch_inspection_post(g, post_mock):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        assert result is None
        assert post_mock.called
        assert post_mock.call_args.kwargs["url"] == MCP_URL
        sent_payload = post_mock.call_args.kwargs["json"]
        assert sent_payload["jsonrpc"] == "2.0"
        assert sent_payload["id"] == "call-42"
        assert sent_payload["method"] == "tools/call"
        assert sent_payload["params"] == {
            "name": "lookup_secret",
            "arguments": {"key": "production"},
        }
        assert sent_payload["result"]["content"][0]["text"] == (
            "Here is the secret API key abc123"
        )
        assert "request" not in sent_payload
        assert "metadata" not in sent_payload

    @pytest.mark.asyncio
    async def test_mcp_response_hook_blocks_violation(self):
        from litellm.types.mcp import MCPPostCallResponseObject

        g = _make_guardrail(
            inspection_type="mcp", event_hook=["pre_mcp_call", "during_mcp_call"]
        )
        response_obj = _mcp_response(
            SimpleNamespace(content=[{"type": "text", "text": "leaked"}])
        )

        post_mock = AsyncMock(return_value=_violation_response(url=MCP_URL))
        with _patch_inspection_post(g, post_mock):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs={"name": "leak", "arguments": {}},
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        assert result is not None, (
            "MCP response block was silently dropped — the litellm "
            "dispatcher swallows raised exceptions, so the hook must "
            "return a non-None MCPPostCallResponseObject to enforce a block."
        )
        assert isinstance(result, MCPPostCallResponseObject)
        replacement = result.mcp_tool_call_response
        assert len(replacement) == 1
        text = _mcp_result_text(replacement)
        assert "Blocked by Cisco AI Defense" in text
        assert "evt_123" in text
        assert "SECURITY_VIOLATION" in text

    @pytest.mark.asyncio
    async def test_mcp_response_hook_skipped_in_chat_mode(self):
        g = _make_guardrail()
        response_obj = _mcp_response(
            SimpleNamespace(content=[{"type": "text", "text": "hi"}])
        )

        post_mock = AsyncMock()
        with _patch_inspection_post(g, post_mock):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs={"name": "tool", "arguments": {}},
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        assert result is None
        post_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_call_skipped_for_mcp_mode_guardrail(self):
        g = _make_guardrail(inspection_type="mcp", event_hook="pre_mcp_call")
        data = {"messages": [{"role": "user", "content": "hi"}]}
        response = _make_model_response_with_content("fine")

        post_mock = AsyncMock()
        with _patch_inspection_post(g, post_mock):
            result = await g.async_post_call_success_hook(
                data=data,
                user_api_key_dict=UserAPIKeyAuth(),
                response=response,
            )
        assert result is response
        post_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_response_hook_runs_with_pre_mcp_call_only(self):
        g = _make_guardrail(inspection_type="mcp", event_hook="pre_mcp_call")
        response_obj = _mcp_response(
            SimpleNamespace(
                content=[{"type": "text", "text": "would have been scanned"}]
            )
        )

        post_mock = AsyncMock(return_value=_safe_response(url=MCP_URL))
        with _patch_inspection_post(g, post_mock):
            await g.async_post_mcp_tool_call_hook(
                kwargs={"name": "lookup", "arguments": {}},
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        assert post_mock.called, (
            "MCP response scan was skipped when only ``pre_mcp_call`` "
            "was configured. Per product decision, pre_mcp_call means "
            "'guard the MCP call' — request AND response."
        )

    @pytest.mark.parametrize(
        "cisco_response_kind,expected_block",
        [("safe", False), ("violation", True)],
    )
    @pytest.mark.asyncio
    async def test_mcp_response_hook_handles_raw_list_content(
        self, cisco_response_kind, expected_block
    ):
        from litellm.types.mcp import MCPPostCallResponseObject

        g = _make_guardrail(
            inspection_type="mcp", event_hook=["pre_mcp_call", "during_mcp_call"]
        )

        text_content = (
            "exfiltrated data: ..."
            if cisco_response_kind == "violation"
            else "Here is the secret API key abc123"
        )
        response_obj = _mcp_response([{"type": "text", "text": text_content}])

        cisco_resp = (
            _violation_response(url=MCP_URL)
            if cisco_response_kind == "violation"
            else _safe_response(url=MCP_URL)
        )
        post_mock = AsyncMock(return_value=cisco_resp)
        kwargs = {
            "name": "leak" if expected_block else "lookup_secret",
            "arguments": {"key": "production"} if not expected_block else {},
            "mcp_server_name": "vault",
            "litellm_call_id": "call-raw-list",
        }
        with _patch_inspection_post(g, post_mock):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        assert post_mock.called, (
            "MCP response inspect was silently skipped for raw-list "
            "shape — _normalize_mcp_response failed."
        )
        assert post_mock.call_args.kwargs["url"] == MCP_URL

        if expected_block:
            assert isinstance(result, MCPPostCallResponseObject)
            replacement = result.mcp_tool_call_response
            assert len(replacement) == 1
            assert "Blocked by Cisco AI Defense" in _mcp_result_text(replacement)
        else:
            sent_payload = post_mock.call_args.kwargs["json"]
            assert sent_payload["jsonrpc"] == "2.0"
            assert sent_payload["id"] == "call-raw-list"
            assert sent_payload["method"] == "tools/call"
            assert sent_payload["params"] == {
                "name": "lookup_secret",
                "arguments": {"key": "production"},
            }
            assert sent_payload["result"]["content"][0]["text"] == text_content
            assert result is None

    @pytest.mark.asyncio
    async def test_mcp_response_hook_through_real_logging_wrapper(self):
        from mcp.types import CallToolResult, TextContent

        from litellm.types.mcp import MCPPostCallResponseObject

        g = _make_guardrail(
            inspection_type="mcp", event_hook=["pre_mcp_call", "during_mcp_call"]
        )

        real_result = CallToolResult(
            content=[TextContent(type="text", text="leak 9045629876")],
            structuredContent={"patient": {"ssn": "123-45-6789"}},
            isError=False,
        )
        wrapped = MCPPostCallResponseObject(
            mcp_tool_call_response=real_result,
            hidden_params={},
        )

        assert isinstance(wrapped.mcp_tool_call_response, list)
        assert all(
            isinstance(item, tuple) and len(item) == 2
            for item in wrapped.mcp_tool_call_response
        ), (
            "Pydantic coercion shape changed — update the normalizer to "
            "match the new wire format."
        )

        post_mock = AsyncMock(return_value=_safe_response(url=MCP_URL))
        with _patch_inspection_post(g, post_mock):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs={
                    "name": "leak_tool",
                    "arguments": {},
                    "mcp_server_name": "vault",
                    "litellm_call_id": "real-wire-call",
                },
                response_obj=wrapped,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        assert post_mock.called, (
            "Inspect API not called for real CallToolResult shape — "
            "_normalize_mcp_response failed to handle Pydantic's "
            "iterated-BaseModel coercion."
        )
        assert post_mock.call_args.kwargs["url"] == MCP_URL
        sent_payload = post_mock.call_args.kwargs["json"]
        content_items = sent_payload["result"]["content"]

        assert len(content_items) == 1, (
            f"expected exactly 1 content item from the real "
            f"CallToolResult.content list, got {len(content_items)}: "
            f"{content_items!r}"
        )
        assert content_items[0].get("text") == "leak 9045629876", (
            f"Cisco wire payload missed the real tool text; got "
            f"{content_items[0]!r}. This means the Pydantic-coerced "
            f"(field_name, value) tuple shape was serialized as text "
            f"content instead of being unwrapped to find the inner "
            f"``content`` field."
        )
        assert content_items[0].get("type") == "text"
        assert sent_payload["result"]["structuredContent"] == {
            "patient": {"ssn": "123-45-6789"}
        }
        assert sent_payload["result"]["isError"] is False
        assert sent_payload["id"] == "real-wire-call"
        assert sent_payload["method"] == "tools/call"
        assert sent_payload["params"] == {"name": "leak_tool", "arguments": {}}
        assert result is None

    @pytest.mark.asyncio
    async def test_mcp_response_hook_uses_standard_logging_tool_metadata(self):
        g = _make_guardrail(inspection_type="mcp", event_hook="pre_mcp_call")
        response_obj = _mcp_response([{"type": "text", "text": "tool output"}])

        post_mock = AsyncMock(return_value=_safe_response(url=MCP_URL))
        with _patch_inspection_post(g, post_mock):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs={
                    "litellm_call_id": "metadata-call",
                    "mcp_tool_call_metadata": {
                        "name": "lookup_secret",
                        "arguments": {"key": "production"},
                        "mcp_server_name": "vault",
                    },
                },
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        assert result is None
        sent_payload = post_mock.call_args.kwargs["json"]
        assert sent_payload["method"] == "tools/call"
        assert sent_payload["params"] == {
            "name": "lookup_secret",
            "arguments": {"key": "production"},
        }
        assert sent_payload["result"]["content"][0]["text"] == "tool output"


class TestCiscoAIDefenseRedactListShape:

    @staticmethod
    def _violation_with_redact_response(text: str = "[REDACTED tool output]"):
        return _mock_inspect_response(
            {
                "is_safe": False,
                "classifications": ["PRIVACY_VIOLATION"],
                "severity": "HIGH",
                "rules": [{"rule_name": "PII", "entity_types": ["SSN"]}],
                "explanation": "PII detected, redaction available",
                "event_id": "evt_redact_1",
                "action": "redact",
                "sanitized_text": text,
            },
            url=MCP_URL,
        )

    @staticmethod
    def _raw_list_factory():
        original_content = [{"type": "text", "text": "Your SSN is 123-45-6789."}]
        return original_content, lambda: original_content[0]["text"]

    @staticmethod
    def _pydantic_tuple_list_factory():
        from mcp.types import TextContent

        inner_content = [TextContent(type="text", text="SSN: 123-45-6789")]
        tuples_list = [
            ("meta", None),
            ("content", inner_content),
            ("structuredContent", {"patient": {"ssn": "123-45-6789"}}),
            ("isError", False),
        ]
        return tuples_list, lambda: inner_content[0].text

    @pytest.mark.parametrize(
        "factory_name",
        ["_raw_list_factory", "_pydantic_tuple_list_factory"],
    )
    @pytest.mark.asyncio
    async def test_redact_rewrites_mcp_response_list_shape(self, factory_name):

        from litellm.types.mcp import MCPPostCallResponseObject

        g = _make_guardrail(
            inspection_type="mcp", event_hook=["pre_mcp_call", "during_mcp_call"]
        )

        content, get_text = getattr(self, factory_name)()
        response_obj = _mcp_response(content)

        with _patch_inspection_post(
            g, AsyncMock(return_value=self._violation_with_redact_response())
        ):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs={"name": "leak", "arguments": {}},
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        assert result is None or not isinstance(result, MCPPostCallResponseObject), (
            f"Redact silently fell through to block for {factory_name}. "
            f"result={result!r}"
        )
        assert get_text() == "[REDACTED tool output]", (
            f"Redact silently failed for {factory_name}; original text "
            f"not rewritten."
        )
        if factory_name == "_pydantic_tuple_list_factory":
            structured_content = dict(content)["structuredContent"]
            assert structured_content == {"result": "[REDACTED tool output]"}
            assert "123-45-6789" not in json.dumps(structured_content)

    @pytest.mark.asyncio
    async def test_redact_rewrites_client_visible_original_response(self):
        from mcp.types import CallToolResult, TextContent

        from litellm.types.llms.base import HiddenParams
        from litellm.types.mcp import MCPPostCallResponseObject

        original_response = CallToolResult(
            content=[TextContent(type="text", text="SSN: 123-45-6789")],
            structuredContent={"patient": {"ssn": "123-45-6789"}},
            isError=False,
        )
        wrapper = MCPPostCallResponseObject(
            mcp_tool_call_response=original_response,
            hidden_params=HiddenParams(),
        )

        g = _make_guardrail(
            inspection_type="mcp", event_hook=["pre_mcp_call", "during_mcp_call"]
        )
        with _patch_inspection_post(
            g, AsyncMock(return_value=self._violation_with_redact_response())
        ):
            await g.async_post_mcp_tool_call_hook(
                kwargs={
                    "name": "leak",
                    "arguments": {},
                    "original_response": original_response,
                },
                response_obj=wrapper,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        assert original_response.content[0].text == "[REDACTED tool output]"
        assert "123-45-6789" not in json.dumps(original_response.structuredContent), (
            "Redact verdict left the client-visible MCP tool output unchanged. "
            "The post-call hook receives a wrapped MCPPostCallResponseObject but "
            "the endpoint returns kwargs['original_response'], so the redaction "
            "must rewrite that object too. structuredContent still leaks: "
            f"{original_response.structuredContent!r}"
        )


class TestCiscoAIDefenseMcpInputRedactionFallback:
    """``sanitized_text``-only redaction of structured MCP arguments."""

    @pytest.mark.asyncio
    async def test_single_string_arg_is_rewritten(self):
        g = _make_guardrail(inspection_type="mcp", event_hook="pre_mcp_call")
        data = _mcp_request(
            name="search", args={"query": "my SSN is 123-45-6789", "limit": 10}
        )
        cisco = _redact_response(sanitized_text="my SSN is [REDACTED]", url=MCP_URL)
        with _patch_inspection_post(g, AsyncMock(return_value=cisco)):
            result = await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="mcp_call",
            )
        assert result == data
        assert data["mcp_arguments"]["query"] == "my SSN is [REDACTED]"
        assert data["mcp_arguments"]["limit"] == 10

    @pytest.mark.asyncio
    async def test_ambiguous_multi_string_args_block_instead_of_leaking(self):
        g = _make_guardrail(
            inspection_type="mcp",
            event_hook="pre_mcp_call",
            on_flagged_action="block",
        )
        original = {"query": "PII data", "filter": "sensitive term", "limit": 10}
        data = _mcp_request(name="search", args=dict(original))
        cisco = _redact_response(sanitized_text="[REDACTED]", url=MCP_URL)
        with _patch_inspection_post(g, AsyncMock(return_value=cisco)):
            with pytest.raises(HTTPException):
                await g.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(),
                    cache=DualCache(),
                    data=data,
                    call_type="mcp_call",
                )
        assert data["mcp_arguments"] == original

    @pytest.mark.asyncio
    async def test_ambiguous_multi_string_args_not_partially_redacted_in_monitor(self):
        g = _make_guardrail(
            inspection_type="mcp",
            event_hook="pre_mcp_call",
            on_flagged_action="monitor",
        )
        original = {"query": "PII data", "filter": "sensitive term"}
        data = _mcp_request(name="search", args=dict(original))
        cisco = _redact_response(sanitized_text="[REDACTED]", url=MCP_URL)
        with _patch_inspection_post(g, AsyncMock(return_value=cisco)):
            result = await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="mcp_call",
            )
        assert result == data
        assert data["mcp_arguments"] == original


class TestCiscoAIDefenseMCPBlockingContract:

    @pytest.mark.asyncio
    async def test_block_response_survives_dispatcher_contract(self):
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.types.mcp import MCPPostCallResponseObject
        from mcp.types import CallToolResult, TextContent

        g = _make_guardrail(
            name="cisco-mcp",
            inspection_type="mcp",
            event_hook=["pre_mcp_call", "during_mcp_call"],
        )
        raw_response = CallToolResult(
            content=[TextContent(type="text", text="exfiltrated")],
            structuredContent={"result": "exfiltrated"},
            isError=False,
        )
        response_obj = MCPPostCallResponseObject(
            mcp_tool_call_response=raw_response,
            hidden_params={},
        )

        post_mock = AsyncMock(return_value=_violation_response(url=MCP_URL))
        captured: Dict[str, Any] = {}
        with _patch_inspection_post(g, post_mock):
            try:
                captured["result"] = await g.async_post_mcp_tool_call_hook(
                    kwargs={
                        "name": "leak",
                        "arguments": {},
                        "original_response": raw_response,
                    },
                    response_obj=response_obj,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                )
            except Exception as e:
                captured["swallowed"] = repr(e)

        assert "swallowed" not in captured, (
            f"async_post_mcp_tool_call_hook raised — the litellm "
            f"dispatcher would swallow this and the block would be lost. "
            f"Got: {captured.get('swallowed')}"
        )
        result = captured["result"]
        assert isinstance(result, MCPPostCallResponseObject), (
            "Hook must keep returning a MCPPostCallResponseObject for "
            "dispatcher paths that do honor returned replacements."
        )
        assert raw_response.isError is True
        assert "Blocked by Cisco AI Defense" in raw_response.content[0].text
        assert raw_response.structuredContent is not None
        assert "Blocked by Cisco AI Defense" in raw_response.structuredContent["result"]
        assert "exfiltrated" not in raw_response.structuredContent["result"]
        logging_stub = Logging.__new__(Logging)
        logging_stub.model_call_details = {}
        parsed = logging_stub._parse_post_mcp_call_hook_response(response=result)
        assert parsed is not None
        assert "Blocked by Cisco AI Defense" in _mcp_result_text(parsed)


class TestCiscoAIDefenseJsonRpcSuccessEnvelope:

    @staticmethod
    def _cisco_mcp_envelope(*, is_safe: bool, action: str = "Block") -> Response:
        return _mock_inspect_response(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "is_safe": is_safe,
                    "action": action,
                    "classifications": [],
                    "rules": [
                        {
                            "rule_name": "PII",
                            "rule_id": 0,
                            "entity_types": [],
                            "classification": "NONE_VIOLATION",
                        }
                    ],
                    "event_id": "645d9d22-b016-47e0-a12c-9d587fb11c57",
                    "detected_pii": [],
                },
            },
            url=MCP_URL,
        )

    @pytest.mark.parametrize(
        "is_safe,action,should_block",
        [
            (False, "Block", True),
            (True, "Allow", False),
            (False, "Allow", False),
            (True, "Block", True),
        ],
    )
    @pytest.mark.asyncio
    async def test_mcp_jsonrpc_envelope_respects_verdict(
        self, is_safe, action, should_block
    ):
        g = _make_guardrail(
            name="cisco-mcp", inspection_type="mcp", event_hook="pre_mcp_call"
        )
        data = _mcp_request(
            name="ask_question",
            args={
                "repoName": "facebook/react",
                "question": "What is React Fiber 9045629876?",
            },
        )
        with _patch_inspection_post(
            g,
            AsyncMock(
                return_value=self._cisco_mcp_envelope(is_safe=is_safe, action=action)
            ),
        ):
            if should_block:
                with pytest.raises(HTTPException) as exc:
                    await g.async_pre_call_hook(
                        user_api_key_dict=UserAPIKeyAuth(),
                        cache=DualCache(),
                        data=data,
                        call_type="mcp_call",
                    )
                assert exc.value.status_code == 400
                assert exc.value.detail["surface"] == "mcp"
                assert (
                    exc.value.detail["event_id"]
                    == "645d9d22-b016-47e0-a12c-9d587fb11c57"
                )
            else:
                result = await g.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(),
                    cache=DualCache(),
                    data=data,
                    call_type="mcp_call",
                )
                assert result == data

    @pytest.mark.parametrize(
        "verdict,expected",
        [
            (
                {
                    "is_safe": False,
                    "classifications": ["SECURITY_VIOLATION"],
                    "action": "block",
                },
                "passthrough",
            ),
            (
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"is_safe": False, "action": "Block"},
                },
                {"is_safe": False, "action": "Block"},
            ),
        ],
    )
    def test_unwrap_verdict_envelope(self, verdict, expected):
        unwrapped = CiscoAIDefenseGuardrail._unwrap_verdict_envelope(verdict)
        if expected == "passthrough":
            assert unwrapped is verdict
        else:
            assert unwrapped == expected
