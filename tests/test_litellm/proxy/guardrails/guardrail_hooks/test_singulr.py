from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.singulr.singulr import SingulrGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
    SingulrGuardrailConfigModel,
)
from litellm.types.utils import ModelResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def singulr_guardrail():
    """Create a SingulrGuardrail instance with test credentials."""
    return SingulrGuardrail(
        singulr_api_base="https://api.test.singulr.ai",
        singulr_api_key="test_token_1234",
        singulr_guardrail_id="test_guardrail_id",
        singulr_application_id="test_enforcement_entity",
        guardrail_name="test-singulr",
        event_hook="pre_call",
        default_on=True,
    )


def _make_response(body: dict) -> MagicMock:
    """Build a mock httpx response with the given JSON body."""
    mock = MagicMock()
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    mock.status_code = 200
    return mock


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestSingulrConfiguration:
    def test_init_with_explicit_credentials(self):
        guardrail = SingulrGuardrail(
            singulr_api_key="test_key",
            singulr_api_base="https://custom.api.local",
            singulr_guardrail_id="id123",
            singulr_application_id="entity123",
            guardrail_name="my-guardrail",
        )
        assert guardrail.singulr_api_key == "test_key"
        assert guardrail.singulr_guardrail_id == "id123"
        assert guardrail.singulr_application_id == "entity123"

    def test_api_base_strips_surrounding_whitespace(self):
        """Regression: a UI-saved api_base with a trailing space
        (e.g. "https://custom.api.local ") broke urlparse's port parsing and
        made every guardrail call fail with a connection error, even though
        the configured host was reachable."""
        guardrail = SingulrGuardrail(
            singulr_api_key="test_key",
            singulr_api_base=" https://custom.api.local ",
        )
        assert guardrail.singulr_api_base == "https://custom.api.local"

    def test_api_base_strips_trailing_slash(self):
        guardrail = SingulrGuardrail(singulr_api_key="test_key", singulr_api_base="https://custom.api.local/")
        assert guardrail.singulr_api_base == "https://custom.api.local"

    def test_non_local_http_api_base_raises(self):
        """Guardrail payloads carry the API token and full conversation
        content, so a non-local endpoint must use HTTPS."""
        with pytest.raises(ValueError, match="HTTPS"):
            SingulrGuardrail(singulr_api_key="test_key", singulr_api_base="http://guardrails.singulr.ai")

    def test_localhost_http_api_base_is_allowed(self):
        guardrail = SingulrGuardrail(singulr_api_key="test_key", singulr_api_base="http://localhost:8003")
        assert guardrail.singulr_api_base == "http://localhost:8003"

    def test_block_on_error_defaults_true(self):
        guardrail = SingulrGuardrail(singulr_api_key="test_key")
        assert guardrail.block_on_error is True

    def test_timeout_defaults_to_30_seconds(self):
        guardrail = SingulrGuardrail(singulr_api_key="test_key")
        assert guardrail.timeout == 30.0

    def test_timeout_uses_configured_value(self):
        guardrail = SingulrGuardrail(singulr_api_key="test_key", timeout=5.0)
        assert guardrail.timeout == 5.0

    def test_supports_pre_call_post_call_logging_and_mcp_hooks(self):
        guardrail = SingulrGuardrail(singulr_api_key="test_key")
        assert guardrail.supported_event_hooks == [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.post_mcp_call,
        ]


# ---------------------------------------------------------------------------
# Payload construction for real proxy requests (request_data present)
# ---------------------------------------------------------------------------


class TestSingulrRequestPayload:
    @pytest.mark.asyncio
    async def test_model_and_messages_are_forwarded(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"model": "gpt-4o", "litellm_call_id": "call-1"}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["How do I reset my password?"], "model": "gpt-4o"},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["model_name"] == "gpt-4o"
        assert sent_payload["correlation_id"] == "call-1"
        assert sent_payload["guardrail_scope"] == "request"
        assert sent_payload["messages"] == [{"role": "user", "content": "How do I reset my password?"}]

    @pytest.mark.asyncio
    async def test_structured_messages_are_forwarded_verbatim(self, singulr_guardrail):
        """When structured_messages are provided (e.g. system + user turns),
        they must be sent as-is instead of being flattened into single
        user-role messages built from texts."""
        resp = _make_response({"should_block": False})
        structured_messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "How do I reset my password?"},
        ]
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["How do I reset my password?"], "structured_messages": structured_messages},
                request_data={},
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["messages"] == structured_messages

    @pytest.mark.asyncio
    async def test_images_are_forwarded(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": [], "images": ["data:image/png;base64,abc123"]},
                request_data={},
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["images"] == ["data:image/png;base64,abc123"]

    @pytest.mark.asyncio
    async def test_no_messages_or_images_skips_the_api_call(self, singulr_guardrail):
        with patch.object(singulr_guardrail.async_handler, "post") as mock_post:
            result = await singulr_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data={},
                input_type="request",
            )
        mock_post.assert_not_called()
        assert result == {"texts": []}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra_inputs",
        [
            {"tools": [{"type": "function", "function": {"name": "delete_file", "description": "", "parameters": {}}}]},
            {"images": ["data:image/png;base64,abc123"]},
        ],
        ids=["tools_alone", "images_alone"],
    )
    async def test_tools_or_images_alone_still_trigger_the_api_call(self, singulr_guardrail, extra_inputs):
        """Regression: a request with only tool definitions or only images and
        no text must still be checked, not skipped for lack of a message."""
        resp = _make_response({"should_block": False})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": [], **extra_inputs},
                request_data={},
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        for key, value in extra_inputs.items():
            assert sent_payload[key] == value

    @pytest.mark.asyncio
    async def test_tools_are_forwarded(self, singulr_guardrail):
        """Regression: tool/function definitions are client-controlled and can
        carry prompt-injection content, so they must reach Singulr for
        inspection instead of only messages and images."""
        resp = _make_response({"should_block": False})
        tools = [
            {
                "type": "function",
                "function": {"name": "search_docs", "description": "Search internal docs", "parameters": {}},
            }
        ]
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["How do I reset my password?"], "tools": tools},
                request_data={},
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["tools"] == tools

    @pytest.mark.asyncio
    async def test_responses_api_mcp_tools_are_forwarded(self, singulr_guardrail):
        """Regression: Responses API tools (e.g. {"type": "mcp", "server_label": ...})
        have no "function" key, unlike Chat Completions tools. SingulrGuardrailPayload
        rejected them with a pydantic ValidationError, turning every Responses API
        request carrying an MCP tool into a 500."""
        resp = _make_response({"should_block": False})
        tools = [
            {
                "type": "mcp",
                "server_label": "docs-server",
                "server_url": "https://mcp.example.com",
                "allowed_tools": ["search_docs"],
            }
        ]
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["How do I reset my password?"], "tools": tools},
                request_data={},
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["tools"] == tools

    @pytest.mark.asyncio
    async def test_user_api_key_alias_is_forwarded_in_metadata(self, singulr_guardrail):
        """Regression: the alias must be sent as {"user_api_key_alias": <alias>},
        not as a dict whose key is the alias value itself."""
        resp = _make_response({"should_block": False})
        request_data = {"litellm_metadata": {"user_api_key_alias": "my-key-alias"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_alias": "my-key-alias"}

    @pytest.mark.asyncio
    async def test_falls_back_to_regular_metadata_for_key_alias(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"metadata": {"user_api_key_alias": "fallback-alias"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_alias": "fallback-alias"}

    @pytest.mark.asyncio
    async def test_user_api_key_user_id_is_forwarded_in_metadata(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"litellm_metadata": {"user_api_key_user_id": "my-user-id"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_user_id": "my-user-id"}

    @pytest.mark.asyncio
    async def test_falls_back_to_regular_metadata_for_user_id(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"metadata": {"user_api_key_user_id": "fallback-user-id"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_user_id": "fallback-user-id"}

    @pytest.mark.asyncio
    async def test_user_api_key_alias_and_user_id_both_forwarded(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {
            "litellm_metadata": {
                "user_api_key_alias": "my-key-alias",
                "user_api_key_user_id": "my-user-id",
            }
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {
            "user_api_key_alias": "my-key-alias",
            "user_api_key_user_id": "my-user-id",
        }

    @pytest.mark.asyncio
    async def test_no_key_alias_available_sends_no_metadata(self, singulr_guardrail):
        """Regression: with no alias found, metadata must be omitted (None),
        not a {None: None} dict that fails payload validation."""
        resp = _make_response({"should_block": False})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data={},
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] is None


# ---------------------------------------------------------------------------
# Payload construction for responses
# ---------------------------------------------------------------------------


class TestSingulrResponsePayload:
    @pytest.mark.asyncio
    async def test_assistant_text_and_tool_calls_are_forwarded(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        inputs = {
            "texts": ["Go to settings."],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": "{}"},
                }
            ],
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="response",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["guardrail_scope"] == "response"
        assert sent_payload["response"]["content"] == "Go to settings."
        assert sent_payload["response"]["tool_calls"][0]["function"]["name"] == "get_current_time"

    @pytest.mark.asyncio
    async def test_response_images_are_forwarded(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        inputs = {"texts": ["ok"], "images": ["data:image/png;base64,xyz"]}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="response",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["images"] == ["data:image/png;base64,xyz"]

    @pytest.mark.asyncio
    async def test_incomplete_tool_calls_are_dropped(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        inputs = {
            "texts": [],
            "tool_calls": [
                {"id": None, "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": None},
            ],
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="response",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["response"]["tool_calls"] == []


# ---------------------------------------------------------------------------
# Allow / block decisions
# ---------------------------------------------------------------------------


class TestSingulrAllowAction:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "guard_response",
        [{"should_block": False}, {}],
        ids=["should_block_false", "should_block_omitted"],
    )
    async def test_should_block_falsy_returns_inputs_unchanged_on_request(self, singulr_guardrail, guard_response):
        """should_block is optional on the wire; a response that omits it
        entirely must be treated as allow, not block."""
        resp = _make_response(guard_response)
        inputs = {"texts": ["How do I reset my password?"]}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            result = await singulr_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"model": "gpt-4o"},
                input_type="request",
            )
            assert result is inputs

    @pytest.mark.asyncio
    async def test_should_block_false_returns_inputs_unchanged_on_response(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        inputs = {"texts": ["Here is your answer."]}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            result = await singulr_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="response",
            )
            assert result is inputs


class TestSingulrBlockAction:
    @pytest.mark.asyncio
    async def test_should_block_true_raises_on_request(self, singulr_guardrail):
        resp = _make_response({"should_block": True, "blocking_due_to": "PII Information detected"})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(GuardrailRaisedException) as exc_info:
                await singulr_guardrail.apply_guardrail(
                    inputs={"texts": ["My SSN is 123-45-6789"]},
                    request_data={"model": "gpt-4o"},
                    input_type="request",
                )
            assert "PII Information detected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_should_block_true_raises_on_response(self, singulr_guardrail):
        """Regression: apply_guardrail's response path compared
        should_block (a bool) against the string "block", which is always
        False, so a should_block=True response never blocked the assistant's
        reply. It must raise on any truthy should_block, matching the
        request path."""
        resp = _make_response({"should_block": True, "blocking_due_to": "Toxic content detected"})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(GuardrailRaisedException) as exc_info:
                await singulr_guardrail.apply_guardrail(
                    inputs={"texts": ["Here is something toxic."]},
                    request_data={},
                    input_type="response",
                )
            assert "Toxic content detected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_block_without_reason_uses_unknown_placeholder(self, singulr_guardrail):
        resp = _make_response({"should_block": True})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(GuardrailRaisedException, match="unknown"):
                await singulr_guardrail.apply_guardrail(
                    inputs={"texts": ["hi"]},
                    request_data={},
                    input_type="request",
                )


# ---------------------------------------------------------------------------
# MCP tool call guardrail (pre_mcp_call / post_mcp_call)
# ---------------------------------------------------------------------------


class TestSingulrMcpRequest:
    @pytest.mark.asyncio
    async def test_mcp_tool_name_routes_to_mcp_request_payload(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {
            "mcp_tool_name": "search_docs",
            "mcp_arguments": {"query": "reset password"},
            "mcp_server_name": "docs-server",
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            result = await singulr_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["guardrail_scope"] == "mcp_request"
        assert sent_payload["tool_name"] == "search_docs"
        assert sent_payload["tool_arguments"] == {"query": "reset password"}
        assert sent_payload["mcp_server_name"] == "docs-server"
        assert result == {"texts": []}

    @pytest.mark.asyncio
    async def test_mcp_request_should_block_true_raises(self, singulr_guardrail):
        resp = _make_response({"should_block": True, "blocking_due_to": "Disallowed tool"})
        request_data = {"mcp_tool_name": "delete_file", "mcp_arguments": {"path": "/etc/passwd"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(GuardrailRaisedException, match="Disallowed tool"):
                await singulr_guardrail.apply_guardrail(
                    inputs={"texts": []},
                    request_data=request_data,
                    input_type="request",
                )


class TestSingulrMcpResponse:
    @pytest.mark.asyncio
    async def test_call_mcp_tool_response_routes_to_mcp_response_payload(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {
            "call_type": "call_mcp_tool",
            "mcp_tool_name": "search_docs",
            "mcp_server_name": "docs-server",
            "model": "MCP: docs-server",
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["Result: password reset link sent."]},
                request_data=request_data,
                input_type="response",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["guardrail_scope"] == "mcp_response"
        assert sent_payload["model_name"] == "MCP: docs-server"
        assert sent_payload["tool_result"] == ["Result: password reset link sent."]

    @pytest.mark.asyncio
    async def test_mcp_response_with_no_texts_skips_the_api_call(self, singulr_guardrail):
        request_data = {"call_type": "call_mcp_tool", "mcp_tool_name": "search_docs"}
        with patch.object(singulr_guardrail.async_handler, "post") as mock_post:
            result = await singulr_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="response",
            )
        mock_post.assert_not_called()
        assert result == {"texts": []}

    @pytest.mark.asyncio
    async def test_mcp_response_should_block_true_raises(self, singulr_guardrail):
        resp = _make_response({"should_block": True, "blocking_due_to": "Sensitive tool output"})
        request_data = {"call_type": "call_mcp_tool", "mcp_tool_name": "search_docs"}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(GuardrailRaisedException, match="Sensitive tool output"):
                await singulr_guardrail.apply_guardrail(
                    inputs={"texts": ["leaked secret"]},
                    request_data=request_data,
                    input_type="response",
                )


# ---------------------------------------------------------------------------
# apply_guardrail dispatch (request vs response vs unknown input_type)
# ---------------------------------------------------------------------------


class TestSingulrApplyGuardrailDispatch:
    @pytest.mark.asyncio
    async def test_unknown_input_type_returns_inputs_unchanged(self, singulr_guardrail):
        with patch.object(singulr_guardrail.async_handler, "post") as mock_post:
            inputs = {"texts": ["hi"]}
            result = await singulr_guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="unsupported",
            )
        mock_post.assert_not_called()
        assert result is inputs


# ---------------------------------------------------------------------------
# logging_only hook
# ---------------------------------------------------------------------------


class TestSingulrLoggingHook:
    @pytest.mark.asyncio
    async def test_forwards_request_messages_and_response_text(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o", "litellm_call_id": "call-1"}
        result = {"choices": [{"finish_reason": "stop", "message": {"content": "hello there"}}]}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.async_logging_hook(kwargs=kwargs, result=result, call_type="acompletion")

        request_payload = mock_post.call_args_list[0].kwargs["json"]
        response_payload = mock_post.call_args_list[1].kwargs["json"]
        assert request_payload["guardrail_scope"] == "request"
        assert request_payload["messages"] == kwargs["messages"]
        assert response_payload["guardrail_scope"] == "response"
        assert response_payload["response"] == result

    @pytest.mark.asyncio
    async def test_forwards_a_real_model_response_without_swallowing_it(self, singulr_guardrail):
        """Regression: a normal completion callback passes a ModelResponse, not a
        dict. The response payload must carry its actual serialized content instead
        of silently dropping it because ModelResponse isn't a Mapping."""
        resp = _make_response({"should_block": False})
        result = ModelResponse(
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello there"}}]
        )
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.async_logging_hook(kwargs={}, result=result, call_type="acompletion")

        response_payload = mock_post.call_args.kwargs["json"]
        assert response_payload["guardrail_scope"] == "response"
        assert response_payload["response"]["choices"][0]["message"]["content"] == "hello there"

    @pytest.mark.asyncio
    async def test_non_serializable_result_falls_back_to_string_report(self, singulr_guardrail):
        """A result that pydantic can't serialize to JSON must still get reported,
        as a stringified fallback, instead of raising out of the logging_only hook."""
        resp = _make_response({"should_block": False})

        class Unserializable:
            def __repr__(self) -> str:
                return "<Unserializable>"

        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.async_logging_hook(kwargs={}, result=Unserializable(), call_type="acompletion")

        response_payload = mock_post.call_args.kwargs["json"]
        assert response_payload["response"] == "<Unserializable>"

    @pytest.mark.asyncio
    async def test_no_messages_and_no_result_skips_both_api_calls(self, singulr_guardrail):
        with patch.object(singulr_guardrail.async_handler, "post") as mock_post:
            returned_kwargs, returned_result = await singulr_guardrail.async_logging_hook(
                kwargs={}, result=None, call_type="acompletion"
            )
        mock_post.assert_not_called()
        assert returned_result is None
        guardrail_information = returned_kwargs["standard_logging_object"]["guardrail_information"]
        assert guardrail_information[0]["guardrail_status"] == "success"

    @pytest.mark.asyncio
    async def test_records_standard_logging_guardrail_information(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            updated_kwargs, _ = await singulr_guardrail.async_logging_hook(
                kwargs=kwargs, result=None, call_type="acompletion"
            )
        guardrail_information = updated_kwargs["standard_logging_object"]["guardrail_information"]
        assert len(guardrail_information) == 1
        assert guardrail_information[0]["guardrail_name"] == "test-singulr"
        assert guardrail_information[0]["guardrail_status"] == "success"

    @pytest.mark.asyncio
    async def test_api_error_marks_guardrail_status_intervened(self, singulr_guardrail):
        """With block_on_error=True (the default), a transport failure while
        reporting to Singulr raises internally; async_logging_hook must catch
        it, mark the status accordingly, and still return (kwargs, result)
        instead of propagating -- logging_only must never block the call."""
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        with patch.object(
            singulr_guardrail.async_handler,
            "post",
            side_effect=httpx.TransportError("connection refused"),
        ):
            updated_kwargs, result = await singulr_guardrail.async_logging_hook(
                kwargs=kwargs, result=None, call_type="acompletion"
            )
        assert result is None
        guardrail_information = updated_kwargs["standard_logging_object"]["guardrail_information"]
        assert guardrail_information[0]["guardrail_status"] == "guardrail_intervened"

    def test_sync_logging_hook_returns_kwargs_and_result_unchanged_when_loop_running(self, singulr_guardrail):
        """logging_hook is the sync entrypoint used outside an event loop;
        inside a running loop it must no-op rather than deadlock or raise."""
        import asyncio

        async def _drive():
            kwargs = {"messages": [{"role": "user", "content": "hi"}]}
            return singulr_guardrail.logging_hook(kwargs=kwargs, result=None, call_type="acompletion")

        returned_kwargs, returned_result = asyncio.run(_drive())
        assert returned_result is None
        assert returned_kwargs == {"messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# HTTP call wiring (endpoint, timeout, headers)
# ---------------------------------------------------------------------------


class TestSingulrRequestWiring:
    @pytest.mark.asyncio
    async def test_sends_configured_timeout_and_calls_the_guard_endpoint(self):
        """litellm_params.timeout must reach the httpx call so operators can
        tighten or loosen the latency budget instead of being stuck with a
        hardcoded 30s regardless of configuration."""
        guardrail = SingulrGuardrail(
            singulr_api_key="test_key",
            singulr_api_base="https://api.test.singulr.ai",
            timeout=5.0,
        )
        resp = _make_response({"should_block": False})
        with patch.object(guardrail.async_handler, "post", return_value=resp) as mock_post:
            await guardrail.apply_guardrail(
                inputs={"texts": ["test"]},
                request_data={},
                input_type="request",
            )
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["timeout"] == 5.0
        assert call_kwargs["url"] == "https://api.test.singulr.ai/api/v1/ai-gateway/litellm"


class TestSingulrBuildHeaders:
    def test_content_type_always_present(self, singulr_guardrail):
        assert singulr_guardrail._build_headers()["Content-Type"] == "application/json"

    def test_all_optional_headers_included_when_set(self, singulr_guardrail):
        headers = singulr_guardrail._build_headers()
        assert headers["X-Singulr-Gateway-Token"] == "test_token_1234"
        assert headers["X-Singulr-Enforcement-Entity-Id"] == "test_enforcement_entity"
        assert headers["X-Singulr-Guardrail-Id"] == "test_guardrail_id"

    def test_optional_headers_absent_when_unset(self):
        guardrail = SingulrGuardrail(guardrail_name="bare")
        headers = guardrail._build_headers()
        assert "X-Singulr-Gateway-Token" not in headers
        assert "X-Singulr-Enforcement-Entity-Id" not in headers
        assert "X-Singulr-Guardrail-Id" not in headers


# ---------------------------------------------------------------------------
# Non-JSON / malformed response handling
# ---------------------------------------------------------------------------


class TestSingulrInvalidResponse:
    @pytest.mark.asyncio
    async def test_non_json_response_block_on_error_false_returns_inputs(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=False,
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")

        inputs = {"texts": ["test"]}
        with patch.object(guardrail.async_handler, "post", return_value=mock_resp):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )
        assert result is inputs

    @pytest.mark.asyncio
    async def test_non_json_response_block_on_error_true_raises(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=True,
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")

        with patch.object(guardrail.async_handler, "post", return_value=mock_resp):
            with pytest.raises(GuardrailRaisedException):
                await guardrail.apply_guardrail(
                    inputs={"texts": ["test"]},
                    request_data={},
                    input_type="request",
                )

    @pytest.mark.asyncio
    async def test_response_missing_expected_fields_block_on_error_true_raises(self):
        """Regression: a response body that fails SingulrGuardrailResponse
        validation must raise GuardrailRaisedException instead of letting
        pydantic.ValidationError propagate unhandled."""
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=True,
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("not valid json")

        with patch.object(guardrail.async_handler, "post", return_value=mock_resp):
            with pytest.raises(GuardrailRaisedException):
                await guardrail.apply_guardrail(
                    inputs={"texts": ["test"]},
                    request_data={},
                    input_type="request",
                )


# ---------------------------------------------------------------------------
# Transport error handling
# ---------------------------------------------------------------------------


class TestSingulrTransportError:
    @pytest.mark.asyncio
    async def test_remote_protocol_error_block_on_error_false_returns_inputs(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=False,
        )
        inputs = {"texts": ["test"]}
        with patch.object(
            guardrail.async_handler,
            "post",
            side_effect=httpx.RemoteProtocolError("malformed HTTP response"),
        ):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )
        assert result is inputs

    @pytest.mark.asyncio
    async def test_remote_protocol_error_block_on_error_true_raises(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=True,
        )
        with patch.object(
            guardrail.async_handler,
            "post",
            side_effect=httpx.RemoteProtocolError("malformed HTTP response"),
        ):
            with pytest.raises(GuardrailRaisedException):
                await guardrail.apply_guardrail(
                    inputs={"texts": ["test"]},
                    request_data={},
                    input_type="request",
                )


# ---------------------------------------------------------------------------
# HTTP status error handling
# ---------------------------------------------------------------------------


class TestSingulrHttpStatusError:
    @pytest.mark.asyncio
    async def test_http_error_message_names_status_code_not_unreachable(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=True,
        )
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        exc = httpx.HTTPStatusError("403 Forbidden", request=MagicMock(), response=mock_response)
        mock_response.raise_for_status.side_effect = exc

        with patch.object(guardrail.async_handler, "post", return_value=mock_response):
            with pytest.raises(GuardrailRaisedException) as exc_info:
                await guardrail.apply_guardrail(
                    inputs={"texts": ["test"]},
                    request_data={},
                    input_type="request",
                )
            msg = str(exc_info.value)
            assert "403" in msg
            assert "unreachable" not in msg.lower()

    @pytest.mark.asyncio
    async def test_http_error_block_on_error_false_returns_inputs(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=False,
        )
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        exc = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_response)
        mock_response.raise_for_status.side_effect = exc

        inputs = {"texts": ["test"]}
        with patch.object(guardrail.async_handler, "post", return_value=mock_response):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )
        assert result is inputs


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------


class TestSingulrConfigModel:
    def test_ui_friendly_name(self):
        assert SingulrGuardrailConfigModel.ui_friendly_name() == "Singulr"


# ---------------------------------------------------------------------------
# Initializer and registry
# ---------------------------------------------------------------------------


class TestSingulrInitializer:
    def test_guardrail_initializer_registry_has_entry(self):
        from litellm.proxy.guardrails.guardrail_hooks.singulr import (
            initialize_guardrail,
        )

        assert callable(initialize_guardrail)

    def test_initialize_guardrail_reads_singulr_prefixed_fields(self):
        """Regression: the UI config form (and YAML config) populate the
        singulr_-prefixed fields declared on SingulrGuardrailConfigModel, not
        the generic api_base/api_key fields. initialize_guardrail must read
        those, or a UI-configured singulr_api_base is silently ignored and
        the guardrail falls back to the localhost default."""
        from litellm.proxy.guardrails.guardrail_hooks.singulr import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import Guardrail, LitellmParams

        litellm_params = LitellmParams(
            guardrail="singulr",
            mode="pre_call",
            singulr_api_base="https://configured.singulr.ai",
            singulr_api_key="configured_key",
            singulr_application_id="configured_app_id",
            singulr_guardrail_id="configured_guardrail_id",
        )
        guardrail: Guardrail = {
            "guardrail_name": "test-singulr",
            "litellm_params": litellm_params,
        }

        cb = initialize_guardrail(litellm_params, guardrail)

        assert cb.singulr_application_id == "configured_app_id"
        assert cb.singulr_guardrail_id == "configured_guardrail_id"

    def test_initialize_guardrail_wires_timeout(self):
        """BaseLitellmParams.timeout exists so operators can override the
        per-request latency budget. initialize_guardrail must forward it to
        SingulrGuardrail instead of leaving every deployment stuck on the
        hardcoded default regardless of configuration."""
        from litellm.proxy.guardrails.guardrail_hooks.singulr import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import Guardrail, LitellmParams

        litellm_params = LitellmParams(
            guardrail="singulr",
            mode="pre_call",
            singulr_api_key="configured_key",
            timeout=12.5,
        )
        guardrail: Guardrail = {
            "guardrail_name": "test-singulr",
            "litellm_params": litellm_params,
        }

        cb = initialize_guardrail(litellm_params, guardrail)

        assert cb.timeout == 12.5
