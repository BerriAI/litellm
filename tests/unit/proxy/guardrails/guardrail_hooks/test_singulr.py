import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.singulr.singulr import SingulrGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
    SingulrGuardrailConfigModel,
)
from litellm.types.utils import ModelResponse


@pytest.fixture
def singulr_guardrail():
    return SingulrGuardrail(
        singulr_api_base="https://api.test.singulr.ai",
        singulr_api_key="test_token_1234",
        singulr_guardrail_id="test_guardrail_id",
        singulr_application_id="test_enforcement_entity",
        guardrail_name="test-singulr",
        event_hook="pre_call",
        default_on=True,
    )


@pytest.fixture
def logging_only_guardrail():
    return SingulrGuardrail(
        singulr_api_base="https://api.test.singulr.ai",
        singulr_api_key="test_token_1234",
        singulr_guardrail_id="test_guardrail_id",
        singulr_application_id="test_enforcement_entity",
        guardrail_name="test-singulr",
        event_hook="logging_only",
        default_on=True,
    )


def _logging_obj(call_type: str) -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.call_type = call_type
    return logging_obj


def _make_response(body: dict) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    mock.status_code = 200
    return mock


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
        guardrail = SingulrGuardrail(
            singulr_api_key="test_key",
            singulr_api_base=" https://custom.api.local ",
        )
        assert guardrail.singulr_api_base == "https://custom.api.local"

    def test_api_base_strips_trailing_slash(self):
        guardrail = SingulrGuardrail(singulr_api_key="test_key", singulr_api_base="https://custom.api.local/")
        assert guardrail.singulr_api_base == "https://custom.api.local"

    def test_non_local_http_api_base_raises(self):
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
    async def test_user_api_key_user_email_is_forwarded_in_metadata(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"litellm_metadata": {"user_api_key_user_email": "user@example.com"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_user_email": "user@example.com"}

    @pytest.mark.asyncio
    async def test_user_api_key_organization_alias_is_forwarded_in_metadata(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"litellm_metadata": {"user_api_key_org_alias": "Acme Org"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_org_alias": "Acme Org"}

    @pytest.mark.asyncio
    async def test_user_api_key_team_alias_is_forwarded_in_metadata(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"litellm_metadata": {"user_api_key_team_alias": "AI Content Security Team"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_team_alias": "AI Content Security Team"}

    @pytest.mark.asyncio
    async def test_user_api_key_org_id_is_forwarded_in_metadata(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"litellm_metadata": {"user_api_key_org_id": "org-123"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_org_id": "org-123"}

    @pytest.mark.asyncio
    async def test_user_api_key_team_id_is_forwarded_in_metadata(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"litellm_metadata": {"user_api_key_team_id": "team-456"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_team_id": "team-456"}

    @pytest.mark.asyncio
    async def test_user_api_key_user_role_is_forwarded_in_metadata(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
        request_data = {"litellm_metadata": {"user_api_key_auth": auth}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_user_role": LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value}

    @pytest.mark.asyncio
    async def test_no_user_role_available_omits_role_from_metadata(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"litellm_metadata": {"user_api_key_alias": "my-key-alias"}}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert "user_api_key_user_role" not in sent_payload["metadata"]

    @pytest.mark.asyncio
    async def test_all_user_metadata_fields_forwarded_together(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
        request_data = {
            "litellm_metadata": {
                "user_api_key_alias": "my-key-alias",
                "user_api_key_user_id": "my-user-id",
                "user_api_key_user_email": "user@example.com",
                "user_api_key_org_id": "org-123",
                "user_api_key_org_alias": "Acme Org",
                "user_api_key_team_id": "team-456",
                "user_api_key_team_alias": "AI Content Security Team",
                "user_api_key_auth": auth,
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
            "user_api_key_user_email": "user@example.com",
            "user_api_key_org_id": "org-123",
            "user_api_key_org_alias": "Acme Org",
            "user_api_key_team_id": "team-456",
            "user_api_key_team_alias": "AI Content Security Team",
            "user_api_key_user_role": LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
        }

    @pytest.mark.asyncio
    async def test_no_key_alias_available_sends_no_metadata(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data={},
                input_type="request",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] is None


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
                {"id": "call_3", "type": "function", "function": {"name": None, "arguments": "{}"}},
                {"id": "call_4", "type": "function", "function": {"name": "f", "arguments": None}},
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw_type, expected_type",
        [(None, "function"), ("custom", "custom")],
        ids=["type_missing", "type_not_function"],
    )
    async def test_tool_call_type_other_than_function_is_still_scanned(
        self, singulr_guardrail, raw_type, expected_type
    ):
        resp = _make_response({"should_block": False})
        tool_call = {"id": "call_1", "function": {"name": "get_current_time", "arguments": "{}"}}
        inputs = {
            "texts": [],
            "tool_calls": [tool_call if raw_type is None else {**tool_call, "type": raw_type}],
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="response")
        sent_tool_calls = mock_post.call_args.kwargs["json"]["response"]["tool_calls"]
        assert [call["type"] for call in sent_tool_calls] == [expected_type]
        assert sent_tool_calls[0]["function"]["name"] == "get_current_time"

    @pytest.mark.asyncio
    async def test_non_string_tool_call_arguments_are_serialized(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        inputs = {
            "texts": [],
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "rm", "arguments": {"path": "/etc/passwd"}}}
            ],
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="response")
        sent_tool_calls = mock_post.call_args.kwargs["json"]["response"]["tool_calls"]
        assert json.loads(sent_tool_calls[0]["function"]["arguments"]) == {"path": "/etc/passwd"}

    @pytest.mark.asyncio
    async def test_block_verdict_still_raises_for_a_non_function_tool_call(self, singulr_guardrail):
        resp = _make_response({"should_block": True, "blocking_due_to": "dangerous_tool"})
        inputs = {
            "texts": [],
            "tool_calls": [{"id": "call_1", "function": {"name": "rm", "arguments": "{}"}}],
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(GuardrailRaisedException) as exc_info:
                await singulr_guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="response")
        assert "dangerous_tool" in str(exc_info.value)


class TestSingulrAllowAction:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "guard_response",
        [{"should_block": False}, {}],
        ids=["should_block_false", "should_block_omitted"],
    )
    async def test_should_block_falsy_returns_inputs_unchanged_on_request(self, singulr_guardrail, guard_response):
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

    @pytest.mark.asyncio
    async def test_response_returns_inputs_unchanged_when_api_unreachable_and_block_on_error_false(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=False,
        )
        inputs = {"texts": ["Here is your answer."]}
        with patch.object(guardrail.async_handler, "post", side_effect=httpx.TransportError("unreachable")):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="response",
            )
            assert result is inputs

    @pytest.mark.asyncio
    async def test_explicit_null_verdict_fails_closed_by_default(self, singulr_guardrail):
        resp = _make_response({"should_block": None})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(GuardrailRaisedException, match="invalid response"):
                await singulr_guardrail.apply_guardrail(
                    inputs={"texts": ["hi"]},
                    request_data={"model": "gpt-4o"},
                    input_type="request",
                )

    @pytest.mark.asyncio
    async def test_explicit_null_verdict_fails_open_when_block_on_error_false(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=False,
        )
        resp = _make_response({"should_block": None})
        inputs = {"texts": ["hi"]}
        with patch.object(guardrail.async_handler, "post", return_value=resp):
            assert await guardrail._call_api({"guardrail_scope": "request"}) is None
            result = await guardrail.apply_guardrail(
                inputs=inputs, request_data={"model": "gpt-4o"}, input_type="request"
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
            assert exc_info.value.blocked_content is True

    @pytest.mark.asyncio
    async def test_should_block_true_raises_on_response(self, singulr_guardrail):
        resp = _make_response({"should_block": True, "blocking_due_to": "Toxic content detected"})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(GuardrailRaisedException) as exc_info:
                await singulr_guardrail.apply_guardrail(
                    inputs={"texts": ["Here is something toxic."]},
                    request_data={},
                    input_type="response",
                )
            assert "Toxic content detected" in str(exc_info.value)
            assert exc_info.value.blocked_content is True

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
            with pytest.raises(GuardrailRaisedException, match="Disallowed tool") as exc_info:
                await singulr_guardrail.apply_guardrail(
                    inputs={"texts": []},
                    request_data=request_data,
                    input_type="request",
                )
            assert exc_info.value.blocked_content is True

    @pytest.mark.asyncio
    async def test_mcp_request_is_a_noop_when_api_unreachable_and_block_on_error_false(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=False,
        )
        request_data = {"mcp_tool_name": "search_docs", "mcp_arguments": {"query": "reset password"}}
        with patch.object(guardrail.async_handler, "post", side_effect=httpx.TransportError("unreachable")):
            result = await guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="request",
            )
        assert result == {"texts": []}

    @pytest.mark.asyncio
    async def test_mcp_rest_body_shape_routes_to_mcp_request_payload(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"name": "echo", "arguments": {"text": "my ssn is 123-45-6789"}, "server_id": "srv-1"}
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["my ssn is 123-45-6789"], "tools": [{"type": "function"}]},
                request_data=request_data,
                input_type="request",
                logging_obj=_logging_obj("call_mcp_tool"),
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["guardrail_scope"] == "mcp_request"
        assert sent_payload["tool_name"] == "echo"
        assert sent_payload["tool_arguments"] == {"text": "my ssn is 123-45-6789"}
        assert "messages" not in sent_payload

    @pytest.mark.asyncio
    async def test_mcp_rest_body_without_arguments_still_routes_to_mcp_request(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": [], "tools": [{"type": "function"}]},
                request_data={"name": "echo", "server_id": "srv-1"},
                input_type="request",
                logging_obj=_logging_obj("call_mcp_tool"),
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["guardrail_scope"] == "mcp_request"
        assert sent_payload["tool_name"] == "echo"
        assert sent_payload["tool_arguments"] is None

    @pytest.mark.asyncio
    async def test_non_mapping_tool_arguments_are_forwarded_verbatim(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["raw text"], "tools": [{"type": "function"}]},
                request_data={"name": "echo", "arguments": "raw text", "server_id": "srv-1"},
                input_type="request",
                logging_obj=_logging_obj("call_mcp_tool"),
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["guardrail_scope"] == "mcp_request"
        assert sent_payload["tool_arguments"] == "raw text"

    @pytest.mark.asyncio
    async def test_llm_request_body_keys_cannot_reroute_the_scan_to_mcp(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "my ssn is 123-45-6789"}],
            "name": "x",
            "arguments": {},
            "mcp_tool_name": "x",
            "call_type": "call_mcp_tool",
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={
                    "texts": ["my ssn is 123-45-6789"],
                    "structured_messages": [{"role": "user", "content": "my ssn is 123-45-6789"}],
                },
                request_data=request_data,
                input_type="request",
                logging_obj=_logging_obj("acompletion"),
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["guardrail_scope"] == "request"
        assert [m["content"] for m in sent_payload["messages"]] == ["my ssn is 123-45-6789"]

    @pytest.mark.asyncio
    async def test_llm_response_with_spoofed_mcp_keys_still_scans_the_tool_calls(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {"model": "gpt-4o", "messages": [], "name": "x", "arguments": {}, "mcp_tool_name": "x"}
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "transfer_funds", "arguments": '{"amount": 5000}'},
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": [], "tool_calls": [tool_call]},
                request_data=request_data,
                input_type="response",
                logging_obj=_logging_obj("acompletion"),
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["guardrail_scope"] == "response"
        assert sent_payload["response"]["tool_calls"][0]["function"]["name"] == "transfer_funds"


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
            with pytest.raises(GuardrailRaisedException, match="Sensitive tool output") as exc_info:
                await singulr_guardrail.apply_guardrail(
                    inputs={"texts": ["leaked secret"]},
                    request_data=request_data,
                    input_type="response",
                )
            assert exc_info.value.blocked_content is True

    @pytest.mark.asyncio
    async def test_mcp_response_resolves_metadata_from_nested_litellm_params(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
        request_data = {
            "call_type": "call_mcp_tool",
            "mcp_tool_name": "search_docs",
            "litellm_params": {
                "metadata": {
                    "user_api_key_alias": "my-key-alias",
                    "user_api_key_user_id": "my-user-id",
                    "user_api_key_user_email": "user@example.com",
                    "user_api_key_org_id": "org-123",
                    "user_api_key_org_alias": "Acme Org",
                    "user_api_key_team_id": "team-456",
                    "user_api_key_team_alias": "AI Content Security Team",
                    "user_api_key_auth": auth,
                }
            },
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["Result: password reset link sent."]},
                request_data=request_data,
                input_type="response",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {
            "user_api_key_alias": "my-key-alias",
            "user_api_key_user_id": "my-user-id",
            "user_api_key_user_email": "user@example.com",
            "user_api_key_org_id": "org-123",
            "user_api_key_org_alias": "Acme Org",
            "user_api_key_team_id": "team-456",
            "user_api_key_team_alias": "AI Content Security Team",
            "user_api_key_user_role": LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
        }

    @pytest.mark.asyncio
    async def test_mcp_response_prefers_top_level_metadata_over_nested_litellm_params(self, singulr_guardrail):
        resp = _make_response({"should_block": False})
        request_data = {
            "call_type": "call_mcp_tool",
            "mcp_tool_name": "search_docs",
            "litellm_metadata": {"user_api_key_alias": "top-level-alias"},
            "litellm_params": {"metadata": {"user_api_key_alias": "nested-alias"}},
        }
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=request_data,
                input_type="response",
            )
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["metadata"] == {"user_api_key_alias": "top-level-alias"}

    @pytest.mark.asyncio
    async def test_mcp_response_returns_inputs_unchanged_when_api_unreachable_and_block_on_error_false(self):
        guardrail = SingulrGuardrail(
            singulr_api_base="https://api.test.singulr.ai",
            singulr_api_key="test_token_1234",
            guardrail_name="test-singulr",
            block_on_error=False,
        )
        request_data = {"call_type": "call_mcp_tool", "mcp_tool_name": "search_docs"}
        inputs = {"texts": ["leaked secret"]}
        with patch.object(guardrail.async_handler, "post", side_effect=httpx.TransportError("unreachable")):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
            )
        assert result is inputs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_data", "logging_obj"),
        [
            ({"call_type": "call_mcp_tool", "model": "MCP: echo"}, None),
            ({"model": "MCP: echo"}, None),
            ({"name": "echo", "arguments": {"text": "hi"}}, _logging_obj("call_mcp_tool")),
        ],
        ids=["post_mcp_call_model_call_details", "logging_only_scratch_request", "rest_pre_call_logger"],
    )
    async def test_mcp_response_is_detected_from_each_producer(self, singulr_guardrail, request_data, logging_obj):
        resp = _make_response({"should_block": False})
        with patch.object(singulr_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await singulr_guardrail.apply_guardrail(
                inputs={"texts": ["tool output"]},
                request_data=request_data,
                input_type="response",
                logging_obj=logging_obj,
            )
        assert mock_post.call_args.kwargs["json"]["guardrail_scope"] == "mcp_response"


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


class TestSingulrLoggingHook:
    @staticmethod
    def _logged_call(**overrides):
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_call_id": "call-1",
            "litellm_params": {"metadata": {"user_api_key_alias": "my-key-alias", "user_api_key_org_id": "org-123"}},
            "standard_logging_object": {"guardrail_information": []},
        }
        return {**kwargs, **overrides}

    @pytest.mark.asyncio
    async def test_scans_request_then_response_as_an_assistant_message(self, logging_only_guardrail):
        resp = _make_response({"should_block": False})
        result = ModelResponse(
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello there"}}]
        )
        with patch.object(logging_only_guardrail.async_handler, "post", return_value=resp) as mock_post:
            updated_kwargs, returned = await logging_only_guardrail.async_logging_hook(
                kwargs=self._logged_call(), result=result, call_type="acompletion"
            )

        assert returned is result
        scopes = [call.kwargs["json"]["guardrail_scope"] for call in mock_post.call_args_list]
        assert scopes == ["request", "response"]
        request_payload = mock_post.call_args_list[0].kwargs["json"]
        response_payload = mock_post.call_args_list[1].kwargs["json"]
        assert request_payload["messages"] == [{"role": "user", "content": "hi"}]
        assert request_payload["correlation_id"] == "call-1"
        assert response_payload["response"] == {"role": "assistant", "content": "hello there", "tool_calls": []}
        expected_metadata = {"user_api_key_alias": "my-key-alias", "user_api_key_org_id": "org-123"}
        assert request_payload["metadata"] == expected_metadata
        assert response_payload["metadata"] == expected_metadata
        statuses = [
            entry["guardrail_status"] for entry in updated_kwargs["standard_logging_object"]["guardrail_information"]
        ]
        assert statuses == ["success", "success"]

    @pytest.mark.asyncio
    async def test_block_verdict_is_recorded_as_intervened_without_failing_the_call(self, logging_only_guardrail):
        resp = _make_response({"should_block": True, "blocking_due_to": "pii"})
        with patch.object(logging_only_guardrail.async_handler, "post", return_value=resp):
            updated_kwargs, returned = await logging_only_guardrail.async_logging_hook(
                kwargs=self._logged_call(messages=[{"role": "user", "content": "my ssn is 123-45-6789"}]),
                result=None,
                call_type="acompletion",
            )
        assert returned is None
        entries = updated_kwargs["standard_logging_object"]["guardrail_information"]
        assert entries[0]["guardrail_status"] == "guardrail_intervened"
        assert entries[0]["guardrail_mode"] == "logging_only"
        assert "Blocking due to pii" in str(entries[0]["guardrail_response"])

    @pytest.mark.asyncio
    async def test_vendor_timeout_is_recorded_as_failed_to_respond(self, logging_only_guardrail):
        timeout = litellm.Timeout("Singulr timed out", model="gpt-4o", llm_provider="singulr")
        with patch.object(logging_only_guardrail.async_handler, "post", side_effect=timeout):
            updated_kwargs, returned = await logging_only_guardrail.async_logging_hook(
                kwargs=self._logged_call(), result=None, call_type="acompletion"
            )
        assert returned is None
        entries = updated_kwargs["standard_logging_object"]["guardrail_information"]
        assert [entry["guardrail_status"] for entry in entries] == ["guardrail_failed_to_respond"]
        assert "timed out" in str(entries[0]["guardrail_response"])

    @pytest.mark.asyncio
    async def test_mcp_tool_result_is_scanned_as_mcp_response(self, logging_only_guardrail):
        from mcp.types import CallToolResult, TextContent

        resp = _make_response({"should_block": False})
        result = CallToolResult(content=[TextContent(type="text", text="ssn 123-45-6789")])
        with patch.object(logging_only_guardrail.async_handler, "post", return_value=resp) as mock_post:
            await logging_only_guardrail.async_logging_hook(
                kwargs=self._logged_call(model="MCP: get_customer_record", messages=None),
                result=result,
                call_type="call_mcp_tool",
            )
        payloads = [call.kwargs["json"] for call in mock_post.call_args_list]
        assert [payload["guardrail_scope"] for payload in payloads] == ["mcp_response"]
        assert payloads[0]["tool_result"] == ["ssn 123-45-6789"]
        assert payloads[0]["model_name"] == "MCP: get_customer_record"

    def test_sync_logging_hook_never_calls_singulr(self, logging_only_guardrail):
        from concurrent.futures import ThreadPoolExecutor

        kwargs = {"messages": [{"role": "user", "content": "hi"}], "standard_logging_object": {}}

        def _run():
            with patch.object(logging_only_guardrail.async_handler, "post") as mock_post:
                returned = logging_only_guardrail.logging_hook(kwargs=kwargs, result=None, call_type="acompletion")
                mock_post.assert_not_called()
                return returned

        with ThreadPoolExecutor(max_workers=1) as pool:
            returned_kwargs, returned_result = pool.submit(_run).result()
        assert returned_result is None
        assert returned_kwargs == {"messages": [{"role": "user", "content": "hi"}], "standard_logging_object": {}}


class TestSingulrRequestWiring:
    @pytest.mark.asyncio
    async def test_sends_configured_timeout_and_calls_the_guard_endpoint(self):
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
        assert call_kwargs["url"] == "https://api.test.singulr.ai/api/v1/ai-gateway/litellm-v2"


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


class TestSingulrConfigModel:
    def test_ui_friendly_name(self):
        assert SingulrGuardrailConfigModel.ui_friendly_name() == "Singulr"

    def test_get_config_model_returns_singulr_config_model(self):
        assert SingulrGuardrail.get_config_model() is SingulrGuardrailConfigModel


class TestSingulrInitializer:
    def test_guardrail_initializer_registry_has_entry(self):
        from litellm.proxy.guardrails.guardrail_hooks.singulr import (
            initialize_guardrail,
        )

        assert callable(initialize_guardrail)

    def test_initialize_guardrail_reads_singulr_prefixed_fields(self):
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
