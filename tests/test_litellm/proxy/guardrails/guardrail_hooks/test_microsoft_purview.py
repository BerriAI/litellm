"""Unit tests for the Microsoft Purview DLP guardrail."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview.base import (
    PurviewGuardrailBase,
)
from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview.purview_dlp import (
    MicrosoftPurviewDLPGuardrail,
)


def _make_guardrail(**kwargs) -> MicrosoftPurviewDLPGuardrail:
    """Helper to construct a guardrail with test defaults."""
    defaults = {
        "guardrail_name": "test-purview",
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
    }
    defaults.update(kwargs)
    return MicrosoftPurviewDLPGuardrail(**defaults)


def _mock_token_response():
    """Mock a successful OAuth2 token response."""
    resp = Mock()
    resp.json.return_value = {
        "access_token": "mock-access-token",
        "expires_in": 3600,
    }
    return resp


def _mock_graph_response(policy_actions=None, protection_scope_state="unchanged"):
    """Mock a processContent Graph API response."""
    resp = Mock()
    body = {
        "protectionScopeState": protection_scope_state,
        "policyActions": policy_actions or [],
        "processingErrors": [],
    }
    resp.json.return_value = body
    resp.headers = {"ETag": "test-etag-123"}
    return resp


def _mock_scope_response():
    """Mock a protectionScopes/compute Graph API response."""
    resp = Mock()
    resp.json.return_value = {
        "value": [
            {
                "activities": "uploadText,downloadText",
                "executionMode": "evaluateInline",
                "policyActions": [],
            }
        ]
    }
    resp.headers = {"ETag": "scope-etag-123"}
    return resp


# ---------------------------------------------------------------
# _should_block
# ---------------------------------------------------------------


class TestShouldBlock:
    def test_empty_policy_actions(self):
        assert PurviewGuardrailBase._should_block({"policyActions": []}) is False

    def test_no_policy_actions_key(self):
        assert PurviewGuardrailBase._should_block({}) is False

    def test_restrict_access_block(self):
        response = {
            "policyActions": [
                {
                    "@odata.type": "#microsoft.graph.restrictAccessAction",
                    "action": "restrictAccess",
                    "restrictionAction": "block",
                }
            ]
        }
        assert PurviewGuardrailBase._should_block(response) is True

    def test_restrict_access_non_block(self):
        response = {
            "policyActions": [
                {
                    "@odata.type": "#microsoft.graph.restrictAccessAction",
                    "action": "restrictAccess",
                    "restrictionAction": "warn",
                }
            ]
        }
        assert PurviewGuardrailBase._should_block(response) is False

    def test_non_restrict_action(self):
        response = {
            "policyActions": [
                {
                    "@odata.type": "#microsoft.graph.auditAction",
                    "action": "audit",
                }
            ]
        }
        assert PurviewGuardrailBase._should_block(response) is False

    def test_multiple_actions_one_blocks(self):
        response = {
            "policyActions": [
                {"action": "audit"},
                {
                    "@odata.type": "#microsoft.graph.restrictAccessAction",
                    "action": "restrictAccess",
                    "restrictionAction": "block",
                },
            ]
        }
        assert PurviewGuardrailBase._should_block(response) is True


# ---------------------------------------------------------------
# completion prompt normalization (text completions API)
# ---------------------------------------------------------------


class TestCompletionPromptToStr:
    def test_string_prompt(self):
        assert PurviewGuardrailBase.completion_prompt_to_str("  hi  ") == "hi"

    def test_list_of_strings(self):
        assert PurviewGuardrailBase.completion_prompt_to_str(["a", "b"]) == "a\nb"

    def test_token_ids_returns_none(self):
        assert PurviewGuardrailBase.completion_prompt_to_str([1, 2, 3]) is None

    def test_empty(self):
        assert PurviewGuardrailBase.completion_prompt_to_str("") is None
        assert PurviewGuardrailBase.completion_prompt_to_str([]) is None


# ---------------------------------------------------------------
# User ID resolution
# ---------------------------------------------------------------


class TestResolveUserId:
    def test_from_metadata_when_no_auth_identity(self):
        guardrail = _make_guardrail()
        data = {"metadata": {"user_id": "entra-user-123"}}
        auth = UserAPIKeyAuth(api_key="test-key-no-user")
        assert guardrail._resolve_user_id(data, auth) == "entra-user-123"

    def test_authenticated_user_id_overrides_metadata(self):
        """Key user_id must win over spoofed metadata[user_id_field]."""
        guardrail = _make_guardrail()
        data = {"metadata": {"user_id": "spoofed-entra-id"}}
        auth = UserAPIKeyAuth(api_key="test", user_id="real-entra-id")
        assert guardrail._resolve_user_id(data, auth) == "real-entra-id"

    def test_user_api_key_metadata_before_custom_field(self):
        """Proxy-injected user_api_key_user_id wins over arbitrary metadata field."""
        guardrail = _make_guardrail(user_id_field="entra_id")
        data = {
            "metadata": {
                "user_api_key_user_id": "from-proxy-111",
                "entra_id": "metadata-222",
            }
        }
        auth = UserAPIKeyAuth(api_key="test")
        assert guardrail._resolve_user_id(data, auth) == "from-proxy-111"

    def test_custom_field_when_no_stronger_source(self):
        guardrail = _make_guardrail(user_id_field="entra_id")
        data = {"metadata": {"entra_id": "custom-user-456"}}
        auth = UserAPIKeyAuth(api_key="test")
        assert guardrail._resolve_user_id(data, auth) == "custom-user-456"

    def test_from_user_api_key_dict_user_id(self):
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test", user_id="key-user-789")
        assert guardrail._resolve_user_id({}, auth) == "key-user-789"

    def test_from_end_user_id(self):
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test", end_user_id="end-user-101")
        assert guardrail._resolve_user_id({}, auth) == "end-user-101"

    def test_end_user_id_after_key_user_id(self):
        """When both key user_id and end_user_id exist, key user_id is used first."""
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(
            api_key="test", user_id="key-owner", end_user_id="end-user-101"
        )
        assert guardrail._resolve_user_id({}, auth) == "key-owner"

    def test_none_when_missing(self):
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test")
        assert guardrail._resolve_user_id({}, auth) is None


# ---------------------------------------------------------------
# Pre-call hook
# ---------------------------------------------------------------


class TestPreCallHook:
    @pytest.mark.asyncio
    async def test_pre_call_allow(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={"messages": [{"role": "user", "content": "Hello, how are you?"}]},
                call_type="completion",
            )

            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["activity"] == "uploadText"
            assert mock_check.call_args.kwargs["block_on_violation"] is True

    @pytest.mark.asyncio
    async def test_pre_call_success_returns_request_data(self):
        """After a successful DLP pass, the hook must return the same data dict (not None)."""
        guardrail = _make_guardrail()
        payload = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "litellm_call_id": "call-abc",
        }

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            out = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data=payload,
                call_type="completion",
            )

            assert out is payload

    @pytest.mark.asyncio
    async def test_pre_call_block(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.side_effect = HTTPException(
                status_code=400,
                detail={"error": "Microsoft Purview DLP: Content blocked by policy"},
            )

            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(
                        api_key="test", user_id="user-123"
                    ),
                    cache=None,
                    data={
                        "messages": [
                            {
                                "role": "user",
                                "content": "SSN: 123-45-6789",
                            }
                        ]
                    },
                    call_type="completion",
                )

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_pre_call_no_user_id_raises(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data={"messages": [{"role": "user", "content": "Hello"}]},
                    call_type="completion",
                )

            mock_check.assert_not_called()
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_pre_call_no_messages_skips(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={},
                call_type="completion",
            )

            mock_check.assert_not_called()


class TestPreCallFullTranscript:
    @pytest.mark.asyncio
    async def test_pre_call_sends_all_message_roles_to_dlp(self):
        """DLP text must include system / prior turns, not only the last user block."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={
                    "messages": [
                        {"role": "system", "content": "SYSTEM_SENSITIVE"},
                        {"role": "user", "content": "EARLIER_USER"},
                        {"role": "assistant", "content": "reply"},
                        {"role": "user", "content": "final benign"},
                    ]
                },
                call_type="completion",
            )

            mock_check.assert_called_once()
            sent = mock_check.call_args.kwargs["text"]
            assert "SYSTEM_SENSITIVE" in sent
            assert "EARLIER_USER" in sent
            assert "final benign" in sent


# ---------------------------------------------------------------
# Post-call hook
# ---------------------------------------------------------------


class TestPostCallHook:
    @pytest.mark.asyncio
    async def test_post_call_allow(self):
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0, message=Message(content="Safe response", role="assistant")
                )
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            result = await guardrail.async_post_call_success_hook(
                data={},
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                response=response,
            )

            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["activity"] == "downloadText"
            assert result is response

    @pytest.mark.asyncio
    async def test_post_call_block(self):
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        content="Credit card: 4532-6677-8521-3500",
                        role="assistant",
                    ),
                )
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.side_effect = HTTPException(
                status_code=400,
                detail={"error": "Microsoft Purview DLP: Content blocked by policy"},
            )

            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_post_call_success_hook(
                    data={},
                    user_api_key_dict=UserAPIKeyAuth(
                        api_key="test", user_id="user-123"
                    ),
                    response=response,
                )

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_post_call_no_user_id_raises(self):
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(index=0, message=Message(content="Response", role="assistant"))
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_post_call_success_hook(
                    data={},
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=response,
                )

            mock_check.assert_not_called()
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_post_call_scans_all_choices(self):
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(content="First completion", role="assistant"),
                ),
                Choices(
                    index=1,
                    message=Message(content="Second completion body", role="assistant"),
                ),
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_post_call_success_hook(
                data={},
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                response=response,
            )

            mock_check.assert_called_once()
            combined = mock_check.call_args.kwargs["text"]
            assert "First completion" in combined
            assert "Second completion body" in combined


class TestTextCompletionHooks:
    @pytest.mark.asyncio
    async def test_pre_call_text_completion_uses_prompt(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={"prompt": "Completions API prompt body"},
                call_type="text_completion",
            )

            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["text"] == "Completions API prompt body"
            assert mock_check.call_args.kwargs["activity"] == "uploadText"

    @pytest.mark.asyncio
    async def test_post_call_text_completion_all_choices(self):
        from litellm.types.utils import TextChoices, TextCompletionResponse

        guardrail = _make_guardrail()
        response = TextCompletionResponse(
            model="gpt-3.5-turbo-instruct",
            choices=[
                TextChoices(text="alpha", index=0),
                TextChoices(text="beta", index=1),
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_post_call_success_hook(
                data={},
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                response=response,
            )

            mock_check.assert_called_once()
            combined = mock_check.call_args.kwargs["text"]
            assert "alpha" in combined
            assert "beta" in combined


# ---------------------------------------------------------------
# Responses API hooks
# ---------------------------------------------------------------


class TestResponsesAPIHooks:
    @pytest.mark.asyncio
    async def test_pre_call_responses_api_string_input(self):
        """Pre-call hook must scan plain-string ``input`` on responses call type."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={"input": "SSN: 123-45-6789"},
                call_type="responses",
            )

            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["activity"] == "uploadText"
            assert "SSN: 123-45-6789" in mock_check.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_pre_call_aresponses_string_input(self):
        """Pre-call hook must scan ``input`` on ``aresponses`` call type too."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={"input": "sensitive content"},
                call_type="aresponses",
            )

            mock_check.assert_called_once()
            assert "sensitive content" in mock_check.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_pre_call_responses_api_list_input(self):
        """Pre-call hook must extract text from structured list ``input``."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={
                    "input": [{"role": "user", "content": "Secret phrase: alpha bravo"}]
                },
                call_type="responses",
            )

            mock_check.assert_called_once()
            assert "Secret phrase: alpha bravo" in mock_check.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_pre_call_responses_api_no_input_skips(self):
        """Pre-call hook must not call _check_content when ``input`` is absent."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={},
                call_type="responses",
            )

            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_pre_call_responses_string_input_includes_instructions(self):
        """Benign string ``input`` must still scan ``instructions`` (system message)."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={
                    "input": "benign user text",
                    "instructions": "SYSTEM_SENSITIVE in instructions",
                },
                call_type="responses",
            )

            mock_check.assert_called_once()
            sent = mock_check.call_args.kwargs["text"]
            assert "benign user text" in sent
            assert "SYSTEM_SENSITIVE in instructions" in sent

    @pytest.mark.asyncio
    async def test_pre_call_responses_instructions_only(self):
        """Requests with only ``instructions`` (no ``input``) must still be scanned."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={"instructions": "policy text in instructions only"},
                call_type="responses",
            )

            mock_check.assert_called_once()
            assert (
                "policy text in instructions only"
                in mock_check.call_args.kwargs["text"]
            )

    @pytest.mark.asyncio
    async def test_post_call_responses_api_output_text(self):
        """Post-call hook must scan text from ``ResponsesAPIResponse.output``."""
        from litellm.types.llms.openai import ResponsesAPIResponse

        guardrail = _make_guardrail()
        response = ResponsesAPIResponse(
            id="resp-1",
            created_at=0,
            output=[
                {
                    "type": "message",
                    "id": "msg-1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "card 4111-1111-1111-1111"}
                    ],
                }
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            result = await guardrail.async_post_call_success_hook(
                data={},
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                response=response,
            )

            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["activity"] == "downloadText"
            assert "card 4111-1111-1111-1111" in mock_check.call_args.kwargs["text"]
            assert result is response

    @pytest.mark.asyncio
    async def test_post_call_responses_api_empty_output_skips(self):
        """Post-call hook must not call _check_content when output has no text."""
        from litellm.types.llms.openai import ResponsesAPIResponse

        guardrail = _make_guardrail()
        response = ResponsesAPIResponse(
            id="resp-2",
            created_at=0,
            output=[],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            await guardrail.async_post_call_success_hook(
                data={},
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                response=response,
            )

            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_logging_hook_responses_api_input_and_output(self):
        """Logging hook must scan both ``input`` and ``ResponsesAPIResponse.output``."""
        from litellm.types.llms.openai import ResponsesAPIResponse

        guardrail = _make_guardrail()
        result_response = ResponsesAPIResponse(
            id="resp-3",
            created_at=0,
            output=[
                {
                    "type": "message",
                    "id": "msg-2",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "response body"}],
                }
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_logging_hook(
                kwargs={
                    "input": "prompt body",
                    "litellm_params": {
                        "metadata": {"user_api_key_user_id": "user-123"}
                    },
                },
                result=result_response,
                call_type="responses",
            )

            assert mock_check.call_count == 2
            activities = {c.kwargs["activity"] for c in mock_check.call_args_list}
            assert activities == {"uploadText", "downloadText"}
            texts = {c.kwargs["text"] for c in mock_check.call_args_list}
            assert any("prompt body" in t for t in texts)
            assert any("response body" in t for t in texts)

    @pytest.mark.asyncio
    async def test_logging_hook_responses_api_with_messages_key_set(self):
        """Responses-API prompt audit must fire even when ``kwargs["messages"]`` is
        also set to the raw responses input.

        litellm's logging pipeline (``function_setup`` +
        ``update_environment_variables``) stores the raw responses ``input``
        under ``model_call_details["messages"]``.  The audit must still extract
        the prompt via the responses-specific path, not silently fall through
        the generic ``messages`` branch with the wrong format.
        """
        from litellm.types.llms.openai import ResponsesAPIResponse

        guardrail = _make_guardrail()
        result_response = ResponsesAPIResponse(
            id="resp-msgkey",
            created_at=0,
            output=[
                {
                    "type": "message",
                    "id": "msg-3",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "response body"}],
                }
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_logging_hook(
                kwargs={
                    "input": "prompt body",
                    "instructions": "system instructions",
                    # Simulate litellm's logging path which mirrors the raw
                    # responses input under "messages".
                    "messages": "prompt body",
                    "litellm_params": {
                        "metadata": {"user_api_key_user_id": "user-123"}
                    },
                },
                result=result_response,
                call_type="aresponses",
            )

            assert mock_check.call_count == 2
            activities = {c.kwargs["activity"] for c in mock_check.call_args_list}
            assert activities == {"uploadText", "downloadText"}
            upload_calls = [
                c
                for c in mock_check.call_args_list
                if c.kwargs["activity"] == "uploadText"
            ]
            assert len(upload_calls) == 1
            upload_text = upload_calls[0].kwargs["text"]
            assert "prompt body" in upload_text
            assert "system instructions" in upload_text


# ---------------------------------------------------------------
# Logging hook user resolution
# ---------------------------------------------------------------


class TestLoggingResolveUserId:
    def test_logging_prefers_user_api_key_user_id_in_metadata(self):
        guardrail = _make_guardrail()
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "user_api_key_user_id": "trusted-from-proxy",
                    "user_id": "metadata-spoof",
                }
            }
        }
        assert (
            guardrail._resolve_user_id_from_logging_kwargs(kwargs)
            == "trusted-from-proxy"
        )

    def test_logging_ignores_caller_supplied_user_id_field(self):
        """Caller-controlled ``metadata[user_id_field]`` must not drive Purview audit attribution."""
        guardrail = _make_guardrail()
        kwargs = {"litellm_params": {"metadata": {"user_id": "only-metadata-user"}}}
        assert guardrail._resolve_user_id_from_logging_kwargs(kwargs) is None

    def test_logging_kwargs_level_user_api_key_user_id(self):
        """Top-level ``kwargs["user_api_key_user_id"]`` is also a proxy-injected source."""
        guardrail = _make_guardrail()
        kwargs = {
            "user_api_key_user_id": "from-top-level",
            "litellm_params": {"metadata": {}},
        }
        assert (
            guardrail._resolve_user_id_from_logging_kwargs(kwargs) == "from-top-level"
        )

    def test_logging_returns_none_when_no_trusted_identity(self):
        guardrail = _make_guardrail()
        kwargs = {"litellm_params": {"metadata": {}}}
        assert guardrail._resolve_user_id_from_logging_kwargs(kwargs) is None


# ---------------------------------------------------------------
# _check_content — integration-level
# ---------------------------------------------------------------


class TestCheckContent:
    @pytest.mark.asyncio
    async def test_check_content_allow(self):
        guardrail = _make_guardrail()

        with (
            patch.object(
                guardrail,
                "_compute_protection_scopes",
                new_callable=AsyncMock,
                return_value=("etag-1", {}),
            ),
            patch.object(
                guardrail,
                "_process_content",
                new_callable=AsyncMock,
                return_value={
                    "protectionScopeState": "unchanged",
                    "policyActions": [],
                },
            ),
        ):
            result = await guardrail._check_content(
                user_id="user-1",
                text="Hello world",
                activity="uploadText",
                request_data={},
                block_on_violation=True,
            )

            assert result["policyActions"] == []

    @pytest.mark.asyncio
    async def test_check_content_block(self):
        guardrail = _make_guardrail()

        with (
            patch.object(
                guardrail,
                "_compute_protection_scopes",
                new_callable=AsyncMock,
                return_value=("etag-1", {}),
            ),
            patch.object(
                guardrail,
                "_process_content",
                new_callable=AsyncMock,
                return_value={
                    "protectionScopeState": "unchanged",
                    "policyActions": [
                        {
                            "@odata.type": "#microsoft.graph.restrictAccessAction",
                            "action": "restrictAccess",
                            "restrictionAction": "block",
                        }
                    ],
                },
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await guardrail._check_content(
                    user_id="user-1",
                    text="SSN: 123-45-6789",
                    activity="uploadText",
                    request_data={},
                    block_on_violation=True,
                )

            assert exc_info.value.status_code == 400
            assert "blocked by policy" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_check_content_logging_only_no_block(self):
        """In logging_only mode, violations should NOT raise."""
        guardrail = _make_guardrail()

        with (
            patch.object(
                guardrail,
                "_compute_protection_scopes",
                new_callable=AsyncMock,
                return_value=("etag-1", {}),
            ),
            patch.object(
                guardrail,
                "_process_content",
                new_callable=AsyncMock,
                return_value={
                    "protectionScopeState": "unchanged",
                    "policyActions": [
                        {
                            "@odata.type": "#microsoft.graph.restrictAccessAction",
                            "action": "restrictAccess",
                            "restrictionAction": "block",
                        }
                    ],
                },
            ),
        ):
            # Should NOT raise even though violation detected
            result = await guardrail._check_content(
                user_id="user-1",
                text="SSN: 123-45-6789",
                activity="uploadText",
                request_data={},
                block_on_violation=False,
            )

            assert len(result["policyActions"]) == 1


# ---------------------------------------------------------------
# Token caching
# ---------------------------------------------------------------


class TestTokenCaching:
    @pytest.mark.asyncio
    async def test_token_cached(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail.async_handler, "post", return_value=_mock_token_response()
        ) as mock_post:
            token1 = await guardrail._get_access_token()
            token2 = await guardrail._get_access_token()

            assert token1 == "mock-access-token"
            assert token2 == "mock-access-token"
            # Should only call the token endpoint once (cached)
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_token_refreshed_on_expiry(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail.async_handler, "post", return_value=_mock_token_response()
        ) as mock_post:
            await guardrail._get_access_token()

            # Expire the token
            guardrail._token_cache = ("old-token", time.time() - 10)

            await guardrail._get_access_token()

            # Should have called token endpoint twice
            assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_token_http_error_propagates(self):
        """OAuth2 4xx/5xx responses must surface as HTTPStatusError, not KeyError."""
        guardrail = _make_guardrail()

        error_resp = Mock()
        error_resp.json.return_value = {
            "error": "invalid_client",
            "error_description": "client secret is wrong",
        }
        error_resp.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "401 Unauthorized",
                request=httpx.Request("POST", "https://login.microsoftonline.com/"),
                response=httpx.Response(401),
            )
        )

        with patch.object(guardrail.async_handler, "post", return_value=error_resp):
            with pytest.raises(httpx.HTTPStatusError):
                await guardrail._get_access_token()

            # Failure must not poison the cache.
            assert guardrail._token_cache is None


# ---------------------------------------------------------------
# Graph POST HTTP error propagation
# ---------------------------------------------------------------


class TestGraphPostHttpError:
    @pytest.mark.asyncio
    async def test_graph_post_http_error_propagates(self):
        """Non-2xx Graph API responses must raise rather than return error JSON."""
        guardrail = _make_guardrail()

        error_resp = Mock()
        error_resp.json.return_value = {
            "error": {"code": "Forbidden", "message": "no access"}
        }
        error_resp.headers = {}
        error_resp.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "403 Forbidden",
                request=httpx.Request("POST", "https://graph.microsoft.com/"),
                response=httpx.Response(403),
            )
        )

        with (
            patch.object(
                guardrail, "_get_access_token", new_callable=AsyncMock
            ) as mock_token,
            patch.object(guardrail.async_handler, "post", return_value=error_resp),
        ):
            mock_token.return_value = "mock-token"

            with pytest.raises(httpx.HTTPStatusError):
                await guardrail._graph_post(
                    "https://graph.microsoft.com/v1.0/users/u/example",
                    {"foo": "bar"},
                )

    @pytest.mark.asyncio
    async def test_compute_protection_scopes_http_error_propagates(self):
        """A Graph error on protectionScopes/compute must not be cached as success."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_graph_post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=httpx.Request("POST", "https://graph.microsoft.com/"),
                response=httpx.Response(429),
            )

            with pytest.raises(httpx.HTTPStatusError):
                await guardrail._compute_protection_scopes("user-err")

            # The failed compute must not populate the scope cache.
            assert "user-err" not in guardrail._scope_cache


# ---------------------------------------------------------------
# Protection scope caching
# ---------------------------------------------------------------


class TestScopeCaching:
    @pytest.mark.asyncio
    async def test_scope_cached(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_graph_post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = (
                {
                    "value": [
                        {"activities": "uploadText", "executionMode": "evaluateInline"}
                    ]
                },
                {"ETag": "scope-etag"},
            )

            etag1, _ = await guardrail._compute_protection_scopes("user-1")
            etag2, _ = await guardrail._compute_protection_scopes("user-1")

            assert etag1 == "scope-etag"
            assert etag2 == "scope-etag"
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_scope_cache_lru_keeps_hot_user_on_eviction(self):
        """Frequently accessed users should not be evicted before cold entries."""
        guardrail = _make_guardrail()
        guardrail._scope_cache_maxsize = 3

        scope_payload = (
            {
                "value": [
                    {"activities": "uploadText", "executionMode": "evaluateInline"}
                ]
            },
            {"ETag": "scope-etag"},
        )

        with patch.object(
            guardrail, "_graph_post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = scope_payload

            await guardrail._compute_protection_scopes("user-a")
            await guardrail._compute_protection_scopes("user-b")
            await guardrail._compute_protection_scopes("user-c")
            assert mock_post.call_count == 3

            await guardrail._compute_protection_scopes("user-a")
            assert mock_post.call_count == 3

            await guardrail._compute_protection_scopes("user-d")
            assert mock_post.call_count == 4

            await guardrail._compute_protection_scopes("user-a")
            assert mock_post.call_count == 4
            assert "user-a" in guardrail._scope_cache
            assert "user-b" not in guardrail._scope_cache

    @pytest.mark.asyncio
    async def test_scope_cache_refresh_moves_to_end_of_lru(self):
        """Refreshing a stale entry must move it to the MRU end of the OrderedDict.

        Before the fix, OrderedDict.__setitem__ preserved the original insertion
        position for existing keys, causing the just-refreshed entry to be the
        next candidate for LRU eviction.
        """
        guardrail = _make_guardrail()
        guardrail._scope_cache_maxsize = 2

        scope_payload = (
            {"value": []},
            {"ETag": "scope-etag"},
        )

        with patch.object(
            guardrail, "_graph_post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = scope_payload

            # Populate cache: user-a (older), user-b (newer)
            await guardrail._compute_protection_scopes("user-a")
            await guardrail._compute_protection_scopes("user-b")
            assert mock_post.call_count == 2

            # Expire user-a's entry so it is re-fetched on the next access.
            old_etag, old_scope, _ = guardrail._scope_cache["user-a"]
            guardrail._scope_cache["user-a"] = (old_etag, old_scope, 0.0)

            # Re-fetch user-a — should move it to the MRU end.
            await guardrail._compute_protection_scopes("user-a")
            assert mock_post.call_count == 3

            # Adding a third user must evict user-b (the true LRU), not user-a.
            await guardrail._compute_protection_scopes("user-c")
            assert mock_post.call_count == 4

            assert "user-a" in guardrail._scope_cache, "user-a was wrongly evicted"
            assert (
                "user-b" not in guardrail._scope_cache
            ), "user-b should have been evicted"
            assert "user-c" in guardrail._scope_cache

    @pytest.mark.asyncio
    async def test_scope_invalidated_on_modified(self):
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_graph_post", new_callable=AsyncMock
        ) as mock_post:
            # First call: compute scopes
            mock_post.return_value = (
                {"value": []},
                {"ETag": "etag-1"},
            )
            await guardrail._compute_protection_scopes("user-1")

            # processContent returns modified
            mock_post.return_value = (
                {"protectionScopeState": "modified", "policyActions": []},
                {},
            )
            await guardrail._process_content("user-1", "text", "uploadText", "etag-1")

            # Scope cache should be invalidated
            assert "user-1" not in guardrail._scope_cache


# ---------------------------------------------------------------
# get_prompt_text_for_dlp — message separator
# ---------------------------------------------------------------


class TestGetPromptTextForDlp:
    def test_single_message_no_extra_separator(self):
        """A single message is returned as-is (no leading/trailing separator)."""
        guardrail = _make_guardrail()
        result = guardrail.get_prompt_text_for_dlp(
            [{"role": "user", "content": "Hello"}]
        )
        assert result == "Hello"

    def test_messages_separated_by_double_newline(self):
        """Adjacent messages must NOT be concatenated without a separator.

        Before the fix, "end of msg1" + "start of msg2" became
        "end of msg1start of msg2", mangling DLP pattern detection.
        """
        guardrail = _make_guardrail()
        result = guardrail.get_prompt_text_for_dlp(
            [
                {"role": "system", "content": "end of msg1"},
                {"role": "user", "content": "start of msg2"},
            ]
        )
        assert result is not None
        assert "end of msg1" in result
        assert "start of msg2" in result
        # Separator must be present between messages
        assert "end of msg1start of msg2" not in result
        assert "end of msg1\n\nstart of msg2" in result

    def test_empty_messages_returns_none(self):
        guardrail = _make_guardrail()
        assert guardrail.get_prompt_text_for_dlp([]) is None

    def test_whitespace_only_messages_skipped(self):
        guardrail = _make_guardrail()
        result = guardrail.get_prompt_text_for_dlp(
            [
                {"role": "system", "content": "   "},
                {"role": "user", "content": "real content"},
            ]
        )
        assert result == "real content"

    def test_multi_role_conversation_preserves_all_content(self):
        guardrail = _make_guardrail()
        result = guardrail.get_prompt_text_for_dlp(
            [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "USER1"},
                {"role": "assistant", "content": "ASSISTANT"},
                {"role": "user", "content": "USER2"},
            ]
        )
        assert result is not None
        for token in ("SYSTEM", "USER1", "ASSISTANT", "USER2"):
            assert token in result


# ---------------------------------------------------------------
# logging_hook — non-blocking fire-and-forget
# ---------------------------------------------------------------


class TestLoggingHookNonBlocking:
    @pytest.mark.asyncio
    async def test_logging_hook_does_not_block_running_loop(self):
        """logging_hook must return immediately without blocking the event loop.

        Before the fix, logging_hook called future.result() which blocked the
        event loop thread for the full round-trip of the two Graph API calls.
        """
        guardrail = _make_guardrail()
        call_count = 0

        async def slow_async_hook(**_kwargs):
            nonlocal call_count
            await asyncio.sleep(0.05)
            call_count += 1
            return _kwargs.get("kwargs", {}), _kwargs.get("result")

        with patch.object(guardrail, "async_logging_hook", side_effect=slow_async_hook):
            # Call logging_hook from within a running event loop
            result = guardrail.logging_hook(
                kwargs={"messages": [{"role": "user", "content": "test"}]},
                result=None,
                call_type="completion",
            )

        # Must return (kwargs, result) unchanged without waiting for async work
        assert result[0]["messages"][0]["content"] == "test"
        assert result[1] is None

    def test_logging_hook_returns_original_kwargs_and_result(self):
        """Return value must be the original (kwargs, result) tuple unchanged."""
        guardrail = _make_guardrail()
        kwargs = {"messages": [{"role": "user", "content": "hello"}]}
        result_obj = {"some": "result"}

        with patch.object(
            guardrail,
            "async_logging_hook",
            new_callable=AsyncMock,
            return_value=(kwargs, result_obj),
        ):
            out = guardrail.logging_hook(
                kwargs=kwargs,
                result=result_obj,
                call_type="completion",
            )

        assert out == (kwargs, result_obj)


# ---------------------------------------------------------------
# Initializer validation
# ---------------------------------------------------------------


class TestInitializerValidation:
    def test_missing_tenant_id(self):
        from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
            initialize_guardrail,
        )

        litellm_params = Mock(
            spec=[
                "tenant_id",
                "client_id",
                "client_secret",
                "purview_app_name",
                "user_id_field",
                "api_key",
                "mode",
                "default_on",
            ]
        )
        litellm_params.tenant_id = None
        litellm_params.client_id = None
        litellm_params.client_secret = None
        litellm_params.api_key = "secret"
        litellm_params.mode = "pre_call"

        with pytest.raises(ValueError, match="tenant_id is required"):
            initialize_guardrail(litellm_params, {"guardrail_name": "test"})

    def test_missing_client_id(self):
        from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
            initialize_guardrail,
        )

        litellm_params = Mock(
            spec=[
                "tenant_id",
                "client_id",
                "client_secret",
                "purview_app_name",
                "user_id_field",
                "api_key",
                "mode",
                "default_on",
            ]
        )
        litellm_params.tenant_id = "test-tenant"
        litellm_params.client_id = None
        litellm_params.client_secret = None
        litellm_params.api_key = "secret"
        litellm_params.mode = "pre_call"

        with pytest.raises(ValueError, match="client_id is required"):
            initialize_guardrail(litellm_params, {"guardrail_name": "test"})

    def test_missing_client_secret(self):
        from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
            initialize_guardrail,
        )

        litellm_params = Mock(
            spec=[
                "tenant_id",
                "client_id",
                "client_secret",
                "purview_app_name",
                "user_id_field",
                "api_key",
                "mode",
                "default_on",
            ]
        )
        litellm_params.tenant_id = "test-tenant"
        litellm_params.client_id = "test-client"
        litellm_params.client_secret = None
        litellm_params.api_key = None
        litellm_params.mode = "pre_call"

        with pytest.raises(ValueError, match="client_secret"):
            initialize_guardrail(litellm_params, {"guardrail_name": "test"})


# ---------------------------------------------------------------
# _check_content — API error handling with block_on_violation=False
# ---------------------------------------------------------------


class TestCheckContentApiErrorHandling:
    @pytest.mark.asyncio
    async def test_api_error_reraises_when_block_on_violation_true(self):
        """API/network errors must surface as HTTPException(400) when block_on_violation=True."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail,
            "_compute_protection_scopes",
            new_callable=AsyncMock,
            side_effect=RuntimeError("network failure"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await guardrail._check_content(
                    user_id="user-1",
                    text="some content",
                    activity="uploadText",
                    request_data={},
                    block_on_violation=True,
                )

        assert exc_info.value.status_code == 400
        assert isinstance(exc_info.value.detail, dict)
        assert "upstream policy evaluation failed" in exc_info.value.detail.get(
            "error", ""
        )
        assert "network failure" in exc_info.value.detail.get("exception", "")
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_http_exception_passes_through_unchanged(self):
        """HTTPException from upstream layers must propagate as-is (not wrapped)."""
        guardrail = _make_guardrail()
        inner = HTTPException(status_code=403, detail="forbidden")

        with patch.object(
            guardrail,
            "_compute_protection_scopes",
            new_callable=AsyncMock,
            side_effect=inner,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await guardrail._check_content(
                    user_id="user-1",
                    text="some content",
                    activity="uploadText",
                    request_data={},
                    block_on_violation=True,
                )

        assert exc_info.value is inner

    @pytest.mark.asyncio
    async def test_api_error_not_reraised_when_block_on_violation_false(self):
        """API/network errors must be swallowed (logged only) when block_on_violation=False."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail,
            "_compute_protection_scopes",
            new_callable=AsyncMock,
            side_effect=RuntimeError("network failure"),
        ):
            # Must NOT raise — should return empty dict
            result = await guardrail._check_content(
                user_id="user-1",
                text="some content",
                activity="uploadText",
                request_data={},
                block_on_violation=False,
            )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_process_content_error_not_reraised_when_block_on_violation_false(
        self,
    ):
        """Errors from _process_content itself must also be suppressed in logging-only mode."""
        guardrail = _make_guardrail()

        with (
            patch.object(
                guardrail,
                "_compute_protection_scopes",
                new_callable=AsyncMock,
                return_value=("etag-1", {}),
            ),
            patch.object(
                guardrail,
                "_process_content",
                new_callable=AsyncMock,
                side_effect=ConnectionError("timeout"),
            ),
        ):
            result = await guardrail._check_content(
                user_id="user-1",
                text="some content",
                activity="uploadText",
                request_data={},
                block_on_violation=False,
            )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_http_status_error_preserves_upstream_status_code(self):
        """Upstream Graph 429 must surface as 429 with Retry-After (not a generic 400)."""
        guardrail = _make_guardrail()
        upstream_resp = httpx.Response(
            status_code=429,
            headers={"Retry-After": "30"},
            request=httpx.Request("POST", "https://graph.microsoft.com/v1.0/x"),
        )
        upstream_err = httpx.HTTPStatusError(
            "rate limited", request=upstream_resp.request, response=upstream_resp
        )

        with patch.object(
            guardrail,
            "_compute_protection_scopes",
            new_callable=AsyncMock,
            side_effect=upstream_err,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await guardrail._check_content(
                    user_id="user-1",
                    text="some content",
                    activity="uploadText",
                    request_data={},
                    block_on_violation=True,
                )

        assert exc_info.value.status_code == 429
        assert exc_info.value.headers == {"Retry-After": "30"}
        assert isinstance(exc_info.value.detail, dict)
        assert exc_info.value.detail.get("upstream_status") == 429
        assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)

    @pytest.mark.asyncio
    async def test_http_status_error_401_maps_to_502(self):
        """Upstream 401/403 (proxy creds problem) should be exposed as 502, not 401/403."""
        guardrail = _make_guardrail()
        upstream_resp = httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "https://graph.microsoft.com/v1.0/x"),
        )
        upstream_err = httpx.HTTPStatusError(
            "unauthorized", request=upstream_resp.request, response=upstream_resp
        )

        with patch.object(
            guardrail,
            "_compute_protection_scopes",
            new_callable=AsyncMock,
            side_effect=upstream_err,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await guardrail._check_content(
                    user_id="user-1",
                    text="some content",
                    activity="uploadText",
                    request_data={},
                    block_on_violation=True,
                )

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail.get("upstream_status") == 401


# ---------------------------------------------------------------
# async_logging_hook — independent prompt/response audit calls
# ---------------------------------------------------------------


class TestAsyncLoggingHookIndependence:
    @pytest.mark.asyncio
    async def test_response_audit_runs_even_if_prompt_audit_fails(self):
        """A failure in the prompt audit must not prevent the response audit from running."""
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(content="response text", role="assistant"),
                )
            ],
        )

        call_activities: list = []

        async def fake_check_content(**kwargs):
            activity = kwargs.get("activity")
            if activity == "uploadText":
                raise RuntimeError("simulated prompt API failure")
            call_activities.append(activity)
            return {"policyActions": []}

        with patch.object(guardrail, "_check_content", side_effect=fake_check_content):
            await guardrail.async_logging_hook(
                kwargs={
                    "messages": [{"role": "user", "content": "prompt"}],
                    "litellm_params": {
                        "metadata": {"user_api_key_user_id": "user-123"}
                    },
                },
                result=response,
                call_type="completion",
            )

        # The response audit must still have been attempted
        assert "downloadText" in call_activities

    @pytest.mark.asyncio
    async def test_prompt_audit_runs_even_if_response_audit_fails(self):
        """A failure in the response audit must not affect the prompt audit result."""
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(content="response text", role="assistant"),
                )
            ],
        )

        call_activities: list = []

        async def fake_check_content(**kwargs):
            activity = kwargs.get("activity")
            if activity == "downloadText":
                raise RuntimeError("simulated response API failure")
            call_activities.append(activity)
            return {"policyActions": []}

        with patch.object(guardrail, "_check_content", side_effect=fake_check_content):
            await guardrail.async_logging_hook(
                kwargs={
                    "messages": [{"role": "user", "content": "prompt"}],
                    "litellm_params": {
                        "metadata": {"user_api_key_user_id": "user-123"}
                    },
                },
                result=response,
                call_type="completion",
            )

        assert "uploadText" in call_activities

    @pytest.mark.asyncio
    async def test_logging_hook_returns_original_when_both_audits_fail(self):
        """async_logging_hook must always return (kwargs, result) even if both audits fail."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail,
            "_check_content",
            new_callable=AsyncMock,
            side_effect=RuntimeError("total failure"),
        ):
            kwargs = {
                "messages": [{"role": "user", "content": "prompt"}],
                "litellm_params": {"metadata": {"user_api_key_user_id": "user-123"}},
            }
            result_obj = {"some": "result"}
            out_kwargs, out_result = await guardrail.async_logging_hook(
                kwargs=kwargs,
                result=result_obj,
                call_type="completion",
            )

        assert out_kwargs is kwargs
        assert out_result is result_obj


# ---------------------------------------------------------------
# Tool-call argument extraction
# ---------------------------------------------------------------


class TestExtractToolCallArgs:
    def test_dict_message_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"arguments": '{"ssn": "123-45-6789"}'}},
                {"function": {"arguments": '{"card": "4111-1111-1111-1111"}'}},
            ],
        }
        args = MicrosoftPurviewDLPGuardrail._extract_tool_call_args_from_message(msg)
        assert '{"ssn": "123-45-6789"}' in args
        assert '{"card": "4111-1111-1111-1111"}' in args

    def test_dict_message_with_function_call(self):
        msg = {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "lookup", "arguments": '{"query": "secret"}'},
        }
        args = MicrosoftPurviewDLPGuardrail._extract_tool_call_args_from_message(msg)
        assert '{"query": "secret"}' in args

    def test_object_message_with_tool_calls(self):
        from litellm.types.utils import Message

        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "fn", "arguments": '{"x": 1}'},
                },
            ],
        )
        args = MicrosoftPurviewDLPGuardrail._extract_tool_call_args_from_message(msg)
        assert '{"x": 1}' in args

    def test_message_with_no_tool_calls(self):
        msg = {"role": "user", "content": "hello"}
        args = MicrosoftPurviewDLPGuardrail._extract_tool_call_args_from_message(msg)
        assert args == []

    def test_empty_arguments_skipped(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"arguments": "  "}}],
        }
        args = MicrosoftPurviewDLPGuardrail._extract_tool_call_args_from_message(msg)
        assert args == []


# ---------------------------------------------------------------
# Tool-call arguments included in DLP text extraction (prompt)
# ---------------------------------------------------------------


class TestGetPromptTextToolCalls:
    def test_tool_call_args_included_in_prompt_scan(self):
        """Sensitive data in tool_calls[].function.arguments must appear in DLP text."""
        guardrail = _make_guardrail()
        messages = [
            {"role": "user", "content": "benign query"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"ssn": "123-45-6789"}',
                        },
                    }
                ],
            },
        ]
        text = guardrail.get_prompt_text_for_dlp(messages)
        assert text is not None
        assert "benign query" in text
        assert '{"ssn": "123-45-6789"}' in text

    def test_function_call_args_included_in_prompt_scan(self):
        """Legacy function_call.arguments must also appear in DLP text."""
        guardrail = _make_guardrail()
        messages = [
            {
                "role": "assistant",
                "content": "Calling function",
                "function_call": {
                    "name": "search",
                    "arguments": '{"credit_card": "4111-1111-1111-1111"}',
                },
            }
        ]
        text = guardrail.get_prompt_text_for_dlp(messages)
        assert text is not None
        assert "Calling function" in text
        assert '{"credit_card": "4111-1111-1111-1111"}' in text

    def test_content_only_message_unchanged(self):
        """Messages without tool calls must still produce the same output."""
        guardrail = _make_guardrail()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me a joke."},
        ]
        text = guardrail.get_prompt_text_for_dlp(messages)
        assert text is not None
        assert "You are helpful." in text
        assert "Tell me a joke." in text

    @pytest.mark.asyncio
    async def test_pre_call_hook_scans_tool_call_args(self):
        """async_pre_call_hook must include tool_call arguments in the text sent to Purview."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_pre_call_hook(
                user_api_key_dict=__import__(
                    "litellm.proxy._types", fromlist=["UserAPIKeyAuth"]
                ).UserAPIKeyAuth(api_key="test", user_id="user-123"),
                cache=None,
                data={
                    "messages": [
                        {"role": "user", "content": "benign"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "tc1",
                                    "type": "function",
                                    "function": {
                                        "name": "do_thing",
                                        "arguments": '{"password": "hunter2"}',
                                    },
                                }
                            ],
                        },
                    ]
                },
                call_type="completion",
            )

            mock_check.assert_called_once()
            sent_text = mock_check.call_args.kwargs["text"]
            assert '{"password": "hunter2"}' in sent_text


# ---------------------------------------------------------------
# Tool-call arguments included in DLP text extraction (response)
# ---------------------------------------------------------------


class TestCompletionResponseTextPartsToolCalls:
    def test_response_tool_call_args_included(self):
        """Model-generated tool_call arguments must appear in the DLP scan text."""
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {
                                    "name": "exfil",
                                    "arguments": '{"data": "secret-value"}',
                                },
                            }
                        ],
                    ),
                )
            ],
        )
        parts = guardrail._completion_response_text_parts(response)
        assert any("secret-value" in p for p in parts)

    def test_response_with_content_and_tool_calls(self):
        """Both message content and tool_call arguments must be included."""
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="Here is the result",
                        tool_calls=[
                            {
                                "id": "tc2",
                                "type": "function",
                                "function": {
                                    "name": "fn",
                                    "arguments": '{"ssn": "123-45-6789"}',
                                },
                            }
                        ],
                    ),
                )
            ],
        )
        parts = guardrail._completion_response_text_parts(response)
        combined = " ".join(parts)
        assert "Here is the result" in combined
        assert '{"ssn": "123-45-6789"}' in combined

    @pytest.mark.asyncio
    async def test_post_call_hook_scans_response_tool_call_args(self):
        """async_post_call_success_hook must send tool_call arguments to Purview."""
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()
        response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": "tc3",
                                "type": "function",
                                "function": {
                                    "name": "retrieve",
                                    "arguments": '{"credit_card": "4111-1111-1111-1111"}',
                                },
                            }
                        ],
                    ),
                )
            ],
        )

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}

            await guardrail.async_post_call_success_hook(
                data={},
                user_api_key_dict=__import__(
                    "litellm.proxy._types", fromlist=["UserAPIKeyAuth"]
                ).UserAPIKeyAuth(api_key="test", user_id="user-123"),
                response=response,
            )

            mock_check.assert_called_once()
            sent_text = mock_check.call_args.kwargs["text"]
            assert '{"credit_card": "4111-1111-1111-1111"}' in sent_text

    def test_responses_api_function_call_args_included(self):
        """Function-call arguments in ``ResponsesAPIResponse.output`` must be DLP-scanned."""
        from litellm.types.llms.openai import ResponsesAPIResponse

        guardrail = _make_guardrail()
        response = ResponsesAPIResponse(
            id="resp-tc-1",
            created_at=0,
            output=[
                {
                    "type": "message",
                    "id": "msg-tc-1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "calling tool"}],
                },
                {
                    "type": "function_call",
                    "id": "fc-1",
                    "call_id": "call-1",
                    "name": "exfil",
                    "arguments": '{"ssn": "123-45-6789"}',
                },
            ],
        )
        parts = guardrail._completion_response_text_parts(response)
        combined = " ".join(parts)
        assert "calling tool" in combined
        assert '{"ssn": "123-45-6789"}' in combined

    def test_responses_api_function_call_args_only(self):
        """Function-call args must be scanned even when no ``output_text`` blocks exist."""
        from litellm.types.llms.openai import ResponsesAPIResponse

        guardrail = _make_guardrail()
        response = ResponsesAPIResponse(
            id="resp-tc-2",
            created_at=0,
            output=[
                {
                    "type": "function_call",
                    "id": "fc-2",
                    "call_id": "call-2",
                    "name": "exfil",
                    "arguments": '{"secret": "hunter2"}',
                }
            ],
        )
        parts = guardrail._completion_response_text_parts(response)
        assert any('{"secret": "hunter2"}' in p for p in parts)


# ---------------------------------------------------------------
# Graph user id path encoding
# ---------------------------------------------------------------


class TestGraphUserIdEncoding:
    def test_encode_graph_user_id_percent_encodes_special_chars(self):
        from urllib.parse import quote

        raw = "user/with%special"
        encoded = PurviewGuardrailBase._encode_graph_user_id(raw)
        assert encoded == quote(raw, safe="")
        assert "/" not in encoded

    @pytest.mark.asyncio
    async def test_compute_protection_scopes_uses_encoded_path(self):
        guardrail = _make_guardrail()
        guardrail._scope_cache.clear()

        mock_resp = _mock_scope_response()

        async def _capture_post(url, **kwargs):
            assert "/users/" in url
            assert "user%2Fwith%25special" in url
            return mock_resp

        guardrail.async_handler.post = AsyncMock(side_effect=_capture_post)

        with patch.object(
            guardrail, "_get_access_token", new_callable=AsyncMock
        ) as mock_token:
            mock_token.return_value = "tok"
            await guardrail._compute_protection_scopes("user/with%special")

        guardrail.async_handler.post.assert_called_once()


# ---------------------------------------------------------------
# _resolve_trusted_user_id
# ---------------------------------------------------------------


class TestResolveTrustedUserId:
    def test_trusted_user_id_from_api_key_dict(self):
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test", user_id="auth-user-111")
        assert guardrail._resolve_trusted_user_id({}, auth) == "auth-user-111"

    def test_end_user_id_not_trusted_for_blocking(self):
        """end_user_id is request-derived; must not be used for blocking DLP."""
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test", end_user_id="end-user-222")
        assert guardrail._resolve_trusted_user_id({}, auth) is None

    def test_metadata_user_api_key_user_id_not_trusted_without_auth(self):
        """Metadata user_api_key_user_id is not trusted when the key has no user_id."""
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test")
        data = {"metadata": {"user_api_key_user_id": "proxy-user-333"}}
        assert guardrail._resolve_trusted_user_id(data, auth) is None

    def test_trusted_user_id_returns_none_for_caller_supplied_only(self):
        """Caller-supplied metadata must NOT be returned by _resolve_trusted_user_id."""
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test")
        data = {"metadata": {"user_id": "caller-supplied-444"}}
        assert guardrail._resolve_trusted_user_id(data, auth) is None

    def test_trusted_prefers_key_user_id_over_end_user_id(self):
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(
            api_key="test", user_id="key-owner", end_user_id="end-user"
        )
        assert guardrail._resolve_trusted_user_id({}, auth) == "key-owner"


# ---------------------------------------------------------------
# _resolve_user_id_from_logging_kwargs — caller-influenceable identity rejected
# ---------------------------------------------------------------


class TestLoggingRejectsCallerInfluenceableIdentity:
    """``end_user_id`` is derived from caller-controllable request fields
    (``user``, ``metadata.user_id``, ``safety_identifier``) so it must not
    drive Purview audit attribution either.
    """

    def test_end_user_id_in_metadata_is_ignored(self):
        guardrail = _make_guardrail()
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "user_api_key_end_user_id": "end-user-from-metadata",
                }
            }
        }
        assert guardrail._resolve_user_id_from_logging_kwargs(kwargs) is None

    def test_end_user_id_at_top_level_kwargs_is_ignored(self):
        guardrail = _make_guardrail()
        kwargs = {
            "user_api_key_end_user_id": "end-user-from-kwargs",
            "litellm_params": {"metadata": {}},
        }
        assert guardrail._resolve_user_id_from_logging_kwargs(kwargs) is None


# ---------------------------------------------------------------
# _resolve_user_id_for_blocking — security warning path
# ---------------------------------------------------------------


class TestResolveUserIdForBlocking:
    def test_trusted_id_returned_without_warning(self, caplog):
        import logging

        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test", user_id="trusted-111")
        with caplog.at_level(logging.WARNING):
            result = guardrail._resolve_user_id_for_blocking({}, auth)
        assert result == "trusted-111"
        assert "SECURITY" not in caplog.text

    def test_caller_supplied_id_raises_http_exception(self):
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test")
        data = {"metadata": {"user_id": "caller-supplied-999"}}
        with pytest.raises(HTTPException) as exc_info:
            guardrail._resolve_user_id_for_blocking(data, auth)
        assert exc_info.value.status_code == 400
        assert "proxy-authenticated" in str(exc_info.value.detail)

    def test_no_id_raises_http_exception(self):
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test")
        with pytest.raises(HTTPException) as exc_info:
            guardrail._resolve_user_id_for_blocking({}, auth)
        assert exc_info.value.status_code == 400
        assert "bind user_id" in str(exc_info.value.detail)

    def test_end_user_id_only_raises_for_blocking(self):
        """Request-derived end_user_id cannot drive blocking Purview checks."""
        guardrail = _make_guardrail()
        auth = UserAPIKeyAuth(api_key="test", end_user_id="caller-end-user")
        with pytest.raises(HTTPException) as exc_info:
            guardrail._resolve_user_id_for_blocking({}, auth)
        assert exc_info.value.status_code == 400
        assert "proxy-authenticated" in str(exc_info.value.detail)


# ---------------------------------------------------------------
# Token-id prompt handling in pre_call blocking mode
# ---------------------------------------------------------------


class TestTokenIdPromptHandling:
    @pytest.mark.asyncio
    async def test_token_id_prompt_raises_in_blocking_mode(self):
        """Pure token-id prompts must be rejected in blocking pre_call mode."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="u1"),
                    cache=None,
                    data={"prompt": [1, 2, 3, 100, 200]},
                    call_type="text_completion",
                )

            mock_check.assert_not_called()
            assert exc_info.value.status_code == 400
            assert "Token-id" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_missing_prompt_skips_without_warning(self, caplog):
        """No prompt at all → silently skip (not a token-id bypass case)."""
        import logging

        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            with caplog.at_level(logging.WARNING):
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="u1"),
                    cache=None,
                    data={},
                    call_type="text_completion",
                )

            mock_check.assert_not_called()
            assert "token-id" not in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_string_prompt_still_scanned(self):
        """Normal string prompts must still be sent to Purview."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = {"policyActions": []}
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="u1"),
                cache=None,
                data={"prompt": "sensitive text"},
                call_type="text_completion",
            )

        mock_check.assert_called_once()
        assert mock_check.call_args.kwargs["text"] == "sensitive text"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty_prompt", ["", "   ", "\n\t  "])
    async def test_empty_or_whitespace_prompt_passes_through(self, empty_prompt):
        """Empty/whitespace-only string prompts must not be flagged as token-id prompts."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            data = {"prompt": empty_prompt}
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="u1"),
                cache=None,
                data=data,
                call_type="text_completion",
            )

        mock_check.assert_not_called()
        assert result is data
        assert result["prompt"] == empty_prompt

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw_prompt",
        [
            [[1, 2, 3]],
            [[1, 2], [3, 4]],
            ["benign text", [99, 100]],
        ],
    )
    async def test_nested_token_id_prompt_raises_in_blocking_mode(self, raw_prompt):
        """Nested/mixed token-id prompts must also be rejected in blocking pre_call mode."""
        guardrail = _make_guardrail()

        with patch.object(
            guardrail, "_check_content", new_callable=AsyncMock
        ) as mock_check:
            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="u1"),
                    cache=None,
                    data={"prompt": raw_prompt},
                    call_type="text_completion",
                )

            mock_check.assert_not_called()
            assert exc_info.value.status_code == 400
            assert "Token-id" in str(exc_info.value.detail)


class TestIsTokenIdPrompt:
    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ([1, 2, 3], True),
            ([[1, 2, 3]], True),
            ([[1, 2], [3, 4]], True),
            (["hi", [1, 2]], True),
            (["a", "b"], False),
            ([], False),
            ("hello", False),
            (None, False),
        ],
    )
    def test_is_token_id_prompt(self, prompt, expected):
        assert PurviewGuardrailBase.is_token_id_prompt(prompt) is expected


# ---------------------------------------------------------------
# Streaming iterator hook
# ---------------------------------------------------------------


class TestStreamingIteratorHook:
    @pytest.mark.asyncio
    async def test_streaming_clean_response_yields_all_chunks(self):
        """Clean stream: all chunks must be re-yielded after DLP passes."""
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()

        assembled_response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(content="safe response", role="assistant"),
                )
            ]
        )

        async def fake_response_stream():
            yield assembled_response

        with (
            patch("litellm.main.stream_chunk_builder", return_value=assembled_response),
            patch(
                "litellm.llms.base_llm.base_model_iterator.MockResponseIterator"
            ) as mock_iterator_cls,
            patch.object(
                guardrail, "_check_content", new_callable=AsyncMock
            ) as mock_check,
        ):
            mock_check.return_value = {"policyActions": []}

            async def _iter_chunks():
                yield assembled_response

            mock_iterator_cls.return_value.__aiter__ = lambda s: _iter_chunks()

            chunks = []
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                response=fake_response_stream(),
                request_data={"metadata": {"user_id": "user-123"}},
            ):
                chunks.append(chunk)

            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["activity"] == "downloadText"
            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_streaming_violation_raises_before_any_chunk(self):
        """A policy violation must raise HTTPException before yielding any chunk."""
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()

        assembled_response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(
                        content="SSN: 123-45-6789",
                        role="assistant",
                    ),
                )
            ]
        )

        async def fake_response_stream():
            yield assembled_response

        with (
            patch("litellm.main.stream_chunk_builder", return_value=assembled_response),
            patch.object(
                guardrail,
                "_check_content",
                new_callable=AsyncMock,
                side_effect=HTTPException(
                    status_code=400,
                    detail={
                        "error": "Microsoft Purview DLP: Content blocked by policy"
                    },
                ),
            ),
        ):
            chunks = []
            with pytest.raises(HTTPException) as exc_info:
                async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(
                        api_key="test", user_id="user-123"
                    ),
                    response=fake_response_stream(),
                    request_data={"metadata": {"user_id": "user-123"}},
                ):
                    chunks.append(chunk)

            assert exc_info.value.status_code == 400
            assert len(chunks) == 0  # No chunks yielded before the block

    @pytest.mark.asyncio
    async def test_streaming_no_user_id_raises_before_yield(self):
        """No resolvable user_id → fail closed before any chunk is yielded."""
        from litellm.types.utils import Choices, Message, ModelResponse

        guardrail = _make_guardrail()

        assembled_response = ModelResponse(
            choices=[
                Choices(
                    index=0,
                    message=Message(content="some content", role="assistant"),
                )
            ]
        )

        async def fake_response_stream():
            yield assembled_response

        with patch(
            "litellm.main.stream_chunk_builder", return_value=assembled_response
        ):
            chunks = []
            with pytest.raises(HTTPException) as exc_info:
                async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),  # no user_id
                    response=fake_response_stream(),
                    request_data={},
                ):
                    chunks.append(chunk)

            assert exc_info.value.status_code == 400
            assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_streaming_text_completion_scans_before_yield(self):
        """Streamed /v1/completions must be DLP-scanned via TextCompletionResponse."""
        from litellm.types.utils import TextChoices, TextCompletionResponse

        guardrail = _make_guardrail()

        assembled_response = TextCompletionResponse(
            model="gpt-3.5-turbo-instruct",
            choices=[TextChoices(text="completion body", index=0)],
        )

        async def fake_response_stream():
            yield assembled_response

        with (
            patch("litellm.main.stream_chunk_builder", return_value=assembled_response),
            patch.object(
                guardrail, "_check_content", new_callable=AsyncMock
            ) as mock_check,
        ):
            mock_check.return_value = {"policyActions": []}

            chunks = []
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                response=fake_response_stream(),
                request_data={},
            ):
                chunks.append(chunk)

            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["text"] == "completion body"
            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_streaming_responses_api_scans_completed_event(self):
        """Streamed Responses API: assembled ResponsesAPIResponse must be DLP-scanned."""
        from litellm.types.llms.openai import (
            ResponseCompletedEvent,
            ResponseCreatedEvent,
            ResponsesAPIResponse,
            ResponsesAPIStreamEvents,
        )

        guardrail = _make_guardrail()

        completed_response = ResponsesAPIResponse(
            id="resp-stream",
            created_at=0,
            output=[
                {
                    "type": "message",
                    "id": "msg-stream",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "streamed output"}],
                }
            ],
        )
        created_event = ResponseCreatedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_CREATED,
            response=completed_response,
        )
        completed_event = ResponseCompletedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
            response=completed_response,
        )

        async def fake_response_stream():
            yield created_event
            yield completed_event

        with (
            patch("litellm.main.stream_chunk_builder") as mock_stream_builder,
            patch.object(
                guardrail, "_check_content", new_callable=AsyncMock
            ) as mock_check,
        ):
            mock_check.return_value = {"policyActions": []}

            chunks = []
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test", user_id="user-123"),
                response=fake_response_stream(),
                request_data={},
            ):
                chunks.append(chunk)

            mock_stream_builder.assert_not_called()
            mock_check.assert_called_once()
            assert mock_check.call_args.kwargs["activity"] == "downloadText"
            assert mock_check.call_args.kwargs["text"] == "streamed output"
            assert chunks == [created_event, completed_event]

    @pytest.mark.asyncio
    async def test_streaming_responses_api_violation_blocks_before_yield(self):
        """Responses API stream with a DLP violation must raise before any chunk is yielded."""
        from litellm.types.llms.openai import (
            ResponseCompletedEvent,
            ResponsesAPIResponse,
            ResponsesAPIStreamEvents,
        )

        guardrail = _make_guardrail()

        completed_response = ResponsesAPIResponse(
            id="resp-stream-block",
            created_at=0,
            output=[
                {
                    "type": "message",
                    "id": "msg-stream-block",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "SSN: 123-45-6789"}],
                }
            ],
        )
        completed_event = ResponseCompletedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
            response=completed_response,
        )

        async def fake_response_stream():
            yield completed_event

        with patch.object(
            guardrail,
            "_check_content",
            new_callable=AsyncMock,
            side_effect=HTTPException(
                status_code=400,
                detail={"error": "Microsoft Purview DLP: Content blocked by policy"},
            ),
        ):
            chunks = []
            with pytest.raises(HTTPException) as exc_info:
                async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(
                        api_key="test", user_id="user-123"
                    ),
                    response=fake_response_stream(),
                    request_data={},
                ):
                    chunks.append(chunk)

            assert exc_info.value.status_code == 400
            assert len(chunks) == 0


# ---------------------------------------------------------------
# Auto-discovery registration
# ---------------------------------------------------------------


class TestRegistration:
    def test_registry_contains_microsoft_purview(self):
        from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
            guardrail_class_registry,
            guardrail_initializer_registry,
        )

        assert "microsoft_purview" in guardrail_initializer_registry
        assert "microsoft_purview" in guardrail_class_registry
        assert (
            guardrail_class_registry["microsoft_purview"]
            is MicrosoftPurviewDLPGuardrail
        )
