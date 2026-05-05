"""
Unit tests for the XecGuard guardrail integration.

Every branch in ``xecguard.py`` is exercised to achieve 100% line +
branch coverage. Network calls are always mocked; the companion live
suite lives in ``test_xecguard_live.py``.
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from fastapi.exceptions import HTTPException
from litellm.proxy.guardrails.guardrail_hooks.xecguard.xecguard import (
    XecGuardGuardrail,
    XecGuardMissingCredentials,
)
from litellm.types.proxy.guardrails.guardrail_hooks.xecguard import (
    XecGuardConfigModel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def xecguard_guardrail():
    return XecGuardGuardrail(
        api_base="https://api.test.xecguard.local",
        api_key="xgs_test_abcdef1234567890_secret",
        guardrail_name="test-xecguard",
        event_hook="pre_call",
        default_on=True,
    )


@pytest.fixture
def mock_request_data():
    return {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I reset my password?"},
        ],
        "metadata": {
            "user_api_key_hash": "abc123",
            "user_api_key_user_id": "user-1",
            "user_api_key_team_id": "team-1",
        },
    }


def _make_response(body: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    mock.status_code = status_code
    return mock


def _build_model_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestXecGuardConfiguration:
    def test_init_with_explicit_credentials(self):
        guardrail = XecGuardGuardrail(
            api_key="xgs_explicit",
            api_base="https://custom.api.local",
            guardrail_name="my-guardrail",
        )
        assert guardrail.api_key == "xgs_explicit"
        assert guardrail.api_base == "https://custom.api.local"

    def test_init_strips_trailing_slash(self):
        guardrail = XecGuardGuardrail(
            api_key="xgs_explicit",
            api_base="https://custom.api.local/",
        )
        assert guardrail.api_base == "https://custom.api.local"

    def test_init_from_env_vars(self):
        with patch.dict(
            os.environ,
            {
                "XECGUARD_API_KEY": "xgs_env_value",
                "XECGUARD_API_BASE": "https://env.api.local",
            },
        ):
            guardrail = XecGuardGuardrail()
            assert guardrail.api_key == "xgs_env_value"
            assert guardrail.api_base == "https://env.api.local"

    def test_init_default_api_base(self):
        guardrail = XecGuardGuardrail(api_key="xgs_default")
        assert guardrail.api_base == "https://api-xecguard.cycraft.ai"

    def test_init_default_model(self):
        guardrail = XecGuardGuardrail(api_key="xgs_default")
        assert guardrail.xecguard_model == "xecguard_v2"

    def test_init_custom_model(self):
        guardrail = XecGuardGuardrail(
            api_key="xgs_default",
            xecguard_model="xecguard_v3",
        )
        assert guardrail.xecguard_model == "xecguard_v3"

    def test_init_missing_api_key_raises(self):
        env_keys = {
            "XECGUARD_API_KEY",
            "XECGUARD_API_BASE",
            "XECGUARD_BLOCK_ON_ERROR",
        }
        cleaned = {k: v for k, v in os.environ.items() if k not in env_keys}
        with patch.dict(os.environ, cleaned, clear=True):
            with pytest.raises(XecGuardMissingCredentials):
                XecGuardGuardrail(api_key=None)

    def test_block_on_error_defaults_true(self):
        env_keys = {"XECGUARD_BLOCK_ON_ERROR"}
        cleaned = {k: v for k, v in os.environ.items() if k not in env_keys}
        with patch.dict(os.environ, cleaned, clear=True):
            guardrail = XecGuardGuardrail(api_key="xgs_default")
            assert guardrail.block_on_error is True

    def test_block_on_error_explicit_false(self):
        guardrail = XecGuardGuardrail(
            api_key="xgs_default",
            block_on_error=False,
        )
        assert guardrail.block_on_error is False

    def test_block_on_error_explicit_true(self):
        guardrail = XecGuardGuardrail(
            api_key="xgs_default",
            block_on_error=True,
        )
        assert guardrail.block_on_error is True

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("", False),
        ],
    )
    def test_block_on_error_from_env(self, value, expected):
        with patch.dict(
            os.environ,
            {
                "XECGUARD_API_KEY": "xgs_env",
                "XECGUARD_BLOCK_ON_ERROR": value,
            },
        ):
            guardrail = XecGuardGuardrail()
            assert guardrail.block_on_error is expected

    def test_grounding_strictness_default_balanced(self):
        guardrail = XecGuardGuardrail(api_key="xgs_default")
        assert guardrail.grounding_strictness == "BALANCED"

    def test_grounding_strictness_strict(self):
        guardrail = XecGuardGuardrail(
            api_key="xgs_default",
            grounding_strictness="STRICT",
        )
        assert guardrail.grounding_strictness == "STRICT"

    def test_policy_names_none_default(self):
        guardrail = XecGuardGuardrail(api_key="xgs_default")
        assert guardrail.policy_names is None

    def test_policy_names_explicit_list(self):
        policies = [
            "Default_Policy_GeneralPromptAttackProtection",
            "Default_Policy_HarmfulContentProtection",
        ]
        guardrail = XecGuardGuardrail(
            api_key="xgs_default",
            policy_names=policies,
        )
        assert guardrail.policy_names == policies

    def test_supported_event_hooks_contains_all_four(self):
        from litellm.types.guardrails import GuardrailEventHooks

        guardrail = XecGuardGuardrail(api_key="xgs_default")
        hooks = guardrail.supported_event_hooks
        assert hooks is not None
        assert GuardrailEventHooks.pre_call in hooks
        assert GuardrailEventHooks.during_call in hooks
        assert GuardrailEventHooks.post_call in hooks
        assert GuardrailEventHooks.logging_only in hooks

    def test_supported_event_hooks_override_preserved(self):
        from litellm.types.guardrails import GuardrailEventHooks

        guardrail = XecGuardGuardrail(
            api_key="xgs_default",
            supported_event_hooks=[GuardrailEventHooks.pre_call],
        )
        assert guardrail.supported_event_hooks == [GuardrailEventHooks.pre_call]

    def test_apply_guardrail_defined_on_class(self):
        """during_call dispatch (proxy/utils.py:1540) requires that
        ``apply_guardrail`` exists on ``type(callback).__dict__`` rather
        than being inherited. Guard against accidental refactors.
        """
        assert "apply_guardrail" in XecGuardGuardrail.__dict__


# ---------------------------------------------------------------------------
# Safe path (both request and response)
# ---------------------------------------------------------------------------


class TestXecGuardApplyGuardrailSafePath:
    @pytest.mark.asyncio
    async def test_request_safe_returns_inputs(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response(
            {"decision": "SAFE", "trace_id": "tr-001", "xecguard_result": []}
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            result = await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["How do I reset my password?"]},
                request_data=mock_request_data,
                input_type="request",
            )
            assert result == {"texts": ["How do I reset my password?"]}

    @pytest.mark.asyncio
    async def test_response_safe_without_documents_skips_grounding(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = _build_model_response(
            "Here is how you reset your password."
        )
        resp = _make_response({"decision": "SAFE", "trace_id": "tr-002"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            result = await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["response text"]},
                request_data=mock_request_data,
                input_type="response",
            )
            assert result == {"texts": ["response text"]}
            assert mock_post.call_count == 1  # only /scan, not /grounding

    @pytest.mark.asyncio
    async def test_response_safe_with_documents_runs_grounding_safe(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = _build_model_response(
            "Peggy Seeger was American."
        )
        mock_request_data["metadata"]["xecguard_grounding_documents"] = [
            {"document_id": "d1", "context": "Peggy Seeger is American."}
        ]
        scan_ok = _make_response({"decision": "SAFE", "trace_id": "tr-003"})
        grounding_ok = _make_response({"decision": "SAFE", "trace_id": "tr-004"})
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_ok],
        ) as mock_post:
            result = await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["text"]},
                request_data=mock_request_data,
                input_type="response",
            )
            assert result == {"texts": ["text"]}
            assert mock_post.call_count == 2
            grounding_call = mock_post.call_args_list[1]
            assert grounding_call.kwargs["url"].endswith("/xecguard/v1/grounding")

    @pytest.mark.asyncio
    async def test_empty_messages_returns_inputs_unchanged(self, xecguard_guardrail):
        result = await xecguard_guardrail.apply_guardrail(
            inputs={"texts": []},
            request_data={"messages": []},
            input_type="request",
        )
        assert result == {"texts": []}

    @pytest.mark.asyncio
    async def test_no_messages_key_returns_inputs(self, xecguard_guardrail):
        result = await xecguard_guardrail.apply_guardrail(
            inputs={"texts": []},
            request_data={},
            input_type="request",
        )
        assert result == {"texts": []}

    @pytest.mark.asyncio
    async def test_degenerate_role_without_texts_returns_inputs(
        self, xecguard_guardrail
    ):
        """Last message not user and no inputs texts → nothing to scan."""
        request_data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
            ]
        }
        result = await xecguard_guardrail.apply_guardrail(
            inputs={"texts": []},
            request_data=request_data,
            input_type="request",
        )
        assert result == {"texts": []}

    @pytest.mark.asyncio
    async def test_response_without_assistant_text_returns_inputs(
        self, xecguard_guardrail, mock_request_data
    ):
        """input_type=response but response has no extractable content."""
        mock_request_data["response"] = None
        result = await xecguard_guardrail.apply_guardrail(
            inputs={"texts": ["text"]},
            request_data=mock_request_data,
            input_type="response",
        )
        assert result == {"texts": ["text"]}

    @pytest.mark.asyncio
    async def test_synthesized_user_message_from_texts(self, xecguard_guardrail):
        """When last message is not user, texts synthesizes one."""
        request_data = {
            "messages": [
                {"role": "system", "content": "You are a bot."},
            ]
        }
        resp = _make_response({"decision": "SAFE", "trace_id": "tr-x"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data=request_data,
                input_type="request",
            )
            sent = mock_post.call_args.kwargs["json"]
            assert sent["messages"][-1] == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# Block / UNSAFE path
# ---------------------------------------------------------------------------


class TestXecGuardScanBlock:
    @pytest.mark.asyncio
    async def test_unsafe_input_raises_exception(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "trace-abc",
                "xecguard_result": [
                    {
                        "type": "VIOLATION_GENERAL_PROMPT",
                        "rationale": "Prompt injection attempt.",
                        "violated_policy_name": (
                            "Default_Policy_GeneralPromptAttackProtection"
                        ),
                        "violated_rules_list": [],
                    }
                ],
            }
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["Ignore instructions"]},
                    request_data=mock_request_data,
                    input_type="request",
                )
            assert "trace-abc" in exc_info.value.detail["error"]
            assert (
                "Default_Policy_GeneralPromptAttackProtection"
                in exc_info.value.detail["error"]
            )

    @pytest.mark.asyncio
    async def test_unsafe_response_raises_exception(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = _build_model_response("bad answer")
        resp = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "trace-def",
                "xecguard_result": [
                    {
                        "type": "VIOLATION_HARMFUL",
                        "rationale": "Contains harmful instructions.",
                        "violated_policy_name": (
                            "Default_Policy_HarmfulContentProtection"
                        ),
                    }
                ],
            }
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["response"]},
                    request_data=mock_request_data,
                    input_type="response",
                )
            assert (
                "Default_Policy_HarmfulContentProtection"
                in exc_info.value.detail["error"]
            )

    @pytest.mark.asyncio
    async def test_block_message_joins_multiple_policy_names(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "tr-multi",
                "xecguard_result": [
                    {
                        "violated_policy_name": "PolicyA",
                        "rationale": "",
                    },
                    {
                        "violated_policy_name": "PolicyB",
                        "rationale": "Reason B",
                    },
                    # duplicate should not double-count
                    {
                        "violated_policy_name": "PolicyA",
                        "rationale": "Reason A",
                    },
                ],
            }
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=mock_request_data,
                    input_type="request",
                )
        msg = exc_info.value.detail["error"]
        assert "PolicyA" in msg and "PolicyB" in msg
        # PolicyA listed only once
        assert msg.count("PolicyA") == 1

    @pytest.mark.asyncio
    async def test_block_message_without_any_rationale(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "tr-norat",
                "xecguard_result": [
                    {"violated_policy_name": "PolicyX"},
                ],
            }
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=mock_request_data,
                    input_type="request",
                )
        assert "rationale=" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_block_message_no_policy_names_uses_unknown(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "tr-u",
                "xecguard_result": [],
            }
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=mock_request_data,
                    input_type="request",
                )
        assert "policies=[unknown]" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_block_message_non_list_xecguard_result(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response(
            {"decision": "UNSAFE", "trace_id": "t", "xecguard_result": "oops"}
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=mock_request_data,
                    input_type="request",
                )
        assert "policies=[unknown]" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_block_message_skips_non_dict_violations(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "t",
                "xecguard_result": [
                    "string-entry",
                    {"violated_policy_name": "PolicyZ"},
                    42,
                ],
            }
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=mock_request_data,
                    input_type="request",
                )
        assert "PolicyZ" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_block_message_rationale_truncated(
        self, xecguard_guardrail, mock_request_data
    ):
        long = "R" * 500
        resp = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "t",
                "xecguard_result": [{"violated_policy_name": "P", "rationale": long}],
            }
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=mock_request_data,
                    input_type="request",
                )
        # Rationale capped at 200 chars
        msg = exc_info.value.detail["error"]
        assert "R" * 200 in msg
        assert "R" * 201 not in msg


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


class TestXecGuardGrounding:
    def _setup_response_with_docs(self, mock_request_data, docs):
        mock_request_data["response"] = _build_model_response(
            "Peggy Seeger was British."
        )
        mock_request_data["metadata"]["xecguard_grounding_documents"] = docs

    @pytest.mark.asyncio
    async def test_grounding_unsafe_raises_exception(
        self, xecguard_guardrail, mock_request_data
    ):
        self._setup_response_with_docs(
            mock_request_data,
            [{"document_id": "d1", "context": "Peggy Seeger is American."}],
        )
        scan_ok = _make_response({"decision": "SAFE", "trace_id": "s"})
        grounding_bad = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "g-trace",
                "xecguard_result": {
                    "violated_policy_name": (
                        "Default_Policy_ContextGroundingValidation"
                    ),
                    "violated_rules_list": ["CONFLICT", "BASELESS"],
                    "rationale": "Contradicts document.",
                    "violated_type": "VIOLATION_CONTEXT_GROUNDING",
                    "metadata": [],
                },
            }
        )
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_bad],
        ):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["a"]},
                    request_data=mock_request_data,
                    input_type="response",
                )
        msg = exc_info.value.detail["error"]
        assert "grounding" in msg
        assert "CONFLICT" in msg
        assert "g-trace" in msg

    @pytest.mark.asyncio
    async def test_grounding_strictness_forwarded(self, mock_request_data):
        guardrail = XecGuardGuardrail(
            api_base="https://api.test.xecguard.local",
            api_key="xgs_test",
            grounding_strictness="STRICT",
        )
        self_ = TestXecGuardGrounding()
        self_._setup_response_with_docs(
            mock_request_data,
            [{"document_id": "d1", "context": "ctx"}],
        )
        scan_ok = _make_response({"decision": "SAFE"})
        grounding_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_ok],
        ) as mock_post:
            await guardrail.apply_guardrail(
                inputs={"texts": ["a"]},
                request_data=mock_request_data,
                input_type="response",
            )
        grounding_payload = mock_post.call_args_list[1].kwargs["json"]
        assert grounding_payload["strictness"] == "STRICT"

    @pytest.mark.asyncio
    async def test_grounding_not_called_on_request_side(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["metadata"]["xecguard_grounding_documents"] = [
            {"document_id": "d", "context": "c"}
        ]
        scan_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=scan_ok
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["a"]},
                request_data=mock_request_data,
                input_type="request",
            )
        # Only /scan called, grounding skipped
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_grounding_skipped_when_docs_empty(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = _build_model_response("answer")
        mock_request_data["metadata"]["xecguard_grounding_documents"] = []
        scan_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=scan_ok
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["a"]},
                request_data=mock_request_data,
                input_type="response",
            )
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_grounding_skipped_when_metadata_absent(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = _build_model_response("answer")
        # no xecguard_grounding_documents in metadata
        scan_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=scan_ok
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["a"]},
                request_data=mock_request_data,
                input_type="response",
            )
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_grounding_malformed_docs_dropped_entirely(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = _build_model_response("answer")
        mock_request_data["metadata"]["xecguard_grounding_documents"] = [
            "string-not-a-dict",
            {"document_id": "only_id"},  # missing context
            {"context": "only_context"},  # missing document_id
            {"document_id": 1, "context": "id not string"},
        ]
        scan_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=scan_ok
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["a"]},
                request_data=mock_request_data,
                input_type="response",
            )
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_grounding_mixed_valid_and_malformed_docs_keeps_valid(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = _build_model_response("answer")
        mock_request_data["metadata"]["xecguard_grounding_documents"] = [
            "bad",
            {"document_id": "good", "context": "good context"},
        ]
        scan_ok = _make_response({"decision": "SAFE"})
        grounding_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_ok],
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["a"]},
                request_data=mock_request_data,
                input_type="response",
            )
        assert mock_post.call_count == 2
        sent_docs = mock_post.call_args_list[1].kwargs["json"]["documents"]
        assert sent_docs == [{"document_id": "good", "context": "good context"}]

    @pytest.mark.asyncio
    async def test_grounding_metadata_falls_back_to_litellm_metadata(
        self, xecguard_guardrail
    ):
        request_data = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
            "response": _build_model_response("a"),
            "litellm_metadata": {
                "xecguard_grounding_documents": [{"document_id": "d", "context": "c"}]
            },
        }
        scan_ok = _make_response({"decision": "SAFE"})
        grounding_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_ok],
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="response",
            )
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_grounding_metadata_missing_returns_empty(self, xecguard_guardrail):
        """No ``metadata`` and no ``litellm_metadata`` keys at all means
        the fallback chain yields None (not a dict) and grounding skips.
        """
        request_data = {
            "messages": [{"role": "user", "content": "q"}],
            "response": _build_model_response("a"),
        }
        scan_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=scan_ok
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="response",
            )
        assert mock_post.call_count == 1

    def test_extract_grounding_documents_metadata_not_dict(self, xecguard_guardrail):
        """Direct coverage of the non-dict metadata branch."""
        assert (
            xecguard_guardrail._extract_grounding_documents({"metadata": "not a dict"})
            == []
        )

    @pytest.mark.asyncio
    async def test_grounding_docs_not_list(self, xecguard_guardrail, mock_request_data):
        mock_request_data["response"] = _build_model_response("answer")
        mock_request_data["metadata"]["xecguard_grounding_documents"] = "not-a-list"
        scan_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=scan_ok
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=mock_request_data,
                input_type="response",
            )
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_grounding_skipped_without_user_or_assistant_message(
        self, xecguard_guardrail
    ):
        """If we cannot extract a user prompt, _call_grounding returns None."""
        request_data = {
            "messages": [],  # empty; build_full_history appends assistant only
            "response": _build_model_response("only assistant"),
            "metadata": {
                "xecguard_grounding_documents": [{"document_id": "d", "context": "c"}]
            },
        }
        scan_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=scan_ok
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="response",
            )
        # Scan ran (assistant-only messages), grounding skipped (no user prompt)
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_grounding_block_message_non_dict_detail(
        self, xecguard_guardrail, mock_request_data
    ):
        """xecguard_result not dict -> formatting yields unknown rules."""
        self._setup_response_with_docs(
            mock_request_data,
            [{"document_id": "d", "context": "c"}],
        )
        scan_ok = _make_response({"decision": "SAFE"})
        grounding_bad = _make_response(
            {"decision": "UNSAFE", "trace_id": "g", "xecguard_result": None}
        )
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_bad],
        ):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["a"]},
                    request_data=mock_request_data,
                    input_type="response",
                )
        assert "rules=[unknown]" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_grounding_block_message_rules_not_list(
        self, xecguard_guardrail, mock_request_data
    ):
        self._setup_response_with_docs(
            mock_request_data,
            [{"document_id": "d", "context": "c"}],
        )
        scan_ok = _make_response({"decision": "SAFE"})
        grounding_bad = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "g",
                "xecguard_result": {
                    "violated_rules_list": "not-list",
                    "rationale": 12345,  # non-string rationale
                },
            }
        )
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_bad],
        ):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["a"]},
                    request_data=mock_request_data,
                    input_type="response",
                )
        assert "rules=[unknown]" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_grounding_block_message_filters_non_string_rules(
        self, xecguard_guardrail, mock_request_data
    ):
        self._setup_response_with_docs(
            mock_request_data,
            [{"document_id": "d", "context": "c"}],
        )
        scan_ok = _make_response({"decision": "SAFE"})
        grounding_bad = _make_response(
            {
                "decision": "UNSAFE",
                "trace_id": "g",
                "xecguard_result": {
                    "violated_rules_list": ["CONFLICT", 1, None, "BASELESS"],
                },
            }
        )
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_bad],
        ):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["a"]},
                    request_data=mock_request_data,
                    input_type="response",
                )
        msg = exc_info.value.detail["error"]
        assert "CONFLICT" in msg and "BASELESS" in msg


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------


class TestXecGuardMessageAssembly:
    @pytest.mark.asyncio
    async def test_full_history_forwarded(self, xecguard_guardrail, mock_request_data):
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["ignored"]},
                request_data=mock_request_data,
                input_type="request",
            )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I reset my password?"},
        ]

    @pytest.mark.asyncio
    async def test_multimodal_content_flattened(self, xecguard_guardrail):
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                        {"type": "text", "text": "world"},
                    ],
                }
            ]
        }
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="request",
            )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["messages"][-1]["content"] == "hello\nworld"

    @pytest.mark.asyncio
    async def test_multimodal_content_no_text_parts_empty_string(
        self, xecguard_guardrail
    ):
        request_data = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "x"}},
                    ],
                },
            ]
        }
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="request",
            )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["messages"][-1] == {"role": "user", "content": ""}

    @pytest.mark.asyncio
    async def test_non_string_non_list_content_becomes_empty_string(
        self, xecguard_guardrail
    ):
        request_data = {"messages": [{"role": "user", "content": 42}]}
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="request",
            )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["messages"][0] == {"role": "user", "content": ""}

    @pytest.mark.asyncio
    async def test_missing_role_defaults_user(self, xecguard_guardrail):
        request_data = {"messages": [{"content": "hi"}]}
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="request",
            )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_messages_non_dict_entries_filtered(self, xecguard_guardrail):
        request_data = {
            "messages": [
                "not a dict",
                {"role": "user", "content": "real"},
                42,
            ]
        }
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="request",
            )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["messages"] == [{"role": "user", "content": "real"}]

    @pytest.mark.asyncio
    async def test_assistant_text_extracted_from_dict_response(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = {
            "choices": [{"message": {"content": "dict-style response"}}]
        }
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=mock_request_data,
                input_type="response",
            )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["messages"][-1] == {
            "role": "assistant",
            "content": "dict-style response",
        }

    @pytest.mark.asyncio
    async def test_assistant_text_extracted_from_list_content(
        self, xecguard_guardrail, mock_request_data
    ):
        msg = MagicMock()
        msg.content = [
            {"type": "text", "text": "partA"},
            {"type": "text", "text": "partB"},
        ]
        choice = MagicMock()
        choice.message = msg
        resp_obj = MagicMock()
        resp_obj.choices = [choice]
        mock_request_data["response"] = resp_obj
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=mock_request_data,
                input_type="response",
            )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["messages"][-1]["content"] == "partA\npartB"

    def test_extract_assistant_text_response_none(self, xecguard_guardrail):
        assert xecguard_guardrail._extract_assistant_text_from_response(None) is None

    def test_extract_assistant_text_no_choices(self, xecguard_guardrail):
        assert xecguard_guardrail._extract_assistant_text_from_response({}) is None

    def test_extract_assistant_text_empty_choices(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_assistant_text_from_response({"choices": []})
            is None
        )

    def test_extract_assistant_text_first_choice_unknown_type(self, xecguard_guardrail):
        resp = MagicMock(spec=[])  # no 'choices'
        assert xecguard_guardrail._extract_assistant_text_from_response(resp) is None

    def test_extract_assistant_text_first_choice_scalar(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_assistant_text_from_response({"choices": [42]})
            is None
        )

    def test_extract_assistant_text_message_none(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_assistant_text_from_response(
                {"choices": [{"message": None}]}
            )
            is None
        )

    def test_extract_assistant_text_message_scalar(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_assistant_text_from_response(
                {"choices": [{"message": 42}]}
            )
            is None
        )

    def test_extract_assistant_text_content_none(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_assistant_text_from_response(
                {"choices": [{"message": {"content": None}}]}
            )
            is None
        )

    def test_extract_assistant_text_content_empty_string(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_assistant_text_from_response(
                {"choices": [{"message": {"content": ""}}]}
            )
            is None
        )

    def test_extract_assistant_text_content_list_all_images(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_assistant_text_from_response(
                {
                    "choices": [
                        {"message": {"content": [{"type": "image_url", "url": "x"}]}}
                    ]
                }
            )
            is None
        )

    def test_extract_assistant_text_content_scalar(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_assistant_text_from_response(
                {"choices": [{"message": {"content": 42}}]}
            )
            is None
        )

    def test_synthesize_user_inputs_not_dict(self, xecguard_guardrail):
        assert xecguard_guardrail._synthesize_user_from_inputs("not-dict") is None

    def test_synthesize_user_no_texts(self, xecguard_guardrail):
        assert xecguard_guardrail._synthesize_user_from_inputs({}) is None

    def test_synthesize_user_texts_filtered_to_empty(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._synthesize_user_from_inputs({"texts": [None, "", 42]})
            is None
        )

    def test_synthesize_user_joins_strings(self, xecguard_guardrail):
        assert xecguard_guardrail._synthesize_user_from_inputs(
            {"texts": ["a", "b"]}
        ) == {"role": "user", "content": "a\nb"}

    def test_extract_last_text_by_role_not_found(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_last_text_by_role(
                [{"role": "user", "content": "hi"}], "assistant"
            )
            is None
        )

    def test_extract_last_text_by_role_empty_content(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_last_text_by_role(
                [{"role": "user", "content": ""}], "user"
            )
            is None
        )

    def test_extract_last_text_by_role_non_string_content(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_last_text_by_role(
                [{"role": "user", "content": 42}], "user"
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_multimodal_text_field_non_string_ignored(self, xecguard_guardrail):
        """A multimodal text part with a non-string ``text`` value is dropped."""
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 123},  # non-string
                        {"type": "text", "text": "keep"},
                    ],
                }
            ]
        }
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=request_data,
                input_type="request",
            )
        sent = mock_post.call_args.kwargs["json"]
        assert sent["messages"][0]["content"] == "keep"


# ---------------------------------------------------------------------------
# Request payload
# ---------------------------------------------------------------------------


class TestXecGuardRequestPayload:
    @pytest.mark.asyncio
    async def test_bearer_auth_header(self, xecguard_guardrail, mock_request_data):
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == ("Bearer xgs_test_abcdef1234567890_secret")
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_scan_url_path(self, xecguard_guardrail, mock_request_data):
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
        assert mock_post.call_args.kwargs["url"] == (
            "https://api.test.xecguard.local/xecguard/v1/scan"
        )

    @pytest.mark.asyncio
    async def test_scan_payload_contains_model_and_scan_type(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "xecguard_v2"
        assert payload["scan_type"] == "input"

    @pytest.mark.asyncio
    async def test_scan_type_response_on_post_call(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = _build_model_response("answer")
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=mock_request_data,
                input_type="response",
            )
        assert mock_post.call_args.kwargs["json"]["scan_type"] == "response"

    @pytest.mark.asyncio
    async def test_policy_names_included_when_set(self, mock_request_data):
        guardrail = XecGuardGuardrail(
            api_base="https://api.test.xecguard.local",
            api_key="xgs_test",
            policy_names=["PolicyA", "PolicyB"],
        )
        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["policy_names"] == ["PolicyA", "PolicyB"]

    @pytest.mark.asyncio
    async def test_policy_names_defaults_when_unconfigured(
        self, xecguard_guardrail, mock_request_data
    ):
        """XecGuard rejects requests without ``policy_names``. When the
        guardrail has no configured policies we fall back to the module
        default set (System Prompt Enforcement + Harmful Content
        Protection) so the request is always acceptable to the server.
        """
        from litellm.proxy.guardrails.guardrail_hooks.xecguard.xecguard import (
            _DEFAULT_POLICIES,
        )

        resp = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["policy_names"] == _DEFAULT_POLICIES

    @pytest.mark.asyncio
    async def test_grounding_url_path(self, xecguard_guardrail, mock_request_data):
        mock_request_data["response"] = _build_model_response("answer")
        mock_request_data["metadata"]["xecguard_grounding_documents"] = [
            {"document_id": "d", "context": "c"}
        ]
        scan_ok = _make_response({"decision": "SAFE"})
        grounding_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_ok],
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=mock_request_data,
                input_type="response",
            )
        grounding_url = mock_post.call_args_list[1].kwargs["url"]
        assert grounding_url == (
            "https://api.test.xecguard.local/xecguard/v1/grounding"
        )

    @pytest.mark.asyncio
    async def test_grounding_payload_shape(self, xecguard_guardrail, mock_request_data):
        mock_request_data["response"] = _build_model_response("response text")
        mock_request_data["metadata"]["xecguard_grounding_documents"] = [
            {"document_id": "d1", "context": "ctx1"}
        ]
        scan_ok = _make_response({"decision": "SAFE"})
        grounding_ok = _make_response({"decision": "SAFE"})
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[scan_ok, grounding_ok],
        ) as mock_post:
            await xecguard_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=mock_request_data,
                input_type="response",
            )
        payload = mock_post.call_args_list[1].kwargs["json"]
        assert payload["model"] == "xecguard_v2"
        assert payload["prompt"] == "How do I reset my password?"
        assert payload["response"] == "response text"
        assert payload["documents"] == [{"document_id": "d1", "context": "ctx1"}]
        assert payload["strictness"] == "BALANCED"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestXecGuardErrorHandling:
    @pytest.mark.asyncio
    async def test_scan_http_error_block_on_error_raises(
        self, xecguard_guardrail, mock_request_data
    ):
        request = httpx.Request("POST", "https://api.test")
        resp = httpx.Response(status_code=500, request=request)
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=httpx.HTTPStatusError("boom", request=request, response=resp),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=mock_request_data,
                    input_type="request",
                )
            assert "block_on_error=True" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_scan_connect_error_block_on_error_raises(
        self, xecguard_guardrail, mock_request_data
    ):
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(HTTPException):
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=mock_request_data,
                    input_type="request",
                )

    @pytest.mark.asyncio
    async def test_scan_http_error_fail_open_returns_inputs(self, mock_request_data):
        guardrail = XecGuardGuardrail(
            api_base="https://api.test.xecguard.local",
            api_key="xgs_test",
            block_on_error=False,
        )
        request = httpx.Request("POST", "https://api.test")
        resp = httpx.Response(status_code=500, request=request)
        with patch.object(
            guardrail.async_handler,
            "post",
            side_effect=httpx.HTTPStatusError("boom", request=request, response=resp),
        ):
            result = await guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
            assert result == {"texts": ["x"]}

    @pytest.mark.asyncio
    async def test_scan_connect_error_fail_open_returns_inputs(self, mock_request_data):
        guardrail = XecGuardGuardrail(
            api_base="https://api.test.xecguard.local",
            api_key="xgs_test",
            block_on_error=False,
        )
        with patch.object(
            guardrail.async_handler,
            "post",
            side_effect=httpx.ConnectError("refused"),
        ):
            result = await guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
            assert result == {"texts": ["x"]}

    @pytest.mark.asyncio
    async def test_grounding_http_error_block_on_error_raises(
        self, xecguard_guardrail, mock_request_data
    ):
        mock_request_data["response"] = _build_model_response("answer")
        mock_request_data["metadata"]["xecguard_grounding_documents"] = [
            {"document_id": "d", "context": "c"}
        ]
        scan_ok = _make_response({"decision": "SAFE"})
        request = httpx.Request("POST", "https://api.test")
        resp = httpx.Response(status_code=500, request=request)
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=[
                scan_ok,
                httpx.HTTPStatusError("boom", request=request, response=resp),
            ],
        ):
            with pytest.raises(HTTPException):
                await xecguard_guardrail.apply_guardrail(
                    inputs={"texts": []},
                    request_data=mock_request_data,
                    input_type="response",
                )

    @pytest.mark.asyncio
    async def test_grounding_http_error_fail_open_returns_inputs(
        self, mock_request_data
    ):
        guardrail = XecGuardGuardrail(
            api_base="https://api.test.xecguard.local",
            api_key="xgs_test",
            block_on_error=False,
        )
        mock_request_data["response"] = _build_model_response("answer")
        mock_request_data["metadata"]["xecguard_grounding_documents"] = [
            {"document_id": "d", "context": "c"}
        ]
        scan_ok = _make_response({"decision": "SAFE"})
        request = httpx.Request("POST", "https://api.test")
        resp = httpx.Response(status_code=500, request=request)
        with patch.object(
            guardrail.async_handler,
            "post",
            side_effect=[
                scan_ok,
                httpx.HTTPStatusError("boom", request=request, response=resp),
            ],
        ):
            result = await guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data=mock_request_data,
                input_type="response",
            )
            assert result == {"texts": []}

    @pytest.mark.asyncio
    async def test_unknown_decision_treated_as_safe(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response({"decision": "MAYBE"})
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            result = await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
            assert result == {"texts": ["x"]}

    @pytest.mark.asyncio
    async def test_missing_decision_treated_as_safe(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response({"trace_id": "t"})
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            result = await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
            assert result == {"texts": ["x"]}

    @pytest.mark.asyncio
    async def test_null_decision_treated_as_safe(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response({"decision": None})
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            result = await xecguard_guardrail.apply_guardrail(
                inputs={"texts": ["x"]},
                request_data=mock_request_data,
                input_type="request",
            )
            assert result == {"texts": ["x"]}


# ---------------------------------------------------------------------------
# Logging-only hook
# ---------------------------------------------------------------------------


class TestXecGuardLoggingHook:
    @pytest.mark.asyncio
    async def test_async_logging_hook_with_response_records_info(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response({"decision": "SAFE", "trace_id": "lg-1"})
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            kwargs = {**mock_request_data, "standard_logging_object": {}}
            result = _build_model_response("some answer")
            out_kwargs, out_result = await xecguard_guardrail.async_logging_hook(
                kwargs=kwargs,
                result=result,
                call_type="acompletion",
            )
            assert out_kwargs is kwargs
            assert out_result is result
        info = kwargs["standard_logging_object"]["guardrail_information"]
        assert info["guardrail_mode"] == "logging_only"
        assert info["guardrail_name"] == "xecguard"
        assert info["guardrail_status"] == "success"
        assert info["guardrail_response"]["trace_id"] == "lg-1"

    @pytest.mark.asyncio
    async def test_async_logging_hook_without_response_records_info(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response({"decision": "SAFE", "trace_id": "lg-2"})
        with patch.object(
            xecguard_guardrail.async_handler, "post", return_value=resp
        ) as mock_post:
            await xecguard_guardrail.async_logging_hook(
                kwargs={**mock_request_data},
                result=None,
                call_type="acompletion",
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["scan_type"] == "input"

    @pytest.mark.asyncio
    async def test_async_logging_hook_unsafe_decision_recorded(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response(
            {"decision": "UNSAFE", "trace_id": "lg-3", "xecguard_result": []}
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            kwargs = {**mock_request_data, "standard_logging_object": {}}
            await xecguard_guardrail.async_logging_hook(
                kwargs=kwargs,
                result=_build_model_response("x"),
                call_type="acompletion",
            )
        info = kwargs["standard_logging_object"]["guardrail_information"]
        assert info["guardrail_status"] == "guardrail_intervened"

    @pytest.mark.asyncio
    async def test_async_logging_hook_does_not_raise_on_http_error(
        self, xecguard_guardrail, mock_request_data
    ):
        result_obj = _build_model_response("x")
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            side_effect=httpx.ConnectError("refused"),
        ):
            out_kwargs, out_result = await xecguard_guardrail.async_logging_hook(
                kwargs=mock_request_data,
                result=result_obj,
                call_type="acompletion",
            )
        assert out_kwargs is mock_request_data
        assert out_result is result_obj

    @pytest.mark.asyncio
    async def test_async_logging_hook_no_messages_returns_unchanged(
        self, xecguard_guardrail
    ):
        kwargs = {"messages": []}
        with patch.object(xecguard_guardrail.async_handler, "post") as mock_post:
            out_kwargs, out_result = await xecguard_guardrail.async_logging_hook(
                kwargs=kwargs, result=None, call_type="acompletion"
            )
        mock_post.assert_not_called()
        assert out_kwargs is kwargs
        assert out_result is None

    @pytest.mark.asyncio
    async def test_async_logging_hook_role_mismatch_returns_unchanged(
        self, xecguard_guardrail
    ):
        kwargs = {
            "messages": [{"role": "system", "content": "sys"}],
        }
        with patch.object(xecguard_guardrail.async_handler, "post") as mock_post:
            await xecguard_guardrail.async_logging_hook(
                kwargs=kwargs, result=None, call_type="acompletion"
            )
        mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_logging_hook_swallows_arbitrary_exception(
        self, xecguard_guardrail, mock_request_data
    ):
        """The hook must never raise. Here we force an unexpected error
        by making ``_build_full_history`` blow up; the outer try/except
        must absorb it and still return (kwargs, result).
        """
        with patch.object(
            xecguard_guardrail.async_handler,
            "post",
            return_value=_make_response({"decision": "SAFE"}),
        ):
            with patch.object(
                xecguard_guardrail,
                "_build_full_history",
                side_effect=RuntimeError("boom"),
            ):
                result_obj = _build_model_response("x")
                out_kwargs, out_result = await xecguard_guardrail.async_logging_hook(
                    kwargs=mock_request_data,
                    result=result_obj,
                    call_type="acompletion",
                )
                assert out_kwargs is mock_request_data
                assert out_result is result_obj

    def test_sync_logging_hook_loop_running_returns_unchanged(
        self, xecguard_guardrail, mock_request_data
    ):
        """When `asyncio.get_event_loop()` returns a running loop, the
        hook returns without driving the async path."""
        fake_loop = MagicMock()
        fake_loop.is_running.return_value = True
        with patch("asyncio.get_event_loop", return_value=fake_loop):
            out = xecguard_guardrail.logging_hook(
                kwargs=mock_request_data,
                result=None,
                call_type="acompletion",
            )
        assert out == (mock_request_data, None)
        fake_loop.run_until_complete.assert_not_called()

    def test_sync_logging_hook_loop_not_running_drives_async(
        self, xecguard_guardrail, mock_request_data
    ):
        """Idle loop path: run_until_complete is driven."""
        fake_loop = MagicMock()
        fake_loop.is_running.return_value = False
        # Close the passed coroutine to silence the un-awaited-coroutine
        # RuntimeWarning (MagicMock doesn't await it for us).
        fake_loop.run_until_complete.side_effect = lambda coro: coro.close()
        with patch("asyncio.get_event_loop", return_value=fake_loop):
            out = xecguard_guardrail.logging_hook(
                kwargs=mock_request_data,
                result=None,
                call_type="acompletion",
            )
        assert out[0] is mock_request_data
        fake_loop.run_until_complete.assert_called_once()

    def test_sync_logging_hook_runtime_error_creates_new_loop(
        self, xecguard_guardrail, mock_request_data
    ):
        new_loop = MagicMock()
        new_loop.is_running.return_value = False
        new_loop.run_until_complete.side_effect = lambda coro: coro.close()
        with patch(
            "asyncio.get_event_loop",
            side_effect=RuntimeError("no current event loop"),
        ):
            with patch("asyncio.new_event_loop", return_value=new_loop):
                with patch("asyncio.set_event_loop") as mock_set:
                    xecguard_guardrail.logging_hook(
                        kwargs=mock_request_data,
                        result=None,
                        call_type="acompletion",
                    )
        new_loop.run_until_complete.assert_called_once()
        mock_set.assert_called_once_with(new_loop)

    def test_sync_logging_hook_swallows_outer_exception(
        self, xecguard_guardrail, mock_request_data
    ):
        """If both get_event_loop and new_event_loop blow up, the outer
        except swallows the error and returns kwargs, result."""
        with patch(
            "asyncio.get_event_loop",
            side_effect=RuntimeError("no loop"),
        ):
            with patch(
                "asyncio.new_event_loop",
                side_effect=OSError("still broken"),
            ):
                out = xecguard_guardrail.logging_hook(
                    kwargs=mock_request_data,
                    result=None,
                    call_type="acompletion",
                )
        assert out == (mock_request_data, None)


# ---------------------------------------------------------------------------
# Config model + registry
# ---------------------------------------------------------------------------


class TestXecGuardConfigModel:
    def test_ui_friendly_name(self):
        assert XecGuardConfigModel.ui_friendly_name() == "XecGuard"

    def test_config_model_default_fields(self):
        model = XecGuardConfigModel()
        assert model.api_key is None
        assert model.api_base is None
        assert model.xecguard_model is None
        assert model.policy_names is None
        assert model.block_on_error is None
        assert model.grounding_strictness is None

    def test_get_config_model_from_guardrail(self, xecguard_guardrail):
        cfg = xecguard_guardrail.get_config_model()
        assert cfg is not None
        assert cfg.ui_friendly_name() == "XecGuard"

    def test_policy_names_exposes_multiselect_options(self):
        """The UI renders policy_names as a multiselect dropdown. Guard
        against accidental removal of the json_schema_extra metadata and
        verify the six default policies are offered."""
        from litellm.types.proxy.guardrails.guardrail_hooks.xecguard import (
            XECGUARD_DEFAULT_POLICY_OPTIONS,
        )

        field = XecGuardConfigModel.model_fields["policy_names"]
        extra = field.json_schema_extra or {}
        assert extra.get("ui_type") == "multiselect"
        assert extra.get("options") == XECGUARD_DEFAULT_POLICY_OPTIONS
        assert (
            "Default_Policy_SystemPromptEnforcement" in XECGUARD_DEFAULT_POLICY_OPTIONS
        )
        assert (
            "Default_Policy_GeneralPromptAttackProtection"
            in XECGUARD_DEFAULT_POLICY_OPTIONS
        )
        assert "Default_Policy_ContentBiasProtection" in XECGUARD_DEFAULT_POLICY_OPTIONS
        assert (
            "Default_Policy_HarmfulContentProtection" in XECGUARD_DEFAULT_POLICY_OPTIONS
        )
        assert "Default_Policy_SkillsProtection" in XECGUARD_DEFAULT_POLICY_OPTIONS
        assert (
            "Default_Policy_PIISensitiveDataProtection"
            in XECGUARD_DEFAULT_POLICY_OPTIONS
        )


class TestXecGuardInitializer:
    def test_initializer_registry_has_entry(self):
        from litellm.proxy.guardrails.guardrail_hooks.xecguard import (
            guardrail_initializer_registry,
        )

        assert "xecguard" in guardrail_initializer_registry

    def test_class_registry_has_entry(self):
        from litellm.proxy.guardrails.guardrail_hooks.xecguard import (
            guardrail_class_registry,
        )

        assert "xecguard" in guardrail_class_registry
        assert guardrail_class_registry["xecguard"] is XecGuardGuardrail

    def test_enum_value_exists(self):
        from litellm.types.guardrails import SupportedGuardrailIntegrations

        assert SupportedGuardrailIntegrations.XECGUARD.value == "xecguard"

    def test_initializer_creates_instance(self):
        from litellm.proxy.guardrails.guardrail_hooks.xecguard import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import LitellmParams

        params = LitellmParams(
            guardrail="xecguard",
            mode="pre_call",
            api_key="xgs_init",
            api_base="https://api.test.xecguard.local",
            default_on=False,
        )
        guardrail = {"guardrail_name": "xg-test"}
        cb = initialize_guardrail(litellm_params=params, guardrail=guardrail)
        assert isinstance(cb, XecGuardGuardrail)
        assert cb.api_key == "xgs_init"
        assert cb.guardrail_name == "xg-test"
