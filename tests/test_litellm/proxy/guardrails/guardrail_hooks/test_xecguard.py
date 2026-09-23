"""
Unit tests for the XecGuard guardrail integration.

Every branch in ``xecguard.py`` is exercised to achieve 100% line +
branch coverage. Network calls are always mocked; the companion live
suite lives in ``test_xecguard_live.py``.
"""

import inspect
import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from fastapi.exceptions import HTTPException
from litellm.proxy.guardrails.guardrail_hooks.xecguard import (
    xecguard as xecguard_module,
)
from litellm.proxy.guardrails.guardrail_hooks.xecguard.xecguard import (
    XecGuardGuardrail,
    XecGuardMissingCredentials,
)
from litellm.types.guardrails import GuardrailEventHooks
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

    def test_extract_assistant_text_combines_all_choices(self, xecguard_guardrail):
        assert (
            xecguard_guardrail._extract_assistant_text_from_response(
                {
                    "choices": [
                        {"message": {"content": "first response"}},
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "second"},
                                    {"type": "text", "text": "response"},
                                ]
                            }
                        },
                    ]
                }
            )
            == "first response\nsecond\nresponse"
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
        ) == {
            "role": "user",
            "content": "a\nb",
        }

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
        info_list = kwargs["standard_logging_object"]["guardrail_information"]
        assert isinstance(info_list, list), "guardrail_information must be a list"
        assert len(info_list) == 1
        info = info_list[0]
        assert info["guardrail_mode"] == "logging_only"
        assert info["guardrail_name"] == "test-xecguard"
        assert info["guardrail_status"] == "success"
        assert info["guardrail_response"]["trace_id"] == "lg-1"

    @pytest.mark.asyncio
    async def test_async_logging_hook_appends_to_existing_guardrail_info(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response({"decision": "SAFE", "trace_id": "lg-4"})
        prior_entry = {"guardrail_name": "other-guardrail"}
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            kwargs = {
                **mock_request_data,
                "standard_logging_object": {"guardrail_information": [prior_entry]},
            }
            await xecguard_guardrail.async_logging_hook(
                kwargs=kwargs,
                result=_build_model_response("some answer"),
                call_type="acompletion",
            )
        info_list = kwargs["standard_logging_object"]["guardrail_information"]
        assert len(info_list) == 2
        assert info_list[0] is prior_entry
        assert info_list[1]["guardrail_name"] == "test-xecguard"
        assert info_list[1]["guardrail_response"]["trace_id"] == "lg-4"

    @pytest.mark.asyncio
    async def test_async_logging_hook_sanitizes_scan_result(
        self, xecguard_guardrail, mock_request_data
    ):
        resp = _make_response(
            {
                "decision": "SAFE",
                "trace_id": "lg-5",
                "secret_fields": {"authorization": "Bearer xgs_raw"},
                "detections": [{"match": "raw matched span", "policy": "pii"}],
                "api_key": "xgs_super_secret_value",
            }
        )
        with patch.object(xecguard_guardrail.async_handler, "post", return_value=resp):
            kwargs = {**mock_request_data, "standard_logging_object": {}}
            await xecguard_guardrail.async_logging_hook(
                kwargs=kwargs,
                result=_build_model_response("some answer"),
                call_type="acompletion",
            )
        info = kwargs["standard_logging_object"]["guardrail_information"][0]
        guardrail_response = info["guardrail_response"]
        assert "secret_fields" not in guardrail_response
        assert guardrail_response["detections"][0]["match"] == "[REDACTED]"
        assert guardrail_response["api_key"] != "xgs_super_secret_value"
        assert guardrail_response["trace_id"] == "lg-5"

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
        info_list = kwargs["standard_logging_object"]["guardrail_information"]
        assert isinstance(info_list, list), "guardrail_information must be a list"
        info = info_list[0]
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

    def test_initializer_forwards_every_configurable_field(self):
        """The initializer names each param explicitly, so a field added to the
        config model but not forwarded here is silently inert: the UI shows the
        control, the operator sets it, and nothing happens. Assert on the config
        model's own field list so adding a field without wiring it fails here."""
        from litellm.proxy.guardrails.guardrail_hooks.xecguard import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import LitellmParams

        params = LitellmParams(
            guardrail="xecguard",
            mode="pre_call",
            api_key="xgs_init",
            api_base="https://api.test.xecguard.local",
            xecguard_model="xecguard_v2",
            policy_names=["Default_Policy_SkillsProtection"],
            apply_to_aliases="key-allow",
            except_aliases="key-deny",
            send_meta=True,
            meta_data_fields="cost_center, owner",
            meta_identity_format="object",
            block_on_error=False,
            grounding_strictness="STRICT",
            default_on=True,
        )
        cb = initialize_guardrail(
            litellm_params=params, guardrail={"guardrail_name": "xg"}
        )

        assert cb.xecguard_model == "xecguard_v2"
        assert cb.policy_names == ["Default_Policy_SkillsProtection"]
        # the alias lists keep the normalized list the config model produced
        assert cb.apply_to_aliases == ["key-allow"]
        assert cb.except_aliases == ["key-deny"]
        assert cb.send_meta is True
        assert cb.meta_data_fields == ("cost_center", "owner")
        assert cb.meta_identity_format == "object"
        assert cb.block_on_error is False
        assert cb.grounding_strictness == "STRICT"

        # Every field the UI renders for this provider must have landed on the
        # instance -- api_key/api_base are asserted by the test above.
        source = inspect.getsource(initialize_guardrail)
        unwired = [
            name
            for name in XecGuardConfigModel.model_fields
            if name not in ("optional_params",)
            and f"litellm_params.{name}" not in source
        ]
        assert unwired == [], (
            f"XecGuardConfigModel fields not forwarded by initialize_guardrail: {unwired}"
        )


# ===========================================================================
# Extension layered on top of the original integration: per-virtual-key
# filtering (key allow/block lists + per-key policy subset via native metadata).
#
# The fixture below is intentionally NOT autouse so the exhaustive suite
# above keeps its original environment handling; it is opted into per class
# via @pytest.mark.usefixtures.
# ===========================================================================


@pytest.fixture
def _clean_env(monkeypatch):
    """Keep XecGuard env vars from leaking into these tests."""
    for var in (
        "XECGUARD_API_KEY",
        "XECGUARD_API_BASE",
        "XECGUARD_BLOCK_ON_ERROR",
        "XECGUARD_SEND_META",
        "XECGUARD_META_IDENTITY_FORMAT",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _extension_guardrail(**overrides):
    params = dict(
        api_base="https://api.test.xecguard.local",
        api_key="xgs_test_scan_secret",
        guardrail_name="test-xecguard",
        event_hook="pre_call",
        default_on=True,
    )
    params.update(overrides)
    return XecGuardGuardrail(**params)


# ---------------------------------------------------------------------------
# Per-virtual-key filtering
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_env")
class TestXecGuardCallingKeyIdentity:
    def test_from_metadata(self):
        data = {"metadata": {"user_api_key_alias": "svc", "user_api_key_hash": "h1"}}
        assert XecGuardGuardrail._calling_key_identity(data) == ("svc", "h1")

    def test_from_litellm_metadata(self):
        data = {"litellm_metadata": {"user_api_key_alias": "svc2"}}
        assert XecGuardGuardrail._calling_key_identity(data) == ("svc2", None)

    def test_master_key_returns_none_none(self):
        assert XecGuardGuardrail._calling_key_identity({}) == (None, None)
        assert XecGuardGuardrail._calling_key_identity(None) == (None, None)


@pytest.mark.usefixtures("_clean_env")
class TestXecGuardKeyIsTargeted:
    def test_no_lists_scans_everything(self):
        gr = _extension_guardrail()
        assert gr._key_is_targeted({"metadata": {"user_api_key_alias": "any"}}) is True

    def test_allowlist_match(self):
        gr = _extension_guardrail(apply_to_aliases=["prod"])
        assert gr._key_is_targeted({"metadata": {"user_api_key_alias": "prod"}}) is True

    def test_allowlist_miss(self):
        gr = _extension_guardrail(apply_to_aliases=["prod"])
        assert gr._key_is_targeted({"metadata": {"user_api_key_alias": "dev"}}) is False

    def test_blocklist_excludes(self):
        gr = _extension_guardrail(except_aliases=["internal"])
        assert (
            gr._key_is_targeted({"metadata": {"user_api_key_alias": "internal"}})
            is False
        )

    def test_blocklist_wins_over_allowlist(self):
        gr = _extension_guardrail(apply_to_aliases=["prod"], except_aliases=["prod"])
        assert (
            gr._key_is_targeted({"metadata": {"user_api_key_alias": "prod"}}) is False
        )

    def test_match_by_hash(self):
        gr = _extension_guardrail(apply_to_aliases=["hash-abc"])
        assert (
            gr._key_is_targeted({"metadata": {"user_api_key_hash": "hash-abc"}}) is True
        )


# ---------------------------------------------------------------------------
# Integration — apply_guardrail respects the key allow/deny lists
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_env")
class TestXecGuardApplyGuardrailKeyTargeting:
    @pytest.mark.asyncio
    async def test_key_not_in_allowlist_skips_scan(self):
        gr = _extension_guardrail(apply_to_aliases=["prod"])
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_api_key_alias": "dev"},
        }
        with patch.object(gr.async_handler, "post") as post:
            result = await gr.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=data,
                input_type="request",
            )
        assert result == {"texts": ["hi"]}
        post.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocklisted_key_skips_scan(self):
        gr = _extension_guardrail(except_aliases=["internal"])
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_api_key_alias": "internal"},
        }
        with patch.object(gr.async_handler, "post") as post:
            result = await gr.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=data,
                input_type="request",
            )
        assert result == {"texts": ["hi"]}
        post.assert_not_called()

    @pytest.mark.asyncio
    async def test_targeted_key_uses_config_policies(self):
        gr = _extension_guardrail(
            apply_to_aliases=["prod"], policy_names=["Config_Level_Policy"]
        )
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_api_key_alias": "prod"},
        }
        resp = _make_response({"decision": "SAFE", "trace_id": "tr"})
        with patch.object(gr.async_handler, "post", return_value=resp) as post:
            out = await gr.apply_guardrail(
                inputs={"texts": ["hi"]},
                request_data=data,
                input_type="request",
            )
        assert out == {"texts": ["hi"]}, "a SAFE verdict hands the inputs back untouched"
        post.assert_called_once()
        assert post.call_args.kwargs["json"]["policy_names"] == ["Config_Level_Policy"]


# ---------------------------------------------------------------------------
# Config-model guards — the alias fields are plain manual text input (no live
# dropdown), so this feature stays entirely off shared UI-renderer / options
# code. These lock in the two facts that guarantee that.
# ---------------------------------------------------------------------------


class TestXecGuardAliasConfigModel:
    def test_validator_splits_comma_separated_string(self):
        # The UI submits a plain text box as a single string.
        cfg = XecGuardConfigModel(apply_to_aliases="prod, staging ,  ,dev")
        assert cfg.apply_to_aliases == ["prod", "staging", "dev"]

    def test_validator_passes_list_through_and_cleans(self):
        # YAML users may write a list; non-str / empty entries are dropped and
        # surviving strings are stripped.
        cfg = XecGuardConfigModel(except_aliases=["  a ", "b", "", 5, None])
        assert cfg.except_aliases == ["a", "b"]

    def test_validator_none_stays_none(self):
        cfg = XecGuardConfigModel()
        assert cfg.apply_to_aliases is None
        assert cfg.except_aliases is None

    def test_ui_type_is_plain_string_not_array(self):
        # Optional[Union[str, List[str]]] must resolve to the "string" UI type
        # (str is first in the Union), so the generic renderer draws a plain
        # <Input> with no options list — never the shared multiselect path.
        from litellm.proxy.guardrails.guardrail_endpoints import (
            _get_field_type_from_annotation,
        )

        for field_name in ("apply_to_aliases", "except_aliases"):
            annotation = XecGuardConfigModel.model_fields[field_name].annotation
            assert _get_field_type_from_annotation(annotation) == "string"


# ---------------------------------------------------------------------------
# Per-virtual-key filtering in logging_only mode
# ---------------------------------------------------------------------------


class TestXecGuardLoggingHookKeyTargeting:
    """logging_only dispatches to async_logging_hook, not apply_guardrail, so the
    allow/deny lists and the per-key opt-out have to be enforced there as well —
    otherwise an excluded key's content is still sent to the XecGuard backend."""

    @staticmethod
    def _kwargs(metadata, nested=True):
        """Build logging-hook kwargs. The proxy puts the injected request
        metadata under ``litellm_params.metadata``; ``nested=False`` exercises
        the top-level ``metadata`` fallback."""
        base = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "standard_logging_object": {},
        }
        if nested:
            base["litellm_params"] = {"metadata": metadata}
        else:
            base["metadata"] = metadata
        return base

    @pytest.mark.asyncio
    async def test_blocklisted_key_is_not_scanned(self):
        gr = _extension_guardrail(except_aliases=["internal"])
        kwargs = self._kwargs({"user_api_key_alias": "internal"})
        with patch.object(gr.async_handler, "post") as post:
            out_kwargs, out_result = await gr.async_logging_hook(
                kwargs=kwargs,
                result=_build_model_response("some answer"),
                call_type="acompletion",
            )
        post.assert_not_called()
        assert out_kwargs is kwargs
        assert "guardrail_information" not in kwargs["standard_logging_object"]

    @pytest.mark.asyncio
    async def test_key_not_in_allowlist_is_not_scanned(self):
        gr = _extension_guardrail(apply_to_aliases=["prod"])
        kwargs = self._kwargs({"user_api_key_alias": "dev"})
        with patch.object(gr.async_handler, "post") as post:
            out_kwargs, _ = await gr.async_logging_hook(
                kwargs=kwargs,
                result=_build_model_response("some answer"),
                call_type="acompletion",
            )
        post.assert_not_called()
        assert out_kwargs is kwargs
        assert "guardrail_information" not in kwargs["standard_logging_object"]

    @pytest.mark.asyncio
    async def test_top_level_metadata_is_also_honoured(self):
        gr = _extension_guardrail(except_aliases=["internal"])
        kwargs = self._kwargs({"user_api_key_alias": "internal"}, nested=False)
        with patch.object(gr.async_handler, "post") as post:
            out_kwargs, _ = await gr.async_logging_hook(
                kwargs=kwargs,
                result=_build_model_response("some answer"),
                call_type="acompletion",
            )
        post.assert_not_called()
        assert out_kwargs is kwargs
        assert "guardrail_information" not in kwargs["standard_logging_object"]

    @pytest.mark.asyncio
    async def test_targeted_key_is_still_scanned(self):
        gr = _extension_guardrail(apply_to_aliases=["prod"])
        kwargs = self._kwargs({"user_api_key_alias": "prod"})
        resp = _make_response({"decision": "SAFE", "trace_id": "lg-key-1"})
        with patch.object(gr.async_handler, "post", return_value=resp) as post:
            await gr.async_logging_hook(
                kwargs=kwargs,
                result=_build_model_response("some answer"),
                call_type="acompletion",
            )
        post.assert_called_once()
        info_list = kwargs["standard_logging_object"]["guardrail_information"]
        assert info_list[0]["guardrail_response"]["trace_id"] == "lg-key-1"

    def test_key_context_lifts_nested_metadata(self):
        # logging path: model_call_details carries the key fields one level down
        gr = _extension_guardrail()
        ctx = gr._key_context(
            {"litellm_params": {"metadata": {"user_api_key_alias": "nested"}}}
        )
        assert ctx["metadata"]["user_api_key_alias"] == "nested"

    def test_key_context_nested_skips_metadata_without_identity(self):
        # logging path: an empty litellm_params.metadata must not shadow the
        # sibling key that actually carries the identity.
        gr = _extension_guardrail()
        ctx = gr._key_context(
            {
                "litellm_params": {
                    "metadata": {},
                    "litellm_metadata": {"user_api_key_alias": "sibling"},
                }
            }
        )
        assert gr._calling_key_identity(ctx) == ("sibling", None)

    def test_key_context_returns_proxy_shape_untouched(self):
        # pre/during/post_call: reshaping to a single key would drop the other
        # location _calling_key_identity also reads.
        gr = _extension_guardrail()
        data = {
            "metadata": {"user_api_key_alias": "prod"},
            "litellm_metadata": {"user_api_key_hash": "hash-abc"},
        }
        assert gr._key_context(data) is data
        assert gr._calling_key_identity(gr._key_context(data)) == ("prod", "hash-abc")

    def test_key_context_top_level_wins_over_nested(self):
        gr = _extension_guardrail()
        ctx = gr._key_context(
            {
                "metadata": {"user_api_key_alias": "top"},
                "litellm_params": {"metadata": {"user_api_key_alias": "nested"}},
            }
        )
        assert ctx["metadata"]["user_api_key_alias"] == "top"

    def test_key_context_passes_through_when_nothing_to_lift(self):
        gr = _extension_guardrail()
        assert gr._key_context(None) is None
        empty: dict = {}
        assert gr._key_context(empty) is empty
        odd = {"litellm_params": {"metadata": "not-a-dict"}}
        assert gr._key_context(odd) is odd


# ---------------------------------------------------------------------------
# should_run_guardrail: gate before LiteLLM records the guardrail as having run
# ---------------------------------------------------------------------------


class TestXecGuardShouldRunGuardrail:
    @staticmethod
    def _data(metadata):
        return {"messages": [{"role": "user", "content": "hi"}], "metadata": metadata}

    def test_targeted_key_runs(self):
        gr = _extension_guardrail(apply_to_aliases=["prod"])
        assert (
            gr.should_run_guardrail(
                self._data({"user_api_key_alias": "prod"}), GuardrailEventHooks.pre_call
            )
            is True
        )

    def test_key_not_in_allowlist_does_not_run(self):
        gr = _extension_guardrail(apply_to_aliases=["prod"])
        assert (
            gr.should_run_guardrail(
                self._data({"user_api_key_alias": "dev"}), GuardrailEventHooks.pre_call
            )
            is False
        )

    def test_blocklisted_key_does_not_run(self):
        gr = _extension_guardrail(except_aliases=["internal"])
        assert (
            gr.should_run_guardrail(
                self._data({"user_api_key_alias": "internal"}),
                GuardrailEventHooks.pre_call,
            )
            is False
        )

    def test_no_lists_configured_runs(self):
        gr = _extension_guardrail()
        assert (
            gr.should_run_guardrail(
                self._data({"user_api_key_alias": "anything"}),
                GuardrailEventHooks.pre_call,
            )
            is True
        )

    def test_native_decision_still_wins(self):
        # super() says no (wrong event type for this hook) -> we must not override it
        gr = _extension_guardrail(apply_to_aliases=["prod"])  # event_hook="pre_call"
        assert (
            gr.should_run_guardrail(
                self._data({"user_api_key_alias": "prod"}),
                GuardrailEventHooks.post_call,
            )
            is False
        )

    def test_native_opt_out_still_wins(self):
        # admin-set disable_global_guardrails is honoured by super() even for a
        # key that our own allow list would otherwise target
        gr = _extension_guardrail(apply_to_aliases=["prod"])
        data = self._data(
            {
                "user_api_key_alias": "prod",
                "user_api_key_metadata": {"disable_global_guardrails": True},
            }
        )
        assert gr.should_run_guardrail(data, GuardrailEventHooks.pre_call) is False

    def test_logging_path_shape_is_understood(self):
        # on the logging path `data` is model_call_details: key fields sit under
        # litellm_params.metadata, and the gate must still find them
        gr = _extension_guardrail(
            except_aliases=["internal"], event_hook="logging_only"
        )
        data = {"litellm_params": {"metadata": {"user_api_key_alias": "internal"}}}
        assert gr.should_run_guardrail(data, GuardrailEventHooks.logging_only) is False
        data_ok = {"litellm_params": {"metadata": {"user_api_key_alias": "other"}}}
        assert (
            gr.should_run_guardrail(data_ok, GuardrailEventHooks.logging_only) is True
        )

    def test_mode_mismatch_is_left_to_super(self):
        # a pre_call guardrail must not run for logging_only, whatever the lists say
        gr = _extension_guardrail(event_hook="pre_call")
        data = {"litellm_params": {"metadata": {"user_api_key_alias": "anything"}}}
        assert gr.should_run_guardrail(data, GuardrailEventHooks.logging_only) is False


# ---------------------------------------------------------------------------
# Scan payload `meta` — caller context for XecGuard's SIEM export.
#
# Contract (POST /xecguard/v1/scan): meta is optional; when present virtualkey
# is required and must match ^[A-Za-z_][A-Za-z0-9_.-]{0,63}$; data is a flat
# string->string map of at most 32 fields, keys on the same pattern, values
# <=512 chars, whole meta <=4096 bytes serialized. A violation is a 400, which
# block_on_error would turn into a user-visible block — so everything is coerced
# or dropped here instead.
# ---------------------------------------------------------------------------


def _meta_request_data(key_metadata=None, alias="team-alpha", **metadata):
    """A pre/during/post_call request_data with the proxy-injected key fields."""
    meta: dict = {"messages": [{"role": "user", "content": "hi"}]}
    injected = {"user_api_key_alias": alias, **metadata}
    if key_metadata is not None:
        injected["user_api_key_metadata"] = key_metadata
    meta["metadata"] = injected
    return meta


def _admin_data(meta):
    """``meta.data`` with the proxy-injected attributes removed.

    ``send_meta`` forwards two merged sources: the attributes the proxy injects
    about the calling key, and the free-form metadata an admin typed on the
    Virtual Keys page. The tests below are about the second source's coercion
    rules, so they filter the first out rather than restating it 20 times --
    ``TestXecGuardScanMetaAutoFields`` covers the injected half on its own.
    """
    auto = {name for name, _ in xecguard_module._META_AUTO_DATA_FIELDS}
    return {k: v for k, v in (meta.get("data") or {}).items() if k not in auto}


async def _sent_payload(gr, request_data):
    """Run one safe scan and return the JSON body that reached the backend."""
    resp = _make_response(
        {"decision": "SAFE", "trace_id": "meta-tr", "xecguard_result": []}
    )
    with patch.object(gr.async_handler, "post", return_value=resp) as mock_post:
        await gr.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=request_data,
            input_type="request",
        )
    return mock_post.call_args.kwargs["json"]


@pytest.mark.usefixtures("_clean_env")
class TestXecGuardScanMeta:
    @pytest.mark.asyncio
    async def test_meta_absent_by_default(self):
        # send_meta is opt-in: enabling it forwards the key alias and the key's
        # admin-set metadata to XecGuard, so an upgrade must not start doing it.
        gr = _extension_guardrail()
        assert gr.send_meta is False
        payload = await _sent_payload(gr, _meta_request_data({"cost_center": "CC-42"}))
        assert "meta" not in payload

    @pytest.mark.asyncio
    async def test_virtualkey_is_the_alias_the_guardrail_filtered_on(self):
        gr = _extension_guardrail(send_meta=True, apply_to_aliases=["team-alpha"])
        payload = await _sent_payload(gr, _meta_request_data(alias="team-alpha"))
        assert payload["meta"]["virtualkey"] == "team-alpha"
        assert _admin_data(payload["meta"]) == {}

    @pytest.mark.asyncio
    async def test_data_is_the_virtual_keys_page_metadata(self):
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _meta_request_data({"cost_center": "CC-42", "owner": "alice@corp"}),
        )
        assert payload["meta"]["virtualkey"] == "team-alpha"
        assert _admin_data(payload["meta"]) == {
            "cost_center": "CC-42",
            "owner": "alice@corp",
        }

    @pytest.mark.asyncio
    async def test_meta_does_not_disturb_the_existing_payload(self):
        gr = _extension_guardrail(send_meta=True, policy_names=["jailbreak"])
        payload = await _sent_payload(gr, _meta_request_data({"a": "b"}))
        assert payload["scan_type"] == "input"
        assert payload["policy_names"] == ["jailbreak"]
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_hash_is_the_fallback_when_the_key_has_no_alias(self):
        gr = _extension_guardrail(send_meta=True)
        data = _meta_request_data(alias=None, user_api_key_hash="abc123def")
        payload = await _sent_payload(gr, data)
        assert payload["meta"]["virtualkey"] == "abc123def"

    @pytest.mark.asyncio
    async def test_meta_omitted_when_no_identity_matches_the_pattern(self):
        # master-key calls carry neither field; a hash starting with a digit
        # cannot satisfy the backend pattern. Omit meta rather than 400 the scan.
        gr = _extension_guardrail(send_meta=True)
        assert "meta" not in await _sent_payload(gr, _meta_request_data(alias=None))
        digit_hash = _meta_request_data(alias=None, user_api_key_hash="9abcdef")
        assert "meta" not in await _sent_payload(gr, digit_hash)

    @pytest.mark.asyncio
    async def test_alias_failing_the_pattern_falls_through_to_the_hash(self):
        gr = _extension_guardrail(send_meta=True)
        data = _meta_request_data(alias="has spaces", user_api_key_hash="deadbeef")
        payload = await _sent_payload(gr, data)
        assert payload["meta"]["virtualkey"] == "deadbeef"

    @pytest.mark.asyncio
    async def test_scalars_are_stringified_and_nested_values_dropped(self):
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _meta_request_data(
                {
                    "tier": 3,
                    "ratio": 1.5,
                    "beta": True,
                    "ga": False,
                    "absent": None,
                    "nested": {"x": 1},
                    "listed": ["a"],
                    "blank": "",
                }
            ),
        )
        assert _admin_data(payload["meta"]) == {
            "tier": "3",
            "ratio": "1.5",
            "beta": "true",
            "ga": "false",
        }

    @pytest.mark.asyncio
    async def test_a_non_string_field_name_is_dropped_not_raised(self):
        """The metadata dict is decoded JSON that nothing validates on the way in.

        `meta.data` promises to coerce or drop, never to fail the scan, and it is
        built outside any try on the pre/during/post_call paths -- so a key that
        reaches `re.match` unchecked would turn one malformed field into a 500 for
        every request from that key.
        """
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr, _meta_request_data({1: "oops", "tier": "gold"})
        )
        assert _admin_data(payload["meta"]) == {"tier": "gold"}

    @pytest.mark.asyncio
    async def test_field_names_off_the_pattern_are_dropped(self):
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _meta_request_data(
                {"ok.name-1": "v", "bad name": "v", "9lead": "v", "": "v"}
            ),
        )
        assert _admin_data(payload["meta"]) == {"ok.name-1": "v"}

    @pytest.mark.asyncio
    async def test_control_characters_are_stripped_and_values_truncated(self):
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _meta_request_data({"note": "a\x00b\x1fc\x7fd", "long": "x" * 700}),
        )
        assert payload["meta"]["data"]["note"] == "abcd"
        assert len(payload["meta"]["data"]["long"]) == 512

    @pytest.mark.asyncio
    async def test_at_most_32_fields(self):
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr, _meta_request_data({f"f{i}": str(i) for i in range(40)})
        )
        assert len(payload["meta"]["data"]) == 32
        assert "f0" in payload["meta"]["data"] and "f39" not in payload["meta"]["data"]

    @pytest.mark.asyncio
    async def test_serialized_cap_sheds_fields_but_keeps_virtualkey(self):
        gr = _extension_guardrail(send_meta=True)
        # 20 x ~500 chars is far past 4096 bytes; a later small field still fits.
        key_metadata = {f"big{i}": "y" * 500 for i in range(20)}
        key_metadata["small"] = "s"
        payload = await _sent_payload(gr, _meta_request_data(key_metadata))
        meta = payload["meta"]
        assert meta["virtualkey"] == "team-alpha"
        assert len(json.dumps(meta, ensure_ascii=False).encode("utf-8")) <= 4096
        assert meta["data"]["small"] == "s"

    @pytest.mark.asyncio
    async def test_utf8_values_are_measured_in_bytes(self):
        gr = _extension_guardrail(send_meta=True)
        # 3 bytes per CJK char: 500 chars pass the char cap but eat 1500 bytes.
        key_metadata = {f"cjk{i}": "資" * 500 for i in range(6)}
        payload = await _sent_payload(gr, _meta_request_data(key_metadata))
        blob = json.dumps(payload["meta"], ensure_ascii=False).encode("utf-8")
        assert len(blob) <= 4096

    @pytest.mark.asyncio
    async def test_meta_data_fields_narrows_the_forwarded_set(self):
        gr = _extension_guardrail(send_meta=True, meta_data_fields=["cost_center"])
        payload = await _sent_payload(
            gr, _meta_request_data({"cost_center": "CC-42", "owner": "alice@corp"})
        )
        assert payload["meta"]["data"] == {"cost_center": "CC-42"}

    @pytest.mark.asyncio
    async def test_callback_credential_slots_are_never_forwarded(self):
        # the proxy strips these before injecting; belt-and-braces here because a
        # leak would ship per-key integration credentials to an external service
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _meta_request_data(
                {
                    "logging": "langfuse-secret",
                    "callback_settings": "s",
                    "secret_manager_settings": "sm",
                    "keep": "v",
                }
            ),
        )
        assert _admin_data(payload["meta"]) == {"keep": "v"}

    @pytest.mark.asyncio
    async def test_credential_slots_cannot_be_opted_back_in(self):
        # meta_data_fields is an admin convenience, not an override for the
        # credential blocklist -- naming one must not spring the leak
        gr = _extension_guardrail(send_meta=True, meta_data_fields=["logging", "keep"])
        payload = await _sent_payload(
            gr, _meta_request_data({"logging": "langfuse-secret", "keep": "v"})
        )
        assert _admin_data(payload["meta"]) == {"keep": "v"}

    @pytest.mark.asyncio
    async def test_proxy_control_settings_are_skipped_by_default(self):
        # The Virtual Keys page writes the proxy's own per-key knobs into the same
        # metadata dict as the admin's fields. They are configuration, not caller
        # identity: noise in a SIEM, they eat the 32-field budget, and
        # disable_global_guardrails describes the key's security posture.
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _meta_request_data(
                {
                    "tag_rpm_limit": "{}",
                    "throttle_on_budget_exceeded": False,
                    "disable_global_guardrails": True,
                    "enforced_params": "x",
                    "cost_center": "CC-42",
                }
            ),
        )
        assert _admin_data(payload["meta"]) == {"cost_center": "CC-42"}

    @pytest.mark.asyncio
    async def test_a_named_control_setting_is_opted_back_in(self):
        # unlike the credential slots, forwarding these is a judgement call --
        # a deployment that wants its rate-limit config in the SIEM can say so
        gr = _extension_guardrail(
            send_meta=True,
            meta_data_fields=["throttle_on_budget_exceeded", "cost_center"],
        )
        payload = await _sent_payload(
            gr,
            _meta_request_data(
                {"throttle_on_budget_exceeded": False, "cost_center": "CC-42"}
            ),
        )
        assert _admin_data(payload["meta"]) == {
            "throttle_on_budget_exceeded": "false",
            "cost_center": "CC-42",
        }

    def test_the_control_field_list_tracks_the_proxys_own(self):
        """The skip list is derived from the proxy's lists, not copied.

        A hardcoded copy silently rots: litellm adds a metadata-backed knob, it
        starts appearing as ctx_<knob> in the customer's SIEM, and nobody
        notices. Assert the derivation instead of the contents.
        """
        from litellm.proxy._types import (
            LiteLLM_ManagementEndpoint_MetadataFields,
            LiteLLM_ManagementEndpoint_MetadataFields_Premium,
        )

        proxy_fields = set(LiteLLM_ManagementEndpoint_MetadataFields) | set(
            LiteLLM_ManagementEndpoint_MetadataFields_Premium
        )
        control = xecguard_module._META_CONTROL_DATA_FIELDS
        excluded = xecguard_module._META_EXCLUDED_DATA_FIELDS
        assert control == proxy_fields - excluded
        # the two tiers must not overlap, or the opt-in path would reach a
        # credential slot
        assert not (control & excluded)
        # sanity: the fields that prompted this are actually covered
        assert {"tag_rpm_limit", "throttle_on_budget_exceeded"} <= control
        assert "logging" in excluded and "logging" not in control

    @pytest.mark.asyncio
    async def test_data_omitted_when_nothing_survives_the_filter(self):
        # empty `data` is not the same as absent: send meta without the key.
        # Reaching that now takes a meta_data_fields matching neither source,
        # since a key with an alias always contributes at least `key_alias`.
        gr = _extension_guardrail(send_meta=True, meta_data_fields=["no_such_field"])
        for request_data in (_meta_request_data(), _meta_request_data({})):
            payload = await _sent_payload(gr, request_data)
            assert payload["meta"] == {"virtualkey": "team-alpha"}

    @pytest.mark.asyncio
    async def test_data_is_only_the_injected_attributes_when_the_key_has_no_metadata(
        self,
    ):
        gr = _extension_guardrail(send_meta=True)
        for request_data in (_meta_request_data(), _meta_request_data({})):
            payload = await _sent_payload(gr, request_data)
            assert payload["meta"]["data"] == {"key_alias": "team-alpha"}

    @pytest.mark.asyncio
    async def test_response_scan_carries_meta_too(self):
        gr = _extension_guardrail(send_meta=True, event_hook="post_call")
        data = _meta_request_data({"cost_center": "CC-42"})
        data["response"] = _build_model_response("the answer")
        resp = _make_response({"decision": "SAFE", "trace_id": "meta-post"})
        with patch.object(gr.async_handler, "post", return_value=resp) as mock_post:
            await gr.apply_guardrail(
                inputs={"texts": ["the answer"]},
                request_data=data,
                input_type="response",
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["scan_type"] == "response"
        assert _admin_data(payload["meta"]) == {"cost_center": "CC-42"}

    @pytest.mark.asyncio
    async def test_logging_only_path_carries_meta_from_nested_metadata(self):
        # logging_only goes through async_logging_hook, where the injected key
        # fields sit under litellm_params.metadata rather than at the top level
        gr = _extension_guardrail(send_meta=True, event_hook="logging_only")
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "standard_logging_object": {},
            "litellm_params": {
                "metadata": {
                    "user_api_key_alias": "team-alpha",
                    "user_api_key_metadata": {"cost_center": "CC-42"},
                }
            },
        }
        resp = _make_response({"decision": "SAFE", "trace_id": "meta-log"})
        with patch.object(gr.async_handler, "post", return_value=resp) as mock_post:
            await gr.async_logging_hook(
                kwargs=kwargs,
                result=_build_model_response("some answer"),
                call_type="acompletion",
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["meta"]["virtualkey"] == "team-alpha"
        assert _admin_data(payload["meta"]) == {"cost_center": "CC-42"}

    @pytest.mark.asyncio
    async def test_grounding_call_does_not_get_meta(self):
        # the contract defines meta for /scan only
        gr = _extension_guardrail(send_meta=True, event_hook="post_call")
        data = _meta_request_data({"cost_center": "CC-42"})
        data["response"] = _build_model_response("Peggy Seeger was American.")
        data["metadata"]["xecguard_grounding_documents"] = [
            {"document_id": "d1", "context": "Peggy Seeger is American."}
        ]
        scan_ok = _make_response({"decision": "SAFE", "trace_id": "g1"})
        grounding_ok = _make_response({"decision": "SAFE", "trace_id": "g2"})
        with patch.object(
            gr.async_handler, "post", side_effect=[scan_ok, grounding_ok]
        ) as mock_post:
            out = await gr.apply_guardrail(
                inputs={"texts": ["text"]},
                request_data=data,
                input_type="response",
            )
        assert out["texts"] == ["text"], "two SAFE verdicts leave the response alone"
        assert "meta" in mock_post.call_args_list[0].kwargs["json"]
        assert "meta" not in mock_post.call_args_list[1].kwargs["json"]

    def test_env_var_enables_meta(self):
        with patch.dict(os.environ, {"XECGUARD_SEND_META": "true"}, clear=False):
            assert _extension_guardrail().send_meta is True
        with patch.dict(os.environ, {"XECGUARD_SEND_META": "false"}, clear=False):
            assert _extension_guardrail().send_meta is False

    def test_explicit_config_beats_the_env_var(self):
        with patch.dict(os.environ, {"XECGUARD_SEND_META": "true"}, clear=False):
            assert _extension_guardrail(send_meta=False).send_meta is False

    def test_config_model_normalizes_meta_data_fields(self):
        cfg = XecGuardConfigModel(meta_data_fields="cost_center, owner ")
        assert cfg.meta_data_fields == ["cost_center", "owner"]
        assert XecGuardConfigModel().meta_data_fields is None

    def test_an_explicit_none_stays_none(self):
        # distinct from omitting the field: a before-validator does not run on an
        # unset default, so this is the only way that branch is reached
        assert XecGuardConfigModel(apply_to_aliases=None).apply_to_aliases is None
        assert XecGuardConfigModel(meta_data_fields=None).meta_data_fields is None

    def test_a_wrong_type_is_handed_back_for_pydantic_to_reject(self):
        """The normalizer returns an unrecognised value unchanged on purpose.

        Coercing it -- to None, or to [] -- would turn a typo in the config into a
        silently empty allowlist, which for `apply_to_aliases` means scanning every
        key instead of the chosen ones. Letting pydantic reject it keeps the
        mistake loud.
        """
        for bad in (5, {"a": 1}, True):
            with pytest.raises(ValidationError):
                XecGuardConfigModel(apply_to_aliases=bad)


# ---------------------------------------------------------------------------
# meta.data also carries what the *proxy* knows about the calling key, not just
# what an admin typed. Without it a SIEM event is a bare alias, and attributing
# it needs a lookup back into the proxy database that whoever reads the SIEM
# usually cannot make. These fields include PII (user_email) and commercials
# (spend, max_budget) and leave the proxy only when send_meta is on.
# ---------------------------------------------------------------------------


def _injected_request_data(**injected):
    """request_data whose metadata holds only proxy-injected key attributes."""
    return {"messages": [{"role": "user", "content": "hi"}], "metadata": injected}


@pytest.mark.usefixtures("_clean_env")
class TestXecGuardScanMetaAutoFields:
    @pytest.mark.asyncio
    async def test_the_full_injected_set_is_forwarded(self):
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _injected_request_data(
                user_api_key_hash="abc123",
                user_api_key_alias="team-alpha",
                user_api_key_team_id="t-1",
                user_api_key_team_alias="platform",
                user_api_key_user_id="u-1",
                user_api_key_user_email="alice@corp",
                user_api_key_org_id="o-1",
                user_api_key_org_alias="acme",
                user_api_key_project_id="p-1",
                user_api_key_project_alias="proj",
                user_api_key_end_user_id="eu-1",
                user_api_key_spend=1.5,
                user_api_key_max_budget=100,
                user_api_key_request_route="/chat/completions",
            ),
        )
        assert payload["meta"]["data"] == {
            "key_id": "abc123",
            "key_alias": "team-alpha",
            "team_id": "t-1",
            "team_alias": "platform",
            "user_id": "u-1",
            "user_email": "alice@corp",
            "org_id": "o-1",
            "org_alias": "acme",
            "project_id": "p-1",
            "project_alias": "proj",
            "end_user_id": "eu-1",
            "spend": "1.5",
            "max_budget": "100",
            "request_route": "/chat/completions",
        }

    @pytest.mark.asyncio
    async def test_absent_and_null_attributes_are_skipped(self):
        # a key with no team must contribute no team_id rather than an empty one:
        # a SIEM query for "scans with no team" should mean it, not match every key
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _injected_request_data(
                user_api_key_alias="team-alpha",
                user_api_key_team_id=None,
                user_api_key_user_email=None,
            ),
        )
        assert payload["meta"]["data"] == {"key_alias": "team-alpha"}

    @pytest.mark.asyncio
    async def test_injected_order_is_the_constants_order(self):
        # identity, then tenancy, then commercials - so the fields that survive
        # the 32-field / 4096-byte caps are the ones worth keeping, whatever
        # order the proxy happened to build its metadata dict in
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _injected_request_data(
                user_api_key_spend=1.5,
                user_api_key_team_id="t-1",
                user_api_key_alias="team-alpha",
                user_api_key_hash="abc123",
            ),
        )
        assert list(payload["meta"]["data"]) == [
            "key_id",
            "key_alias",
            "team_id",
            "spend",
        ]

    @pytest.mark.asyncio
    async def test_an_admin_cannot_shadow_an_injected_attribute(self):
        # otherwise a key whose owner controls its metadata could write
        # key_id/team_id of someone else and mislead an investigation
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _meta_request_data(
                {"key_id": "not-mine", "team_id": "not-mine", "mine": "ok"},
                alias="team-alpha",
                user_api_key_hash="abc123",
                user_api_key_team_id="t-1",
            ),
        )
        data = payload["meta"]["data"]
        assert data["key_id"] == "abc123"
        assert data["team_id"] == "t-1"
        assert data["mine"] == "ok"

    @pytest.mark.asyncio
    async def test_meta_data_fields_narrows_the_injected_set_too(self):
        # the allowlist is what a deployment forbidden to egress PII or spend
        # figures uses, so it has to bind the injected half as well
        gr = _extension_guardrail(
            send_meta=True, meta_data_fields=["key_id", "cost_center"]
        )
        payload = await _sent_payload(
            gr,
            _meta_request_data(
                {"cost_center": "CC-42", "owner": "alice@corp"},
                alias="team-alpha",
                user_api_key_hash="abc123",
                user_api_key_user_email="alice@corp",
                user_api_key_spend=1.5,
            ),
        )
        assert payload["meta"]["data"] == {"key_id": "abc123", "cost_center": "CC-42"}

    @pytest.mark.asyncio
    async def test_nested_metadata_is_read_on_the_logging_only_path(self):
        # async_logging_hook sees the injected fields under litellm_params.metadata
        gr = _extension_guardrail(send_meta=True, event_hook="logging_only")
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "standard_logging_object": {},
            "litellm_params": {
                "metadata": {
                    "user_api_key_alias": "team-alpha",
                    "user_api_key_hash": "abc123",
                    "user_api_key_team_alias": "platform",
                }
            },
        }
        resp = _make_response({"decision": "SAFE", "trace_id": "meta-auto-log"})
        with patch.object(gr.async_handler, "post", return_value=resp) as mock_post:
            await gr.async_logging_hook(
                kwargs=kwargs,
                result=_build_model_response("some answer"),
                call_type="acompletion",
            )
        info_list = kwargs["standard_logging_object"]["guardrail_information"]
        assert info_list[0]["guardrail_response"]["trace_id"] == "meta-auto-log"
        assert mock_post.call_args.kwargs["json"]["meta"]["data"] == {
            "key_id": "abc123",
            "key_alias": "team-alpha",
            "team_alias": "platform",
        }

    @pytest.mark.asyncio
    async def test_injected_attributes_obey_the_value_coercion_rules(self):
        gr = _extension_guardrail(send_meta=True)
        payload = await _sent_payload(
            gr,
            _injected_request_data(
                user_api_key_alias="team-alpha",
                user_api_key_user_email="a\x00b@corp",
                user_api_key_team_alias="y" * 700,
                user_api_key_model_max_budget={"gpt-4o": 1},
            ),
        )
        data = payload["meta"]["data"]
        assert data["user_email"] == "ab@corp"
        assert len(data["team_alias"]) == 512
        # model_max_budget is not in the forwarded set, and would be dropped as a
        # nested value even if it were
        assert "model_max_budget" not in data

    @pytest.mark.asyncio
    async def test_nothing_is_forwarded_while_send_meta_is_off(self):
        gr = _extension_guardrail()
        payload = await _sent_payload(
            gr,
            _injected_request_data(
                user_api_key_alias="team-alpha", user_api_key_hash="abc123"
            ),
        )
        assert "meta" not in payload


# ---------------------------------------------------------------------------
# meta.virtualkey has two wire shapes. The string form is the alias, which is
# all the deployed backend accepts; the object form carries {alias, key_id} so a
# scan stays attributable when the alias is absent, renamed, or reused. Sending
# the object form to a string-only backend is a 400, which block_on_error turns
# into a block for every request - hence a switch, defaulting to string.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_env")
class TestXecGuardScanMetaVirtualkeyObject:
    def test_string_is_the_default(self):
        assert _extension_guardrail(send_meta=True).meta_identity_format == "string"

    def test_env_var_selects_the_format(self):
        with patch.dict(
            os.environ, {"XECGUARD_META_IDENTITY_FORMAT": "OBJECT"}, clear=False
        ):
            assert _extension_guardrail(send_meta=True).meta_identity_format == "object"

    def test_explicit_config_beats_the_env_var(self):
        with patch.dict(
            os.environ, {"XECGUARD_META_IDENTITY_FORMAT": "object"}, clear=False
        ):
            gr = _extension_guardrail(send_meta=True, meta_identity_format="string")
            assert gr.meta_identity_format == "string"

    def test_an_unknown_value_falls_back_instead_of_raising(self):
        # a typo in the UI must not take the gateway down at startup
        gr = _extension_guardrail(send_meta=True, meta_identity_format="objekt")
        assert gr.meta_identity_format == "string"

    def test_config_model_offers_both_shapes_to_the_ui(self):
        assert XecGuardConfigModel().meta_identity_format is None
        assert (
            XecGuardConfigModel(meta_identity_format="object").meta_identity_format
            == "object"
        )
        with pytest.raises(ValidationError):
            XecGuardConfigModel(meta_identity_format="objekt")

    def test_no_non_secret_field_is_masked_on_the_way_back_to_the_ui(self):
        """Only `api_key` may be masked when the proxy serves this provider's
        params back to the UI.

        ``_get_masked_values`` matches on *substrings* of the field name --
        "key", "token", "secret", "credentials", "password" -- and the guardrail
        endpoints run every ``litellm_params`` dict through it. A non-secret
        field caught by that heuristic is served to the UI as "ob****ct"; the
        edit form prefills it, saving writes the masked string back, and the
        value is silently wrong from then on. That is why this field is
        ``meta_identity_format`` and not ``meta_virtualkey_format``: the latter
        matches on "virtual*key*".

        Enum and free-text fields are the dangerous ones. Booleans and lists
        survive by type (``_mask_value`` returns non-str unchanged), so a name
        collision there is latent rather than live -- still worth failing on,
        since changing such a field to a string would spring the trap.
        """
        from litellm.litellm_core_utils.litellm_logging import _get_masked_values

        probe = {name: "sentinel" for name in XecGuardConfigModel.model_fields}
        served = _get_masked_values(probe, unmasked_length=4, number_of_asterisks=4)
        masked = {name for name, value in served.items() if value != "sentinel"}
        assert masked == {"api_key"}, (
            "these XecGuard params would reach the UI masked and be corrupted by a "
            f"form save: {sorted(masked - {'api_key'})}. Rename them off the "
            "substrings _get_masked_values matches on."
        )

    @pytest.mark.asyncio
    async def test_object_form_carries_alias_and_key_id(self):
        gr = _extension_guardrail(send_meta=True, meta_identity_format="object")
        payload = await _sent_payload(
            gr, _meta_request_data(alias="team-alpha", user_api_key_hash="abc123")
        )
        assert payload["meta"]["virtualkey"] == {
            "alias": "team-alpha",
            "key_id": "abc123",
        }

    @pytest.mark.asyncio
    async def test_either_member_may_be_absent(self):
        gr = _extension_guardrail(send_meta=True, meta_identity_format="object")
        no_alias = await _sent_payload(
            gr, _injected_request_data(user_api_key_hash="abc123")
        )
        assert no_alias["meta"]["virtualkey"] == {"key_id": "abc123"}
        no_hash = await _sent_payload(gr, _meta_request_data(alias="team-alpha"))
        assert no_hash["meta"]["virtualkey"] == {"alias": "team-alpha"}

    @pytest.mark.asyncio
    async def test_meta_is_omitted_when_the_key_has_neither(self):
        # a master-key call: nothing to correlate on, so omit meta rather than
        # send an empty object and collect a 400
        gr = _extension_guardrail(send_meta=True, meta_identity_format="object")
        assert "meta" not in await _sent_payload(gr, _injected_request_data())

    @pytest.mark.asyncio
    async def test_the_object_form_lifts_the_identifier_pattern_on_aliases(self):
        # the pattern exists because a bare string becomes a SIEM field *value*
        # directly; an object member is sanitized like a meta.data value instead,
        # so aliases with spaces or CJK become correlatable
        gr = _extension_guardrail(send_meta=True, meta_identity_format="object")
        payload = await _sent_payload(
            gr, _meta_request_data(alias="研發 team", user_api_key_hash="abc123")
        )
        assert payload["meta"]["virtualkey"] == {
            "alias": "研發 team",
            "key_id": "abc123",
        }
        # the string form cannot: it falls through the failing alias to the hash
        string_gr = _extension_guardrail(send_meta=True)
        string_payload = await _sent_payload(
            string_gr, _meta_request_data(alias="研發 team", user_api_key_hash="abc123")
        )
        assert string_payload["meta"]["virtualkey"] == "abc123"

    @pytest.mark.asyncio
    async def test_object_members_are_sanitized_and_truncated(self):
        gr = _extension_guardrail(send_meta=True, meta_identity_format="object")
        payload = await _sent_payload(
            gr, _meta_request_data(alias="a\x00b\x1fc", user_api_key_hash="h" * 700)
        )
        virtualkey = payload["meta"]["virtualkey"]
        assert virtualkey["alias"] == "abc"
        assert len(virtualkey["key_id"]) == 512

    @pytest.mark.asyncio
    async def test_the_serialized_cap_still_holds_with_the_object_form(self):
        gr = _extension_guardrail(send_meta=True, meta_identity_format="object")
        key_metadata = {f"big{i}": "y" * 500 for i in range(20)}
        payload = await _sent_payload(
            gr,
            _meta_request_data(
                key_metadata, alias="team-alpha", user_api_key_hash="abc123"
            ),
        )
        meta = payload["meta"]
        assert meta["virtualkey"] == {"alias": "team-alpha", "key_id": "abc123"}
        assert len(json.dumps(meta, ensure_ascii=False).encode("utf-8")) <= 4096

    @pytest.mark.asyncio
    async def test_data_is_unaffected_by_the_format(self):
        gr = _extension_guardrail(send_meta=True, meta_identity_format="object")
        payload = await _sent_payload(
            gr,
            _meta_request_data(
                {"cost_center": "CC-42"}, alias="team-alpha", user_api_key_hash="abc123"
            ),
        )
        assert _admin_data(payload["meta"]) == {"cost_center": "CC-42"}
        assert payload["meta"]["data"]["key_id"] == "abc123"
