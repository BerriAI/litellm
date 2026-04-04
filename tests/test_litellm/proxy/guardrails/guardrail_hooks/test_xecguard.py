import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm

from litellm.proxy.guardrails.guardrail_hooks.xecguard import (
    XecGuardGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.proxy.guardrails.guardrail_hooks.xecguard.xecguard import (
    DEFAULT_POLICY_NAMES,
    GUARDRAIL_NAME,
    GROUNDING_ENDPOINT,
    SCAN_ENDPOINT,
    _extract_text,
    _last_role,
    _litellm_messages_to_xecguard,
    _pre_register_guardrail_info,
)
from litellm.proxy.guardrails.guardrail_registry import (
    guardrail_class_registry as global_class_registry,
)
from litellm.proxy.guardrails.guardrail_registry import (
    guardrail_initializer_registry as global_initializer_registry,
)
from litellm.types.proxy.guardrails.guardrail_hooks.xecguard import (
    XecGuardConfigModel,
    XecGuardUIConfigModel,
)


# ---------------------------------------------------------------------------
#  Registry tests
# ---------------------------------------------------------------------------


def test_xecguard_in_local_initializer_registry():
    assert "xecguard" in guardrail_initializer_registry


def test_xecguard_in_local_class_registry():
    assert "xecguard" in guardrail_class_registry
    assert guardrail_class_registry["xecguard"] is XecGuardGuardrail


def test_xecguard_in_global_initializer_registry():
    assert "xecguard" in global_initializer_registry


def test_xecguard_in_global_class_registry():
    assert "xecguard" in global_class_registry
    assert global_class_registry["xecguard"] is XecGuardGuardrail


# ---------------------------------------------------------------------------
#  Enum test
# ---------------------------------------------------------------------------


def test_xecguard_enum_value():
    from litellm.types.guardrails import SupportedGuardrailIntegrations

    assert SupportedGuardrailIntegrations.XECGUARD.value == "xecguard"


# ---------------------------------------------------------------------------
#  Config model tests
# ---------------------------------------------------------------------------


class TestXecGuardConfigModel:
    def test_ui_friendly_name(self):
        assert XecGuardConfigModel.ui_friendly_name() == "XecGuard"

    def test_default_values(self):
        config = XecGuardConfigModel()
        assert config.api_key is None
        assert config.api_base is None
        assert config.model == "xecguard_v2"
        assert config.policy_names is None
        assert config.grounding_enabled is False
        assert config.grounding_strictness == "BALANCED"
        assert config.grounding_documents is None

    def test_custom_values(self):
        config = XecGuardConfigModel(
            api_key="test-key",
            api_base="https://custom.example.com",
            model="xecguard_v3",
            policy_names=["policy1"],
            grounding_enabled=True,
            grounding_strictness="STRICT",
            grounding_documents=[{"document_id": "0", "context": "some context"}],
        )
        assert config.api_key == "test-key"
        assert config.api_base == "https://custom.example.com"
        assert config.model == "xecguard_v3"
        assert config.policy_names == ["policy1"]
        assert config.grounding_enabled is True
        assert config.grounding_strictness == "STRICT"
        assert config.grounding_documents == [{"document_id": "0", "context": "some context"}]


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def xecguard():
    """XecGuardGuardrail configured for during_call."""
    return XecGuardGuardrail(
        api_key="test-token",
        api_base="https://api-xecguard.test.com",
        guardrail_name="test-xecguard",
        event_hook="during_call",
    )


@pytest.fixture
def xecguard_pre_call():
    """XecGuardGuardrail configured for pre_call."""
    return XecGuardGuardrail(
        api_key="test-token",
        api_base="https://api-xecguard.test.com",
        guardrail_name="test-xecguard-pre",
        event_hook="pre_call",
    )


@pytest.fixture
def xecguard_post_call():
    """XecGuardGuardrail configured for post_call with grounding."""
    return XecGuardGuardrail(
        api_key="test-token",
        api_base="https://api-xecguard.test.com",
        grounding_enabled=True,
        grounding_documents=[{"document_id": "d1", "context": "X is Y"}],
        guardrail_name="test-xecguard-post",
        event_hook="post_call",
    )


def _mock_safe_scan_response():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {
        "decision": "SAFE",
        "trace_id": "trace-123",
        "xecguard_result": [],
    }
    return mock


def _mock_unsafe_scan_response():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {
        "decision": "UNSAFE",
        "trace_id": "trace-456",
        "xecguard_result": [
            {
                "type": "VIOLATION_HARMFUL",
                "violated_policy_name": "Default_Policy_HarmfulContentProtection",
                "rationale": "Content contains harmful instructions",
            }
        ],
    }
    return mock


def _mock_safe_grounding_response():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {
        "decision": "SAFE",
        "trace_id": "trace-789",
        "xecguard_result": {},
    }
    return mock


def _mock_unsafe_grounding_response():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {
        "decision": "UNSAFE",
        "trace_id": "trace-999",
        "xecguard_result": {
            "rationale": "Response is not grounded",
            "violated_rules_list": ["rule1"],
        },
    }
    return mock


def _mock_error_response(status_code=500, text="Internal Server Error"):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.text = text
    return mock


# ---------------------------------------------------------------------------
#  Initialization tests
# ---------------------------------------------------------------------------


class TestXecGuardInitialization:
    def test_explicit_params(self):
        docs = [{"document_id": "0", "context": "test context"}]
        g = XecGuardGuardrail(
            api_key="my-key",
            api_base="https://custom.example.com/",
            model="xecguard_v3",
            policy_names=["policy_a"],
            grounding_enabled=True,
            grounding_strictness="STRICT",
            grounding_documents=docs,
            guardrail_name="test",
            event_hook="pre_call",
        )
        assert g.api_key == "my-key"
        assert g.api_base == "https://custom.example.com"  # trailing slash stripped
        assert g.model == "xecguard_v3"
        assert g.policy_names == ["policy_a"]
        assert g.grounding_enabled is True
        assert g.grounding_strictness == "STRICT"
        assert g.grounding_documents == docs
        assert g.guardrail_provider == GUARDRAIL_NAME

    def test_default_grounding_documents_empty(self):
        g = XecGuardGuardrail(
            api_key="k",
            guardrail_name="test",
            event_hook="pre_call",
        )
        assert g.grounding_documents == []

    def test_env_fallback(self):
        with patch.dict(
            os.environ,
            {
                "XECGUARD_SERVICE_TOKEN": "env-token",
                "XECGUARD_API_BASE": "https://env-api.example.com",
            },
        ):
            g = XecGuardGuardrail(
                guardrail_name="test",
                event_hook="pre_call",
            )
            assert g.api_key == "env-token"
            assert g.api_base == "https://env-api.example.com"

    def test_default_policies(self):
        g = XecGuardGuardrail(
            api_key="k",
            guardrail_name="test",
            event_hook="pre_call",
        )
        assert g.policy_names == list(DEFAULT_POLICY_NAMES)

    def test_default_api_base(self):
        g = XecGuardGuardrail(
            api_key="k",
            guardrail_name="test",
            event_hook="pre_call",
        )
        assert g.api_base == "https://api-xecguard.cycraft.ai"

    def test_get_config_model_returns_ui_model(self):
        """get_config_model() returns the UI model which excludes grounding fields."""
        assert XecGuardGuardrail.get_config_model() is XecGuardUIConfigModel

    def test_ui_config_model_excludes_grounding_fields(self):
        """XecGuardUIConfigModel must NOT expose grounding_enabled / grounding_strictness."""
        ui_fields = set(XecGuardUIConfigModel.model_fields.keys())
        assert "grounding_enabled" not in ui_fields
        assert "grounding_strictness" not in ui_fields

    def test_full_config_model_includes_grounding_fields(self):
        """XecGuardConfigModel (API-level) must still have grounding fields."""
        api_fields = set(XecGuardConfigModel.model_fields.keys())
        assert "grounding_enabled" in api_fields
        assert "grounding_strictness" in api_fields


# ---------------------------------------------------------------------------
#  Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_text_string(self):
        assert _extract_text("hello") == "hello"

    def test_extract_text_list(self):
        content = [
            {"type": "text", "text": "part1"},
            {"type": "text", "text": "part2"},
        ]
        assert _extract_text(content) == "part1\npart2"

    def test_extract_text_empty(self):
        assert _extract_text(None) == ""
        assert _extract_text("") == ""

    def test_litellm_messages_to_xecguard(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = _litellm_messages_to_xecguard(messages)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_litellm_messages_skips_empty(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "hello"},
        ]
        result = _litellm_messages_to_xecguard(messages)
        assert len(result) == 1

    def test_last_role(self):
        assert _last_role([{"role": "user"}]) == "user"
        assert _last_role([{"role": "assistant"}]) == "assistant"
        assert _last_role([]) == "user"


# ---------------------------------------------------------------------------
#  Decision helper tests
# ---------------------------------------------------------------------------


class TestDecisionHelpers:
    def test_safe_scan_no_raise(self):
        XecGuardGuardrail._raise_if_unsafe_scan(
            {"decision": "SAFE", "xecguard_result": []}
        )

    def test_unsafe_scan_raises(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            XecGuardGuardrail._raise_if_unsafe_scan(
                {
                    "decision": "UNSAFE",
                    "trace_id": "t1",
                    "xecguard_result": [
                        {
                            "type": "VIOLATION_HARMFUL",
                            "violated_policy_name": "policy1",
                            "rationale": "bad content",
                        }
                    ],
                }
            )
        assert exc_info.value.status_code == 400
        assert "XecGuard scan blocked" in exc_info.value.detail["error"]

    def test_safe_grounding_no_raise(self):
        XecGuardGuardrail._raise_if_unsafe_grounding(
            {"decision": "SAFE", "xecguard_result": {}}
        )

    def test_unsafe_grounding_raises(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            XecGuardGuardrail._raise_if_unsafe_grounding(
                {
                    "decision": "UNSAFE",
                    "trace_id": "t2",
                    "xecguard_result": {
                        "rationale": "not grounded",
                        "violated_rules_list": ["rule1"],
                    },
                }
            )
        assert exc_info.value.status_code == 400
        assert "XecGuard grounding failed" in exc_info.value.detail["error"]


# ---------------------------------------------------------------------------
#  Scan API call tests
# ---------------------------------------------------------------------------


class TestScanAPI:
    @pytest.mark.asyncio
    async def test_scan_safe(self, xecguard):
        xecguard.async_handler.post = AsyncMock(
            return_value=_mock_safe_scan_response()
        )

        result = await xecguard._call_scan(
            scan_type="input",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result["decision"] == "SAFE"

        call_kwargs = xecguard.async_handler.post.call_args
        assert SCAN_ENDPOINT in call_kwargs.kwargs["url"]
        body = call_kwargs.kwargs["json"]
        assert body["scan_type"] == "input"
        assert body["model"] == "xecguard_v2"
        assert body["policy_names"] == list(DEFAULT_POLICY_NAMES)

    @pytest.mark.asyncio
    async def test_scan_unsafe(self, xecguard):
        xecguard.async_handler.post = AsyncMock(
            return_value=_mock_unsafe_scan_response()
        )

        result = await xecguard._call_scan(
            scan_type="input",
            messages=[{"role": "user", "content": "harmful request"}],
        )
        assert result["decision"] == "UNSAFE"

    @pytest.mark.asyncio
    async def test_scan_413_error(self, xecguard):
        from fastapi import HTTPException

        xecguard.async_handler.post = AsyncMock(
            return_value=_mock_error_response(413, "Content Too Large")
        )

        with pytest.raises(HTTPException) as exc_info:
            await xecguard._call_scan(
                scan_type="input",
                messages=[{"role": "user", "content": "huge content"}],
            )
        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    async def test_scan_500_error(self, xecguard):
        from fastapi import HTTPException

        xecguard.async_handler.post = AsyncMock(
            return_value=_mock_error_response(500, "Server Error")
        )

        with pytest.raises(HTTPException) as exc_info:
            await xecguard._call_scan(
                scan_type="input",
                messages=[{"role": "user", "content": "test"}],
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_scan_headers(self, xecguard):
        xecguard.async_handler.post = AsyncMock(
            return_value=_mock_safe_scan_response()
        )

        await xecguard._call_scan(
            scan_type="input",
            messages=[{"role": "user", "content": "test"}],
        )

        call_kwargs = xecguard.async_handler.post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
#  Grounding API call tests
# ---------------------------------------------------------------------------


class TestGroundingAPI:
    @pytest.mark.asyncio
    async def test_grounding_safe(self, xecguard_post_call):
        xecguard_post_call.async_handler.post = AsyncMock(
            return_value=_mock_safe_grounding_response()
        )

        result = await xecguard_post_call._call_grounding(
            prompt="What is X?",
            response_text="X is Y.",
            documents=[{"document_id": "d1", "context": "X is Y"}],
        )
        assert result["decision"] == "SAFE"

        call_kwargs = xecguard_post_call.async_handler.post.call_args
        assert GROUNDING_ENDPOINT in call_kwargs.kwargs["url"]
        body = call_kwargs.kwargs["json"]
        assert body["prompt"] == "What is X?"
        assert body["response"] == "X is Y."
        assert body["strictness"] == "BALANCED"

    @pytest.mark.asyncio
    async def test_grounding_413(self, xecguard_post_call):
        from fastapi import HTTPException

        xecguard_post_call.async_handler.post = AsyncMock(
            return_value=_mock_error_response(413, "Content Too Large")
        )

        with pytest.raises(HTTPException) as exc_info:
            await xecguard_post_call._call_grounding(
                prompt="p", response_text="r"
            )
        assert exc_info.value.status_code == 413


# ---------------------------------------------------------------------------
#  No apply_guardrail defined – verify direct hook dispatch
# ---------------------------------------------------------------------------


class TestNoApplyGuardrail:
    """XecGuardGuardrail must NOT define apply_guardrail.

    When ``apply_guardrail`` exists on a guardrail class the framework
    routes ALL modes (pre_call, during_call, post_call) through the
    unified guardrail path, which bypasses the guardrail's own hook
    implementations.  This breaks grounding (post_call) and
    pre-registration (during_call).

    XecGuard implements its own async_pre_call_hook,
    async_moderation_hook, and async_post_call_success_hook, so
    ``apply_guardrail`` must NOT be present.
    """

    def test_no_apply_guardrail_on_class(self):
        """apply_guardrail must not be in XecGuardGuardrail's own __dict__."""
        assert "apply_guardrail" not in XecGuardGuardrail.__dict__


# ---------------------------------------------------------------------------
#  Initializer tests
# ---------------------------------------------------------------------------


class TestInitializer:
    @patch("litellm.logging_callback_manager.add_litellm_callback")
    def test_initialize_guardrail(self, mock_add_callback):
        from litellm.proxy.guardrails.guardrail_hooks.xecguard import (
            initialize_guardrail,
        )

        litellm_params = MagicMock()
        litellm_params.api_key = "test-key"
        litellm_params.api_base = "https://test.example.com"
        litellm_params.model = "xecguard_v2"
        litellm_params.policy_names = None
        litellm_params.grounding_enabled = False
        litellm_params.grounding_strictness = "BALANCED"
        litellm_params.grounding_documents = None
        litellm_params.mode = "during_call"
        litellm_params.default_on = False

        guardrail = {"guardrail_name": "xecguard-test"}

        result = initialize_guardrail(litellm_params, guardrail)

        assert isinstance(result, XecGuardGuardrail)
        assert result.api_key == "test-key"
        assert result.api_base == "https://test.example.com"
        assert result.grounding_documents == []
        mock_add_callback.assert_called_once_with(result)

    @patch("litellm.logging_callback_manager.add_litellm_callback")
    def test_initialize_guardrail_with_grounding_documents(self, mock_add_callback):
        from litellm.proxy.guardrails.guardrail_hooks.xecguard import (
            initialize_guardrail,
        )

        docs = [{"document_id": "0", "context": "test context"}]
        litellm_params = MagicMock()
        litellm_params.api_key = "test-key"
        litellm_params.api_base = "https://test.example.com"
        litellm_params.model = "xecguard_v2"
        litellm_params.policy_names = None
        litellm_params.grounding_enabled = True
        litellm_params.grounding_strictness = "BALANCED"
        litellm_params.grounding_documents = docs
        litellm_params.mode = "post_call"
        litellm_params.default_on = True

        guardrail = {"guardrail_name": "xecguard-grounding"}

        result = initialize_guardrail(litellm_params, guardrail)

        assert isinstance(result, XecGuardGuardrail)
        assert result.grounding_enabled is True
        assert result.grounding_documents == docs
        mock_add_callback.assert_called_once_with(result)


# ---------------------------------------------------------------------------
#  Pre-registration helper tests
# ---------------------------------------------------------------------------


class TestPreRegisterGuardrailInfo:
    def test_creates_metadata_key_when_missing(self):
        from litellm.types.guardrails import GuardrailEventHooks

        data: dict = {"metadata": {}}
        placeholder = _pre_register_guardrail_info(
            data=data,
            guardrail_name="xg",
            event_type=GuardrailEventHooks.during_call,
            start_time=100.0,
        )
        key = "standard_logging_guardrail_information"
        assert key in data["metadata"]
        assert len(data["metadata"][key]) == 1
        assert data["metadata"][key][0] is placeholder
        assert placeholder["guardrail_status"] == "success"
        assert placeholder["guardrail_name"] == "xg"
        assert placeholder["start_time"] == 100.0

    def test_creates_metadata_dict_when_absent(self):
        from litellm.types.guardrails import GuardrailEventHooks

        data: dict = {}
        placeholder = _pre_register_guardrail_info(
            data=data,
            guardrail_name="xg",
            event_type=GuardrailEventHooks.during_call,
            start_time=1.0,
        )
        assert "metadata" in data
        assert data["metadata"]["standard_logging_guardrail_information"] == [
            placeholder
        ]

    def test_appends_to_existing_list(self):
        from litellm.types.guardrails import GuardrailEventHooks

        existing_entry = {"guardrail_name": "other", "guardrail_status": "success"}
        data: dict = {
            "metadata": {
                "standard_logging_guardrail_information": [existing_entry],
            }
        }
        placeholder = _pre_register_guardrail_info(
            data=data,
            guardrail_name="xg",
            event_type=GuardrailEventHooks.during_call,
            start_time=2.0,
        )
        info_list = data["metadata"]["standard_logging_guardrail_information"]
        assert len(info_list) == 2
        assert info_list[0] is existing_entry
        assert info_list[1] is placeholder

    def test_placeholder_is_same_object_in_metadata(self):
        """Mutating the returned placeholder updates the metadata entry."""
        from litellm.types.guardrails import GuardrailEventHooks

        data: dict = {"metadata": {}}
        placeholder = _pre_register_guardrail_info(
            data=data,
            guardrail_name="xg",
            event_type=GuardrailEventHooks.during_call,
            start_time=0.0,
        )
        # Simulate what async_moderation_hook does after scan completes
        placeholder["guardrail_status"] = "guardrail_intervened"
        placeholder["end_time"] = 5.0
        placeholder["duration"] = 5.0

        stored = data["metadata"]["standard_logging_guardrail_information"][0]
        assert stored["guardrail_status"] == "guardrail_intervened"
        assert stored["end_time"] == 5.0


# ---------------------------------------------------------------------------
#  during_call guardrail-info logging tests
# ---------------------------------------------------------------------------


@pytest.fixture
def xecguard_default_on():
    """XecGuardGuardrail configured for during_call with default_on=True."""
    return XecGuardGuardrail(
        api_key="test-token",
        api_base="https://api-xecguard.test.com",
        guardrail_name="test-xecguard",
        event_hook="during_call",
        default_on=True,
    )


class TestDuringCallGuardrailInfoLogging:
    """Verify async_moderation_hook pre-registers guardrail info so the
    Guardrails Monitor counts passed requests even when the LLM call
    completes before the guardrail scan."""

    @pytest.mark.asyncio
    async def test_passed_scan_registers_success(self, xecguard_default_on):
        """SAFE scan → placeholder stays guardrail_status='success'."""
        xecguard_default_on.async_handler.post = AsyncMock(
            return_value=_mock_safe_scan_response()
        )

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {},
        }
        await xecguard_default_on.async_moderation_hook(
            data=data,
            user_api_key_dict=MagicMock(),
            call_type="acompletion",
        )

        key = "standard_logging_guardrail_information"
        info_list = data["metadata"][key]
        assert len(info_list) == 1
        entry = info_list[0]
        assert entry["guardrail_status"] == "success"
        assert entry["guardrail_name"] == "test-xecguard"
        assert entry["guardrail_provider"] == GUARDRAIL_NAME
        assert entry["guardrail_response"]["decision"] == "SAFE"
        assert entry["start_time"] is not None
        assert entry["end_time"] is not None
        assert entry["duration"] > 0

    @pytest.mark.asyncio
    async def test_blocked_scan_registers_intervened(self, xecguard_default_on):
        """UNSAFE scan → placeholder updated to 'guardrail_intervened'."""
        from fastapi import HTTPException

        xecguard_default_on.async_handler.post = AsyncMock(
            return_value=_mock_unsafe_scan_response()
        )

        data = {
            "messages": [{"role": "user", "content": "harmful"}],
            "metadata": {},
        }
        with pytest.raises(HTTPException):
            await xecguard_default_on.async_moderation_hook(
                data=data,
                user_api_key_dict=MagicMock(),
                call_type="acompletion",
            )

        key = "standard_logging_guardrail_information"
        info_list = data["metadata"][key]
        assert len(info_list) == 1
        entry = info_list[0]
        assert entry["guardrail_status"] == "guardrail_intervened"
        assert entry["end_time"] is not None

    @pytest.mark.asyncio
    async def test_api_error_registers_failed(self, xecguard_default_on):
        """Non-200 API error → 'guardrail_failed_to_respond'."""
        from fastapi import HTTPException

        xecguard_default_on.async_handler.post = AsyncMock(
            return_value=_mock_error_response(500, "Server Error")
        )

        data = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": {},
        }
        with pytest.raises(HTTPException):
            await xecguard_default_on.async_moderation_hook(
                data=data,
                user_api_key_dict=MagicMock(),
                call_type="acompletion",
            )

        key = "standard_logging_guardrail_information"
        entry = data["metadata"][key][0]
        assert entry["guardrail_status"] == "guardrail_failed_to_respond"

    @pytest.mark.asyncio
    async def test_info_available_before_scan_completes(self, xecguard_default_on):
        """Guardrail info is in metadata immediately after pre-registration,
        before the scan HTTP call returns."""
        captured_metadata_snapshot = {}

        async def _capturing_post(**kwargs):
            # At this point the scan is in-flight; snapshot metadata
            key = "standard_logging_guardrail_information"
            info = captured_metadata_snapshot.setdefault("info", [])
            meta = data.get("metadata", {})
            info.extend(meta.get(key, []))
            return _mock_safe_scan_response()

        xecguard_default_on.async_handler.post = _capturing_post

        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {},
        }
        await xecguard_default_on.async_moderation_hook(
            data=data,
            user_api_key_dict=MagicMock(),
            call_type="acompletion",
        )

        # The snapshot captured during the HTTP call should already have
        # the pre-registered guardrail info
        assert len(captured_metadata_snapshot["info"]) == 1
        assert captured_metadata_snapshot["info"][0]["guardrail_status"] == "success"

    @pytest.mark.asyncio
    async def test_no_messages_skips_registration(self, xecguard_default_on):
        """No messages → no guardrail info registered."""
        data = {"messages": [], "metadata": {}}
        await xecguard_default_on.async_moderation_hook(
            data=data,
            user_api_key_dict=MagicMock(),
            call_type="acompletion",
        )
        assert "standard_logging_guardrail_information" not in data["metadata"]


# ---------------------------------------------------------------------------
#  pre_call hook tests
# ---------------------------------------------------------------------------


class TestPreCallHook:
    """Verify async_pre_call_hook sends full chat history to XecGuard."""

    @pytest.mark.asyncio
    async def test_pre_call_sends_full_chat_history(self, xecguard_pre_call):
        xecguard_pre_call.async_handler.post = AsyncMock(
            return_value=_mock_safe_scan_response()
        )

        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "How's weather in Taipei?"},
            ],
            "metadata": {"guardrails": ["test-xecguard-pre"]},
        }
        result = await xecguard_pre_call.async_pre_call_hook(
            user_api_key_dict=MagicMock(),
            cache=MagicMock(),
            data=data,
            call_type="acompletion",
        )
        assert result is not None

        call_kwargs = xecguard_pre_call.async_handler.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["scan_type"] == "input"
        assert body["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How's weather in Taipei?"},
        ]

    @pytest.mark.asyncio
    async def test_pre_call_unsafe_raises(self, xecguard_pre_call):
        from fastapi import HTTPException

        xecguard_pre_call.async_handler.post = AsyncMock(
            return_value=_mock_unsafe_scan_response()
        )

        data = {
            "messages": [{"role": "user", "content": "harmful request"}],
            "metadata": {"guardrails": ["test-xecguard-pre"]},
        }
        with pytest.raises(HTTPException) as exc_info:
            await xecguard_pre_call.async_pre_call_hook(
                user_api_key_dict=MagicMock(),
                cache=MagicMock(),
                data=data,
                call_type="acompletion",
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_pre_call_no_messages_returns_data(self, xecguard_pre_call):
        data = {
            "messages": [],
            "metadata": {"guardrails": ["test-xecguard-pre"]},
        }
        result = await xecguard_pre_call.async_pre_call_hook(
            user_api_key_dict=MagicMock(),
            cache=MagicMock(),
            data=data,
            call_type="acompletion",
        )
        assert result == data


# ---------------------------------------------------------------------------
#  post_call hook tests
# ---------------------------------------------------------------------------


class TestPostCallHook:
    """Verify async_post_call_success_hook sends full chat history
    with assistant response and performs grounding when enabled."""

    @pytest.mark.asyncio
    async def test_post_call_sends_full_history_with_response(
        self, xecguard_post_call
    ):
        # Two calls: scan + grounding (fixture has grounding_documents)
        xecguard_post_call.async_handler.post = AsyncMock(
            side_effect=[
                _mock_safe_scan_response(),
                _mock_safe_grounding_response(),
            ]
        )

        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "How's weather in Taipei?"},
            ],
            "metadata": {"guardrails": ["test-xecguard-post"]},
        }

        response = MagicMock(spec=["choices"])
        from litellm.types.utils import Choices, Message

        choice = Choices(
            finish_reason="stop",
            index=0,
            message=Message(role="assistant", content="It is hot and sunny"),
        )
        response.choices = [choice]
        response.__class__ = litellm.ModelResponse

        await xecguard_post_call.async_post_call_success_hook(
            user_api_key_dict=MagicMock(),
            data=data,
            response=response,
        )

        # First call is the scan
        scan_call = xecguard_post_call.async_handler.post.call_args_list[0]
        body = scan_call.kwargs["json"]
        assert body["scan_type"] == "response"
        assert body["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How's weather in Taipei?"},
            {"role": "assistant", "content": "It is hot and sunny"},
        ]

    @pytest.mark.asyncio
    async def test_post_call_with_grounding_from_config(self, xecguard_post_call):
        """Grounding is called when enabled and documents come from config."""
        data = {
            "messages": [
                {"role": "user", "content": "What is X?"},
            ],
            "metadata": {
                "guardrails": ["test-xecguard-post"],
            },
        }

        response = MagicMock(spec=["choices"])
        from litellm.types.utils import Choices, Message

        choice = Choices(
            finish_reason="stop",
            index=0,
            message=Message(role="assistant", content="X is Y."),
        )
        response.choices = [choice]
        response.__class__ = litellm.ModelResponse

        # The handler is called twice: once for scan, once for grounding
        xecguard_post_call.async_handler.post = AsyncMock(
            side_effect=[
                _mock_safe_scan_response(),
                _mock_safe_grounding_response(),
            ]
        )

        await xecguard_post_call.async_post_call_success_hook(
            user_api_key_dict=MagicMock(),
            data=data,
            response=response,
        )

        # Verify both scan and grounding were called
        assert xecguard_post_call.async_handler.post.call_count == 2
        # Verify grounding call used config documents
        grounding_call = xecguard_post_call.async_handler.post.call_args_list[1]
        grounding_body = grounding_call.kwargs["json"]
        assert grounding_body["documents"] == [{"document_id": "d1", "context": "X is Y"}]
