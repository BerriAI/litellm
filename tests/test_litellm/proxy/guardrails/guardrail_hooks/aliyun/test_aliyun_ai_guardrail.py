"""
Unit tests for Aliyun AI Security Guardrail.

Tests cover:
- Registration in guardrail system (enum, initializer, registry)
- Constructor validation (credentials, level, configurable service codes)
- Helper functions (level_to_int, _split_text)
- Blocking logic (_should_block_by_level, _parse_response_and_check)
- Pre-call hook (text + multimodal image URL detection)
- Post-call hook (blocks violations in response, passes clean response)
- Image URL extraction (get_image_urls)
- ServiceParameters construction (text / image / mixed combos)
- Config model (get_config_model, ui_friendly_name)
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.aliyun.aliyun_ai_guardrail import (
    CONTENT_MODERATION_TYPE,
    PROMPT_ATTACK_TYPE,
    SENSITIVE_DATA_TYPE,
    AliyunAIGuardrail,
    level_to_int,
)

FAKE_AK = "test-access-key-id"
FAKE_SK = "test-access-key-secret"

IMG_A = "https://example.com/a.png"
IMG_B = "http://example.com/b.jpg"
DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def _make_guardrail(**kwargs) -> AliyunAIGuardrail:
    defaults = dict(
        guardrail_name="test-aliyun",
        access_key_id=FAKE_AK,
        access_key_secret=FAKE_SK,
        level="medium",
    )
    defaults.update(kwargs)
    return AliyunAIGuardrail(**defaults)


def _make_aliyun_api_response(
    suggestion: str = "pass",
    detail: list = None,
    code: int = 200,
) -> MagicMock:
    """Build a mock httpx.Response mimicking Aliyun API output."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "Code": code,
        "RequestId": "test-req-id",
        "Message": None,
        "Data": {
            "Suggestion": suggestion,
            "Detail": detail or [],
        },
    }
    return mock


def _make_detail(
    detection_type: str = CONTENT_MODERATION_TYPE,
    level: str = "high",
    suggestion: str = "block",
    results: list = None,
) -> dict:
    """Build a Detail item. Pass level=None to omit the Level field entirely,
    mimicking the response shape documented for MultiModalGuard, which carries
    the severity as Result[].RiskLevel instead."""
    detail = {
        "Type": detection_type,
        "Suggestion": suggestion,
        "Result": results or [],
    }
    if level is not None:
        detail["Level"] = level
    return detail


def _captured_service_parameters(mock_post: AsyncMock):
    """Return (ServiceParameters dict, Service code) from a mocked post call."""
    _, kwargs = mock_post.call_args
    params = kwargs["data"]
    return json.loads(params["ServiceParameters"]), params["Service"]


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestAliyunGuardrailRegistration:
    def test_supported_guardrail_enum_entry(self):
        from litellm.types.guardrails import SupportedGuardrailIntegrations

        assert hasattr(SupportedGuardrailIntegrations, "ALIYUN_AI_GUARDRAIL")
        assert SupportedGuardrailIntegrations.ALIYUN_AI_GUARDRAIL.value == "aliyun_ai_guardrail"

    def test_initialize_guardrail_function_exists(self):
        from litellm.proxy.guardrails.guardrail_hooks.aliyun import (
            guardrail_initializer_registry,
            initialize_guardrail,
        )

        assert initialize_guardrail is not None
        assert "aliyun_ai_guardrail" in guardrail_initializer_registry

    def test_guardrail_class_registry_exists(self):
        from litellm.proxy.guardrails.guardrail_hooks.aliyun import (
            guardrail_class_registry,
        )

        assert "aliyun_ai_guardrail" in guardrail_class_registry
        assert guardrail_class_registry["aliyun_ai_guardrail"] is AliyunAIGuardrail

    def test_aliyun_in_global_registry(self):
        from litellm.proxy.guardrails.guardrail_registry import (
            guardrail_initializer_registry,
        )

        assert "aliyun_ai_guardrail" in guardrail_initializer_registry

    def test_initialize_guardrail_creates_instance(self):
        from litellm.proxy.guardrails.guardrail_hooks.aliyun import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import LitellmParams

        litellm_params = LitellmParams(
            guardrail="aliyun_ai_guardrail",
            mode="pre_call",
            level="medium",
            access_key_id=FAKE_AK,
            access_key_secret=FAKE_SK,
        )
        guardrail_config = {"guardrail_name": "test-aliyun-guard"}

        with patch("litellm.logging_callback_manager.add_litellm_callback") as mock_add:
            result = initialize_guardrail(litellm_params, guardrail_config)

            assert isinstance(result, AliyunAIGuardrail)
            assert result.guardrail_name == "test-aliyun-guard"
            assert result.level == "medium"
            assert result.access_key_id == FAKE_AK
            assert result.access_key_secret == FAKE_SK
            mock_add.assert_called_once_with(result)

    def test_initialize_guardrail_resolves_os_environ_reference(self):
        from litellm.proxy.guardrails.guardrail_hooks.aliyun import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import LitellmParams

        litellm_params = LitellmParams(
            guardrail="aliyun_ai_guardrail",
            mode="pre_call",
            access_key_id="os.environ/GUARD_ACCESS_KEY_ID",
            access_key_secret="os.environ/GUARD_ACCESS_KEY_SECRET",
        )
        guardrail_config = {"guardrail_name": "test-aliyun-guard"}

        with (
            patch.dict(
                "os.environ",
                {
                    "GUARD_ACCESS_KEY_ID": FAKE_AK,
                    "GUARD_ACCESS_KEY_SECRET": FAKE_SK,
                },
            ),
            patch("litellm.logging_callback_manager.add_litellm_callback"),
        ):
            result = initialize_guardrail(litellm_params, guardrail_config)

            assert result.access_key_id == FAKE_AK
            assert result.access_key_secret == FAKE_SK

    def test_initialize_guardrail_forwards_service_mcp(self):
        from litellm.proxy.guardrails.guardrail_hooks.aliyun import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import LitellmParams

        litellm_params = LitellmParams(
            guardrail="aliyun_ai_guardrail",
            mode="pre_mcp_call",
            access_key_id=FAKE_AK,
            access_key_secret=FAKE_SK,
            service_mcp="text_img_mix_guard",
        )
        guardrail_config = {"guardrail_name": "test-aliyun-guard"}

        with patch("litellm.logging_callback_manager.add_litellm_callback"):
            result = initialize_guardrail(litellm_params, guardrail_config)

            assert result.service_mcp == "text_img_mix_guard"


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestAliyunGuardrailConstructor:
    def test_init_with_explicit_credentials(self):
        g = _make_guardrail()
        assert g.access_key_id == FAKE_AK
        assert g.access_key_secret == FAKE_SK
        assert g.level == "medium"
        assert g.region_id == "cn-shanghai"

    def test_init_credentials_from_config(self):
        g = AliyunAIGuardrail(
            guardrail_name="config-test",
            access_key_id="cfg-ak",
            access_key_secret="cfg-sk",
            level="low",
        )
        assert g.access_key_id == "cfg-ak"
        assert g.access_key_secret == "cfg-sk"
        # region defaults to cn-shanghai when not provided via config
        assert g.region_id == "cn-shanghai"

    def test_init_raises_without_api_key(self):
        with pytest.raises(ValueError, match="ak is required"):
            AliyunAIGuardrail(guardrail_name="test")

    def test_init_raises_without_api_secret(self):
        with pytest.raises(ValueError, match="sk is required"):
            AliyunAIGuardrail(guardrail_name="test", access_key_id=FAKE_AK)

    def test_init_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Invalid level"):
            _make_guardrail(level="invalid")

    def test_init_default_service_codes_domestic(self):
        g = _make_guardrail()
        assert g.service_input == "query_security_check_pro"
        assert g.service_output == "response_security_check_pro"

    def test_init_service_codes_are_configurable(self):
        g = _make_guardrail(
            service_input="text_img_mix_guard",
            service_output="response_security_check_cb",
        )
        assert g.service_input == "text_img_mix_guard"
        assert g.service_output == "response_security_check_cb"

    def test_init_region_is_configurable(self):
        g = _make_guardrail(region_id="eu-central-1")
        assert g.region_id == "eu-central-1"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestLevelToInt:
    def test_standard_levels(self):
        assert level_to_int("none") == 0
        assert level_to_int("low") == 1
        assert level_to_int("medium") == 2
        assert level_to_int("high") == 3

    def test_sensitive_data_levels(self):
        assert level_to_int("S0") == 0
        assert level_to_int("S1") == 1
        assert level_to_int("S2") == 2
        assert level_to_int("S3") == 3
        assert level_to_int("S4") == 3

    def test_case_insensitive(self):
        assert level_to_int("HIGH") == 3
        assert level_to_int("Low") == 1

    def test_empty_string_defaults_to_zero(self):
        assert level_to_int("") == 0
        assert level_to_int(None) == 0

    def test_unknown_level_defaults_to_zero(self):
        assert level_to_int("unknown") == 0


class TestSplitText:
    def test_short_text_returns_single_segment(self):
        g = _make_guardrail()
        result = g._split_text("short text", max_length=100)
        assert result == ("short text",)

    def test_long_text_splits_at_sentence_boundary(self):
        g = _make_guardrail()
        text = "Hello world. This is a test. Final sentence."
        result = g._split_text(text, max_length=20)
        assert len(result) >= 2
        assert "".join(result) == text

    def test_long_text_without_boundary_splits_at_max_length(self):
        g = _make_guardrail()
        text = "a" * 100
        result = g._split_text(text, max_length=30)
        assert len(result) >= 3
        assert "".join(result) == text

    def test_empty_text_returns_no_segments(self):
        g = _make_guardrail()
        result = g._split_text("", max_length=100)
        assert result == ()


# ---------------------------------------------------------------------------
# Blocking logic tests
# ---------------------------------------------------------------------------


class TestShouldBlockByLevel:
    def test_low_protection_blocks_all_risks(self):
        g = _make_guardrail(level="low")
        assert g._should_block_by_level("low") is True
        assert g._should_block_by_level("medium") is True
        assert g._should_block_by_level("high") is True
        assert g._should_block_by_level("none") is False

    def test_medium_protection_blocks_medium_and_high(self):
        g = _make_guardrail(level="medium")
        assert g._should_block_by_level("low") is False
        assert g._should_block_by_level("medium") is True
        assert g._should_block_by_level("high") is True

    def test_high_protection_blocks_high_only(self):
        g = _make_guardrail(level="high")
        assert g._should_block_by_level("medium") is False
        assert g._should_block_by_level("high") is True

    def test_max_observation_never_blocks(self):
        g = _make_guardrail(level="max")
        assert g._should_block_by_level("high") is False
        assert g._should_block_by_level("S4") is False


class TestParseResponseAndCheck:
    def test_blocks_when_level_meets_threshold(self):
        g = _make_guardrail(level="medium")
        response = {
            "Data": {
                "Suggestion": "block",
                "Detail": [_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
            },
        }
        with pytest.raises(HTTPException) as exc_info:
            g._parse_response_and_check(response, check_type="input")
        assert exc_info.value.status_code == 400
        # Message uses the raw detection type returned by Aliyun
        assert CONTENT_MODERATION_TYPE in str(exc_info.value.detail)

    def test_passes_when_level_below_threshold(self):
        g = _make_guardrail(level="high")
        response = {
            "Data": {
                "Suggestion": "pass",
                "Detail": [_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="low")],
            },
        }
        result = g._parse_response_and_check(response, check_type="input")
        assert result["flagged"] is False

    def test_empty_data_returns_pass(self):
        g = _make_guardrail()
        response = {"Data": {}}
        result = g._parse_response_and_check(response, check_type="input")
        assert result["flagged"] is False
        assert result["suggestion"] == "pass"

    def test_extracts_desensitization_for_sensitive_data(self):
        # level=max never blocks, so the parsed result (with desensitization) is returned
        g = _make_guardrail(level="max")
        response = {
            "Data": {
                "Suggestion": "mask",
                "Detail": [
                    {
                        "Type": SENSITIVE_DATA_TYPE,
                        "Level": "S2",
                        "Suggestion": "mask",
                        "Result": [{"Ext": {"Desensitization": "masked_text"}}],
                    },
                ],
            },
        }
        result = g._parse_response_and_check(response, check_type="input")
        assert result["desensitization"] == "masked_text"

    def test_prompt_attack_block_message(self):
        g = _make_guardrail(level="medium")
        response = {
            "Data": {
                "Suggestion": "block",
                "Detail": [_make_detail(detection_type=PROMPT_ATTACK_TYPE, level="medium")],
            },
        }
        with pytest.raises(HTTPException) as exc_info:
            g._parse_response_and_check(response, check_type="input")
        assert PROMPT_ATTACK_TYPE in str(exc_info.value.detail)


class TestRiskLevelResolution:
    """MultiModalGuard reports severity in two shapes: Detail[].Level (returned by
    the _pro service codes) and Detail[].Result[].RiskLevel (the documented shape).
    Reading only the former silently downgrades the latter to "none" and lets
    Aliyun's own block decision through.
    """

    def test_falls_back_to_result_risk_level_when_level_absent(self):
        g = _make_guardrail(level="medium")
        response = {
            "Data": {
                "Suggestion": "block",
                "Detail": [
                    _make_detail(
                        detection_type=CONTENT_MODERATION_TYPE,
                        level=None,
                        results=[{"Label": "violence", "Confidence": 99.5, "RiskLevel": "high"}],
                    )
                ],
            },
        }
        with pytest.raises(HTTPException) as exc_info:
            g._parse_response_and_check(response, check_type="input")
        assert CONTENT_MODERATION_TYPE in str(exc_info.value.detail)

    def test_result_risk_level_still_respects_threshold(self):
        g = _make_guardrail(level="high")
        response = {
            "Data": {
                "Suggestion": "pass",
                "Detail": [
                    _make_detail(
                        detection_type=CONTENT_MODERATION_TYPE,
                        level=None,
                        suggestion="pass",
                        results=[{"Label": "spam", "RiskLevel": "low"}],
                    )
                ],
            },
        }
        result = g._parse_response_and_check(response, check_type="input")
        assert result["flagged"] is False

    def test_highest_result_risk_level_wins(self):
        g = _make_guardrail(level="high")
        response = {
            "Data": {
                "Suggestion": "block",
                "Detail": [
                    _make_detail(
                        detection_type=CONTENT_MODERATION_TYPE,
                        level=None,
                        results=[
                            {"Label": "spam", "RiskLevel": "low"},
                            {"Label": "violence", "RiskLevel": "high"},
                        ],
                    )
                ],
            },
        }
        with pytest.raises(HTTPException):
            g._parse_response_and_check(response, check_type="input")

    def test_detail_level_takes_precedence_over_result_risk_level(self):
        g = _make_guardrail(level="medium")
        response = {
            "Data": {
                "Suggestion": "pass",
                "Detail": [
                    _make_detail(
                        detection_type=CONTENT_MODERATION_TYPE,
                        level="none",
                        suggestion="pass",
                        results=[{"Label": "spam", "RiskLevel": "low"}],
                    )
                ],
            },
        }
        result = g._parse_response_and_check(response, check_type="input")
        assert result["flagged"] is False

    def test_blocks_on_detail_block_suggestion_without_any_risk_level(self):
        g = _make_guardrail(level="medium")
        response = {
            "Data": {
                "Suggestion": "pass",
                "Detail": [_make_detail(detection_type=PROMPT_ATTACK_TYPE, level=None, suggestion="block")],
            },
        }
        with pytest.raises(HTTPException) as exc_info:
            g._parse_response_and_check(response, check_type="input")
        assert PROMPT_ATTACK_TYPE in str(exc_info.value.detail)

    def test_blocks_on_overall_block_suggestion_without_any_risk_level(self):
        g = _make_guardrail(level="medium")
        response = {
            "Data": {
                "Suggestion": "block",
                "Detail": [_make_detail(detection_type=CONTENT_MODERATION_TYPE, level=None, suggestion="pass")],
            },
        }
        with pytest.raises(HTTPException):
            g._parse_response_and_check(response, check_type="input")

    def test_blocks_on_overall_block_suggestion_with_empty_detail(self):
        g = _make_guardrail(level="medium")
        response = {"Data": {"Suggestion": "block", "Detail": []}}
        with pytest.raises(HTTPException):
            g._parse_response_and_check(response, check_type="input")

    def test_observation_mode_never_blocks_on_block_suggestion(self):
        """level=max is an explicit opt-in to logging only; fail-closed handling of an
        unparseable severity must still honour it rather than bypass the threshold."""
        g = _make_guardrail(level="max")
        response = {
            "Data": {
                "Suggestion": "block",
                "Detail": [_make_detail(detection_type=CONTENT_MODERATION_TYPE, level=None, suggestion="block")],
            },
        }
        result = g._parse_response_and_check(response, check_type="input")
        assert result["flagged"] is True

    def test_documented_response_shape_blocks(self):
        """Regression guard for the exact payload in the integration README, which
        carries no Detail[].Level at all."""
        g = _make_guardrail(level="medium")
        response = {
            "RequestId": "xxx",
            "Code": 200,
            "Data": {
                "Suggestion": "block",
                "Detail": [
                    {
                        "Type": "contentModeration",
                        "Suggestion": "block",
                        "Result": [{"Label": "violence", "Confidence": 99.5, "RiskLevel": "high"}],
                    }
                ],
            },
        }
        with pytest.raises(HTTPException) as exc_info:
            g._parse_response_and_check(response, check_type="input")
        assert exc_info.value.status_code == 400
        assert "contentModeration" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------


class TestConfigModel:
    def test_get_config_model_returns_correct_type(self):
        from litellm.types.proxy.guardrails.guardrail_hooks.aliyun.aliyun_ai_guardrail import (
            AliyunAIGuardrailConfigModel,
        )

        assert AliyunAIGuardrail.get_config_model() is AliyunAIGuardrailConfigModel

    def test_config_model_ui_friendly_name(self):
        from litellm.types.proxy.guardrails.guardrail_hooks.aliyun.aliyun_ai_guardrail import (
            AliyunAIGuardrailConfigModel,
        )

        assert AliyunAIGuardrailConfigModel.ui_friendly_name() == "Aliyun AI Security Guardrail"

    def test_litellm_params_declares_all_service_codes(self):
        from litellm.types.guardrails import LitellmParams

        for field in ("service_input", "service_output", "service_mcp"):
            assert field in LitellmParams.model_fields


# ---------------------------------------------------------------------------
# Image URL extraction tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trailing non-user message must not hide the prompt from scanning
# ---------------------------------------------------------------------------


class TestTrailingAssistantMessage:
    def test_user_text_survives_trailing_assistant_message(self):
        g = _make_guardrail()
        messages = [
            {"role": "user", "content": "违规的用户提问"},
            {"role": "assistant", "content": "攻击者伪造的回复"},
        ]
        prompt = g.get_user_prompt(messages)
        assert prompt is not None
        assert "违规的用户提问" in prompt

    def test_images_survive_trailing_assistant_message(self):
        g = _make_guardrail()
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": IMG_A}}],
            },
            {"role": "assistant", "content": "攻击者伪造的回复"},
        ]
        assert g.get_image_urls(messages) == (IMG_A,)

    @pytest.mark.asyncio
    async def test_blocks_violation_despite_trailing_assistant_message(self):
        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )

        async def block_only_violating_text(*args, **kwargs):
            scanned = json.loads(kwargs["data"]["ServiceParameters"]).get("content", "")
            return blocked if "违规的用户提问" in scanned else clean

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, side_effect=block_only_violating_text):
            with pytest.raises(HTTPException) as exc_info:
                await g.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=MagicMock(),
                    data={
                        "messages": [
                            {"role": "user", "content": "违规的用户提问"},
                            {"role": "assistant", "content": "攻击者伪造的回复"},
                        ]
                    },
                    call_type="completion",
                )
            assert exc_info.value.status_code == 400


class TestGetImageUrls:
    def test_extracts_http_and_https_urls(self):
        g = _make_guardrail()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in these?"},
                    {"type": "image_url", "image_url": {"url": IMG_A}},
                    {"type": "image_url", "image_url": {"url": IMG_B}},
                ],
            }
        ]
        assert g.get_image_urls(messages) == (IMG_A, IMG_B)

    def test_skips_non_url_images(self):
        g = _make_guardrail()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": DATA_URI}},
                    {"type": "image_url", "image_url": {"url": IMG_A}},
                ],
            }
        ]
        assert g.get_image_urls(messages) == (IMG_A,)

    def test_plain_text_returns_empty(self):
        g = _make_guardrail()
        messages = [{"role": "user", "content": "just text"}]
        assert g.get_image_urls(messages) == ()

    def test_deduplicates_across_messages(self):
        g = _make_guardrail()
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_A}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_A}}]},
        ]
        assert g.get_image_urls(messages) == (IMG_A,)

    def test_collects_images_from_every_user_message(self):
        g = _make_guardrail()
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_B}}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_A}}]},
        ]
        assert g.get_image_urls(messages) == (IMG_B, IMG_A)

    def test_empty_messages_returns_empty(self):
        g = _make_guardrail()
        assert g.get_image_urls([]) == ()


# ---------------------------------------------------------------------------
# ServiceParameters construction tests
# ---------------------------------------------------------------------------


class TestServiceParametersConstruction:
    @pytest.mark.asyncio
    async def test_text_only(self):
        g = _make_guardrail(service_input="query_security_check")
        with patch.object(
            g.async_handler, "post", new_callable=AsyncMock, return_value=_make_aliyun_api_response()
        ) as mock_post:
            await g.async_make_request(text="hello", service_type="input")
        sp, service = _captured_service_parameters(mock_post)
        assert sp == {"requestFrom": "LiteLLM", "content": "hello"}
        assert service == "query_security_check"

    @pytest.mark.asyncio
    async def test_text_and_images(self):
        g = _make_guardrail(service_input="text_img_mix_guard")
        with patch.object(
            g.async_handler, "post", new_callable=AsyncMock, return_value=_make_aliyun_api_response()
        ) as mock_post:
            await g.async_make_request(text="hello", service_type="input", image_urls=[IMG_A, IMG_B])
        sp, service = _captured_service_parameters(mock_post)
        assert sp == {"requestFrom": "LiteLLM", "content": "hello", "imageUrls": [IMG_A, IMG_B]}
        assert service == "text_img_mix_guard"

    @pytest.mark.asyncio
    async def test_images_only(self):
        g = _make_guardrail(service_input="img_query_security_check")
        with patch.object(
            g.async_handler, "post", new_callable=AsyncMock, return_value=_make_aliyun_api_response()
        ) as mock_post:
            await g.async_make_request(service_type="input", image_urls=[IMG_A])
        sp, service = _captured_service_parameters(mock_post)
        assert sp == {"requestFrom": "LiteLLM", "imageUrls": [IMG_A]}
        assert service == "img_query_security_check"


# ---------------------------------------------------------------------------
# Pre-call hook tests (text + multimodal)
# ---------------------------------------------------------------------------


class TestPreCallHook:
    @pytest.mark.asyncio
    async def test_scans_responses_api_string_input(self):
        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data={"input": "违规的 responses 输入"},
                call_type="responses",
            )

        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"]).get("content", "") for call in mock_post.call_args_list
        )
        assert "违规的 responses 输入" in scanned

    @pytest.mark.asyncio
    async def test_blocks_violating_responses_api_input(self):
        g = _make_guardrail(level="medium")
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=blocked):
            with pytest.raises(HTTPException) as exc_info:
                await g.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=MagicMock(),
                    data={"input": "违规的 responses 输入"},
                    call_type="responses",
                )
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_scans_responses_api_structured_input_with_image(self):
        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        data = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "结构化输入文本"},
                        {"type": "input_image", "image_url": IMG_A, "detail": "auto"},
                    ],
                }
            ]
        }
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data=data,
                call_type="responses",
            )

        sent = [json.loads(call.kwargs["data"]["ServiceParameters"]) for call in mock_post.call_args_list]
        assert any("结构化输入文本" in (sp.get("content") or "") for sp in sent)
        assert any(IMG_A in (sp.get("imageUrls") or []) for sp in sent)

    @pytest.mark.asyncio
    async def test_scans_responses_api_function_call_output(self):
        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        data = {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "工具输出里的违规内容",
                }
            ]
        }
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data=data,
                call_type="responses",
            )

        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"]).get("content", "") for call in mock_post.call_args_list
        )
        assert "工具输出里的违规内容" in scanned

    @pytest.mark.asyncio
    async def test_scans_responses_api_function_call_output_image(self):
        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        data = {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_image", "image_url": IMG_A, "detail": "auto"},
                    ],
                }
            ]
        }
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data=data,
                call_type="responses",
            )

        sent = [json.loads(call.kwargs["data"]["ServiceParameters"]) for call in mock_post.call_args_list]
        assert any(IMG_A in (service_parameters.get("imageUrls") or []) for service_parameters in sent)

    @pytest.mark.asyncio
    async def test_scans_responses_api_function_call_arguments(self):
        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        data = {
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "send_message",
                    "arguments": '{"text":"函数调用里的违规参数"}',
                }
            ]
        }
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data=data,
                call_type="responses",
            )

        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"]).get("content", "") for call in mock_post.call_args_list
        )
        assert "函数调用里的违规参数" in scanned

    @pytest.mark.asyncio
    async def test_blocks_violation(self):
        g = _make_guardrail(level="medium")
        mock_api_response = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response):
            with pytest.raises(HTTPException) as exc_info:
                await g.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=MagicMock(),
                    data={"messages": [{"role": "user", "content": "违规内容"}]},
                    call_type="completion",
                )
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_passes_clean_content(self):
        g = _make_guardrail(level="medium")
        mock_api_response = _make_aliyun_api_response(suggestion="pass", detail=[])
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response):
            result = await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data={"messages": [{"role": "user", "content": "你好"}]},
                call_type="completion",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_no_messages_returns_data(self):
        g = _make_guardrail()
        result = await g.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test"),
            cache=MagicMock(),
            data={},
            call_type="completion",
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_user_prompt_returns_none(self):
        g = _make_guardrail()
        result = await g.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test"),
            cache=MagicMock(),
            data={"messages": [{"role": "system", "content": "system prompt only"}]},
            call_type="completion",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_long_text_splits_into_multiple_requests(self):
        g = _make_guardrail(max_text_length=10, level="medium")
        mock_api_response = _make_aliyun_api_response(suggestion="pass", detail=[])
        long_content = "This is a very long text that exceeds the max_text_length limit."
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response) as mock_post:
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data={"messages": [{"role": "user", "content": long_content}]},
                call_type="completion",
            )
            assert mock_post.call_count >= 2

    @pytest.mark.asyncio
    async def test_blocks_image_violation(self):
        g = _make_guardrail(level="medium", service_input="text_img_mix_guard")
        mock_resp = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": IMG_A}},
                ],
            }
        ]
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(HTTPException) as exc_info:
                await g.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=MagicMock(),
                    data={"messages": messages},
                    call_type="completion",
                )
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_image_only_request_sends_imageurls(self):
        g = _make_guardrail(level="medium", service_input="img_query_security_check")
        mock_resp = _make_aliyun_api_response(suggestion="pass", detail=[])
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_A}}]}]
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data={"messages": messages},
                call_type="completion",
            )
            assert mock_post.call_count == 1
            sp, _ = _captured_service_parameters(mock_post)
            assert sp == {"requestFrom": "LiteLLM", "imageUrls": [IMG_A]}

    @pytest.mark.asyncio
    async def test_first_segment_carries_images(self):
        g = _make_guardrail(max_text_length=10, level="medium", service_input="text_img_mix_guard")
        mock_resp = _make_aliyun_api_response(suggestion="pass", detail=[])
        long_text = "This is a very long text exceeding the limit for sure."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": long_text},
                    {"type": "image_url", "image_url": {"url": IMG_A}},
                ],
            }
        ]
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data={"messages": messages},
                call_type="completion",
            )
            assert mock_post.call_count >= 2
            image_carrying = 0
            for call in mock_post.call_args_list:
                sp = json.loads(call.kwargs["data"]["ServiceParameters"])
                if "imageUrls" in sp:
                    image_carrying += 1
            assert image_carrying == 1


# ---------------------------------------------------------------------------
# Post-call hook tests
# ---------------------------------------------------------------------------


class TestPostCallHook:
    @pytest.mark.asyncio
    async def test_blocks_violation_in_response(self):
        import litellm

        g = _make_guardrail(level="medium")
        mock_api_response = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        response = litellm.ModelResponse(
            id="test-id",
            choices=[
                litellm.Choices(
                    index=0,
                    message=litellm.Message(role="assistant", content="违规响应内容"),
                )
            ],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response):
            with pytest.raises(HTTPException) as exc_info:
                await g.async_post_call_success_hook(
                    data={"messages": [{"role": "user", "content": "hi"}]},
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=response,
                )
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_passes_clean_response(self):
        import litellm

        g = _make_guardrail(level="medium")
        mock_api_response = _make_aliyun_api_response(suggestion="pass", detail=[])
        response = litellm.ModelResponse(
            id="test-id",
            choices=[
                litellm.Choices(
                    index=0,
                    message=litellm.Message(role="assistant", content="正常的回复内容"),
                )
            ],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response):
            result = await g.async_post_call_success_hook(
                data={"messages": [{"role": "user", "content": "hi"}]},
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=response,
            )
            assert result is response

    @pytest.mark.asyncio
    async def test_blocks_violation_in_later_choice(self):
        import litellm

        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )

        async def block_only_violating_text(*args, **kwargs):
            scanned = json.loads(kwargs["data"]["ServiceParameters"])["content"]
            return blocked if "违规响应内容" in scanned else clean

        response = litellm.ModelResponse(
            id="test-id",
            choices=[
                litellm.Choices(
                    index=0,
                    message=litellm.Message(role="assistant", content="正常的回复内容"),
                ),
                litellm.Choices(
                    index=1,
                    message=litellm.Message(role="assistant", content="违规响应内容"),
                ),
            ],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, side_effect=block_only_violating_text):
            with pytest.raises(HTTPException) as exc_info:
                await g.async_post_call_success_hook(
                    data={"messages": [{"role": "user", "content": "hi"}]},
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=response,
                )
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_scans_content_from_every_choice(self):
        import litellm

        g = _make_guardrail(level="medium", max_text_length=10000)
        mock_api_response = _make_aliyun_api_response(suggestion="pass", detail=[])
        response = litellm.ModelResponse(
            id="test-id",
            choices=[
                litellm.Choices(index=0, message=litellm.Message(role="assistant", content="第一个回复")),
                litellm.Choices(index=1, message=litellm.Message(role="assistant", content="第二个回复")),
            ],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response) as mock_post:
            await g.async_post_call_success_hook(
                data={"messages": [{"role": "user", "content": "hi"}]},
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=response,
            )

        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"])["content"] for call in mock_post.call_args_list
        )
        assert "第一个回复" in scanned
        assert "第二个回复" in scanned

    @pytest.mark.asyncio
    async def test_non_model_response_passthrough(self):
        g = _make_guardrail()
        result = await g.async_post_call_success_hook(
            data={},
            user_api_key_dict=UserAPIKeyAuth(api_key="test"),
            response="not a model response",
        )
        assert result == "not a model response"

    @pytest.mark.asyncio
    async def test_empty_content_response_passthrough(self):
        import litellm

        g = _make_guardrail()
        response = litellm.ModelResponse(
            id="test-id",
            choices=[
                litellm.Choices(
                    index=0,
                    message=litellm.Message(role="assistant", content=""),
                )
            ],
        )
        result = await g.async_post_call_success_hook(
            data={},
            user_api_key_dict=UserAPIKeyAuth(api_key="test"),
            response=response,
        )
        assert result is response


def _make_tool_call_response(arguments: str, name: str = "send_email"):
    """Build a non-streaming response whose only output is a tool call."""
    import litellm
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    return litellm.ModelResponse(
        id="test-id",
        choices=[
            litellm.Choices(
                index=0,
                message=litellm.Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name=name, arguments=arguments),
                        )
                    ],
                ),
            )
        ],
    )


def _make_responses_api_response(text: str):
    """Build a non-streaming /v1/responses body carrying assistant text."""
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.responses.main import GenericResponseOutputItem, OutputText

    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        model="gpt-4o",
        object="response",
        output=[
            GenericResponseOutputItem(
                type="message",
                id="m1",
                status="completed",
                role="assistant",
                content=[OutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        temperature=1.0,
        top_p=1.0,
    )


class TestPostCallStructuredFields:
    """The streaming path already audits tool calls, reasoning text and
    /v1/responses output. Auditing only ``message.content`` here would let the
    very same content reach the client unchecked whenever stream=False."""

    @pytest.mark.asyncio
    async def test_scans_tool_call_arguments(self):
        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        response = _make_tool_call_response('{"body": "违规的工具参数"}')
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            await g.async_post_call_success_hook(
                data={"messages": [{"role": "user", "content": "hi"}]},
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=response,
            )
        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"])["content"] for call in mock_post.call_args_list
        )
        assert "违规的工具参数" in scanned

    @pytest.mark.asyncio
    async def test_blocks_violation_in_tool_call_arguments(self):
        g = _make_guardrail(level="medium")
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        response = _make_tool_call_response('{"body": "违规的工具参数"}')
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=blocked):
            with pytest.raises(HTTPException) as exc_info:
                await g.async_post_call_success_hook(
                    data={"messages": [{"role": "user", "content": "hi"}]},
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=response,
                )
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_scans_reasoning_content(self):
        import litellm

        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        response = litellm.ModelResponse(
            id="test-id",
            choices=[
                litellm.Choices(
                    index=0,
                    message=litellm.Message(
                        role="assistant",
                        content="正常的回复内容",
                        reasoning_content="推理过程里的违规内容",
                    ),
                )
            ],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            await g.async_post_call_success_hook(
                data={"messages": [{"role": "user", "content": "hi"}]},
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=response,
            )
        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"])["content"] for call in mock_post.call_args_list
        )
        assert "推理过程里的违规内容" in scanned

    @pytest.mark.asyncio
    async def test_scans_responses_api_output(self):
        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        response = _make_responses_api_response("响应体里的违规内容")
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            await g.async_post_call_success_hook(
                data={"input": "hi"},
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=response,
            )
        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"])["content"] for call in mock_post.call_args_list
        )
        assert "响应体里的违规内容" in scanned

    @pytest.mark.asyncio
    async def test_blocks_violation_in_responses_api_output(self):
        g = _make_guardrail(level="medium")
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        response = _make_responses_api_response("响应体里的违规内容")
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=blocked):
            with pytest.raises(HTTPException) as exc_info:
                await g.async_post_call_success_hook(
                    data={"input": "hi"},
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=response,
                )
            assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Post-MCP hook tests
# ---------------------------------------------------------------------------


def _make_call_tool_result(text: str = "tool output"):
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


class TestExtractMcpToolText:
    """A tool result carries text outside of ``content[].text``. Auditing only that
    field would release structured payloads and embedded resources unchecked."""

    def test_collects_structured_content(self):
        from mcp.types import CallToolResult

        g = _make_guardrail()
        result = CallToolResult(content=[], structuredContent={"note": "结构化字段里的违规内容"}, isError=False)
        assert "结构化字段里的违规内容" in g._extract_mcp_tool_text(result)

    def test_collects_structured_content_alongside_text(self):
        from mcp.types import CallToolResult, TextContent

        g = _make_guardrail()
        result = CallToolResult(
            content=[TextContent(type="text", text="正常的工具输出")],
            structuredContent={"note": "结构化字段里的违规内容"},
            isError=False,
        )
        extracted = g._extract_mcp_tool_text(result)
        assert "正常的工具输出" in extracted
        assert "结构化字段里的违规内容" in extracted

    def test_collects_embedded_resource_text(self):
        from mcp.types import CallToolResult, EmbeddedResource, TextResourceContents

        g = _make_guardrail()
        result = CallToolResult(
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="file:///tmp/note.txt",
                        mimeType="text/plain",
                        text="内嵌资源里的违规内容",
                    ),
                )
            ],
            isError=False,
        )
        assert "内嵌资源里的违规内容" in g._extract_mcp_tool_text(result)

    def test_collects_structured_content_from_dict_payload(self):
        g = _make_guardrail()
        payload = {"content": [], "structuredContent": {"note": "结构化字段里的违规内容"}}
        assert "结构化字段里的违规内容" in g._extract_mcp_tool_text(payload)

    def test_collects_structured_content_from_coerced_tuple_list(self):
        """MCPPostCallResponseObject coerces a CallToolResult into (field, value) pairs."""
        g = _make_guardrail()
        payload = [("content", []), ("structuredContent", {"note": "结构化字段里的违规内容"}), ("isError", False)]
        assert "结构化字段里的违规内容" in g._extract_mcp_tool_text(payload)

    @pytest.mark.asyncio
    async def test_blocks_violation_in_structured_content(self):
        from mcp.types import CallToolResult, TextContent

        g = _make_guardrail(level="medium")
        tool_result = CallToolResult(
            content=[TextContent(type="text", text="正常的工具输出")],
            structuredContent={"note": "结构化字段里的违规内容"},
            isError=False,
        )
        kwargs, response_obj = _make_post_mcp_hook_args(tool_result)
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )

        async def block_only_structured_content(*args, **kwargs):
            scanned = json.loads(kwargs["data"]["ServiceParameters"]).get("content", "")
            return blocked if "结构化字段里的违规内容" in scanned else clean

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, side_effect=block_only_structured_content):
            await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        remaining = " ".join(getattr(item, "text", "") for item in tool_result.content)
        assert CONTENT_MODERATION_TYPE in remaining
        assert tool_result.isError is True
        assert tool_result.structuredContent is None


# ---------------------------------------------------------------------------
# Streaming hook tests
# ---------------------------------------------------------------------------


def _make_stream_chunk(content=None, tool_call_arguments=None):
    """Build a ModelResponseStream chunk carrying content and/or tool call arguments."""
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
    )

    tool_calls = None
    if tool_call_arguments is not None:
        tool_calls = [
            ChatCompletionDeltaToolCall(
                id="call_1",
                type="function",
                index=0,
                function=Function(name="send_message", arguments=tool_call_arguments),
            )
        ]
    return ModelResponseStream(
        id="test-id",
        choices=[StreamingChoices(index=0, delta=Delta(content=content, tool_calls=tool_calls))],
    )


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# MCP pre-call hook tests
# ---------------------------------------------------------------------------


class TestMcpPreCallCheck:
    @pytest.mark.asyncio
    async def test_blocks_violating_tool_arguments(self):
        g = _make_guardrail(level="medium")
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=blocked):
            with pytest.raises(HTTPException) as exc_info:
                await g.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=MagicMock(),
                    data={"messages": [{"role": "user", "content": "delete_all_files 违规参数"}]},
                    call_type="call_mcp_tool",
                )
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_uses_mcp_service_code(self):
        g = _make_guardrail(level="medium", service_mcp="text_img_mix_guard")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            result = await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data={"messages": [{"role": "user", "content": "send_message hello"}]},
                call_type="call_mcp_tool",
            )

        assert result is None
        service_parameters, service = _captured_service_parameters(mock_post)
        assert service == "text_img_mix_guard"
        assert service_parameters["content"] == "send_message hello"

    @pytest.mark.asyncio
    async def test_logs_mcp_content_length_without_content(self):
        g = _make_guardrail(level="medium")
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        sensitive_content = "send_message token=secret-value"
        logger_path = "litellm.proxy.guardrails.guardrail_hooks.aliyun.aliyun_ai_guardrail.verbose_proxy_logger.info"
        with (
            patch(logger_path) as mock_info,
            patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean),
        ):
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data={"messages": [{"role": "user", "content": sensitive_content}]},
                call_type="call_mcp_tool",
            )

        mock_info.assert_any_call(
            "Aliyun AI Guardrail: ★ MCP pre-call check started, content length: %d",
            len(sensitive_content),
        )
        assert sensitive_content not in str(mock_info.call_args_list)

    @pytest.mark.asyncio
    async def test_empty_content_skips_check(self):
        g = _make_guardrail()
        with patch.object(g.async_handler, "post", new_callable=AsyncMock) as mock_post:
            result = await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data={"messages": [{"role": "user", "content": ""}]},
                call_type="call_mcp_tool",
            )

        assert result is None
        mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_content_is_split_into_multiple_requests(self):
        g = _make_guardrail(max_text_length=10)
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            await g.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=MagicMock(),
                data={"messages": [{"role": "user", "content": "一二三四五六七八九十。壹贰叁肆伍陆柒捌玖抾。"}]},
                call_type="call_mcp_tool",
            )

        assert mock_post.call_count > 1


# ---------------------------------------------------------------------------
# post_mcp_call event hook gating tests
# ---------------------------------------------------------------------------


class TestShouldRunPostMcpCall:
    def test_no_event_hook_runs(self):
        g = _make_guardrail()
        g.event_hook = None
        assert g._should_run_post_mcp_call() is True

    def test_list_containing_post_mcp_call_runs(self):
        g = _make_guardrail()
        g.event_hook = ["pre_call", "post_mcp_call"]
        assert g._should_run_post_mcp_call() is True

    def test_list_without_post_mcp_call_skips(self):
        g = _make_guardrail()
        g.event_hook = ["pre_call", "post_call"]
        assert g._should_run_post_mcp_call() is False

    def test_plain_string_is_matched(self):
        g = _make_guardrail()
        g.event_hook = "post_mcp_call"
        assert g._should_run_post_mcp_call() is True
        g.event_hook = "pre_call"
        assert g._should_run_post_mcp_call() is False

    def test_mode_tags_are_matched(self):
        from litellm.types.guardrails import Mode

        g = _make_guardrail()
        g.event_hook = Mode(tags={"team-a": ["pre_call", "post_mcp_call"]})
        assert g._should_run_post_mcp_call() is True

        g.event_hook = Mode(tags={"team-a": "post_mcp_call"})
        assert g._should_run_post_mcp_call() is True

    def test_mode_falls_back_to_default(self):
        from litellm.types.guardrails import Mode

        g = _make_guardrail()
        g.event_hook = Mode(tags={"team-a": "pre_call"}, default=["post_mcp_call"])
        assert g._should_run_post_mcp_call() is True

        g.event_hook = Mode(tags={"team-a": "pre_call"}, default="pre_call")
        assert g._should_run_post_mcp_call() is False

    def test_mode_without_match_or_default_skips(self):
        from litellm.types.guardrails import Mode

        g = _make_guardrail()
        g.event_hook = Mode(tags={"team-a": "pre_call"})
        assert g._should_run_post_mcp_call() is False


# ---------------------------------------------------------------------------
# MCP payload shape handling tests
# ---------------------------------------------------------------------------


class TestIterMcpContentItems:
    def test_plain_string_is_wrapped(self):
        g = _make_guardrail()
        assert g._iter_mcp_content_items("hello") == ("hello",)

    def test_object_with_content_list(self):
        g = _make_guardrail()
        payload = MagicMock()
        payload.content = ["a", "b"]
        assert g._iter_mcp_content_items(payload) == ("a", "b")

    def test_dict_with_content_list(self):
        g = _make_guardrail()
        assert g._iter_mcp_content_items({"content": ["a"]}) == ("a",)

    def test_dict_without_content_returns_itself(self):
        g = _make_guardrail()
        payload = {"text": "no content key"}
        assert g._iter_mcp_content_items(payload) == (payload,)

    def test_coerced_tuple_pairs_recover_real_content(self):
        g = _make_guardrail()
        payload = [("meta", None), ("content", ["real"]), ("isError", False)]
        assert g._iter_mcp_content_items(payload) == ("real",)

    def test_plain_list_passes_through(self):
        g = _make_guardrail()
        assert g._iter_mcp_content_items(["a", "b"]) == ("a", "b")

    def test_unsupported_payload_returns_empty(self):
        g = _make_guardrail()
        assert g._iter_mcp_content_items(123) == ()


class TestReplaceToolOutputInPlace:
    def test_none_target_is_rejected(self):
        g = _make_guardrail()
        assert g._replace_tool_output_in_place(None, ["blocked"]) is False

    def test_object_content_is_overwritten_and_flagged(self):
        g = _make_guardrail()
        target = MagicMock()
        target.content = ["original"]
        target.isError = False
        assert g._replace_tool_output_in_place(target, ["blocked"]) is True
        assert target.content == ["blocked"]
        assert target.isError is True

    def test_list_target_is_overwritten(self):
        g = _make_guardrail()
        target = ["original"]
        assert g._replace_tool_output_in_place(target, ["blocked"]) is True
        assert target == ["blocked"]

    def test_nested_result_content_is_overwritten(self):
        g = _make_guardrail()
        target = {"result": {"content": ["original"]}}
        assert g._replace_tool_output_in_place(target, ["blocked"]) is True
        assert target["result"]["content"] == ["blocked"]

    def test_dict_content_is_overwritten(self):
        g = _make_guardrail()
        target = {"content": ["original"]}
        assert g._replace_tool_output_in_place(target, ["blocked"]) is True
        assert target["content"] == ["blocked"]

    def test_unsupported_shape_is_rejected(self):
        g = _make_guardrail()
        assert g._replace_tool_output_in_place({"unrelated": 1}, ["blocked"]) is False


class TestStreamingHook:
    @pytest.mark.asyncio
    async def test_scans_tool_call_arguments(self):
        g = _make_guardrail(level="medium", stream_first_check_step=1, stream_slide_step=1)
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        chunk = _make_stream_chunk(tool_call_arguments='{"text": "违规工具参数"}')

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            async for _ in g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=_aiter([chunk]),
                request_data={},
            ):
                pass

        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"])["content"] for call in mock_post.call_args_list
        )
        assert "违规工具参数" in scanned

    @pytest.mark.asyncio
    async def test_blocks_violating_tool_call_arguments(self):
        g = _make_guardrail(level="medium", stream_first_check_step=1, stream_slide_step=1)
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )

        async def block_only_violating_text(*args, **kwargs):
            scanned = json.loads(kwargs["data"]["ServiceParameters"])["content"]
            return blocked if "违规工具参数" in scanned else clean

        chunk = _make_stream_chunk(tool_call_arguments='{"text": "违规工具参数"}')

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, side_effect=block_only_violating_text):
            emitted = [
                chunk
                async for chunk in g.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=_aiter([chunk]),
                    request_data={},
                )
            ]

        # Streaming blocks by emitting an SSE error event instead of raising
        assert not any(getattr(item, "choices", None) for item in emitted if hasattr(item, "choices"))

    @pytest.mark.asyncio
    async def test_scans_plain_content(self):
        g = _make_guardrail(level="medium", stream_first_check_step=1, stream_slide_step=1)
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        chunk = _make_stream_chunk(content="普通流式内容")

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            emitted = [
                item
                async for item in g.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=_aiter([chunk]),
                    request_data={},
                )
            ]

        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"])["content"] for call in mock_post.call_args_list
        )
        assert "普通流式内容" in scanned
        assert emitted == [chunk]

    @pytest.mark.asyncio
    async def test_blocks_violation_in_oversized_delta_prefix(self):
        g = _make_guardrail(
            level="medium",
            stream_window_size=10,
            stream_first_check_step=1,
            stream_slide_step=6,
        )
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        chunk = _make_stream_chunk(content="违规前缀" + "正常内容" * 8)

        async def block_prefix(*args, **kwargs):
            scanned = json.loads(kwargs["data"]["ServiceParameters"])["content"]
            return blocked if "违规前缀" in scanned else clean

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, side_effect=block_prefix):
            emitted = [
                item
                async for item in g.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=_aiter([chunk]),
                    request_data={},
                )
            ]

        assert chunk not in emitted

    @pytest.mark.asyncio
    async def test_scans_all_windows_before_emitting_oversized_delta_once(self):
        g = _make_guardrail(
            level="medium",
            stream_window_size=10,
            stream_first_check_step=1,
            stream_slide_step=6,
        )
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        chunk = _make_stream_chunk(content="开头内容" + "中间内容" * 6 + "结尾内容")

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            emitted = [
                item
                async for item in g.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=_aiter([chunk]),
                    request_data={},
                )
            ]

        scanned_windows = tuple(
            json.loads(call.kwargs["data"]["ServiceParameters"])["content"] for call in mock_post.call_args_list
        )
        assert all(len(window) <= g.stream_window_size for window in scanned_windows)
        assert any("开头内容" in window for window in scanned_windows)
        assert any("结尾内容" in window for window in scanned_windows)
        assert emitted == [chunk]

    @pytest.mark.asyncio
    async def test_scans_responses_api_text_delta(self):
        from litellm.types.llms.openai import OutputTextDeltaEvent

        g = _make_guardrail(level="medium", stream_first_check_step=1, stream_slide_step=1)
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        chunk = OutputTextDeltaEvent(
            type="response.output_text.delta",
            item_id="item_1",
            output_index=0,
            content_index=0,
            delta="违规的 responses 输出",
        )

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            async for _ in g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=_aiter([chunk]),
                request_data={},
            ):
                pass

        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"])["content"] for call in mock_post.call_args_list
        )
        assert "违规的 responses 输出" in scanned

    @pytest.mark.asyncio
    async def test_blocks_violating_responses_api_text_delta(self):
        from litellm.types.llms.openai import OutputTextDeltaEvent

        g = _make_guardrail(level="medium", stream_first_check_step=1, stream_slide_step=1)
        blocked = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        chunk = OutputTextDeltaEvent(
            type="response.output_text.delta",
            item_id="item_1",
            output_index=0,
            content_index=0,
            delta="违规的 responses 输出",
        )

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=blocked):
            emitted = [
                item
                async for item in g.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    response=_aiter([chunk]),
                    request_data={},
                )
            ]

        assert chunk not in emitted

    @pytest.mark.asyncio
    async def test_scans_responses_api_completed_event(self):
        from litellm.types.llms.openai import ResponseCompletedEvent

        g = _make_guardrail(level="medium", stream_first_check_step=1, stream_slide_step=1)
        clean = _make_aliyun_api_response(suggestion="pass", detail=[])
        content_part = MagicMock()
        content_part.text = "完成事件里的违规内容"
        output_item = MagicMock()
        output_item.content = [content_part]
        chunk = MagicMock(spec=ResponseCompletedEvent)
        chunk.type = "response.completed"
        chunk.response = MagicMock()
        chunk.response.output = [output_item]

        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=clean) as mock_post:
            async for _ in g.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                response=_aiter([chunk]),
                request_data={},
            ):
                pass

        scanned = "".join(
            json.loads(call.kwargs["data"]["ServiceParameters"])["content"] for call in mock_post.call_args_list
        )
        assert "完成事件里的违规内容" in scanned


def _make_post_mcp_hook_args(tool_result):
    """Mimic how litellm_logging dispatches the post-MCP hook.

    The live CallToolResult is stored in model_call_details["original_response"] and
    the same object is wrapped into MCPPostCallResponseObject, whose
    mcp_tool_call_response field is declared as a list - so pydantic coerces the
    CallToolResult by iterating it into (key, value) tuples.
    """
    from litellm.types.llms.base import HiddenParams
    from litellm.types.mcp import MCPPostCallResponseObject

    kwargs = {"original_response": tool_result}
    response_obj = MCPPostCallResponseObject(mcp_tool_call_response=tool_result, hidden_params=HiddenParams())
    return kwargs, response_obj


class TestPostMcpToolCallHook:
    @pytest.mark.asyncio
    async def test_audits_tool_text_not_pydantic_tuple_repr(self):
        """The wrapped response degrades into (key, value) tuples, so auditing it
        verbatim would send Python reprs - and MCP envelope fields - to Aliyun
        instead of the tool's own output."""
        g = _make_guardrail(level="medium")
        tool_result = _make_call_tool_result("工具返回的内容")
        kwargs, response_obj = _make_post_mcp_hook_args(tool_result)
        mock_api_response = _make_aliyun_api_response(suggestion="pass", detail=[])
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response) as mock_post:
            await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        sp, service = _captured_service_parameters(mock_post)
        assert sp["content"] == "工具返回的内容"
        for envelope_field in ("isError", "structuredContent", "annotations", "TextContent"):
            assert envelope_field not in sp["content"]

    @pytest.mark.asyncio
    async def test_violation_replaces_tool_output_in_place(self):
        """Both dispatch sites discard this hook's return value and hand the original
        CallToolResult to the client, so the violation must be written into it."""
        g = _make_guardrail(level="medium")
        tool_result = _make_call_tool_result("违规的工具输出")
        kwargs, response_obj = _make_post_mcp_hook_args(tool_result)
        mock_api_response = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=CONTENT_MODERATION_TYPE, level="high")],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response):
            await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        remaining = " ".join(getattr(item, "text", "") for item in tool_result.content)
        assert "违规的工具输出" not in remaining
        assert CONTENT_MODERATION_TYPE in remaining
        assert tool_result.isError is True

    @pytest.mark.asyncio
    async def test_violation_returns_replacement_object_without_raising(self):
        """Raising is a no-op here: the dispatcher catches every callback exception as a
        non-blocking logging error and returns the untouched tool result."""
        from litellm.types.mcp import MCPPostCallResponseObject

        g = _make_guardrail(level="medium")
        tool_result = _make_call_tool_result("违规的工具输出")
        kwargs, response_obj = _make_post_mcp_hook_args(tool_result)
        mock_api_response = _make_aliyun_api_response(
            suggestion="block",
            detail=[_make_detail(detection_type=PROMPT_ATTACK_TYPE, level="high")],
        )
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        assert isinstance(result, MCPPostCallResponseObject)
        replaced = " ".join(getattr(item, "text", "") for item in result.mcp_tool_call_response)
        assert PROMPT_ATTACK_TYPE in replaced

    @pytest.mark.asyncio
    async def test_clean_output_leaves_tool_result_untouched(self):
        g = _make_guardrail(level="medium")
        tool_result = _make_call_tool_result("正常的工具输出")
        kwargs, response_obj = _make_post_mcp_hook_args(tool_result)
        mock_api_response = _make_aliyun_api_response(suggestion="pass", detail=[])
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, return_value=mock_api_response):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        assert result is None
        assert tool_result.content[0].text == "正常的工具输出"
        assert tool_result.isError is False

    @pytest.mark.asyncio
    async def test_api_network_failure_fails_closed_in_place(self):
        """A network error is swallowed by the dispatcher as a non-blocking logging
        error, so unaudited tool output would reach the client untouched. The hook
        must fail closed by replacing the live tool result instead of raising."""
        import httpx

        g = _make_guardrail(level="medium")
        tool_result = _make_call_tool_result("未拦截将泄漏的敏感内容")
        kwargs, response_obj = _make_post_mcp_hook_args(tool_result)
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, side_effect=httpx.ConnectError("boom")):
            await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        remaining = " ".join(getattr(item, "text", "") for item in tool_result.content)
        assert "未拦截将泄漏的敏感内容" not in remaining
        assert "未经审核" in remaining
        assert tool_result.isError is True

    @pytest.mark.asyncio
    async def test_api_network_failure_returns_replacement_object(self):
        """The failure notice must also be returned as the documented replacement
        object, mirroring the violation path."""
        import httpx

        from litellm.types.mcp import MCPPostCallResponseObject

        g = _make_guardrail(level="medium")
        tool_result = _make_call_tool_result("未拦截将泄漏的敏感内容")
        kwargs, response_obj = _make_post_mcp_hook_args(tool_result)
        with patch.object(g.async_handler, "post", new_callable=AsyncMock, side_effect=httpx.ConnectError("boom")):
            result = await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        assert isinstance(result, MCPPostCallResponseObject)
        replaced = " ".join(getattr(item, "text", "") for item in result.mcp_tool_call_response)
        assert "未经审核" in replaced

    @pytest.mark.asyncio
    async def test_skipped_when_post_mcp_call_not_configured(self):
        g = _make_guardrail(level="medium", event_hook=["pre_call"])
        tool_result = _make_call_tool_result("违规的工具输出")
        kwargs, response_obj = _make_post_mcp_hook_args(tool_result)
        with patch.object(g.async_handler, "post", new_callable=AsyncMock) as mock_post:
            result = await g.async_post_mcp_tool_call_hook(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        assert result is None
        mock_post.assert_not_called()
        assert tool_result.content[0].text == "违规的工具输出"
