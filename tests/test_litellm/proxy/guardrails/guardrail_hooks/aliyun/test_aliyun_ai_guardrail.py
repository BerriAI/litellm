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
    return {
        "Type": detection_type,
        "Level": level,
        "Suggestion": suggestion,
        "Result": results or [],
    }


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

        with patch.dict("os.environ", {
            "GUARD_ACCESS_KEY_ID": FAKE_AK,
            "GUARD_ACCESS_KEY_SECRET": FAKE_SK,
        }), patch("litellm.logging_callback_manager.add_litellm_callback"):
            result = initialize_guardrail(litellm_params, guardrail_config)

            assert result.access_key_id == FAKE_AK
            assert result.access_key_secret == FAKE_SK


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
        assert result == ["short text"]

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

    def test_empty_text_returns_empty_list(self):
        g = _make_guardrail()
        result = g._split_text("", max_length=100)
        assert result == []


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
                        "Result": [
                            {"Ext": {"Desensitization": "masked_text"}}
                        ],
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


# ---------------------------------------------------------------------------
# Image URL extraction tests
# ---------------------------------------------------------------------------


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
        assert g.get_image_urls(messages) == [IMG_A, IMG_B]

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
        assert g.get_image_urls(messages) == [IMG_A]

    def test_plain_text_returns_empty(self):
        g = _make_guardrail()
        messages = [{"role": "user", "content": "just text"}]
        assert g.get_image_urls(messages) == []

    def test_deduplicates_across_messages(self):
        g = _make_guardrail()
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_A}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_A}}]},
        ]
        assert g.get_image_urls(messages) == [IMG_A]

    def test_only_last_consecutive_user_block(self):
        g = _make_guardrail()
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_B}}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_A}}]},
        ]
        assert g.get_image_urls(messages) == [IMG_A]

    def test_empty_messages_returns_empty(self):
        g = _make_guardrail()
        assert g.get_image_urls([]) == []


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
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG_A}}]}
        ]
        with patch.object(
            g.async_handler, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
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
        with patch.object(
            g.async_handler, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
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
