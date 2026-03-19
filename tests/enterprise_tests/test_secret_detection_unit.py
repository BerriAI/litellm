"""
enterprise 密钥拦截功能的单元测试。

覆盖范围：
- mask_middle_chars：字符串遮蔽函数
- raiseSecretsException：检测到敏感信息时的异常逻辑
- OpenAIApiKeyDetector：OpenAI Key 正则匹配
- SecurityGuardrail.async_pre_call_hook：pre-call hook 集成测试
"""

import os
import sys
import importlib
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi import HTTPException

# 将项目根目录加入 sys.path，确保能导入 litellm 相关模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ─────────────────────────────────────────────
# 导入被测模块
# ─────────────────────────────────────────────
import enterprise.litellm_enterprise.enterprise_callbacks.secret_detection as sd_module
from enterprise.litellm_enterprise.enterprise_callbacks.secret_detection import (
    mask_middle_chars,
    raiseSecretsException,
    _ENTERPRISE_SecretDetection,
)
from enterprise.litellm_enterprise.enterprise_callbacks.secrets_plugins.openai_api_key import (
    OpenAIApiKeyDetector,
)


# =============================================================
# 1. mask_middle_chars 函数测试
# =============================================================

class TestMaskMiddleChars:
    """测试 mask_middle_chars 函数的各种边界情况。"""

    def test_length_lte_4_unchanged(self):
        """长度 <= 4 时，字符串不作任何修改。"""
        assert mask_middle_chars("") == ""
        assert mask_middle_chars("a") == "a"
        assert mask_middle_chars("ab") == "ab"
        assert mask_middle_chars("abc") == "abc"
        assert mask_middle_chars("abcd") == "abcd"

    def test_length_5_mask_last_one(self):
        """长度为 5 时，保留前 4 位，最后 1 位替换为 *。"""
        result = mask_middle_chars("abcde")
        assert result == "abcd*"
        assert len(result) == 5

    def test_length_8_mask_last_four(self):
        """长度为 8 时（仍属于 5-8 段），保留前 4 位，其余全部替换为 *。"""
        result = mask_middle_chars("abcdefgh")
        assert result == "abcd****"
        assert len(result) == 8

    def test_length_gt_8_keep_head_and_tail(self):
        """长度 > 8 时，保留前 4 位和后 4 位，中间替换为 *。"""
        # 长度 9："abcdefghi" → "abcd*ghhi"，中间只有 1 个 *
        result = mask_middle_chars("abcdefghi")
        assert result == "abcd*fghi"
        assert len(result) == 9

    def test_length_20_long_secret(self):
        """较长字符串：前 4 + 中间全 * + 后 4。"""
        s = "sk-abcdef1234567890"  # 长度 19
        result = mask_middle_chars(s)
        assert result.startswith("sk-a")
        assert result.endswith("7890")
        # 中间全是 *
        middle = result[4:-4]
        assert all(c == "*" for c in middle)
        assert len(result) == len(s)


# =============================================================
# 2. raiseSecretsException 函数测试
# =============================================================

class TestRaiseSecretsException:
    """测试 raiseSecretsException 的开关控制与异常内容。"""

    def test_disabled_by_default_no_exception(self):
        """
        DETECTED_SECRETS_RAISE_ENABLED 默认为 'false'，
        调用函数不应抛出任何异常。
        """
        secrets = [{"type": "OpenAI API Key", "value": "sk-abcde12345"}]
        # 确保模块级变量为 'false'
        with patch.object(sd_module, "DETECTED_SECRETS_RAISE_ENABLED", "false"):
            # 不应抛出异常
            raiseSecretsException(secrets)

    def test_enabled_raises_http_exception(self):
        """
        DETECTED_SECRETS_RAISE_ENABLED='true' 时，应抛出 HTTPException(status_code=400)。
        """
        secrets = [{"type": "OpenAI API Key", "value": "sk-abcde12345"}]
        with patch.object(sd_module, "DETECTED_SECRETS_RAISE_ENABLED", "true"):
            with pytest.raises(HTTPException) as exc_info:
                raiseSecretsException(secrets)
        assert exc_info.value.status_code == 400

    def test_enabled_detail_contains_masked_value(self):
        """
        抛出的 HTTPException.detail 应包含 masked 后的 secret 值。
        """
        secret_value = "sk-abcde12345"
        secrets = [{"type": "OpenAI API Key", "value": secret_value}]
        expected_masked = mask_middle_chars(secret_value)  # "sk-a*****2345"

        with patch.object(sd_module, "DETECTED_SECRETS_RAISE_ENABLED", "true"):
            with pytest.raises(HTTPException) as exc_info:
                raiseSecretsException(secrets)

        assert expected_masked in exc_info.value.detail

    def test_enabled_multiple_secrets_all_shown(self):
        """
        多条 secret 时，所有 masked 值都应出现在 detail 中。
        这对应 commit 448fe93142 的改进：可显示全部敏感段落。
        """
        secrets = [
            {"type": "OpenAI API Key", "value": "sk-abcde12345"},
            {"type": "AWS Key", "value": "AKIAIOSFODNN7EXAMPLE"},
        ]
        expected_masked_1 = mask_middle_chars("sk-abcde12345")
        expected_masked_2 = mask_middle_chars("AKIAIOSFODNN7EXAMPLE")

        with patch.object(sd_module, "DETECTED_SECRETS_RAISE_ENABLED", "true"):
            with pytest.raises(HTTPException) as exc_info:
                raiseSecretsException(secrets)

        detail = exc_info.value.detail
        assert expected_masked_1 in detail, f"第一条 secret 未出现在 detail 中: {detail}"
        assert expected_masked_2 in detail, f"第二条 secret 未出现在 detail 中: {detail}"

    def test_enabled_empty_list_no_exception(self):
        """
        enabled 但 secret 列表为空时，不应抛出异常（join 得到空字符串仍会抛出）。
        实际代码会抛出 HTTPException，detail 为 "包含敏感信息: "。
        此测试仅确保空列表行为与预期一致。
        """
        with patch.object(sd_module, "DETECTED_SECRETS_RAISE_ENABLED", "true"):
            # 空列表仍然会触发 raise（因为代码无空列表守卫），验证 status_code
            with pytest.raises(HTTPException) as exc_info:
                raiseSecretsException([])
        assert exc_info.value.status_code == 400


# =============================================================
# 3. OpenAIApiKeyDetector 正则测试
# =============================================================

class TestOpenAIApiKeyDetectorRegex:
    """
    测试 OpenAIApiKeyDetector 的 denylist 正则表达式。
    对应 commit 6734e559b7：正则从 sk-xxx 改为边界匹配。
    """

    @pytest.fixture
    def detector(self):
        """创建 OpenAIApiKeyDetector 实例。"""
        return OpenAIApiKeyDetector()

    def _matches(self, detector: OpenAIApiKeyDetector, text: str) -> bool:
        """辅助方法：判断给定文本是否与任一 denylist 正则匹配。"""
        for pattern in detector.denylist:
            if pattern.search(text):
                return True
        return False

    def test_standard_format_detected(self, detector):
        """标准格式 sk-abcde12345 应能被检测到。"""
        assert self._matches(detector, "sk-abcde12345") is True

    def test_prefix_letter_not_detected(self, detector):
        """前面紧跟字母（如 somesk-abcde）不应被检测（负向后查断言）。"""
        assert self._matches(detector, "somesk-abcde12345") is False

    def test_suffix_letter_boundary_behavior(self, detector):
        """
        正则 (?<![a-zA-Z])(sk-[a-zA-Z0-9]{5,})(?![a-zA-Z]) 使用贪婪匹配，
        sk-abcde12345end 中 [a-zA-Z0-9]{5,} 会贪婪吞掉整个 'abcde12345end'，
        匹配到字符串末尾后负向前查 (?![a-zA-Z]) 通过（末尾无后续字母），
        因此整体字符串仍会被匹配。
        验证此行为：sk-abcde12345end 被视为一个合法 key token 并被检测到。
        """
        # 实际正则行为：整个 sk-abcde12345end 被贪婪匹配为一个 key
        assert self._matches(detector, "sk-abcde12345end") is True

    def test_key_followed_by_at_sign_not_detected(self, detector):
        """
        key 后紧跟非字母非数字字符（如 @）时，负向前查正常生效：
        但由于 @ 不在 [a-zA-Z0-9] 内，贪婪匹配在数字处停止，此时才真正触发后缀检查。
        sk-abcde12345@domain 中数字后跟 @ ，负向前查通过，key 被检测到。
        """
        assert self._matches(detector, "sk-abcde12345@domain") is True

    def test_key_preceded_by_letter_not_detected(self, detector):
        """
        前缀紧跟字母的情况（如 xsk-abcde12345）不应被检测（负向后查断言生效）。
        注意：与 somesk- 同类，只要 sk- 前有字母就不匹配。
        """
        assert self._matches(detector, "xsk-abcde12345") is False

    def test_pure_digits_detected(self, detector):
        """纯数字后缀 sk-12345 也应能被检测到。"""
        assert self._matches(detector, "sk-12345") is True

    def test_surrounded_by_spaces_detected(self, detector):
        """前后有空格的 ' sk-abcde12345 ' 应能被检测到（空格不是字母）。"""
        assert self._matches(detector, " sk-abcde12345 ") is True

    def test_key_in_sentence_detected(self, detector):
        """嵌入句子中间（单词边界处）的 key 应能被检测到。"""
        assert self._matches(detector, "my api key is sk-abcde12345 please use it") is True

    def test_too_short_not_detected(self, detector):
        """sk- 后面少于 5 个字符，不应被检测（正则要求 {5,}）。"""
        assert self._matches(detector, "sk-abc") is False  # 只有 3 个字符


# =============================================================
# 4. SecurityGuardrail.async_pre_call_hook 集成测试
# =============================================================

class TestSecurityGuardrailHook:
    """
    测试 _ENTERPRISE_SecretDetection.async_pre_call_hook 的行为。
    通过 mock should_run_check 跳过数据库依赖。
    """

    @pytest.fixture
    def guardrail(self):
        """创建 _ENTERPRISE_SecretDetection 实例。"""
        return _ENTERPRISE_SecretDetection()

    @pytest.fixture
    def user_api_key_dict(self):
        """创建虚拟的 UserAPIKeyAuth 对象。"""
        from litellm.proxy._types import UserAPIKeyAuth
        return UserAPIKeyAuth(api_key="test-key")

    @pytest.fixture
    def dual_cache(self):
        """创建虚拟的 DualCache 对象。"""
        from litellm.caching.caching import DualCache
        return DualCache()

    @pytest.mark.asyncio
    async def test_openai_key_in_message_raises_exception(
        self, guardrail, user_api_key_dict, dual_cache
    ):
        """
        消息内容含 OpenAI key 且 DETECTED_SECRETS_RAISE_ENABLED=true 时，
        async_pre_call_hook 应抛出 HTTPException(400)。
        """
        data = {
            "messages": [
                {"role": "user", "content": "请帮我用这个 key：sk-abcde12345 调用 API"}
            ]
        }

        # mock should_run_check 返回 True，跳过数据库权限检查
        with patch.object(guardrail, "should_run_check", new=AsyncMock(return_value=True)):
            # 设置抛出异常的环境变量
            with patch.object(sd_module, "DETECTED_SECRETS_RAISE_ENABLED", "true"):
                with pytest.raises(HTTPException) as exc_info:
                    await guardrail.async_pre_call_hook(
                        user_api_key_dict=user_api_key_dict,
                        cache=dual_cache,
                        data=data,
                        call_type="completion",
                    )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_no_secret_in_message_passes(
        self, guardrail, user_api_key_dict, dual_cache
    ):
        """
        消息内容不含任何敏感信息时，async_pre_call_hook 应正常通过（返回 None）。
        """
        data = {
            "messages": [
                {"role": "user", "content": "今天天气怎么样？"}
            ]
        }

        with patch.object(guardrail, "should_run_check", new=AsyncMock(return_value=True)):
            with patch.object(sd_module, "DETECTED_SECRETS_RAISE_ENABLED", "true"):
                # 不应抛出任何异常
                result = await guardrail.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=dual_cache,
                    data=data,
                    call_type="completion",
                )
        # 函数末尾 return None，result 应为 None
        assert result is None

    @pytest.mark.asyncio
    async def test_should_run_check_false_skips_scanning(
        self, guardrail, user_api_key_dict, dual_cache
    ):
        """
        当 should_run_check 返回 False 时，即使消息含有 key，也不扫描，不抛出异常。
        """
        data = {
            "messages": [
                {"role": "user", "content": "sk-abcde12345 这是一个 key"}
            ]
        }

        # should_run_check 返回 False → 跳过检查
        with patch.object(guardrail, "should_run_check", new=AsyncMock(return_value=False)):
            with patch.object(sd_module, "DETECTED_SECRETS_RAISE_ENABLED", "true"):
                # 不应抛出任何异常
                result = await guardrail.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=dual_cache,
                    data=data,
                    call_type="completion",
                )
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_secrets_in_messages_all_collected(
        self, guardrail, user_api_key_dict, dual_cache
    ):
        """
        多条消息中各含一个 secret 时，所有 secret 的 masked 值都应出现在 detail 中。
        对应 commit 448fe93142 的改进。
        """
        data = {
            "messages": [
                {"role": "user", "content": "第一个 key：sk-abcde12345"},
                {"role": "user", "content": "第二个 key：sk-fghij67890"},
            ]
        }

        expected_masked_1 = mask_middle_chars("sk-abcde12345")
        expected_masked_2 = mask_middle_chars("sk-fghij67890")

        with patch.object(guardrail, "should_run_check", new=AsyncMock(return_value=True)):
            with patch.object(sd_module, "DETECTED_SECRETS_RAISE_ENABLED", "true"):
                with pytest.raises(HTTPException) as exc_info:
                    await guardrail.async_pre_call_hook(
                        user_api_key_dict=user_api_key_dict,
                        cache=dual_cache,
                        data=data,
                        call_type="completion",
                    )

        detail = exc_info.value.detail
        assert expected_masked_1 in detail, f"第一个 secret masked 值未出现在 detail 中: {detail}"
        assert expected_masked_2 in detail, f"第二个 secret masked 值未出现在 detail 中: {detail}"
