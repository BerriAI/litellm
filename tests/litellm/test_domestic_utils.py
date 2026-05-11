"""
Tests for domestic_utils.py - Chinese model compatibility detection
"""

import os
import pytest

from litellm.llms.domestic.domestic_utils import (
    _is_domestic_compatibility_disabled,
    is_domestic_model,
    is_domestic_endpoint,
    is_domestic_model_or_endpoint,
)


class TestIsDomesticCompatibilityDisabled:
    """Tests for _is_domestic_compatibility_disabled function"""

    def test_default_disabled(self):
        """By default, domestic compatibility is enabled (function returns False)"""
        # Ensure env var is not set
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)
        assert _is_domestic_compatibility_disabled() == False

    def test_disabled_with_true(self):
        """Setting to 'true' should disable domestic compatibility"""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "true"
        assert _is_domestic_compatibility_disabled() == True
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)

    def test_disabled_with_1(self):
        """Setting to '1' should disable domestic compatibility"""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "1"
        assert _is_domestic_compatibility_disabled() == True
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)

    def test_disabled_with_yes(self):
        """Setting to 'yes' should disable domestic compatibility"""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "yes"
        assert _is_domestic_compatibility_disabled() == True
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)

    def test_not_disabled_with_false(self):
        """Setting to 'false' should NOT disable (compatibility enabled)"""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "false"
        assert _is_domestic_compatibility_disabled() == False
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)

    def test_not_disabled_with_random_value(self):
        """Random values should NOT disable"""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "random"
        assert _is_domestic_compatibility_disabled() == False
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)

    def test_case_insensitive_true(self):
        """'TRUE' (uppercase) should also work"""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "TRUE"
        assert _is_domestic_compatibility_disabled() == True
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)


class TestEnvironmentVariableOptOut:
    """Tests for LITELLM_DISABLE_DOMESTIC_COMPATIBILITY opt-out mechanism"""

    def test_disable_env_bypasses_model_check(self):
        """When disabled, even domestic model names return False"""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "true"
        assert is_domestic_model("qwen3.5-plus") == False
        assert is_domestic_model("MiniMax-M2.7") == False
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)

    def test_disable_env_bypasses_endpoint_check(self):
        """When disabled, even domestic endpoints return False"""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "true"
        assert is_domestic_endpoint("https://dashscope.aliyuncs.com/api/v1") == False
        assert is_domestic_endpoint("https://api.deepseek.com/v1") == False
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)

    def test_disable_env_bypasses_combined_check(self):
        """When disabled, combined check also returns False"""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "true"
        assert is_domestic_model_or_endpoint("qwen3.5-plus", "https://dashscope.aliyuncs.com/api/v1") == False
        os.environ.pop("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", None)


class TestIsDomesticModel:
    """Tests for is_domestic_model function"""

    def test_none_model(self):
        """None model should return False"""
        assert is_domestic_model(None) == False

    def test_empty_model(self):
        """Empty string should return False"""
        assert is_domestic_model("") == False

    def test_qwen_model(self):
        """Qwen models should be detected"""
        assert is_domestic_model("qwen3.5-plus") == True
        assert is_domestic_model("qwen-max") == True
        assert is_domestic_model("QWEN-TURBO") == True  # case insensitive

    def test_glm_model(self):
        """GLM models should be detected"""
        assert is_domestic_model("glm-5") == True
        assert is_domestic_model("GLM-4") == True

    def test_doubao_model(self):
        """Doubao models should be detected"""
        assert is_domestic_model("doubao-seed-2.0-pro") == True
        assert is_domestic_model("DOUBAO-PRO") == True

    def test_minimax_model(self):
        """MiniMax models should be detected"""
        assert is_domestic_model("MiniMax-M2.7") == True
        assert is_domestic_model("minimax-01") == True

    def test_mimo_model(self):
        """MiMo models should be detected"""
        assert is_domestic_model("mimo-v2.5-pro") == True
        assert is_domestic_model("MIMO-7B") == True

    def test_deepseek_model(self):
        """DeepSeek models should be detected"""
        assert is_domestic_model("deepseek-chat") == True
        assert is_domestic_model("DeepSeek-V3") == True

    def test_kimi_model(self):
        """Kimi models should be detected"""
        assert is_domestic_model("kimi-chat") == True
        assert is_domestic_model("KIMI-MOONSHOT") == True

    def test_prefixed_model(self):
        """Models with provider prefix should be detected"""
        assert is_domestic_model("openai/MiniMax-M2.7") == True
        assert is_domestic_model("openai/qwen3.5-plus") == True

    def test_non_domestic_model(self):
        """Non-domestic models should return False"""
        assert is_domestic_model("gpt-4") == False
        assert is_domestic_model("claude-3-opus") == False
        assert is_domestic_model("gemini-pro") == False


class TestIsDomesticEndpoint:
    """Tests for is_domestic_endpoint function"""

    def test_none_endpoint(self):
        """None endpoint should return False"""
        assert is_domestic_endpoint(None) == False

    def test_empty_endpoint(self):
        """Empty string should return False"""
        assert is_domestic_endpoint("") == False

    def test_dashscope_endpoint(self):
        """Alibaba DashScope endpoint should be detected"""
        assert is_domestic_endpoint("https://dashscope.aliyuncs.com/api/v1") == True

    def test_volcengine_endpoint(self):
        """Volcengine (ByteDance) endpoint should be detected"""
        assert is_domestic_endpoint("https://ark.cn-beijing.volces.com/api/v3") == True

    def test_minimax_endpoint(self):
        """MiniMax endpoint should be detected"""
        assert is_domestic_endpoint("https://api.minimaxi.com/v1/chat") == True

    def test_mimo_endpoint(self):
        """MiMo endpoint should be detected"""
        assert is_domestic_endpoint("https://xiaomimimo.com/api/chat") == True

    def test_deepseek_endpoint(self):
        """DeepSeek endpoint should be detected"""
        assert is_domestic_endpoint("https://api.deepseek.com/v1") == True

    def test_moonshot_endpoint(self):
        """Moonshot (Kimi) endpoint should be detected"""
        assert is_domestic_endpoint("https://moonshot.cn/v1/chat") == True

    def test_bigmodel_endpoint(self):
        """Zhipu GLM endpoint should be detected"""
        assert is_domestic_endpoint("https://bigmodel.cn/api/paas/v4") == True

    def test_non_domestic_endpoint(self):
        """Non-domestic endpoints should return False"""
        assert is_domestic_endpoint("https://api.openai.com/v1") == False
        assert is_domestic_endpoint("https://api.anthropic.com/v1") == False


class TestIsDomesticModelOrEndpoint:
    """Tests for is_domestic_model_or_endpoint function"""

    def test_none_both(self):
        """None model and endpoint should return False"""
        assert is_domestic_model_or_endpoint(None, None) == False

    def test_domestic_model_only(self):
        """Domestic model with None endpoint should return True"""
        assert is_domestic_model_or_endpoint("qwen3.5-plus", None) == True
        assert is_domestic_model_or_endpoint("MiniMax-M2.7", None) == True

    def test_domestic_endpoint_only(self):
        """Non-domestic model name but domestic endpoint should return True"""
        # This covers the case where model name is just a group name like "codex-model"
        assert (
            is_domestic_model_or_endpoint(
                "my-model", "https://dashscope.aliyuncs.com/api/v1"
            )
            == True
        )
        assert (
            is_domestic_model_or_endpoint(
                "custom-llm", "https://ark.cn-beijing.volces.com/api/v3"
            )
            == True
        )

    def test_both_domestic(self):
        """Both domestic model and endpoint should return True"""
        assert (
            is_domestic_model_or_endpoint(
                "qwen3.5-plus", "https://dashscope.aliyuncs.com/api/v1"
            )
            == True
        )

    def test_both_non_domestic(self):
        """Both non-domestic should return False"""
        assert (
            is_domestic_model_or_endpoint("gpt-4", "https://api.openai.com/v1") == False
        )

    def test_model_priority(self):
        """Model name check has priority"""
        # Even if endpoint is non-domestic, if model name is domestic, return True
        assert (
            is_domestic_model_or_endpoint("qwen3.5-plus", "https://api.openai.com/v1")
            == True
        )

    def test_endpoint_fallback(self):
        """Endpoint check is fallback when model name is non-domestic"""
        assert (
            is_domestic_model_or_endpoint("custom-model", "https://api.deepseek.com/v1")
            == True
        )
