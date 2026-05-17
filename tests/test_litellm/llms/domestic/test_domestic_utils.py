"""
Tests for domestic model compatibility utilities.

Tests the functions that detect domestic (Chinese) models and endpoints
for compatibility filtering in the Responses API handler.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.llms.domestic.domestic_utils import (
    _is_domestic_compatibility_disabled,
    is_domestic_endpoint,
    is_domestic_model,
    is_domestic_model_or_endpoint,
)


class TestIsDomesticModel(unittest.TestCase):
    """Test is_domestic_model function."""

    def test_qwen_model(self):
        """Test Qwen series models are detected."""
        self.assertTrue(is_domestic_model("qwen3.5-plus"))
        self.assertTrue(is_domestic_model("qwen-max"))
        self.assertTrue(is_domestic_model("openai/qwen3.5-plus"))

    def test_glm_model(self):
        """Test GLM series models are detected."""
        self.assertTrue(is_domestic_model("glm-4"))
        self.assertTrue(is_domestic_model("glm-5"))
        self.assertTrue(is_domestic_model("zhipu/glm-4"))

    def test_doubao_model(self):
        """Test Doubao series models are detected."""
        self.assertTrue(is_domestic_model("doubao-pro-32k"))
        self.assertTrue(is_domestic_model("Doubao-Seed-2.0-Code"))

    def test_minimax_model(self):
        """Test MiniMax series models are detected."""
        self.assertTrue(is_domestic_model("MiniMax-M2.7"))
        self.assertTrue(is_domestic_model("minimax-01"))

    def test_mimo_model(self):
        """Test Xiaomi MiMo series models are detected."""
        self.assertTrue(is_domestic_model("mimo-v2.5-pro"))
        self.assertTrue(is_domestic_model("MiMo-7B"))

    def test_deepseek_model(self):
        """Test DeepSeek series models are detected."""
        self.assertTrue(is_domestic_model("deepseek-coder"))
        self.assertTrue(is_domestic_model("deepseek-chat"))
        self.assertTrue(is_domestic_model("openai/deepseek-pro"))

    def test_kimi_model(self):
        """Test Kimi series models are detected."""
        self.assertTrue(is_domestic_model("kimi-k2"))
        self.assertTrue(is_domestic_model("kimi-coding"))

    def test_non_domestic_model(self):
        """Test non-domestic models are not detected."""
        self.assertFalse(is_domestic_model("gpt-4"))
        self.assertFalse(is_domestic_model("claude-3-opus"))
        self.assertFalse(is_domestic_model("gemini-pro"))
        self.assertFalse(is_domestic_model("llama-3"))

    def test_none_model(self):
        """Test None model name returns False."""
        self.assertFalse(is_domestic_model(None))

    def test_empty_model(self):
        """Test empty model name returns False."""
        self.assertFalse(is_domestic_model(""))
        self.assertFalse(is_domestic_model("   "))

    def test_case_insensitive(self):
        """Test detection is case insensitive."""
        self.assertTrue(is_domestic_model("QWEN-PLUS"))
        self.assertTrue(is_domestic_model("DEEPSEEK-CODER"))
        self.assertTrue(is_domestic_model("MINIMAX-M2"))


class TestIsDomesticEndpoint(unittest.TestCase):
    """Test is_domestic_endpoint function."""

    def test_alibaba_endpoint(self):
        """Test Alibaba DashScope endpoint."""
        self.assertTrue(is_domestic_endpoint("https://dashscope.aliyuncs.com/v1"))
        self.assertTrue(is_domestic_endpoint("dashscope.aliyuncs.com"))

    def test_volcengine_endpoint(self):
        """Test Volcengine endpoint."""
        self.assertTrue(is_domestic_endpoint("https://ark.cn-beijing.volces.com/v1"))
        self.assertTrue(is_domestic_endpoint("ark.cn-beijing.volces.com"))

    def test_minimax_endpoint(self):
        """Test MiniMax endpoint."""
        self.assertTrue(is_domestic_endpoint("https://api.minimaxi.com/v1"))
        self.assertTrue(is_domestic_endpoint("api.minimaxi.com"))

    def test_xiaomi_endpoint(self):
        """Test Xiaomi MiMo endpoint."""
        self.assertTrue(is_domestic_endpoint("https://xiaomimimo.com/api/v1"))
        self.assertTrue(is_domestic_endpoint("xiaomimimo.com"))

    def test_deepseek_endpoint(self):
        """Test DeepSeek endpoint."""
        self.assertTrue(is_domestic_endpoint("https://api.deepseek.com/v1"))
        self.assertTrue(is_domestic_endpoint("api.deepseek.com"))

    def test_moonshot_endpoint(self):
        """Test Moonshot Kimi endpoint."""
        self.assertTrue(is_domestic_endpoint("https://api.moonshot.cn/v1"))
        self.assertTrue(is_domestic_endpoint("moonshot.cn"))

    def test_zhipu_endpoint(self):
        """Test Zhipu GLM endpoint."""
        self.assertTrue(is_domestic_endpoint("https://open.bigmodel.cn/api/v1"))
        self.assertTrue(is_domestic_endpoint("bigmodel.cn"))

    def test_non_domestic_endpoint(self):
        """Test non-domestic endpoints are not detected."""
        self.assertFalse(is_domestic_endpoint("https://api.openai.com/v1"))
        self.assertFalse(is_domestic_endpoint("api.anthropic.com"))
        self.assertFalse(is_domestic_endpoint("generativelanguage.googleapis.com"))

    def test_none_endpoint(self):
        """Test None endpoint returns False."""
        self.assertFalse(is_domestic_endpoint(None))

    def test_empty_endpoint(self):
        """Test empty endpoint returns False."""
        self.assertFalse(is_domestic_endpoint(""))
        self.assertFalse(is_domestic_endpoint("   "))


class TestIsDomesticModelOrEndpoint(unittest.TestCase):
    """Test is_domestic_model_or_endpoint function."""

    def test_domestic_model_non_domestic_endpoint(self):
        """Test domestic model with non-domestic endpoint."""
        self.assertTrue(
            is_domestic_model_or_endpoint("qwen3.5-plus", "https://api.openai.com/v1")
        )

    def test_non_domestic_model_domestic_endpoint(self):
        """Test non-domestic model with domestic endpoint."""
        self.assertTrue(
            is_domestic_model_or_endpoint("gpt-4", "https://dashscope.aliyuncs.com/v1")
        )

    def test_both_domestic(self):
        """Test both model and endpoint are domestic."""
        self.assertTrue(
            is_domestic_model_or_endpoint(
                "deepseek-coder", "https://api.deepseek.com/v1"
            )
        )

    def test_both_non_domestic(self):
        """Test both model and endpoint are non-domestic."""
        self.assertFalse(
            is_domestic_model_or_endpoint("gpt-4", "https://api.openai.com/v1")
        )

    def test_none_values(self):
        """Test None values."""
        self.assertFalse(is_domestic_model_or_endpoint(None, None))
        self.assertTrue(is_domestic_model_or_endpoint("minimax", None))
        self.assertFalse(
            is_domestic_model_or_endpoint(None, "https://api.openai.com/v1")
        )

    def test_empty_values(self):
        """Test empty values."""
        self.assertFalse(is_domestic_model_or_endpoint("", ""))
        self.assertTrue(is_domestic_model_or_endpoint("mimo", ""))


class TestDisableCompatibility(unittest.TestCase):
    """Test _is_domestic_compatibility_disabled function."""

    def setUp(self):
        """Save original env value."""
        self.original_value = os.environ.get("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY")

    def tearDown(self):
        """Restore original env value."""
        if self.original_value is not None:
            os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = self.original_value
        elif "LITELLM_DISABLE_DOMESTIC_COMPATIBILITY" in os.environ:
            del os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"]

    def test_disabled_true(self):
        """Test when disabled is set to 'true'."""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "true"
        self.assertTrue(_is_domestic_compatibility_disabled())

    def test_disabled_1(self):
        """Test when disabled is set to '1'."""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "1"
        self.assertTrue(_is_domestic_compatibility_disabled())

    def test_disabled_yes(self):
        """Test when disabled is set to 'yes'."""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "yes"
        self.assertTrue(_is_domestic_compatibility_disabled())

    def test_disabled_false(self):
        """Test when disabled is set to 'false'."""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "false"
        self.assertFalse(_is_domestic_compatibility_disabled())

    def test_not_set(self):
        """Test when env var is not set."""
        if "LITELLM_DISABLE_DOMESTIC_COMPATIBILITY" in os.environ:
            del os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"]
        self.assertFalse(_is_domestic_compatibility_disabled())

    def test_disable_affects_is_domestic_model(self):
        """Test that disable flag affects is_domestic_model."""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "true"
        # Even for domestic model names, should return False when disabled
        self.assertFalse(is_domestic_model("qwen3.5-plus"))
        self.assertFalse(is_domestic_model("deepseek-coder"))

    def test_disable_affects_is_domestic_endpoint(self):
        """Test that disable flag affects is_domestic_endpoint."""
        os.environ["LITELLM_DISABLE_DOMESTIC_COMPATIBILITY"] = "true"
        # Even for domestic endpoints, should return False when disabled
        self.assertFalse(is_domestic_endpoint("https://dashscope.aliyuncs.com/v1"))
        self.assertFalse(is_domestic_endpoint("https://api.deepseek.com/v1"))


if __name__ == "__main__":
    unittest.main()
