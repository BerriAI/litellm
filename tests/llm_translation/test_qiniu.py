"""
Integration tests for Qiniu provider.

Run with:
    QINIU_API_KEY=<your-key> .venv/bin/pytest tests/llm_translation/test_qiniu.py -v

Requires QINIU_API_KEY environment variable to be set.
"""

import os

import pytest

import litellm
from base_llm_unit_tests import BaseLLMChatTest

pytestmark = pytest.mark.skipif(
    not os.environ.get("QINIU_API_KEY"),
    reason="QINIU_API_KEY not set",
)


class TestQiniu(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "qiniu/deepseek/deepseek-v3.1-terminus-thinking",
        }

    # ── Unsupported features ──────────────────────────────────────────────────

    def test_image_url(self, **kwargs):
        pytest.skip("Qiniu does not support vision")

    def test_image_url_string(self):
        pytest.skip("Qiniu does not support vision")

    def test_audio_input(self):
        pytest.skip("Qiniu does not support audio input")

    def test_pdf_handling(self, **kwargs):
        pytest.skip("Qiniu does not support PDF input")

    def test_async_pdf_handling_with_file_id(self):
        pytest.skip("Qiniu does not support PDF input")

    def test_file_data_unit_test(self, pdf_messages):
        pytest.skip("Qiniu does not support file data")

    def test_prompt_caching(self):
        pytest.skip("Qiniu does not support prompt caching")

    def test_web_search(self):
        pytest.skip("Qiniu does not support web search")

    def test_url_context(self):
        pytest.skip("Qiniu does not support URL context")

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        pytest.skip("Skipped: not applicable")

    def test_tool_call_with_empty_enum_property(self):
        pytest.skip("Skipped: not applicable")

    def test_reasoning_effort(self):
        pytest.skip("Qiniu does not support reasoning_effort")
