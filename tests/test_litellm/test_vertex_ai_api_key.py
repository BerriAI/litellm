"""
Vertex AI api_key 支持单元测试

覆盖 VertexBase._check_custom_proxy 中的 api_key 追加逻辑：
- vertex_ai + 无 auth_header + 有 gemini_api_key → ?key= 或 &key= 追加到 URL
- vertex_ai + 有 auth_header → URL 不变
- gemini provider + 有 gemini_api_key → URL 不变（只对 vertex_ai 生效）
"""

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

import pytest
from unittest.mock import patch

from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


@pytest.fixture
def vertex_base():
    """创建 VertexBase 实例"""
    return VertexBase()


class TestCheckCustomProxyApiKey:
    """测试 _check_custom_proxy 中 api_key 追加到 URL 的逻辑"""

    # ─────────────────────────────────────────────────────────
    # 场景1：vertex_ai + 无 auth_header + 有 gemini_api_key
    #        URL 无问号 → 追加 ?key=
    # ─────────────────────────────────────────────────────────

    def test_vertex_ai_no_auth_header_appends_key_with_question_mark(self, vertex_base):
        """vertex_ai provider，无 auth_header，URL 无 ?，应使用 ?key=... 追加 api_key"""
        base_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-pro"
        gemini_api_key = "AIzaSy_test_key_12345"

        auth_header, result_url = vertex_base._check_custom_proxy(
            api_base=None,                      # 无自定义 api_base
            custom_llm_provider="vertex_ai",
            gemini_api_key=gemini_api_key,
            endpoint="generateContent",
            stream=None,
            auth_header=None,                   # 无 auth_header
            url=base_url,
        )

        # URL 应包含 ?key=
        assert f"?key={gemini_api_key}" in result_url, (
            f"期望 URL 包含 ?key={gemini_api_key}，实际 URL={result_url}"
        )
        # 不应使用 & 拼接（原 URL 无 ?）
        assert f"&key={gemini_api_key}" not in result_url

    # ─────────────────────────────────────────────────────────
    # 场景2：vertex_ai + 无 auth_header + 有 gemini_api_key
    #        URL 已含问号 → 追加 &key=
    # ─────────────────────────────────────────────────────────

    def test_vertex_ai_no_auth_header_appends_key_with_ampersand_when_url_has_params(
        self, vertex_base
    ):
        """vertex_ai provider，无 auth_header，URL 已有查询参数，应使用 &key=... 追加"""
        base_url = (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj"
            "/locations/us-central1/publishers/google/models/gemini-pro?alt=sse"
        )
        gemini_api_key = "AIzaSy_test_key_99999"

        auth_header, result_url = vertex_base._check_custom_proxy(
            api_base=None,
            custom_llm_provider="vertex_ai",
            gemini_api_key=gemini_api_key,
            endpoint="generateContent",
            stream=None,
            auth_header=None,
            url=base_url,
        )

        # URL 已有 ?，应用 & 拼接 key
        assert f"&key={gemini_api_key}" in result_url, (
            f"期望 URL 包含 &key={gemini_api_key}，实际 URL={result_url}"
        )
        # 不应再用 ? 拼接
        assert f"?key={gemini_api_key}" not in result_url
        # 原有参数 alt=sse 应保留
        assert "alt=sse" in result_url

    # ─────────────────────────────────────────────────────────
    # 场景3：vertex_ai + 有 auth_header → URL 不追加 key
    # ─────────────────────────────────────────────────────────

    def test_vertex_ai_with_auth_header_does_not_append_key(self, vertex_base):
        """vertex_ai provider，有 auth_header 时，URL 不应追加 key 参数"""
        base_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-pro"
        gemini_api_key = "AIzaSy_should_not_appear"

        auth_header, result_url = vertex_base._check_custom_proxy(
            api_base=None,
            custom_llm_provider="vertex_ai",
            gemini_api_key=gemini_api_key,
            endpoint="generateContent",
            stream=None,
            auth_header="Bearer my-valid-token",  # 已有 auth_header
            url=base_url,
        )

        # 有 auth_header 时不应追加 key
        assert "key=" not in result_url, (
            f"有 auth_header 时 URL 不应含 key=，实际 URL={result_url}"
        )
        assert result_url == base_url, (
            f"有 auth_header 时 URL 应保持不变，期望={base_url}，实际={result_url}"
        )

    # ─────────────────────────────────────────────────────────
    # 场景4：gemini provider + 有 gemini_api_key → URL 不追加 key
    #        （api_key 追加逻辑只对 vertex_ai 生效）
    # ─────────────────────────────────────────────────────────

    def test_gemini_provider_does_not_append_key_to_url(self, vertex_base):
        """gemini provider 使用 x-goog-api-key header，不应把 key 追加到 URL"""
        api_base = "https://generativelanguage.googleapis.com"
        gemini_api_key = "AIzaSy_gemini_key"
        model = "gemini-pro"
        endpoint = "generateContent"

        # gemini provider 带 api_base 时走 custom_proxy 路径
        auth_header, result_url = vertex_base._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider="gemini",
            gemini_api_key=gemini_api_key,
            endpoint=endpoint,
            stream=None,
            auth_header=None,
            url="https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
            model=model,
        )

        # gemini provider 不走 vertex_ai 的 api_key 追加逻辑，URL 不应含 ?key=
        assert "?key=" not in result_url, (
            f"gemini provider URL 不应追加 ?key=，实际 URL={result_url}"
        )
        assert "&key=" not in result_url, (
            f"gemini provider URL 不应追加 &key=，实际 URL={result_url}"
        )
        # auth_header 应被设置为包含 x-goog-api-key 的 dict
        assert auth_header is not None
        assert isinstance(auth_header, dict)
        assert auth_header.get("x-goog-api-key") == gemini_api_key

    # ─────────────────────────────────────────────────────────
    # 补充场景：无 api_base 时 vertex_ai 的 auth_header=None + 无 gemini_api_key
    # ─────────────────────────────────────────────────────────

    def test_vertex_ai_no_gemini_api_key_url_unchanged(self, vertex_base):
        """vertex_ai provider，无 auth_header，也无 gemini_api_key，URL 不变"""
        base_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-pro"

        auth_header, result_url = vertex_base._check_custom_proxy(
            api_base=None,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,             # 无 api_key
            endpoint="generateContent",
            stream=None,
            auth_header=None,
            url=base_url,
        )

        # 没有 gemini_api_key，URL 不应追加 key 参数
        assert "key=" not in result_url
        assert result_url == base_url

    # ─────────────────────────────────────────────────────────
    # 参数化测试：对多种 URL 格式验证 ? 与 & 的使用
    # ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "input_url, expected_separator",
        [
            # URL 无查询参数，应使用 ?
            (
                "https://us-central1-aiplatform.googleapis.com/v1/models/gemini-pro",
                "?",
            ),
            # URL 已有 alt=sse（来自 stream=True），应使用 &
            (
                "https://us-central1-aiplatform.googleapis.com/v1/models/gemini-pro?alt=sse",
                "&",
            ),
            # URL 已有多个参数，应使用 &
            (
                "https://us-central1-aiplatform.googleapis.com/v1/models/gemini-pro?foo=1&bar=2",
                "&",
            ),
        ],
    )
    def test_api_key_separator_depends_on_url_format(
        self, vertex_base, input_url, expected_separator
    ):
        """验证 URL 有无 ? 时分隔符的选择是正确的"""
        gemini_api_key = "test_key_abc"

        auth_header, result_url = vertex_base._check_custom_proxy(
            api_base=None,
            custom_llm_provider="vertex_ai",
            gemini_api_key=gemini_api_key,
            endpoint="generateContent",
            stream=None,
            auth_header=None,
            url=input_url,
        )

        expected_suffix = f"{expected_separator}key={gemini_api_key}"
        assert expected_suffix in result_url, (
            f"期望 URL 含 {expected_suffix!r}，实际 URL={result_url}"
        )
        # api_key 只追加一次
        assert result_url.count(f"key={gemini_api_key}") == 1, (
            f"api_key 应只追加一次，实际 URL={result_url}"
        )
