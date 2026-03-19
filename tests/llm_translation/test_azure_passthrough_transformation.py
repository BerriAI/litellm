"""
Azure Passthrough Transformation 单元测试

覆盖：
1. AzurePassthroughConfig.is_streaming_request - stream=True/False/None/缺失
2. BaseAzureLLM._get_base_azure_url - request_query_params 合并逻辑
"""

import sys
import os

sys.path.insert(0, os.path.abspath("../../"))

import pytest
from unittest.mock import patch

import httpx

from litellm.llms.azure.passthrough.transformation import AzurePassthroughConfig
from litellm.llms.azure.common_utils import BaseAzureLLM


# ─────────────────────────────────────────────────────────────
# 测试 AzurePassthroughConfig.is_streaming_request
# ─────────────────────────────────────────────────────────────

class TestIsStreamingRequest:
    """测试 is_streaming_request 方法的各种入参场景"""

    def setup_method(self):
        self.config = AzurePassthroughConfig()

    @pytest.mark.parametrize(
        "request_data, expected",
        [
            # stream=True 应返回 True
            ({"stream": True}, True),
            # stream=False 应返回 False
            ({"stream": False}, False),
            # stream=None 应返回 False（修复前可能返回 None）
            ({"stream": None}, False),
            # 不含 stream 键时应返回默认值 False
            ({}, False),
            # 含有其他字段但无 stream 字段
            ({"model": "gpt-4", "messages": []}, False),
        ],
    )
    def test_is_streaming_request(self, request_data, expected):
        """验证 is_streaming_request 在各种输入下都返回正确的布尔值"""
        result = self.config.is_streaming_request(
            endpoint="/openai/deployments/gpt-4/chat/completions",
            request_data=request_data,
        )
        assert result == expected, (
            f"期望 is_streaming_request 返回 {expected}，"
            f"实际返回 {result}，输入 request_data={request_data}"
        )

    def test_stream_none_returns_bool_not_none(self):
        """修复验证：stream=None 时必须返回 False 而非 None"""
        result = self.config.is_streaming_request(
            endpoint="/openai/deployments/gpt-4/chat/completions",
            request_data={"stream": None},
        )
        # 修复前 dict.get('stream', False) 在 stream=None 时会返回 None
        # 修复后 stream=None → False（因为 None or False 不行，这里用 bool 默认值保证）
        assert result is False, (
            f"stream=None 时应返回 False，而不是 {result!r}"
        )
        # 同时确保返回值是真正的 False（非 None）
        assert result is not None


# ─────────────────────────────────────────────────────────────
# 测试 BaseAzureLLM._get_base_azure_url
# ─────────────────────────────────────────────────────────────

BASE_URL = "https://my-resource.openai.azure.com"
BASE_URL_WITH_PARAMS = "https://my-resource.openai.azure.com?existing=1"


class TestGetBaseAzureUrl:
    """测试 _get_base_azure_url 的 query param 合并逻辑"""

    # ── 无 request_query_params ──────────────────────────────

    def test_no_request_query_params_adds_api_version(self):
        """无 request_query_params 时，api_version 应自动追加到 URL"""
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL,
            litellm_params={"api_version": "2024-02-01"},
            route="/openai/deployments/gpt-4/chat/completions",
            request_query_params=None,
        )
        parsed = httpx.URL(result)
        assert parsed.params["api-version"] == "2024-02-01"

    def test_no_request_query_params_and_no_api_version(self):
        """无 request_query_params 且无 api_version 时，URL 不含 api-version"""
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL,
            litellm_params={},
            route="/openai/deployments/gpt-4/chat/completions",
            request_query_params=None,
        )
        parsed = httpx.URL(result)
        assert "api-version" not in dict(parsed.params)

    # ── 有 request_query_params ──────────────────────────────

    def test_request_query_params_merged_into_url(self):
        """request_query_params 中的参数应出现在最终 URL 中"""
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL,
            litellm_params={},
            route="/openai/deployments/gpt-4/chat/completions",
            request_query_params={"custom_param": "hello"},
        )
        parsed = httpx.URL(result)
        assert parsed.params["custom_param"] == "hello"

    def test_request_query_params_multiple_params_merged(self):
        """request_query_params 中多个参数都应合并到 URL"""
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL,
            litellm_params={"api_version": "2024-02-01"},
            route="/openai/deployments/gpt-4/chat/completions",
            request_query_params={"foo": "bar", "baz": "qux"},
        )
        parsed = httpx.URL(result)
        assert parsed.params["foo"] == "bar"
        assert parsed.params["baz"] == "qux"
        assert parsed.params["api-version"] == "2024-02-01"

    # ── URL 自带 params 应被保留 ──────────────────────────────

    def test_url_existing_params_preserved(self):
        """api_base 中已有的 query params 应保留在最终 URL 中"""
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL_WITH_PARAMS,
            litellm_params={},
            route="/openai/deployments/gpt-4/chat/completions",
            request_query_params=None,
        )
        parsed = httpx.URL(result)
        assert parsed.params["existing"] == "1"

    def test_url_existing_params_and_request_params_both_present(self):
        """api_base 中已有的 params 与 request_query_params 应同时出现"""
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL_WITH_PARAMS,
            litellm_params={"api_version": "2024-02-01"},
            route="/openai/deployments/gpt-4/chat/completions",
            request_query_params={"custom": "value"},
        )
        parsed = httpx.URL(result)
        # URL 自带的 existing=1 应保留
        assert parsed.params["existing"] == "1"
        # request_query_params 的 custom=value 应合并
        assert parsed.params["custom"] == "value"
        # api_version 也应出现
        assert parsed.params["api-version"] == "2024-02-01"

    # ── api-version 合并优先级 ──────────────────────────────

    def test_api_version_from_litellm_params(self):
        """litellm_params 中的 api_version 应优先于 default_api_version"""
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL,
            litellm_params={"api_version": "2024-02-01"},
            route="/openai/deployments/gpt-4/chat/completions",
            default_api_version="2023-05-15",
            request_query_params=None,
        )
        parsed = httpx.URL(result)
        # litellm_params 中的 api_version 应生效
        assert parsed.params["api-version"] == "2024-02-01"

    def test_default_api_version_used_when_no_litellm_params_version(self):
        """无 litellm_params api_version 时，使用 default_api_version"""
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL,
            litellm_params={},
            route="/openai/deployments/gpt-4/chat/completions",
            default_api_version="2023-05-15",
            request_query_params=None,
        )
        parsed = httpx.URL(result)
        assert parsed.params["api-version"] == "2023-05-15"

    def test_existing_api_version_in_url_not_overwritten_by_default(self):
        """api_base 中已有 api-version 时，不被 default_api_version 覆盖"""
        url_with_api_version = f"{BASE_URL}?api-version=2023-01-01"
        result = BaseAzureLLM._get_base_azure_url(
            api_base=url_with_api_version,
            litellm_params={},
            route="/openai/deployments/gpt-4/chat/completions",
            default_api_version="9999-99-99",
            request_query_params=None,
        )
        parsed = httpx.URL(result)
        # URL 自带的 api-version 不应被覆盖
        assert parsed.params["api-version"] == "2023-01-01"

    # ── request_query_params 与 URL 自带 params 的覆盖顺序 ──

    def test_url_params_override_request_query_params_for_same_key(self):
        """同名参数：URL 自带 params 优先于 request_query_params（右侧 | 右值胜）

        合并逻辑为：dict(request_query_params) | dict(original_url.params)
        所以 URL 自带的 params 会覆盖 request_query_params 中的同名键
        """
        url_with_key = f"{BASE_URL}?mykey=from_url"
        result = BaseAzureLLM._get_base_azure_url(
            api_base=url_with_key,
            litellm_params={},
            route="/openai/deployments/gpt-4/chat/completions",
            request_query_params={"mykey": "from_request"},
        )
        parsed = httpx.URL(result)
        # URL 自带的 mykey 应覆盖 request_query_params 中的 mykey
        assert parsed.params["mykey"] == "from_url"

    # ── 路由路径处理 ──────────────────────────────────────────

    def test_route_appended_to_base_url(self):
        """route 路径应正确追加到 api_base"""
        route = "/openai/deployments/gpt-4/chat/completions"
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL,
            litellm_params={},
            route=route,
            request_query_params=None,
        )
        parsed = httpx.URL(result)
        assert route in parsed.path

    @pytest.mark.parametrize(
        "request_query_params",
        [
            None,
            {},
            {"key1": "val1"},
            {"key1": "val1", "key2": "val2"},
        ],
    )
    def test_result_is_valid_url_string(self, request_query_params):
        """_get_base_azure_url 应始终返回有效的 URL 字符串"""
        result = BaseAzureLLM._get_base_azure_url(
            api_base=BASE_URL,
            litellm_params={"api_version": "2024-02-01"},
            route="/openai/deployments/gpt-4/chat/completions",
            request_query_params=request_query_params,
        )
        assert isinstance(result, str)
        # 确保可以被 httpx.URL 解析
        parsed = httpx.URL(result)
        assert parsed.host == "my-resource.openai.azure.com"
