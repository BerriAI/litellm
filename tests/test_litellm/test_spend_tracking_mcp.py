"""
单元测试：spend_tracking 对 MCP tool response 的处理逻辑
对应 commit db26ea1d15 "支持记录MCP tools response到spend_log"

核心逻辑（spend_tracking_utils.py get_logging_payload 函数）：
  当满足以下全部条件时，将 original_response_obj 序列化为 JSON 并写入 response 字段：
    1. original_response_obj 不为 None
    2. mcp_namespaced_tool_name 不为 None
    3. standard_logging_payload 不为 None
    4. standard_logging_payload["response"] 为假值（空/None）
    5. original_response_obj 是 list 且非空
"""

import datetime
import json
import sys
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 初始化 mock，避免导入时依赖真实代理服务器
# ---------------------------------------------------------------------------
_proxy_server_mock = MagicMock()
_proxy_server_mock.general_settings = {}
_proxy_server_mock.master_key = None

# 必须在导入 spend_tracking_utils 之前 mock，否则模块级 import 会失败
sys.modules.setdefault("litellm.proxy.proxy_server", _proxy_server_mock)


from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助函数：构造最小可用的 StandardLoggingPayload
# ---------------------------------------------------------------------------

def _make_standard_logging_payload(
    mcp_tool_call_metadata: Optional[dict] = None,
    response: Any = None,
) -> dict:
    """构造一个符合 StandardLoggingPayload 结构的最小字典，方便测试复用。"""
    return {
        "id": "test-id-001",
        "trace_id": "trace-001",
        "call_type": "mcp_tool_call",
        "stream": False,
        "response_cost": 0.0,
        "cost_breakdown": None,
        "response_cost_failure_debug_info": None,
        "status": "success",
        "status_fields": {},
        "custom_llm_provider": None,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "startTime": 0.0,
        "endTime": 1.0,
        "completionStartTime": 0.5,
        "response_time": 1.0,
        "model_map_information": {"model_map_key": "gpt-4", "model_map_value": None},
        "model": "gpt-4",
        "model_id": None,
        "model_group": None,
        "api_base": "",
        "metadata": {
            # 标准 API key 元数据字段
            "user_api_key": None,
            "user_api_key_alias": None,
            "user_api_key_team_id": None,
            "user_api_key_org_id": None,
            "user_api_key_user_id": None,
            "user_api_key_team_alias": None,
            "spend_logs_metadata": None,
            "requester_ip_address": None,
            "requester_metadata": None,
            "requester_custom_headers": None,
            "prompt_management_metadata": None,
            # MCP 相关
            "mcp_tool_call_metadata": mcp_tool_call_metadata,
            "vector_store_request_metadata": None,
            "applied_guardrails": None,
            "usage_object": None,
            "cold_storage_object_key": None,
        },
        "cache_hit": False,
        "cache_key": None,
        "saved_cache_cost": 0.0,
        "request_tags": [],
        "end_user": None,
        "requester_ip_address": None,
        "messages": None,
        # response 字段：None 表示"没有已有 response"，有值则不应被覆盖
        "response": response,
        "error_str": None,
        "error_information": None,
        "model_parameters": {},
        "hidden_params": {
            "model_id": None,
            "cache_key": None,
            "api_base": None,
            "response_cost": None,
            "litellm_overhead_time_ms": None,
            "additional_headers": None,
            "batch_models": None,
            "litellm_model_name": None,
            "usage_object": None,
        },
        "guardrail_information": None,
        "standard_built_in_tools_params": None,
    }


def _make_kwargs(standard_logging_payload: Optional[dict] = None) -> dict:
    """构造传给 get_logging_payload 的 kwargs 字典。"""
    return {
        "standard_logging_object": standard_logging_payload,
        "litellm_params": {},
        "call_type": "mcp_tool_call",
        "model": "gpt-4",
        "litellm_call_id": "call-mcp-001",
    }


def _now() -> datetime.datetime:
    """返回当前 UTC 时间，方便测试使用。"""
    return datetime.datetime.now(datetime.timezone.utc)


# ---------------------------------------------------------------------------
# 辅助：给 general_settings 打 patch，避免 _should_store_prompts_and_responses_in_spend_logs
# 被真实配置干扰（默认 False，即不存储 prompt/response 到 spendlogs）
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_proxy_server_mock():
    """每个测试前重置 general_settings，确保隔离。"""
    _proxy_server_mock.general_settings = {}
    _proxy_server_mock.master_key = None
    yield
    _proxy_server_mock.general_settings = {}


# ---------------------------------------------------------------------------
# 测试类
# ---------------------------------------------------------------------------


class TestMCPToolResponseInSpendLogs:
    """
    测试 get_logging_payload 中针对 MCP tool response 的特殊处理分支。
    """

    def test_mcp_namespaced_tool_name_not_none_and_valid_list_sets_response(self):
        """
        测试点1：当 mcp_namespaced_tool_name 不为 None 且 original_response_obj
        是有效列表时，response 字段被设置为 MCP tool response 的 JSON 序列化结果。
        """
        # 构造 MCP 元数据，包含 namespaced_tool_name
        mcp_meta = {
            "namespaced_tool_name": "my-server/get_info",
            "name": "get_info",
            "arguments": {"query": "test"},
            "result": {},
        }
        slp = _make_standard_logging_payload(
            mcp_tool_call_metadata=mcp_meta,
            response=None,  # 没有已有 response
        )
        kwargs = _make_kwargs(standard_logging_payload=slp)
        now = _now()

        # original_response_obj 是字符串列表
        original_response_obj = ["result content", "second item"]

        payload = get_logging_payload(kwargs, original_response_obj, now, now)

        # response 字段不应为 "{}"（空）
        assert payload["response"] != "{}"

        # response 应为合法 JSON，且包含 mcp_tool_response 键
        parsed = json.loads(payload["response"])
        assert "mcp_tool_response" in parsed

        # 列表内容应被完整序列化
        mcp_responses = parsed["mcp_tool_response"]
        assert mcp_responses == ["result content", "second item"]

        # mcp_namespaced_tool_name 字段应被正确填充
        assert payload["mcp_namespaced_tool_name"] == "my-server/get_info"

    def test_original_response_obj_none_skips_mcp_handling(self):
        """
        测试点2：当 original_response_obj 为 None 时，不触发 MCP 特殊处理。
        response 字段应保持为 "{}"（因为 standard_logging_payload 的 response 也为空）。
        """
        mcp_meta = {
            "namespaced_tool_name": "my-server/get_info",
            "name": "get_info",
            "arguments": {},
            "result": {},
        }
        slp = _make_standard_logging_payload(
            mcp_tool_call_metadata=mcp_meta,
            response=None,
        )
        kwargs = _make_kwargs(standard_logging_payload=slp)
        now = _now()

        # original_response_obj 为 None
        payload = get_logging_payload(kwargs, None, now, now)

        # None 不是 list，MCP 分支不应被触发；response 为默认空值
        assert payload["response"] == "{}"

    def test_existing_response_not_overwritten(self):
        """
        测试点3：当 standard_logging_payload["response"] 已有值时，不覆盖 response 字段。

        条件检查：not standard_logging_payload.get("response", {}) 为 False 时跳过 MCP 处理。
        注意：_get_response_for_spend_logs_payload 仅在 store_prompts_in_spend_logs=True 时
        才读取 payload["response"]，默认返回 "{}"。所以此测试需要开启该开关。
        """
        mcp_meta = {
            "namespaced_tool_name": "my-server/get_info",
            "name": "get_info",
            "arguments": {},
            "result": {},
        }
        # standard_logging_payload 已有 response 值（非空）
        existing_response = {"choices": [{"message": {"content": "existing answer"}}]}
        slp = _make_standard_logging_payload(
            mcp_tool_call_metadata=mcp_meta,
            response=existing_response,  # 已有 response
        )
        kwargs = _make_kwargs(standard_logging_payload=slp)
        now = _now()
        original_response_obj = ["new mcp result"]

        # 开启 store_prompts_in_spend_logs，使 _get_response_for_spend_logs_payload
        # 能够读取并返回 payload 中已有的 response。
        # 使用 patch 直接让 _should_store_prompts_and_responses_in_spend_logs 返回 True，
        # 避免依赖 general_settings 的内部注入机制。
        with patch(
            "litellm.proxy.spend_tracking.spend_tracking_utils._should_store_prompts_and_responses_in_spend_logs",
            return_value=True,
        ):
            payload = get_logging_payload(kwargs, original_response_obj, now, now)

        # 解析 response 字段
        parsed = json.loads(payload["response"])

        # MCP 分支不应触发，所以不应有 mcp_tool_response 键
        assert "mcp_tool_response" not in parsed, (
            "当 standard_logging_payload 已有 response 时，不应触发 MCP 覆盖逻辑"
        )

        # 原有的 choices 内容应被保留
        assert "choices" in parsed

    def test_base_model_items_use_model_dump(self):
        """
        测试点4：当 original_response_obj 中的元素是 BaseModel 时，
        调用 .model_dump() 序列化，而非 str() 转换。
        """
        # 定义一个简单的 Pydantic BaseModel
        class MockToolContent(BaseModel):
            type: str
            text: str

        mcp_meta = {
            "namespaced_tool_name": "my-server/read_file",
            "name": "read_file",
            "arguments": {"path": "/tmp/test.txt"},
            "result": {},
        }
        slp = _make_standard_logging_payload(
            mcp_tool_call_metadata=mcp_meta,
            response=None,
        )
        kwargs = _make_kwargs(standard_logging_payload=slp)
        now = _now()

        # 使用 BaseModel 实例作为 original_response_obj 的元素
        content_item = MockToolContent(type="text", text="file content here")
        original_response_obj = [content_item]

        payload = get_logging_payload(kwargs, original_response_obj, now, now)

        parsed = json.loads(payload["response"])
        assert "mcp_tool_response" in parsed

        # 元素应该是通过 model_dump() 序列化的字典，而非字符串形式
        serialized_item = parsed["mcp_tool_response"][0]
        assert isinstance(serialized_item, dict), (
            "BaseModel 元素应通过 model_dump() 序列化为字典，而不是 str()"
        )
        assert serialized_item["type"] == "text"
        assert serialized_item["text"] == "file content here"

    def test_empty_list_does_not_trigger_mcp_handling(self):
        """
        额外测试：original_response_obj 是空列表时不触发 MCP 分支。
        条件要求 len(original_response_obj) > 0。
        """
        mcp_meta = {
            "namespaced_tool_name": "my-server/get_info",
            "name": "get_info",
            "arguments": {},
            "result": {},
        }
        slp = _make_standard_logging_payload(
            mcp_tool_call_metadata=mcp_meta,
            response=None,
        )
        kwargs = _make_kwargs(standard_logging_payload=slp)
        now = _now()

        # 空列表不满足 len > 0 条件
        payload = get_logging_payload(kwargs, [], now, now)

        # 不触发 MCP 分支，response 应为默认空值
        assert payload["response"] == "{}"

    def test_no_mcp_metadata_skips_mcp_handling(self):
        """
        额外测试：mcp_tool_call_metadata 为 None（即 mcp_namespaced_tool_name 为 None）时，
        不触发 MCP 特殊处理，即使 original_response_obj 是有效列表。
        """
        # 没有 MCP 元数据
        slp = _make_standard_logging_payload(
            mcp_tool_call_metadata=None,
            response=None,
        )
        kwargs = _make_kwargs(standard_logging_payload=slp)
        now = _now()

        original_response_obj = ["some result"]

        payload = get_logging_payload(kwargs, original_response_obj, now, now)

        # mcp_namespaced_tool_name 为 None，不触发 MCP 分支
        assert payload["response"] == "{}"
        assert payload.get("mcp_namespaced_tool_name") is None

    def test_mixed_type_list_serializes_correctly(self):
        """
        额外测试：original_response_obj 包含混合类型（BaseModel + 普通字符串）时，
        BaseModel 用 model_dump()，其他用 str() 转换。
        """
        class TextContent(BaseModel):
            type: str
            text: str

        mcp_meta = {
            "namespaced_tool_name": "mixed-server/tool",
            "name": "tool",
            "arguments": {},
            "result": {},
        }
        slp = _make_standard_logging_payload(
            mcp_tool_call_metadata=mcp_meta,
            response=None,
        )
        kwargs = _make_kwargs(standard_logging_payload=slp)
        now = _now()

        # 混合类型列表：第一个是 BaseModel，第二个是普通字符串
        original_response_obj = [
            TextContent(type="text", text="model content"),
            "plain string content",
        ]

        payload = get_logging_payload(kwargs, original_response_obj, now, now)

        parsed = json.loads(payload["response"])
        assert "mcp_tool_response" in parsed
        items = parsed["mcp_tool_response"]
        assert len(items) == 2

        # 第一个元素是 dict（来自 model_dump）
        assert isinstance(items[0], dict)
        assert items[0]["type"] == "text"

        # 第二个元素是字符串（来自 str()）
        assert items[1] == "plain string content"
