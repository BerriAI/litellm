"""
DashScope Realtime API E2E Tests

Tests Alibaba Cloud Model Studio (DashScope) Qwen-Omni-Realtime through LiteLLM's
realtime interface. Uses the base test class to ensure consistent behavior across
providers.

Requires DASHSCOPE_API_KEY; skipped otherwise. Pass api_base to exercise a
workspace-scoped host such as wss://{workspace_id}.cn-beijing.maas.aliyuncs.com.
"""

from tests.llm_translation.realtime.base_realtime_tests import BaseRealtimeTest


class TestDashScopeRealtime(BaseRealtimeTest):
    """
    E2E tests for DashScope's Qwen-Omni-Realtime WebSocket API.

    The API speaks OpenAI-compatible realtime events:
    - Endpoint: wss://dashscope.aliyuncs.com/api-ws/v1/realtime
    - Initial event: "session.created"
    """

    def get_model(self) -> str:
        return "dashscope/qwen3.5-omni-plus-realtime"

    def get_api_key_env_var(self) -> str:
        return "DASHSCOPE_API_KEY"

    def get_initial_event_type(self) -> tuple[str, ...]:
        return ("session.created",)
