from datetime import datetime

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    native_success,
    series,
    standard_payload,
)


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


@pytest.fixture
def logger():
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    return PrometheusLogger()


CALLS = "litellm_mcp_tool_calls_total"
SPEND = "litellm_mcp_tool_call_spend_metric_total"


def _make_payload(
    *,
    mcp_tool_name: str = "get_weather",
    mcp_server_name: str = "weather-server",
    response_cost: float = 0.005,
) -> dict:
    return standard_payload(
        metadata={
            "user_api_key_hash": "sk-hash-123",
            "user_api_key_alias": "test-key",
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "Test Team",
            "user_api_key_user_id": "user-1",
            "user_api_key_user_email": None,
            "user_api_key_org_id": None,
            "user_api_key_org_alias": None,
            "mcp_tool_call_metadata": {
                "name": mcp_tool_name,
                "mcp_server_name": mcp_server_name,
                "namespaced_tool_name": f"{mcp_server_name}/{mcp_tool_name}",
                "arguments": {"city": "SF"},
                "result": {"temp": 72},
            },
        },
        model="gpt-4o",
        model_group="gpt-4o",
        model_id="model-123",
        api_base="https://api.openai.com",
        custom_llm_provider="openai",
        response_cost=response_cost,
        completion_tokens=50,
        prompt_tokens=100,
        total_tokens=150,
        request_tags=[],
        stream=False,
    )


async def _log(logger: PrometheusLogger, payload: dict, end_user: str = "end-user-1") -> None:
    now = datetime.now()
    kwargs = {
        "model": "gpt-4o",
        "litellm_params": {"metadata": {"user_api_key_end_user_id": end_user}},
        "start_time": now,
        "end_time": now,
        "standard_logging_object": payload,
    }
    await native_success(logger, kwargs, None)


class TestIncrementMCPToolCallMetrics:
    @pytest.mark.asyncio
    async def test_increments_calls_counter_when_mcp_metadata_present(self, logger):
        payload = _make_payload()

        await _log(logger, payload)

        assert len(series(CALLS)) == 1
        assert [value for _, value in series(CALLS)] == [1.0]

    @pytest.mark.asyncio
    async def test_increments_spend_counter_when_cost_positive(self, logger):
        payload = _make_payload(response_cost=0.01)

        await _log(logger, payload)

        assert len(series(SPEND)) == 1
        assert [value for _, value in series(SPEND)] == [0.01]

    @pytest.mark.asyncio
    async def test_skips_spend_counter_when_cost_zero(self, logger):
        payload = _make_payload(response_cost=0.0)

        await _log(logger, payload)

        assert len(series(CALLS)) == 1
        assert series(SPEND) == []

    @pytest.mark.asyncio
    async def test_noop_when_no_mcp_metadata(self, logger):
        payload = _make_payload()
        payload["metadata"]["mcp_tool_call_metadata"] = None

        await _log(logger, payload)

        assert series(CALLS) == []
        assert series(SPEND) == []

    @pytest.mark.asyncio
    async def test_label_values_carry_tool_and_server_name(self, logger):
        payload = _make_payload(
            mcp_tool_name="search_docs",
            mcp_server_name="docs-mcp",
        )

        await _log(logger, payload)

        [(labels_passed, _)] = series(CALLS)
        assert labels_passed["mcp_tool_name"] == "search_docs"
        assert labels_passed["mcp_server_name"] == "docs-mcp"

    @pytest.mark.asyncio
    async def test_label_values_carry_team_and_key_from_parent(self, logger):
        payload = _make_payload()
        payload["metadata"].update(
            {
                "user_api_key_hash": "sk-parent-key",
                "user_api_key_alias": "parent-alias",
                "user_api_key_team_id": "parent-team",
                "user_api_key_team_alias": "Parent Team",
                "user_api_key_user_id": "parent-user",
            }
        )

        await _log(logger, payload, end_user="parent-end-user")

        [(labels_passed, _)] = series(CALLS)
        assert labels_passed["hashed_api_key"] == "sk-parent-key"
        assert labels_passed["team"] == "parent-team"
        assert labels_passed["team_alias"] == "Parent Team"
        assert labels_passed["user"] == "parent-user"

    @pytest.mark.asyncio
    async def test_handles_missing_server_name_gracefully(self, logger):
        payload = _make_payload(response_cost=0.0)
        payload["metadata"]["mcp_tool_call_metadata"] = {
            "name": "standalone_tool",
            "arguments": {},
            "result": {},
        }

        await _log(logger, payload)

        [(labels_passed, _)] = series(CALLS)
        assert labels_passed["mcp_tool_name"] == "standalone_tool"
        assert labels_passed["mcp_server_name"] == "None"
