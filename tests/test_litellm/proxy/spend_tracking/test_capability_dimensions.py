"""Tests for capability-dimension attribution in spend logs (S2-10 / S6-01)."""

from datetime import datetime
from unittest.mock import MagicMock, patch


def _build_kwargs(metadata_extra=None):
    """Minimal kwargs that get_logging_payload tolerates without blowing up."""
    md = {
        "user_api_key_team_id": "t-1",
        "user_api_key_user_id": "u-1",
    }
    if metadata_extra:
        md.update(metadata_extra)
    return {
        "litellm_call_id": "req-1",
        "call_type": "completion",
        "model": "gpt-4o",
        "litellm_params": {
            "api_base": "http://example/",
            "metadata": md,
        },
        "response_cost": 0.001,
        "cache_hit": False,
        "standard_logging_object": {
            "metadata": md,
            "request_tags": [],
            "model_map_information": {"model_map_value": {}},
            "total_tokens": 10,
            "prompt_tokens": 4,
            "completion_tokens": 6,
        },
    }


def _call(metadata=None):
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    kwargs = _build_kwargs(metadata)
    start = datetime(2026, 5, 19, 6, 0, 0)
    end = datetime(2026, 5, 19, 6, 0, 1)
    return get_logging_payload(
        kwargs=kwargs, response_obj={}, start_time=start, end_time=end
    )


def test_plain_chat_call_attributes_as_model():
    payload = _call()
    assert payload["entity_type"] == "model"
    assert payload["entity_id"] == "gpt-4o"
    assert payload["skill_ids"] == []


def test_skill_call_attributes_as_skill():
    payload = _call(metadata={"skill_ids": ["fact-check"]})
    assert payload["entity_type"] == "skill"
    assert payload["entity_id"] == "fact-check"
    assert payload["skill_ids"] == ["fact-check"]


def test_agent_call_attributes_as_agent():
    """agent_id present → entity_type = 'agent'."""
    payload = _call(metadata={"agent_id": "agt-77"})
    assert payload["entity_type"] == "agent"
    assert payload["entity_id"] == "agt-77"
    # agent_id beats skill_ids if both happen to be set
    payload2 = _call(metadata={"agent_id": "agt-77", "skill_ids": ["s"]})
    assert payload2["entity_type"] == "agent"


def test_mcp_call_attributes_as_mcp():
    payload = _call(
        metadata={"mcp_tool_call_metadata": {"namespaced_tool_name": "github:search"}}
    )
    assert payload["entity_type"] == "mcp"
    assert payload["entity_id"] == "github:search"


def test_app_id_picked_up_from_metadata():
    payload = _call(metadata={"user_api_key_app_id": "xct-chat"})
    assert payload["app_id"] == "xct-chat"
