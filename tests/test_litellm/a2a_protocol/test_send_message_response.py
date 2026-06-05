"""Tests for LiteLLMSendMessageResponse JSON-RPC normalization."""

from litellm.types.agents import LiteLLMSendMessageResponse


def test_from_dict_backfills_id_on_agent_error_response():
    agent_error = {
        "jsonrpc": "2.0",
        "error": {"code": -32054, "message": "Session not found"},
    }

    response = LiteLLMSendMessageResponse.from_dict(
        agent_error, request_id="r1"
    )

    assert response.id == "r1"
    assert response.error == {"code": -32054, "message": "Session not found"}
    assert response.result is None


def test_from_dict_preserves_existing_id():
    payload = {
        "id": "upstream-id",
        "jsonrpc": "2.0",
        "error": {"code": -32001, "message": "Task not found"},
    }

    response = LiteLLMSendMessageResponse.from_dict(
        payload, request_id="r1"
    )

    assert response.id == "upstream-id"


def test_from_dict_without_request_id_still_requires_id():
    try:
        LiteLLMSendMessageResponse.from_dict(
            {"jsonrpc": "2.0", "error": {"code": -32054, "message": "x"}}
        )
    except Exception as exc:
        assert "id" in str(exc).lower()
    else:
        raise AssertionError("expected validation error when id and request_id missing")
