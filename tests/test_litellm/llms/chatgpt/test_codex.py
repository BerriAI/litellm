import httpx
import pytest

from litellm.llms.chatgpt.codex import CodexRealtimeCall, build_sideband_request, parse_call_response


@pytest.mark.parametrize("location", ["", "/v1/realtime/calls/foreign-id"])
def test_signaling_rejects_invalid_upstream_call_id(location):
    response = httpx.Response(201, headers={"Location": location},
        extensions={"chatgpt_realtime": {"model": "gpt-live-1-codex"}})
    with pytest.raises(ValueError, match="String should match pattern"):
        parse_call_response(response, "voice", "owner", 1000)


@pytest.mark.parametrize("extra_query", [None, {"gateway_token": "opaque +/& value"}])
def test_signaling_preserves_selected_model_for_sideband(extra_query):
    response = httpx.Response(
        201,
        headers={"Location": "/v1/realtime/calls/rtc_provider"},
        extensions={
            "chatgpt_realtime": {
                "model": "gpt-live-1-codex",
                "api_base": "https://voice.example/codex",
                "extra_headers": {"x-gateway-route": "voice"},
                **({"extra_query": extra_query} if extra_query is not None else {}),
            }
        },
    )
    call = parse_call_response(response, "voice", "owner", 1000)
    request = build_sideband_request(CodexRealtimeCall.model_validate_json(call.model_dump_json(exclude_none=True)))
    assert request["api_base"] == "https://voice.example/codex"
    assert request["model"] == "chatgpt/gpt-live-1-codex"
    assert request["chatgpt_realtime_call_id"] == "rtc_provider"
    assert request["query_params"] == {"model": "gpt-live-1-codex"}
    assert request["extra_headers"] == {"x-gateway-route": "voice"}
    assert request["extra_query"] == extra_query
