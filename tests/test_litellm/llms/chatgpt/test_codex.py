import httpx
import pytest

from litellm.llms.chatgpt.codex import build_sideband_request, parse_call_response


@pytest.mark.parametrize("location", ["", "/v1/realtime/calls/foreign-id"])
def test_signaling_rejects_invalid_upstream_call_id(location):
    response = httpx.Response(201, headers={"Location": location},
        extensions={"chatgpt_realtime": {"model": "gpt-live-1-codex"}})
    with pytest.raises(ValueError, match="String should match pattern"):
        parse_call_response(response, "voice", "owner", 1000)


def test_signaling_preserves_selected_model_for_sideband():
    response = httpx.Response(201, headers={"Location": "/v1/realtime/calls/rtc_provider"},
        extensions={"chatgpt_realtime": {"model": "gpt-live-1-codex"}})
    call = parse_call_response(response, "voice", "owner", 1000)
    request = build_sideband_request(call)
    assert request["model"] == "chatgpt/gpt-live-1-codex"
    assert request["chatgpt_realtime_call_id"] == "rtc_provider"
    assert request["query_params"] == {"model": "gpt-live-1-codex"}
