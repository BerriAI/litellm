import json
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

import litellm
from litellm.integrations.prompt_layer import PromptLayerLogger
from litellm.llms.custom_httpx.http_handler import HTTPHandler

CALL_TIME: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
REQUEST_KWARGS: Final = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}],
    "optional_params": {"temperature": 0.2},
    "litellm_params": {"metadata": {"pl_tags": ["prod"], "team": "search"}},
}


def _log_to_promptlayer(
    monkeypatch: pytest.MonkeyPatch, track_request_body: object
) -> tuple[list[tuple[str, object]], list[str]]:
    posted: Final[list[tuple[str, object]]] = []
    logged: Final[list[str]] = []

    def promptlayer(request: httpx.Request) -> httpx.Response:
        posted.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/rest/track-request":
            return httpx.Response(200, json=track_request_body)
        return httpx.Response(200, json={"success": True})

    monkeypatch.setenv("PROMPTLAYER_API_KEY", "pl-key")
    monkeypatch.setattr(
        litellm, "module_level_client", HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(promptlayer)))
    )
    PromptLayerLogger().log_event(
        kwargs=REQUEST_KWARGS,
        response_obj={"id": "chatcmpl-1"},
        start_time=CALL_TIME,
        end_time=CALL_TIME,
        print_verbose=logged.append,
    )
    return posted, logged


def test_prompt_layer_tracks_metadata_under_the_request_id_it_got_back(monkeypatch: pytest.MonkeyPatch) -> None:
    posted, _ = _log_to_promptlayer(monkeypatch, {"success": True, "request_id": 42})

    assert [path for path, _ in posted] == ["/rest/track-request", "/rest/track-metadata"]
    track_request: Final = posted[0][1]
    assert isinstance(track_request, dict)
    assert track_request["tags"] == ["prod"]
    assert track_request["kwargs"] == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
    }
    assert posted[1][1] == {"request_id": 42, "api_key": "pl-key", "metadata": {"team": "search"}}


@pytest.mark.parametrize(
    ("track_request_body", "logged_error"),
    [
        ({"success": False, "request_id": 42}, "Promptlayer did not successfully log the response!"),
        (["success", "request_id"], "'list' object has no attribute 'get'"),
    ],
)
def test_prompt_layer_skips_metadata_and_logs_the_error_when_tracking_did_not_succeed(
    monkeypatch: pytest.MonkeyPatch, track_request_body: object, logged_error: str
) -> None:
    posted, logged = _log_to_promptlayer(monkeypatch, track_request_body)

    assert [path for path, _ in posted] == ["/rest/track-request"]
    assert logged_error in logged[-1]
