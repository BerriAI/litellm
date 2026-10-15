from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer

_HEADERS: Final = TypeAdapter(Mapping[str, object])


def _assert_marked_native(response: object) -> None:
    headers: Final = _HEADERS.validate_python(get_hidden_params_dict(response).get("additional_headers"))
    assert headers.get("x-litellm-rust") == "true"


def assert_served_natively(server: RecordingServer, *responses: object, count: int = 1) -> None:
    assert len(server.requests) == (len(responses) or count)
    if not responses:
        assert not any(request.headers.get("user-agent", "").startswith("python-httpx") for request in server.requests)
    for response in responses:
        _assert_marked_native(response)


def assert_nested_call_served_natively(server: RecordingServer, recorder: RecordingLogger) -> None:
    assert len(server.requests) == 1
    for event in recorder.wait_for("async_log_success_event"):
        _assert_marked_native(event.response)
