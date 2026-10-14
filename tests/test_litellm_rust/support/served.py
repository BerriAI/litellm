from typing import Final

from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from tests.test_litellm_rust.support.recording_server import RecordingServer


def assert_served_natively(server: RecordingServer, response: object) -> None:
    assert len(server.requests) == 1
    additional_headers: Final = get_hidden_params_dict(response).get("additional_headers")
    assert isinstance(additional_headers, dict)
    assert additional_headers.get("x-litellm-rust") == "true"
