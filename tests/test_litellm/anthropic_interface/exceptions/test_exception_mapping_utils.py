import json
from typing import Final

import pytest

from litellm.anthropic_interface.exceptions import AnthropicErrorSseFrame, anthropic_error_sse_frame


class TestAnthropicErrorSseFrame:
    @pytest.mark.parametrize(
        ("status_code", "expected_error_type"),
        [(429, "rate_limit_error"), (503, "api_error"), (400, "invalid_request_error")],
    )
    def test_the_frame_is_one_error_event_carrying_the_anthropic_envelope(
        self, status_code: int, expected_error_type: str
    ) -> None:
        frame: Final = anthropic_error_sse_frame(status_code=status_code, raw_message="upstream unavailable")

        event_line, data_line, first_blank, second_blank = frame.split("\n")
        assert event_line == "event: error"
        assert (first_blank, second_blank) == ("", "")
        assert json.loads(data_line.removeprefix("data: ")) == {
            "type": "error",
            "error": {"type": expected_error_type, "message": "upstream unavailable"},
        }

    def test_the_frame_remembers_the_status_and_body_it_was_built_from(self) -> None:
        frame: Final = anthropic_error_sse_frame(status_code=503, raw_message="upstream unavailable")

        assert isinstance(frame, AnthropicErrorSseFrame)
        assert frame.status_code == 503
        data_line: Final = frame.split("\n")[1]
        assert data_line == f"data: {json.dumps(frame.json_body(call_id=None))}"

    def test_the_json_body_names_the_call_only_when_asked(self) -> None:
        frame: Final = anthropic_error_sse_frame(status_code=503, raw_message="upstream unavailable")

        assert frame.json_body(call_id="call-1") == {
            "type": "error",
            "error": {"type": "api_error", "message": "upstream unavailable", "litellm_call_id": "call-1"},
        }
        assert frame.json_body(call_id=None) == {
            "type": "error",
            "error": {"type": "api_error", "message": "upstream unavailable"},
        }
