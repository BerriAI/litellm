import pytest

from litellm.llms.vllm.passthrough.transformation import VLLMPassthroughConfig


@pytest.mark.parametrize(
    "request_data, expected",
    [({"stream": True}, True), ({"stream": 1}, True), ({"stream": False}, False), ({}, False)],
)
def test_is_streaming_request_reads_the_stream_flag(request_data: dict[str, object], expected: bool) -> None:
    assert VLLMPassthroughConfig().is_streaming_request("v1/chat/completions", request_data) is expected
