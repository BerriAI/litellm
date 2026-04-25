import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
from litellm.types.utils import Usage, ServerToolUse


def test_usage_coercion_from_dict():
    """
    Verify that Usage correctly coerces server_tool_use from a dict to a ServerToolUse object.
    """
    usage = Usage(
        prompt_tokens=10,
        completion_tokens=20,
        server_tool_use={"web_search_requests": 5},
    )

    assert isinstance(usage.server_tool_use, ServerToolUse)
    assert usage.server_tool_use.web_search_requests == 5


def test_chunk_processor_coerces_dict_server_tool_use_from_stream_usage():
    """
    Verify streaming usage assembly handles server_tool_use from a dict.

    Anthropic streaming chunks can arrive with usage as plain dicts. This
    exercises the exact dict path that previously raised AttributeError when
    server_tool_use was accessed with attribute syntax.
    """
    chunks = [
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 0,
                "total_tokens": 10,
                "server_tool_use": {"web_search_requests": 5},
            }
        },
        {
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 20,
                "total_tokens": 20,
            }
        },
    ]

    usage = ChunkProcessor(chunks=chunks).calculate_usage(
        chunks=chunks,
        model="claude-sonnet-4-5-20250929",
        completion_output="hello",
    )

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30
    assert isinstance(usage.server_tool_use, ServerToolUse)
    assert usage.server_tool_use.web_search_requests == 5


if __name__ == "__main__":
    test_usage_coercion_from_dict()
    test_chunk_processor_coerces_dict_server_tool_use_from_stream_usage()
