import os
import sys
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.bedrock.chat.converse_handler import make_sync_call
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def _stream_completion_with_spied_iter_bytes(model: str, **kwargs) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        **kwargs,
    )
    return mock_response.iter_bytes


def test_make_sync_call_does_not_rechunk_stream_by_default():
    """Re-chunking the event stream into fixed 1024-byte blocks holds small
    early events in httpx's ByteChunker until 1024 bytes accumulate, delaying
    time-to-first-chunk by the whole generation when Bedrock trickles bytes
    (e.g. buffered tool-use streams)."""
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = MagicMock(return_value=response)

    make_sync_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
    )

    response.iter_bytes.assert_called_once_with(chunk_size=None)


def test_make_sync_call_honors_explicit_stream_chunk_size():
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = MagicMock(return_value=response)

    make_sync_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
        stream_chunk_size=2048,
    )

    response.iter_bytes.assert_called_once_with(chunk_size=2048)


def test_completion_plumbs_stream_chunk_size_through_converse():
    iter_bytes_spy = _stream_completion_with_spied_iter_bytes(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    iter_bytes_spy.assert_called_once_with(chunk_size=None)

    iter_bytes_spy = _stream_completion_with_spied_iter_bytes(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        stream_chunk_size=2048,
    )
    iter_bytes_spy.assert_called_once_with(chunk_size=2048)
