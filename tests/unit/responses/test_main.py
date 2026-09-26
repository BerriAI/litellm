from typing import Final
from unittest.mock import MagicMock

import httpx
import openai
import pytest

import litellm


def test_responses_bridged_to_chat_still_rejects_an_invalid_stream_chunk_size() -> None:
    send: Final = MagicMock(return_value=httpx.Response(200))
    client: Final = openai.OpenAI(api_key="fake-key", http_client=httpx.Client(transport=httpx.MockTransport(send)))

    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.responses(
            model="openai/gpt-4.1-mini",
            input="hi",
            use_chat_completions_api=True,
            stream_chunk_size="sixty-four",
            client=client,
            num_retries=0,
        )

    assert exc_info.value.param == "stream_chunk_size"
    send.assert_not_called()
