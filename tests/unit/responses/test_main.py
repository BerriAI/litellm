import pytest

import litellm


def test_responses_bridged_to_chat_still_rejects_an_invalid_stream_chunk_size() -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.responses(
            model="openai/gpt-4.1-mini",
            input="hi",
            use_chat_completions_api=True,
            stream_chunk_size="sixty-four",
            api_key="fake-key",
            api_base="http://127.0.0.1:9/v1",
            num_retries=0,
        )

    assert exc_info.value.param == "stream_chunk_size"
