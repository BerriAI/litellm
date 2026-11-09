import json
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import respx

import litellm
from litellm.llms.oobabooga.chat import oobabooga

MOCK_COMPLETION_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


def _mock_post_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "ok"
    mock_response.json.return_value = MOCK_COMPLETION_RESPONSE
    return mock_response


def test_model_name_with_https_substring_uses_api_base():
    api_base = "https://legit.example"

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_post.return_value = _mock_post_response()

        litellm.completion(
            model="oobabooga/my-https-model",
            messages=[{"role": "user", "content": "hello"}],
            api_base=api_base,
        )

    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert called_url == f"{api_base}/v1/chat/completions"


def test_url_valued_model_still_targets_that_url():
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_post.return_value = _mock_post_response()

        litellm.completion(
            model="oobabooga/https://sdk-user.example",
            messages=[{"role": "user", "content": "hello"}],
        )

    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert called_url == "https://sdk-user.example/v1/chat/completions"


def test_stream_keeps_unicode_line_separators_inside_a_data_line(respx_mock: respx.MockRouter) -> None:
    text: Final = "before\u2028between\u2029after"
    chunk: Final = {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    data_line: Final = f"data: {json.dumps(chunk, ensure_ascii=False)}"
    respx_mock.post("https://ooba.example/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=f"{data_line}\n\ndata: [DONE]\n\n".encode())
    )

    lines: Final = oobabooga.completion(
        model="my-model",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://ooba.example",
        model_response=litellm.ModelResponse(),
        print_verbose=print,
        encoding=None,
        api_key=None,
        logging_obj=MagicMock(),
        optional_params={"stream": True},
        litellm_params={},
    )

    assert [line for line in lines if line] == [data_line, "data: [DONE]"]
