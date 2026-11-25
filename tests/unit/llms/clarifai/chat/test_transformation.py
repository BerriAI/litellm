from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.clarifai.chat.transformation import ClarifaiConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.utils import ModelResponse


def _transform(raw_response: httpx.Response) -> ModelResponse:
    return ClarifaiConfig().transform_response(
        model="user.app.model",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=Mock(),
        request_data={},
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_response_builds_the_model_response_from_the_body():
    response = _transform(
        httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "created": 1700000000,
                "model": "upstream-model",
                "system_fingerprint": "fp_1",
                "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "Hi"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "vendor_field": {"kept": True},
            },
        )
    )

    assert response.id == "chatcmpl-1"
    assert response.created == 1700000000
    assert response.model == "clarifai/user.app.model"
    assert response.system_fingerprint == "fp_1"
    assert [(choice.finish_reason, choice.message.content) for choice in response.choices] == [("length", "Hi")]
    assert response.usage.model_dump()["total_tokens"] == 5
    assert response.vendor_field == {"kept": True}


def test_transform_response_keeps_a_missing_model_unset():
    response = _transform(httpx.Response(200, json={"choices": [{"message": {"content": "Hi"}}]}))

    assert response.model is None
    assert response.choices[0].message.content == "Hi"


@pytest.mark.parametrize("body", [b'["prompt-text"]', b'"prompt-text"', b"7", b"null"])
def test_transform_response_rejects_a_body_that_is_not_an_object_without_echoing_it(body: bytes):
    with pytest.raises(ValidationError) as exc_info:
        _transform(httpx.Response(200, content=body))

    assert "prompt-text" not in str(exc_info.value)


def test_transform_response_reports_an_unparseable_body_as_an_openai_error():
    with pytest.raises(OpenAIError, match="Failed to parse Clarifai response") as exc_info:
        _transform(httpx.Response(502, content=b"<html>bad gateway</html>"))

    assert exc_info.value.status_code == 502
