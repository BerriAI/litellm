from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.replicate.chat.transformation import ReplicateConfig
from litellm.types.utils import ModelResponse


def _transform(raw_response: httpx.Response) -> ModelResponse:
    return ReplicateConfig().transform_response(
        model="acme/echo-model",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=Mock(),
        request_data={"input": {"prompt": "Hello"}},
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


@pytest.mark.parametrize(
    ("output", "content"),
    [
        (["Hello", ", ", "world"], "Hello, world"),
        ("Hello", "Hello"),
        ([], " "),
        ("", " "),
        ([""], " "),
    ],
)
def test_transform_response_joins_the_prediction_output_into_the_message_content(output: object, content: str):
    response = _transform(httpx.Response(200, json={"status": "succeeded", "output": output}))

    assert response.choices[0].message.content == content
    assert response.model == "replicate/acme/echo-model"
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


def test_transform_response_uses_a_blank_message_when_the_prediction_has_no_output():
    response = _transform(httpx.Response(200, json={"status": "succeeded"}))

    assert response.choices[0].message.content == " "


@pytest.mark.parametrize("output", [None, 7, True, ["secret-output", 7], ["secret-output", None], [["secret-output"]]])
def test_transform_response_rejects_an_output_that_is_not_made_of_strings_without_echoing_it(output: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(httpx.Response(200, json={"status": "succeeded", "output": output}))

    assert "secret-output" not in str(exc_info.value)
