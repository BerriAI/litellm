from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.nlp_cloud.chat.transformation import NLPCloudConfig
from litellm.llms.nlp_cloud.common_utils import NLPCloudError
from litellm.types.utils import ModelResponse


@pytest.mark.parametrize(
    ("supplied", "stored"),
    [
        pytest.param({}, {}, id="nothing-supplied"),
        pytest.param(
            {"max_length": 0, "length_no_input": False, "end_sequence": "", "temperature": 0.0},
            {"max_length": 0, "length_no_input": False, "end_sequence": "", "temperature": 0.0},
            id="falsy-values-are-kept",
        ),
        pytest.param({"top_k": 40, "top_p": None, "num_beams": None}, {"top_k": 40}, id="none-is-skipped"),
    ],
)
def test_init_stores_each_supplied_value_except_none_as_class_level_config(
    supplied: dict[str, object], stored: dict[str, object]
) -> None:
    class ScopedNLPCloudConfig(NLPCloudConfig):
        pass

    ScopedNLPCloudConfig(**supplied)

    assert ScopedNLPCloudConfig.get_config() == stored


def test_init_stores_the_supplied_object_itself_rather_than_a_copy() -> None:
    class ScopedNLPCloudConfig(NLPCloudConfig):
        pass

    bad_words = ["darn", "heck"]

    ScopedNLPCloudConfig(bad_words=bad_words)

    assert ScopedNLPCloudConfig.get_config()["bad_words"] is bad_words


def test_init_called_again_with_none_keeps_the_value_stored_earlier() -> None:
    class ScopedNLPCloudConfig(NLPCloudConfig):
        pass

    ScopedNLPCloudConfig(top_k=40)
    ScopedNLPCloudConfig(top_k=None, top_p=0.5)

    assert ScopedNLPCloudConfig.get_config() == {"top_k": 40, "top_p": 0.5}


def _transform(body: object) -> ModelResponse:
    return NLPCloudConfig().transform_response(
        model="finetuned-llama-3-70b",
        raw_response=httpx.Response(200, json=body),
        model_response=ModelResponse(),
        logging_obj=Logging(
            model="finetuned-llama-3-70b",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            call_type="completion",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            litellm_call_id="nlp-cloud-transform",
            function_id="nlp-cloud-transform",
        ),
        request_data={"text": "Hi"},
        messages=[{"role": "user", "content": "Hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_response_returns_the_generated_text_and_reported_token_counts() -> None:
    response: Final = _transform({"generated_text": "Hello there", "nb_input_tokens": 3, "nb_generated_tokens": 2})

    assert response.choices[0].message.content == "Hello there"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (3, 2, 5)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"nb_input_tokens": 3, "nb_generated_tokens": 0}, id="missing-generated-text"),
        pytest.param(
            {"generated_text": 42, "nb_input_tokens": 3, "nb_generated_tokens": 1}, id="generated-text-without-a-length"
        ),
    ],
)
def test_transform_response_without_usable_generated_text_raises_an_nlp_cloud_error(body: dict[str, object]) -> None:
    with pytest.raises(NLPCloudError) as raised:
        _transform(body)

    assert raised.value.status_code == 200
