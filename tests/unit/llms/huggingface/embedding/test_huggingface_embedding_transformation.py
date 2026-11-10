from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.huggingface.embedding.transformation import HuggingFaceEmbeddingConfig
from litellm.types.utils import ModelResponse


def _transform(
    body: object, *, task: str | None = None, optional_params: dict[str, object] | None = None
) -> ModelResponse:
    return HuggingFaceEmbeddingConfig().transform_response(
        model="tgi-endpoint",
        raw_response=httpx.Response(200, json=body),
        model_response=ModelResponse(),
        logging_obj=Logging(
            model="tgi-endpoint",
            messages=[{"role": "user", "content": "Capital of France?"}],
            stream=False,
            call_type="completion",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            litellm_call_id="huggingface-transform",
            function_id="huggingface-transform",
        ),
        request_data={"inputs": "Capital of France?"},
        messages=[{"role": "user", "content": "Capital of France?"}],
        optional_params={} if optional_params is None else optional_params,
        litellm_params={} if task is None else {"task": task},
        encoding=None,
    )


def test_tgi_response_strips_chat_tokens_and_sums_the_token_logprobs() -> None:
    response: Final = _transform(
        [
            {
                "generated_text": "<s>Paris</s>",
                "details": {
                    "finish_reason": "length",
                    "tokens": [{"logprob": -0.5}, {"logprob": None}, {"logprob": -0.25}],
                },
            }
        ]
    )

    assert response.choices[0].message.content == "Paris"
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message._logprob == -0.75


def test_tgi_response_with_best_of_adds_one_choice_per_extra_sequence() -> None:
    response: Final = _transform(
        [
            {
                "generated_text": "Paris",
                "details": {
                    "finish_reason": "length",
                    "tokens": [{"logprob": -0.5}],
                    "best_of_sequences": [
                        {
                            "generated_text": "<s>Lyon",
                            "finish_reason": "length",
                            "tokens": [{"logprob": -1.0}, {"logprob": None}, {"logprob": -0.5}],
                        },
                        {"generated_text": "", "finish_reason": "stop", "tokens": []},
                    ],
                },
            }
        ],
        optional_params={"best_of": 2},
    )

    assert [(choice.index, choice.message.content, choice.finish_reason) for choice in response.choices] == [
        (0, "Paris", "length"),
        (1, "Lyon", "length"),
        (2, None, "stop"),
    ]
    assert response.choices[1].message.logprobs == -1.5


def test_text_generation_task_reads_the_first_generated_text() -> None:
    response: Final = _transform([{"generated_text": "<|assistant|>Paris"}], task="text-generation")

    assert response.choices[0].message.content == "Paris"


@pytest.mark.parametrize(
    ("body", "error"),
    [
        pytest.param([{"generated_text": 42}], TypeError, id="generated-text-without-a-length"),
        pytest.param([{"generated_text": ["Paris"]}], AttributeError, id="generated-text-that-is-not-a-string"),
        pytest.param(
            [{"generated_text": "Paris", "details": {"finish_reason": "length", "tokens": [{"logprob": "-0.5"}]}}],
            TypeError,
            id="logprob-that-is-not-a-number",
        ),
    ],
)
def test_tgi_response_with_malformed_fields_raises_the_python_error_for_that_field(
    body: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _transform(body)
