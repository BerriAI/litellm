from datetime import datetime
from typing import Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.amazon_nova.chat.transformation import AmazonNovaChatConfig
from litellm.types.utils import ModelResponse

REQUESTED_MODEL: Final = "nova-micro-v1"
MESSAGES: Final = [{"role": "user", "content": "hi"}]


def test_map_openai_params_sends_max_completion_tokens_as_max_tokens_and_drops_unsupported_params() -> None:
    optional_params = AmazonNovaChatConfig().map_openai_params(
        non_default_params={
            "max_completion_tokens": 64,
            "temperature": 0.2,
            "reasoning_effort": "low",
            "frequency_penalty": 0.5,
        },
        optional_params={},
        model=REQUESTED_MODEL,
        drop_params=False,
    )

    assert optional_params == {"max_tokens": 64, "temperature": 0.2, "reasoning_effort": "low"}


def test_transform_response_reports_the_requested_model_under_the_amazon_nova_prefix() -> None:
    upstream_body = {
        "id": "chatcmpl-nova",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "upstream-reported-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hello from nova"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }

    response = AmazonNovaChatConfig().transform_response(
        model=REQUESTED_MODEL,
        raw_response=httpx.Response(200, json=upstream_body),
        model_response=ModelResponse(),
        logging_obj=Logging(
            model=REQUESTED_MODEL,
            messages=MESSAGES,
            stream=False,
            call_type="completion",
            start_time=datetime(2026, 1, 1),
            litellm_call_id="nova-call",
            function_id="nova-function",
        ),
        request_data={},
        messages=MESSAGES,
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.model == "amazon-nova/nova-micro-v1"
    assert response.choices[0].message.content == "hello from nova"
    assert response.usage.total_tokens == 7
