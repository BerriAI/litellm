import json
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_ENDPOINT
from litellm.llms.alibaba_token_plan.image_generation.transformation import AlibabaTokenPlanImageGenerationConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import ImageResponse


def _logging_obj() -> Logging:
    return Logging(
        model="qwen-image-3.0-pro",
        messages=[],
        stream=False,
        call_type="image_generation",
        start_time=datetime(2026, 10, 2, tzinfo=timezone.utc),
        litellm_call_id="test-call",
        function_id="test-function",
    )


def _image_response(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    assert str(request.url) == f"https://gateway.example/token-plan/{IMAGE_ENDPOINT}"
    assert request.headers["Authorization"] == "Bearer explicit-key"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {
        "model": "qwen-image-3.0-pro",
        "input": {"messages": [{"role": "user", "content": [{"text": "Draw a lighthouse"}]}]},
        "parameters": {"size": "1536*1024", "n": 2, "prompt_extend_mode": "agent", "enable_thinking": False},
    }
    return httpx.Response(
        200,
        json={
            "output": {
                "choices": [
                    {"message": {"content": [{"text": "Generated"}, {"image": "https://images.example/one.png"}]}},
                    {"message": {"content": [{"image": "https://images.example/two.png"}]}},
                ]
            }
        },
    )


@pytest.mark.parametrize("path", [IMAGE_ENDPOINT, "compatible-mode/v1", "apps/anthropic"])
def test_sync_image_request_and_response_through_public_api(path: str) -> None:
    with httpx.Client(transport=httpx.MockTransport(_image_response)) as client:
        response: Final = litellm.image_generation(
            model="alibaba_token_plan/qwen-image-3.0-pro",
            prompt="Draw a lighthouse",
            n=2,
            size="1536x1024",
            response_format="url",
            prompt_extend_mode="agent",
            enable_thinking=False,
            api_base=f"https://gateway.example/token-plan/{path}",
            timeout=10,
            max_retries=0,
            api_key="explicit-key",
            client=HTTPHandler(client=client),
        )
    assert isinstance(response, ImageResponse)
    assert response.data is not None
    assert [item.url for item in response.data] == ["https://images.example/one.png", "https://images.example/two.png"]


@pytest.mark.asyncio
async def test_async_image_request_and_response_through_public_api() -> None:
    with httpx.Client(transport=httpx.MockTransport(_image_response)) as client:
        response: Final = await litellm.aimage_generation(
            model="alibaba_token_plan/qwen-image-3.0-pro",
            prompt="Draw a lighthouse",
            n=2,
            size="1536x1024",
            prompt_extend_mode="agent",
            enable_thinking=False,
            api_base=f"https://gateway.example/token-plan/{IMAGE_ENDPOINT}",
            timeout=10,
            max_retries=0,
            api_key="explicit-key",
            client=HTTPHandler(client=client),
        )
    assert response.data is not None
    assert [item.url for item in response.data] == ["https://images.example/one.png", "https://images.example/two.png"]


@pytest.mark.parametrize("model", ["qwen-image-3.0-pro", "wan2.7-image", "wan2.7-image-pro"])
def test_provider_parameters_and_defaults(model: str) -> None:
    config: Final = AlibabaTokenPlanImageGenerationConfig()
    params: Final = config.map_openai_params({"n": 2, "size": "2048x1024"}, {"watermark": False}, model, False)
    assert config.transform_image_generation_request(model, "Draw a lighthouse", params, {}, {}) == {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{"text": "Draw a lighthouse"}]}]},
        "parameters": {"size": "2048*1024", "n": 2, "watermark": False},
    }
    assert config.transform_image_generation_request(model, "Draw a lighthouse", {}, {}, {})["parameters"] == {
        "size": "1024*1024"
    }


def test_base64_response_format_is_not_silently_ignored() -> None:
    config: Final = AlibabaTokenPlanImageGenerationConfig()
    with pytest.raises(litellm.UnsupportedParamsError, match="response_format='url'"):
        config.map_openai_params({"response_format": "b64_json"}, {}, "qwen-image-3.0-pro", False)
    assert config.map_openai_params({"response_format": "b64_json"}, {}, "qwen-image-3.0-pro", True) == {}


@pytest.mark.parametrize(
    ("status_code", "body", "expected_status", "message"),
    [
        (401, '{"message":"Invalid API key"}', 401, "Invalid API key"),
        (429, '{"message":"Quota exceeded"}', 429, "Quota exceeded"),
        (200, '{"code":"InvalidParameter","message":"Size not supported"}', 502, "Size not supported"),
        (200, "not json", 502, "Invalid Alibaba Token Plan image response"),
        (200, '{"output":{"choices":[]}}', 502, "returned no images"),
        (200, "{}", 502, "returned no image output"),
    ],
)
def test_image_errors_are_not_returned_as_success(
    status_code: int, body: str, expected_status: int, message: str
) -> None:
    with pytest.raises(BaseLLMException, match=message) as error:
        AlibabaTokenPlanImageGenerationConfig().transform_image_generation_response(
            "qwen-image-3.0-pro",
            httpx.Response(status_code, text=body),
            ImageResponse(),
            _logging_obj(),
            {},
            {},
            {},
            None,
        )
    assert error.value.status_code == expected_status


def test_malformed_image_output_is_not_exposed() -> None:
    secret: Final = "sensitive-presigned-value"
    with pytest.raises(BaseLLMException, match="Invalid Alibaba Token Plan image response") as error:
        AlibabaTokenPlanImageGenerationConfig().transform_image_generation_response(
            "qwen-image-3.0-pro",
            httpx.Response(
                200,
                json={"output": {"choices": [{"message": {"content": f"bad?Signature={secret}"}}]}},
            ),
            ImageResponse(),
            _logging_obj(),
            {},
            {},
            {},
            None,
        )
    assert secret not in str(error.value)
