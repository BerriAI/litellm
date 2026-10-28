import httpx
import pytest

import litellm
from litellm.llms.fal_ai.chat.transformation import FalAIChatConfig, FalAIError
from litellm.types.utils import LlmProviders, ModelResponse
from litellm.utils import ProviderConfigManager

MODEL = "fal-ai/moondream3-preview/query"


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _messages(*content):
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": text} for text in content[:1]]
            + [{"type": "image_url", "image_url": {"url": c}} for c in content[1:]],
        }
    ]


def test_provider_config_manager_resolves_fal_ai_chat_config():
    config = ProviderConfigManager.get_provider_chat_config(model=MODEL, provider=LlmProviders.FAL_AI)
    assert isinstance(config, FalAIChatConfig)


def test_get_complete_url_targets_fal_endpoint():
    assert (
        FalAIChatConfig().get_complete_url(
            api_base=None, api_key=None, model=MODEL, optional_params={}, litellm_params={}
        )
        == "https://fal.run/fal-ai/moondream3-preview/query"
    )


def test_get_complete_url_strips_fal_ai_model_prefix():
    assert (
        FalAIChatConfig().get_complete_url(
            api_base=None, api_key=None, model=f"fal_ai/{MODEL}", optional_params={}, litellm_params={}
        )
        == "https://fal.run/fal-ai/moondream3-preview/query"
    )


def test_validate_environment_uses_fal_key_scheme():
    headers = FalAIChatConfig().validate_environment(
        headers={}, model=MODEL, messages=[], optional_params={}, litellm_params={}, api_key="secret"
    )
    assert headers["Authorization"] == "Key secret"


def test_transform_request_joins_text_parts_and_extracts_image_url():
    body = FalAIChatConfig().transform_request(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is"},
                    {"type": "text", "text": "in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}},
                ],
            }
        ],
        optional_params={"temperature": 0.2, "top_p": 0.9, "reasoning": False},
        litellm_params={},
        headers={},
    )
    assert body == {
        "prompt": "what is\nin this image?",
        "image_url": "https://example.com/pic.png",
        "temperature": 0.2,
        "top_p": 0.9,
        "reasoning": False,
    }


def test_transform_request_passes_data_url_through():
    body = FalAIChatConfig().transform_request(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert body["image_url"] == "data:image/png;base64,AAAA"


def test_transform_request_accepts_single_user_message():
    body = FalAIChatConfig().transform_request(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": "https://a"}],
            }
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert body["prompt"] == "describe"
    assert body["image_url"] == "https://a"


def test_transform_request_rejects_system_message():
    with pytest.raises(FalAIError, match="exactly one user message") as exc_info:
        FalAIChatConfig().transform_request(
            model=MODEL,
            messages=[
                {"role": "system", "content": "be terse"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": "https://a"}],
                },
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )
    assert exc_info.value.status_code == 400


def test_transform_request_rejects_multi_turn_history():
    with pytest.raises(FalAIError, match="exactly one user message") as exc_info:
        FalAIChatConfig().transform_request(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "first"}, {"type": "image_url", "image_url": "https://a"}],
                },
                {"role": "assistant", "content": "an answer"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "second"}, {"type": "image_url", "image_url": "https://b"}],
                },
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )
    assert exc_info.value.status_code == 400


def test_transform_request_rejects_zero_images():
    with pytest.raises(FalAIError, match="exactly one image_url"):
        FalAIChatConfig().transform_request(
            model=MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": "describe"}]}],
            optional_params={},
            litellm_params={},
            headers={},
        )


def test_transform_request_rejects_two_images():
    with pytest.raises(FalAIError, match="exactly one image_url"):
        FalAIChatConfig().transform_request(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "compare"},
                        {"type": "image_url", "image_url": {"url": "https://a"}},
                        {"type": "image_url", "image_url": {"url": "https://b"}},
                    ],
                }
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )


def test_transform_request_rejects_missing_text():
    with pytest.raises(FalAIError, match="require text"):
        FalAIChatConfig().transform_request(
            model=MODEL,
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://a"}}]}],
            optional_params={},
            litellm_params={},
            headers={},
        )


def test_transform_request_rejects_streaming():
    with pytest.raises(FalAIError, match="streaming"):
        FalAIChatConfig().transform_request(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://a"}},
                    ],
                }
            ],
            optional_params={"stream": True},
            litellm_params={},
            headers={},
        )


def test_completion_dispatch_rejects_streaming():
    with pytest.raises(litellm.BadRequestError):
        litellm.completion(
            model=MODEL,
            custom_llm_provider="fal_ai",
            stream=True,
            messages=[{"role": "user", "content": "describe"}],
        )


@pytest.mark.parametrize(
    "effort,expected",
    [("none", False), ("minimal", False), ("low", True), ("medium", True), ("high", True)],
)
def test_map_openai_params_maps_reasoning_effort(effort, expected):
    mapped = FalAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": effort}, optional_params={}, model=MODEL, drop_params=False
    )
    assert mapped["reasoning"] is expected


def test_map_openai_params_drops_unknown_reasoning_effort_when_dropping():
    mapped = FalAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": "extreme"}, optional_params={}, model=MODEL, drop_params=True
    )
    assert "reasoning" not in mapped


@pytest.mark.parametrize("effort", [{"level": "low"}, ["low"], 1])
def test_map_openai_params_rejects_non_string_reasoning_effort(effort: object) -> None:
    with pytest.raises(FalAIError) as exc_info:
        FalAIChatConfig().map_openai_params(
            non_default_params={"reasoning_effort": effort}, optional_params={}, model=MODEL, drop_params=False
        )
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("effort", [{"level": "low"}, ["low"], 1])
def test_map_openai_params_drops_non_string_reasoning_effort_when_dropping(effort: object) -> None:
    mapped = FalAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": effort}, optional_params={}, model=MODEL, drop_params=True
    )
    assert "reasoning" not in mapped


def test_map_openai_params_maps_sampling_params():
    mapped = FalAIChatConfig().map_openai_params(
        non_default_params={"temperature": 0.5, "top_p": 0.7, "max_tokens": 10},
        optional_params={},
        model=MODEL,
        drop_params=False,
    )
    assert mapped == {"temperature": 0.5, "top_p": 0.7}


def test_transform_response_maps_output_reasoning_usage_and_finish_reason():
    raw = httpx.Response(
        200,
        json={
            "output": "a red circle",
            "reasoning": "looked at shapes",
            "finish_reason": "stop",
            "usage_info": {
                "input_tokens": 11,
                "output_tokens": 4,
                "prefill_time_ms": 1.0,
                "decode_time_ms": 2.0,
                "ttft_ms": 1.5,
            },
        },
    )
    response = FalAIChatConfig().transform_response(
        model=MODEL,
        raw_response=raw,
        model_response=ModelResponse(),
        logging_obj=None,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.choices[0].message.content == "a red circle"
    assert response.choices[0].message.reasoning_content == "looked at shapes"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 15
    assert response.model == MODEL


def test_transform_response_omits_reasoning_when_null():
    raw = httpx.Response(
        200,
        json={
            "output": "a red circle",
            "reasoning": None,
            "finish_reason": "stop",
            "usage_info": {"input_tokens": 3, "output_tokens": 2},
        },
    )
    response = FalAIChatConfig().transform_response(
        model=MODEL,
        raw_response=raw,
        model_response=ModelResponse(),
        logging_obj=None,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.choices[0].message.content == "a red circle"
    assert getattr(response.choices[0].message, "reasoning_content", None) is None
    assert response.usage.total_tokens == 5


def test_transform_response_rejects_body_missing_output():
    raw = httpx.Response(
        200,
        json={"reasoning": "looked", "usage_info": {"input_tokens": 3, "output_tokens": 2}},
    )
    with pytest.raises(FalAIError) as exc_info:
        FalAIChatConfig().transform_response(
            model=MODEL,
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
    assert exc_info.value.status_code == 422


def test_transform_response_rejects_body_missing_usage_info():
    raw = httpx.Response(200, json={"output": "a red circle"})
    with pytest.raises(FalAIError) as exc_info:
        FalAIChatConfig().transform_response(
            model=MODEL,
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
    assert exc_info.value.status_code == 422
