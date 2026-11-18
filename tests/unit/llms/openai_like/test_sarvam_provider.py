import json
import platform
from typing import Final

import httpx
import openai
import pytest
import respx

import litellm
from litellm._version import version as LITELLM_VERSION
from litellm.litellm_core_utils.reasoning_effort_strength import REASONING_EFFORT_STRENGTH_ORDER
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

SARVAM_USER_AGENT: Final = f"litellm/{LITELLM_VERSION} Python/{platform.python_version()}"


def _sarvam_config() -> SimpleProviderConfig:
    config: Final = JSONProviderRegistry.get("sarvam")
    assert config is not None
    return config


def _sarvam_105b_base_url() -> str:
    return _sarvam_config().default_base_url("sarvam-105b")


def _sarvam_chat_url() -> str:
    return f"{_sarvam_105b_base_url()}/chat/completions"


def _chat_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "sarvam-105b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_sarvam_attribution_headers_report_running_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)

    assert dict(_sarvam_config().get_attribution_headers()) == {"User-Agent": SARVAM_USER_AGENT}


@pytest.mark.parametrize("override", ["my-gateway/2.0", ""])
def test_litellm_user_agent_env_replaces_the_template(monkeypatch: pytest.MonkeyPatch, override: str) -> None:
    monkeypatch.setenv("LITELLM_USER_AGENT", override)

    assert dict(_sarvam_config().get_attribution_headers()) == {"User-Agent": override}


def test_generated_config_class_forwards_attribution_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)
    instance: Final = create_config_class(_sarvam_config())()

    assert dict(instance.get_attribution_headers()) == {"User-Agent": SARVAM_USER_AGENT}


def test_json_provider_without_user_agent_sends_no_attribution_headers() -> None:
    config: Final = JSONProviderRegistry.get("aiand")
    assert config is not None

    assert dict(config.get_attribution_headers()) == {}


def test_sarvam_chat_completion_sends_the_litellm_user_agent(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)
    route: Final = respx_mock.post(_sarvam_chat_url()).mock(return_value=httpx.Response(200, json=_chat_response()))

    litellm.completion(
        model="sarvam/sarvam-105b",
        api_key="test-key",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert route.called
    assert route.calls.last.request.headers["User-Agent"] == SARVAM_USER_AGENT


def test_sarvam_caller_user_agent_wins(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_sarvam_chat_url()).mock(return_value=httpx.Response(200, json=_chat_response()))

    litellm.completion(
        model="sarvam/sarvam-105b",
        api_key="test-key",
        messages=[{"role": "user", "content": "Hello"}],
        headers={"User-Agent": "caller-app/1.0"},
    )

    assert route.called
    assert route.calls.last.request.headers["User-Agent"] == "caller-app/1.0"


class _BrandedClient(openai.OpenAI):
    @property
    def user_agent(self) -> str:
        return "branded-app/3.0"


class _UnbrandedSubclass(openai.OpenAI):
    pass


@pytest.mark.parametrize(
    ("client_class", "client_default_headers", "expected_user_agent"),
    [
        pytest.param(
            openai.OpenAI, {"User-Agent": "client-default/1.0"}, "client-default/1.0", id="client_user_agent_wins"
        ),
        pytest.param(_BrandedClient, {}, "branded-app/3.0", id="subclass_user_agent_wins"),
        pytest.param(openai.OpenAI, {}, SARVAM_USER_AGENT, id="sdk_default_user_agent_replaced"),
        pytest.param(_UnbrandedSubclass, {}, SARVAM_USER_AGENT, id="subclass_sdk_default_user_agent_replaced"),
    ],
)
def test_sarvam_user_agent_with_a_caller_client(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    client_class: type[openai.OpenAI],
    client_default_headers: dict[str, str],
    expected_user_agent: str,
) -> None:
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)
    route: Final = respx_mock.post(_sarvam_chat_url()).mock(return_value=httpx.Response(200, json=_chat_response()))
    client: Final = client_class(
        api_key="test-key", base_url=_sarvam_105b_base_url(), default_headers=client_default_headers
    )

    litellm.completion(
        model="sarvam/sarvam-105b",
        api_key="test-key",
        messages=[{"role": "user", "content": "Hello"}],
        client=client,
    )

    assert route.called
    assert route.calls.last.request.headers["User-Agent"] == expected_user_agent


SARVAM_MODELS: Final = (
    "sarvam/sarvam-105b",
    "sarvam/sarvam-105b-conversations",
    "sarvam/glm5.3",
    "sarvam/glm5.3-flash",
    "sarvam/gemma4",
    "sarvam/deepseekv4.1-flash-1017",
    "sarvam/deepseekv4.1-flash-1017-interactive",
)


def _sarvam_response(model: str, cached_tokens: int) -> ModelResponse:
    return ModelResponse(
        model=model,
        usage=Usage(
            prompt_tokens=1000,
            completion_tokens=5,
            total_tokens=1005,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
        ),
    )


@pytest.mark.parametrize(
    ("model", "api_base", "expected_api_base"),
    [
        pytest.param("sarvam/sarvam-105b", None, "https://api.sarvam.ai/v1", id="sarvam_105b_stays_on_v1"),
        pytest.param(
            "sarvam/sarvam-105b-conversations", None, "https://api.sarvam.ai/v1", id="conversations_stays_on_v1"
        ),
        pytest.param("sarvam/glm5.3", None, "https://api.sarvam.ai/v2", id="token_factory_model_uses_v2"),
        pytest.param("sarvam/unlisted-model", None, "https://api.sarvam.ai/v2", id="unlisted_model_uses_v2"),
        pytest.param(
            "sarvam/sarvam-105b", "https://api.sarvam.ai/v2", "https://api.sarvam.ai/v2", id="deployment_api_base_wins"
        ),
    ],
)
def test_sarvam_provider_resolution(model: str, api_base: str | None, expected_api_base: str) -> None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    resolved_model, provider, _, resolved_api_base = get_llm_provider(
        model=model,
        custom_llm_provider=None,
        api_base=api_base,
        api_key=None,
    )

    assert resolved_model == model.removeprefix("sarvam/")
    assert provider == "sarvam"
    assert resolved_api_base == expected_api_base


@pytest.mark.parametrize(
    ("model", "url"),
    [
        pytest.param("sarvam-105b", "https://api.sarvam.ai/v1/chat/completions", id="sarvam_105b_on_v1"),
        pytest.param("glm5.3", "https://api.sarvam.ai/v2/chat/completions", id="glm_on_v2"),
    ],
)
def test_sarvam_chat_completion_request(model: str, url: str) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(url).respond(
            200,
            json={
                "id": "chatcmpl_sarvam",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model=f"sarvam/{model}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sarvam-test-key",
            max_completion_tokens=200,
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == url
    assert request.headers["authorization"] == "Bearer sarvam-test-key"
    assert body == {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 200,
    }
    assert response.choices[0].message.content == "Hello"


@pytest.mark.parametrize("model", ["sarvam-105b", "sarvam-105b-conversations"])
def test_sarvam_v1_models_receive_text_content_lists_as_strings(model: str) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.sarvam.ai/v1/chat/completions").respond(200, json=_chat_response())
        litellm.completion(
            model=f"sarvam/{model}",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": "Answer in English."},
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Say"}, {"type": "text", "text": "hello"}]},
            ],
            api_key="sarvam-test-key",
        )

    assert json.loads(route.calls.last.request.content)["messages"] == [
        {"role": "system", "content": "Be terse.\nAnswer in English."},
        {"role": "user", "content": "Say\nhello"},
    ]


@pytest.mark.parametrize("model", ["sarvam-105b", "sarvam-105b-conversations"])
def test_sarvam_v1_models_send_lists_with_images_unchanged(model: str) -> None:
    content: Final = [
        {"type": "text", "text": "Describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ]
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.sarvam.ai/v1/chat/completions").respond(200, json=_chat_response())
        litellm.completion(
            model=f"sarvam/{model}",
            messages=[{"role": "user", "content": content}],
            api_key="sarvam-test-key",
        )

    assert json.loads(route.calls.last.request.content)["messages"] == [{"role": "user", "content": content}]


def test_sarvam_v2_models_keep_content_lists() -> None:
    image_part: Final = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.sarvam.ai/v2/chat/completions").respond(200, json=_chat_response())
        litellm.completion(
            model="sarvam/gemma4",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Describe this"}, image_part]}],
            api_key="sarvam-test-key",
        )

    assert json.loads(route.calls.last.request.content)["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Describe this"}, image_part]}
    ]


def _sent_body(model: str, **params: object) -> dict[str, object]:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_sarvam_config().default_base_url(model)}/chat/completions").respond(
            200, json=_chat_response()
        )
        litellm.completion(
            model=f"sarvam/{model}",
            messages=[{"role": "user", "content": "Is 221 a prime number?"}],
            api_key="sarvam-test-key",
            **params,
        )
    return json.loads(route.calls.last.request.content)


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("effort", ["none", *REASONING_EFFORT_STRENGTH_ORDER])
def test_sarvam_forwards_reasoning_effort_without_allowed_openai_params(effort: str) -> None:
    assert _sent_body("gemma4", reasoning_effort=effort)["reasoning_effort"] == effort


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", ["sarvam-105b", "glm5.3", "glm5.3-flash"])
@pytest.mark.parametrize("effort", REASONING_EFFORT_STRENGTH_ORDER)
def test_sarvam_sends_only_a_declared_reasoning_effort_at_least_as_strong(model: str, effort: str) -> None:
    declared: Final = litellm.model_cost[f"sarvam/{model}"]["reasoning_effort_levels"]
    strongest: Final = max(REASONING_EFFORT_STRENGTH_ORDER.index(level) for level in declared)
    sent: Final = _sent_body(model, reasoning_effort=effort)["reasoning_effort"]

    assert sent in declared
    assert effort not in declared or sent == effort
    assert REASONING_EFFORT_STRENGTH_ORDER.index(sent) >= min(REASONING_EFFORT_STRENGTH_ORDER.index(effort), strongest)


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    ("per_call", "global_drop_params", "expected"),
    [
        pytest.param({"drop_params": True}, False, None, id="per_call"),
        pytest.param({}, True, None, id="global"),
        pytest.param({"drop_params": False}, False, "none", id="off"),
    ],
)
def test_sarvam_105b_drops_reasoning_off_only_under_drop_params(
    monkeypatch: pytest.MonkeyPatch, per_call: dict[str, bool], global_drop_params: bool, expected: str | None
) -> None:
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)
    assert _sent_body("sarvam-105b", reasoning_effort="none", **per_call).get("reasoning_effort") == expected


@pytest.mark.usefixtures("local_model_cost_map")
def test_a_sibling_deployments_declared_levels_leave_the_sarvam_request_alone() -> None:
    deployment: Final = {"model": "sarvam/sarvam-105b", "api_key": "sarvam-test-key"}
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "declared",
                "litellm_params": deployment,
                "model_info": {"reasoning_effort_levels": ["low", "high"]},
            },
            {"model_name": "sibling", "litellm_params": deployment},
        ]
    )
    with respx.mock() as upstream:
        route: Final = upstream.post(_sarvam_chat_url()).respond(200, json=_chat_response())
        router.completion(
            model="sibling",
            messages=[{"role": "user", "content": "Is 221 a prime number?"}],
            reasoning_effort="medium",
        )

    assert json.loads(route.calls.last.request.content)["reasoning_effort"] == "medium"


def test_sarvam_responses_request_is_bridged_to_chat_completions() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.sarvam.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_sarvam_bridge",
                "object": "chat.completion",
                "created": 1,
                "model": "sarvam-105b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="sarvam/sarvam-105b",
            input="Say hello",
            api_key="sarvam-test-key",
        )

    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer sarvam-test-key"
    assert json.loads(request.content) == {
        "model": "sarvam-105b",
        "messages": [{"role": "user", "content": "Say hello"}],
    }
    assert response.output_text == "Hello"


@pytest.mark.usefixtures("local_model_cost_map")
class TestSarvamCostMap:
    @pytest.mark.parametrize("model", SARVAM_MODELS)
    def test_cached_prompt_tokens_cost_less_than_uncached(self, model: str) -> None:
        uncached_cost: Final = litellm.completion_cost(completion_response=_sarvam_response(model, cached_tokens=0))
        cached_cost: Final = litellm.completion_cost(completion_response=_sarvam_response(model, cached_tokens=800))

        assert 0 < cached_cost < uncached_cost

    def test_cached_tokens_bill_at_the_cached_rate(self) -> None:
        cached_tokens: Final = 800
        with respx.mock() as upstream:
            route: Final = upstream.post("https://api.sarvam.ai/v1/chat/completions").respond(
                200,
                json={
                    "id": "chatcmpl_sarvam_cache",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "sarvam-105b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 5,
                        "total_tokens": 1005,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                },
            )
            response: Final = litellm.completion(
                model="sarvam/sarvam-105b",
                messages=[{"role": "user", "content": "Say hello"}],
                api_key="sarvam-test-key",
            )

        rates: Final = litellm.model_cost["sarvam/sarvam-105b"]
        expected: Final = (
            (1000 - cached_tokens) * rates["input_cost_per_token"]
            + cached_tokens * rates["cache_read_input_token_cost"]
            + 5 * rates["output_cost_per_token"]
        )
        assert route.call_count == 1
        assert response.usage.prompt_tokens_details.cached_tokens == cached_tokens
        assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "model",
        (
            "sarvam/deepseekv4.1-flash-1017",
            "sarvam/deepseekv4.1-flash-1017-interactive",
        ),
    )
    def test_deepseek_skus_bill_at_their_own_rates(self, model: str) -> None:
        cached_tokens: Final = 800
        with respx.mock() as upstream:
            route: Final = upstream.post("https://api.sarvam.ai/v2/chat/completions").respond(
                200,
                json={
                    "id": "chatcmpl_sarvam_deepseek",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model.split("/", 1)[1],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 5,
                        "total_tokens": 1005,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                },
            )
            response: Final = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Say hello"}],
                api_key="sarvam-test-key",
            )

        rates: Final = litellm.model_cost[model]
        expected: Final = (
            (1000 - cached_tokens) * rates["input_cost_per_token"]
            + cached_tokens * rates["cache_read_input_token_cost"]
            + 5 * rates["output_cost_per_token"]
        )
        assert route.call_count == 1
        assert response.usage.prompt_tokens_details.cached_tokens == cached_tokens
        assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)

    def test_sarvam_m_kept_for_deprecation(self) -> None:
        # Source: https://docs.sarvam.ai/changelog/2026/6/5.md (2026-06-05)
        entry: Final = litellm.model_cost["sarvam/sarvam-m"]
        assert entry["deprecation_date"] == "2026-06-05"
        assert entry["cache_creation_input_token_cost_above_1hr"] == 0
