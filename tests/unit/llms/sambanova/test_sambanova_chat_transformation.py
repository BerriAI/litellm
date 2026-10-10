import pytest

from litellm.llms.sambanova.chat import SambanovaConfig
from litellm.utils import get_optional_params

_ALWAYS_SUPPORTED = (
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop",
    "stream",
    "stream_options",
    "response_format",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
    "n",
    "logit_bias",
    "seed",
)


@pytest.mark.parametrize("param", _ALWAYS_SUPPORTED)
def test_supported_params_include_spec_params(param):
    assert param in SambanovaConfig().get_supported_openai_params("MiniMax-M3")


def test_reasoning_effort_only_for_reasoning_models():
    config = SambanovaConfig()
    assert "reasoning_effort" in config.get_supported_openai_params("gpt-oss-120b")
    assert "reasoning_effort" not in config.get_supported_openai_params("Meta-Llama-3.3-70B-Instruct")


def test_tools_only_for_function_calling_models():
    config = SambanovaConfig()
    for param in ("tools", "tool_choice", "parallel_tool_calls"):
        assert param in config.get_supported_openai_params("MiniMax-M3")
        assert param not in config.get_supported_openai_params("gemma-4-31B-it")


def test_max_completion_tokens_mapped_to_max_tokens():
    optional_params = SambanovaConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 128},
        optional_params={},
        model="MiniMax-M3",
        drop_params=False,
    )
    assert optional_params == {"max_tokens": 128}


def test_new_params_forwarded_via_get_optional_params():
    optional_params = get_optional_params(
        model="MiniMax-M3",
        custom_llm_provider="sambanova",
        seed=7,
        presence_penalty=0.5,
        frequency_penalty=0.25,
        reasoning_effort="low",
    )
    assert optional_params["seed"] == 7
    assert optional_params["presence_penalty"] == 0.5
    assert optional_params["frequency_penalty"] == 0.25
    assert optional_params["reasoning_effort"] == "low"


def _transform(optional_params):
    return SambanovaConfig().transform_request(
        model="MiniMax-M3",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )


def test_integration_source_defaults_to_litellm(monkeypatch):
    monkeypatch.delenv("SAMBANOVA_INTEGRATION_SOURCE", raising=False)
    request = _transform({})
    assert request["extra_headers"]["X-Integration-Source"] == "litellm"


def test_integration_source_env_overrides_default(monkeypatch):
    monkeypatch.setenv("SAMBANOVA_INTEGRATION_SOURCE", "langflow")
    request = _transform({})
    assert request["extra_headers"]["X-Integration-Source"] == "langflow"


def test_integration_source_param_overrides_env(monkeypatch):
    monkeypatch.setenv("SAMBANOVA_INTEGRATION_SOURCE", "langflow")
    optional_params = {"extra_body": {"integration_source": "my-framework"}}
    request = _transform(optional_params)
    assert request["extra_headers"]["X-Integration-Source"] == "my-framework"
    assert "integration_source" not in request.get("extra_body", {})
    assert "integration_source" in optional_params["extra_body"], "caller's dict must not be mutated"


def test_integration_source_explicit_header_wins(monkeypatch):
    monkeypatch.setenv("SAMBANOVA_INTEGRATION_SOURCE", "langflow")
    request = _transform({"extra_headers": {"X-Integration-Source": "custom", "X-Other": "1"}})
    assert request["extra_headers"] == {"X-Integration-Source": "custom", "X-Other": "1"}


def test_integration_source_removed_from_body_when_header_also_set():
    request = _transform(
        {
            "extra_headers": {"X-Integration-Source": "custom"},
            "extra_body": {"integration_source": "ignored", "other": 1},
        }
    )
    assert request["extra_headers"] == {"X-Integration-Source": "custom"}
    assert request["extra_body"] == {"other": 1}


def test_integration_source_header_matched_case_insensitively(monkeypatch):
    monkeypatch.setenv("SAMBANOVA_INTEGRATION_SOURCE", "langflow")
    request = _transform({"extra_headers": {"x-integration-source": "custom"}})
    assert request["extra_headers"] == {"x-integration-source": "custom"}


def test_content_list_converted_to_string():
    messages = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    out = SambanovaConfig()._transform_messages(messages, model="MiniMax-M3")
    assert isinstance(out[0]["content"], str)
    assert "a" in out[0]["content"] and "b" in out[0]["content"]
