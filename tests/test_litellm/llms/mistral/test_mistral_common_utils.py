import pytest

import litellm
from litellm.llms.mistral.common_utils import MistralModelInfo, is_web_search_request


@pytest.fixture(autouse=True)
def add_mistral_api_key_to_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral-api-key-12345")


@pytest.fixture
def local_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


@pytest.mark.parametrize(
    "optional_params, expected",
    [
        ({"web_search_options": {}}, True),
        ({"tools": [{"type": "web_search"}]}, True),
        ({"tools": [{"type": "web_search_premium"}]}, True),
        ({"tools": [{"type": "function", "function": {"name": "x"}}]}, False),
        ({"tools": []}, False),
        ({}, False),
    ],
)
def test_is_web_search_request(optional_params, expected):
    assert is_web_search_request(optional_params) is expected


class TestMistralModelInfo:
    """Capability reporting and environment resolution for Mistral models."""

    @pytest.mark.parametrize(
        "model",
        ["mistral/mistral-medium-latest", "mistral/mistral-large-latest", "mistral/mistral-small-latest"],
    )
    def test_supports_web_search_true_for_chat_models(self, model, local_cost_map):
        assert litellm.supports_web_search(model) is True

    @pytest.mark.parametrize("model", ["mistral/mistral-embed", "mistral/mistral-ocr-latest"])
    def test_no_web_search_for_non_chat_models(self, model, local_cost_map):
        assert litellm.supports_web_search(model) is False

    def test_get_api_base_default_arg_and_env(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_BASE", raising=False)
        assert MistralModelInfo.get_api_base() == "https://api.mistral.ai"
        assert MistralModelInfo.get_api_base("https://custom.example/v1") == "https://custom.example/v1"
        monkeypatch.setenv("MISTRAL_API_BASE", "https://env.example")
        assert MistralModelInfo.get_api_base() == "https://env.example"

    def test_get_api_key_from_arg_and_env(self):
        assert MistralModelInfo.get_api_key("explicit-key") == "explicit-key"
        assert MistralModelInfo.get_api_key() == "fake-mistral-api-key-12345"

    def test_get_base_model_strips_prefix(self):
        assert MistralModelInfo.get_base_model("mistral/mistral-large-latest") == "mistral-large-latest"

    def test_validate_environment_sets_bearer_and_content_type(self):
        headers = MistralModelInfo().validate_environment(
            headers={},
            model="mistral-medium-latest",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-abc",
        )
        assert headers["Authorization"] == "Bearer sk-abc"
        assert headers["Content-Type"] == "application/json"

    def test_get_models_returns_prefixed_ids(self, respx_mock):
        litellm.disable_aiohttp_transport = True
        respx_mock.get("https://api.mistral.ai/v1/models").respond(
            json={"data": [{"id": "mistral-medium-latest"}, {"id": "mistral-large-latest"}]}
        )
        assert MistralModelInfo().get_models() == [
            "mistral/mistral-medium-latest",
            "mistral/mistral-large-latest",
        ]

    def test_get_models_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MISTRAL_API_BASE or MISTRAL_API_KEY"):
            MistralModelInfo().get_models()

    def test_get_models_raises_on_http_error(self, respx_mock):
        litellm.disable_aiohttp_transport = True
        respx_mock.get("https://api.mistral.ai/v1/models").respond(status_code=500, text="boom")
        with pytest.raises(Exception, match="Status code: 500"):
            MistralModelInfo().get_models()
