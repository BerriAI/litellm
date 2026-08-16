"""
Apodex Responses API transformation.

The core models expose a stateless subset of /v1/responses while the Deep
Research tiers keep server-side state, so the parameter contract is keyed off
the model rather than applied provider-wide.
"""

import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

CORE_MODEL = "apodex/apodex-1.1"
CORE_MINI_MODEL = "apodex/apodex-1.1-mini"
DEEP_RESEARCH_MODEL = "apodex/apodex-1-1-deep-research"


@pytest.fixture(autouse=True)
def _apodex_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APODEX_API_KEY", "sk-apodex-test")
    monkeypatch.delenv("APODEX_API_BASE", raising=False)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(litellm, "drop_params", False)
    yield


_SENTINEL = "apodex-request-captured"


def _capture(**kwargs) -> dict:
    """Run litellm.responses() and return the request it would have sent.

    Validation errors raised before the request is built propagate to the caller.
    """
    captured: dict = {}

    class CapturingHandler(HTTPHandler):
        def post(self, *args, **post_kwargs):
            captured.update(url=post_kwargs.get("url"), body=post_kwargs.get("json"))
            raise RuntimeError(_SENTINEL)

    try:
        litellm.responses(client=CapturingHandler(), **kwargs)
    except Exception as exc:
        if _SENTINEL not in str(exc):
            raise
    assert captured, "no request was sent"
    return captured


def _responses_config(model: str):
    return ProviderConfigManager.get_provider_responses_api_config(model=model, provider=LlmProviders.APODEX)


class TestConfigSelection:
    def test_python_config_is_used_for_every_apodex_model(self):
        for model in ("apodex-1.1", "apodex-1.1-mini", "apodex-1-1-deep-research"):
            config = _responses_config(model)
            assert type(config).__name__ == "ApodexResponsesConfig"

    def test_auth_uses_the_apodex_key(self):
        config = _responses_config("apodex-1.1")
        assert config.validate_environment(headers={}, model="apodex-1.1", litellm_params=None) == {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-apodex-test",
        }

    def test_auth_does_not_fall_back_to_an_openai_key(self, monkeypatch: pytest.MonkeyPatch):
        """The inherited OpenAI config would forward OPENAI_API_KEY to Apodex."""
        monkeypatch.delenv("APODEX_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")

        config = _responses_config("apodex-1.1")
        assert config.validate_environment(headers={}, model="apodex-1.1", litellm_params=None) == {}

    def test_request_targets_the_apodex_responses_url(self):
        assert _capture(model=CORE_MODEL, input="hi")["url"] == "https://api.apodex.ai/v1/responses"

    def test_request_honours_an_api_base_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APODEX_API_BASE", "https://env.apodex.test/v1")
        assert _capture(model=CORE_MODEL, input="hi")["url"] == "https://env.apodex.test/v1/responses"


class TestStreamDefault:
    def test_non_streaming_pins_stream_false(self):
        captured = _capture(model=DEEP_RESEARCH_MODEL, input="hi")
        assert captured["url"] == "https://api.apodex.ai/v1/responses"
        assert captured["body"]["stream"] is False

    @pytest.mark.asyncio
    async def test_streaming_sends_stream_true(self):
        captured: dict = {}

        class CapturingHandler(AsyncHTTPHandler):
            async def post(self, *args, **kwargs):
                captured.update(body=kwargs.get("json"))
                raise RuntimeError("captured")

        with pytest.raises(Exception, match="captured"):
            await litellm.aresponses(
                model=DEEP_RESEARCH_MODEL, input="hi", stream=True, client=CapturingHandler()
            )

        assert captured["body"]["stream"] is True


class TestCoreModelStatelessSubset:
    """Apodex core models reject anything that would persist state on their side."""

    @pytest.mark.parametrize("model", [CORE_MODEL, CORE_MINI_MODEL])
    def test_store_is_pinned_false(self, model: str):
        captured = _capture(model=model, input="hi")
        assert captured["body"]["store"] is False

    def test_store_true_raises(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="store=True"):
            _capture(model=CORE_MODEL, input="hi", store=True)

    def test_background_raises(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="background"):
            _capture(model=CORE_MODEL, input="hi", background=True)

    def test_previous_response_id_raises(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="previous_response_id"):
            _capture(model=CORE_MODEL, input="hi", previous_response_id="resp_1")

    def test_drop_params_strips_all_three(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(litellm, "drop_params", True)
        captured = _capture(
            model=CORE_MODEL,
            input="hi",
            store=True,
            background=True,
            previous_response_id="resp_1",
        )

        assert captured["body"]["store"] is False
        assert "background" not in captured["body"]
        assert "previous_response_id" not in captured["body"]

    def test_stateful_params_are_not_advertised(self):
        supported = _responses_config("apodex-1.1").get_supported_openai_params("apodex-1.1")
        assert "background" not in supported
        assert "previous_response_id" not in supported
        assert "max_output_tokens" in supported

    def test_max_output_tokens_still_passes_through(self):
        captured = _capture(model=CORE_MODEL, input="hi", max_output_tokens=512)
        assert captured["body"]["max_output_tokens"] == 512


class TestDeepResearchKeepsState:
    """The agent tiers survive client disconnects, so none of this may be stripped."""

    def test_background_passes_through(self):
        captured = _capture(model=DEEP_RESEARCH_MODEL, input="hi", background=True)
        assert captured["body"]["background"] is True

    def test_store_and_previous_response_id_pass_through(self):
        captured = _capture(model=DEEP_RESEARCH_MODEL, input="hi", store=True, previous_response_id="resp_1")
        assert captured["body"]["store"] is True
        assert captured["body"]["previous_response_id"] == "resp_1"

    def test_store_is_not_pinned_when_unset(self):
        captured = _capture(model=DEEP_RESEARCH_MODEL, input="hi")
        assert "store" not in captured["body"]

    def test_stateful_params_are_advertised(self):
        supported = _responses_config("apodex-1-1-deep-research").get_supported_openai_params(
            "apodex-1-1-deep-research"
        )
        assert "background" in supported
        assert "previous_response_id" in supported
