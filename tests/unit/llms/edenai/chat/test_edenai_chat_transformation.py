"""Eden AI (`edenai/...`) chat provider: an OpenAI-compatible gateway that reports the real
per-request cost at the top level of every response instead of leaving it to the price map."""

import json
from pathlib import Path

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params, response_cost_calculator
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.edenai.chat.transformation import EdenAIChatCompletionStreamingHandler, EdenAIChatConfig
from litellm.llms.edenai.common_utils import EdenAIException
from litellm.proxy.auth.model_checks import get_provider_models
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

REPO_ROOT = Path(__file__).resolve().parents[5]
EDEN_BASE = "https://api.edenai.run/v3"
EDEN_EU_BASE = "https://api.eu.edenai.run/v3"
EDEN_CHAT_URL = f"{EDEN_BASE}/chat/completions"
EDEN_REPORTED_COST = 0.0042
EDEN_USAGE = {"completion_tokens": 1, "prompt_tokens": 9, "total_tokens": 10}
MESSAGES = [{"role": "user", "content": "Say OK"}]


def _eden_chat_completion(cost: float | None = EDEN_REPORTED_COST) -> dict:
    """Live `/v3/chat/completions` body: OpenAI shape plus Eden's top-level `cost`, `provider`
    and `status`, with `model` echoing the seller's bare model name."""
    body = {
        "status": "success",
        "id": "chatcmpl-eden-1",
        "created": 1788347376,
        "model": "gpt-4.1-nano",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "OK", "role": "assistant"}}],
        "usage": EDEN_USAGE,
        "provider": "openai",
    }
    return body if cost is None else {**body, "cost": cost}


def _eden_stream_chunk(
    delta: dict, finish_reason: str | None = None, usage: dict | None = None, cost: float | None = None
) -> dict:
    chunk = {
        "id": "chatcmpl-eden-stream",
        "created": 1788347377,
        "model": "openai/gpt-4.1-nano",
        "object": "chat.completion.chunk",
        "choices": [{"finish_reason": finish_reason, "index": 0, "delta": delta, "logprobs": None}],
    }
    if usage is not None:
        chunk["usage"] = usage
    if cost is not None:
        chunk["cost"] = cost
    return chunk


def _eden_stream_frames(cost: float | None = EDEN_REPORTED_COST) -> tuple[dict, ...]:
    """Live stream with `stream_options.include_usage`: the usage frame comes after the
    finish_reason frame, keeps one empty choice, and carries Eden's `cost` at the top level."""
    return (
        _eden_stream_chunk({"role": "assistant", "content": ""}),
        _eden_stream_chunk({"content": "OK"}),
        _eden_stream_chunk({"content": None}, finish_reason="stop"),
        _eden_stream_chunk({"content": None, "role": None}, usage=EDEN_USAGE, cost=cost),
    )


def _sse(frames: tuple[dict, ...]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _request_body(respx_mock) -> dict:
    return json.loads(respx_mock.calls.last.request.content)


class TestProviderResolution:
    @pytest.mark.parametrize(
        "requested, sent_to_eden",
        [
            ("edenai/openai/gpt-4.1-nano", "openai/gpt-4.1-nano"),
            ("edenai/gpt-4o", "gpt-4o"),
            ("edenai/vertex/gemini-3.7-flash@eu", "vertex/gemini-3.7-flash@eu"),
            ("edenai/fireworks_ai/accounts/fireworks/models/glm-5p3", "fireworks_ai/accounts/fireworks/models/glm-5p3"),
            ("edenai/cloudflare/@cf/qwen/qwen3.8-27b", "cloudflare/@cf/qwen/qwen3.8-27b"),
        ],
    )
    def test_strips_only_the_edenai_prefix(self, eden_key, requested, sent_to_eden):
        model, provider, api_key, api_base = get_llm_provider(requested)

        assert (model, provider, api_key, api_base) == (sent_to_eden, "edenai", eden_key, EDEN_BASE)

    def test_env_api_base_moves_the_key_to_the_eu_endpoint(self, eden_key, monkeypatch):
        monkeypatch.setenv("EDENAI_API_BASE", EDEN_EU_BASE)

        _, provider, api_key, api_base = get_llm_provider("edenai/openai/gpt-4.1-nano")

        assert (provider, api_key, api_base) == ("edenai", eden_key, EDEN_EU_BASE)

    def test_explicit_credentials_win_over_env(self, eden_key):
        _, _, api_key, api_base = get_llm_provider(
            "edenai/openai/gpt-4.1-nano", api_key="explicit-key", api_base="https://eden.internal/v3"
        )

        assert (api_key, api_base) == ("explicit-key", "https://eden.internal/v3")

    def test_eden_api_base_is_recognised_without_the_prefix(self, eden_key):
        model, provider, api_key, api_base = get_llm_provider("gpt-4.1-nano", api_base=EDEN_BASE)

        assert (model, provider, api_key, api_base) == ("gpt-4.1-nano", "edenai", eden_key, EDEN_BASE)


class TestRegistration:
    def test_provider_is_registered_everywhere_routing_looks(self):
        assert LlmProviders.EDENAI.value == "edenai"
        assert "edenai" in litellm.provider_list
        assert "edenai" in litellm.openai_compatible_providers
        assert EDEN_BASE in litellm.openai_compatible_endpoints
        assert isinstance(
            ProviderConfigManager.get_provider_chat_config(model="openai/gpt-4.1-nano", provider=LlmProviders.EDENAI),
            EdenAIChatConfig,
        )

    def test_supported_params_are_the_openai_chat_params(self):
        supported = litellm.get_supported_openai_params(model="openai/gpt-4.1-nano", custom_llm_provider="edenai")

        assert supported is not None
        assert {"tools", "tool_choice", "response_format", "stream_options", "max_completion_tokens"} <= set(supported)

    def test_reasoning_effort_is_supported_only_for_models_the_price_map_flags_as_reasoning(self):
        reasoning = litellm.get_supported_openai_params(model="openai/gpt-5-mini", custom_llm_provider="edenai")
        plain = litellm.get_supported_openai_params(model="openai/gpt-4.1-nano", custom_llm_provider="edenai")

        assert reasoning is not None and plain is not None
        assert "reasoning_effort" in reasoning
        assert "reasoning_effort" not in plain

    def test_validate_environment_names_the_eden_key(self, monkeypatch):
        monkeypatch.delenv("EDENAI_API_KEY", raising=False)
        missing = litellm.validate_environment(model="edenai/openai/gpt-4.1-nano")
        monkeypatch.setenv("EDENAI_API_KEY", "eden-test-key")
        present = litellm.validate_environment(model="edenai/openai/gpt-4.1-nano")

        assert (missing["keys_in_environment"], missing["missing_keys"]) == (False, ["EDENAI_API_KEY"])
        assert (present["keys_in_environment"], present["missing_keys"]) == (True, [])

    def test_a_model_registered_from_a_cost_map_still_asks_for_the_eden_key(self, monkeypatch):
        """A cost map may name an Eden model without the `edenai/` prefix, leaving the provider
        registry as the only way key validation can tell whose key the model needs."""
        alias = "eden-cost-map-alias"
        litellm.register_model(
            {alias: {"litellm_provider": "edenai", "mode": "chat", "input_cost_per_token": 1e-06}},
            persist_across_reloads=False,
        )
        try:
            monkeypatch.delenv("EDENAI_API_KEY", raising=False)
            missing = litellm.validate_environment(model=alias)
            monkeypatch.setenv("EDENAI_API_KEY", "eden-test-key")
            present = litellm.validate_environment(model=alias)
        finally:
            litellm.edenai_models.discard(alias)
            litellm.model_cost.pop(alias, None)
            litellm.add_known_models(model_cost_map={})

        assert (missing["keys_in_environment"], missing["missing_keys"]) == (False, ["EDENAI_API_KEY"])
        assert (present["keys_in_environment"], present["missing_keys"]) == (True, [])

    def test_a_cost_map_reload_reaches_wildcard_expansion(self, eden_key):
        """Wildcard expansion reads the provider registry, which a cost map reload rebuilds in
        place, so models added after startup have to show up without a restart."""
        alias = "edenai/openai/gpt-4.1-nano-from-cost-map"
        wildcard = LiteLLM_Params(model="edenai/*", api_key="wildcard-key")
        assert alias not in (get_provider_models("edenai", wildcard) or [])

        litellm.add_known_models(model_cost_map={alias: {"litellm_provider": "edenai", "mode": "chat"}})
        try:
            expanded = get_provider_models("edenai", wildcard)
        finally:
            litellm.edenai_models.discard(alias)
            litellm.add_known_models(model_cost_map={})

        assert expanded is not None
        assert alias in expanded
        assert alias not in (get_provider_models("edenai", wildcard) or [])


class TestRequestTransformation:
    def _request(self, optional_params: dict) -> dict:
        return EdenAIChatConfig().transform_request(
            model="openai/gpt-4.1-nano",
            messages=MESSAGES,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

    def test_streaming_request_asks_eden_for_the_usage_frame(self):
        assert self._request({"stream": True})["stream_options"] == {"include_usage": True}

    def test_streaming_request_overrides_a_caller_opt_out(self):
        body = self._request({"stream": True, "stream_options": {"include_usage": False}})

        assert body["stream_options"] == {"include_usage": True}

    def test_non_streaming_request_carries_no_stream_options(self):
        assert "stream_options" not in self._request({"max_tokens": 5})


class TestCompletion:
    def test_posts_to_eden_with_the_bearer_key_and_the_seller_model_id(self, eden_key, respx_mock):
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=httpx.Response(200, json=_eden_chat_completion()))

        response = litellm.completion(model="edenai/openai/gpt-4.1-nano", messages=MESSAGES, max_tokens=5)

        assert response.choices[0].message.content == "OK"
        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {eden_key}"
        body = _request_body(respx_mock)
        assert (body["model"], body["messages"], body["max_tokens"]) == ("openai/gpt-4.1-nano", MESSAGES, 5)

    def test_reasoning_effort_reaches_eden_without_drop_params(self, eden_key, respx_mock):
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=httpx.Response(200, json=_eden_chat_completion()))

        litellm.completion(model="edenai/openai/gpt-5-mini", messages=MESSAGES, reasoning_effort="low")

        assert _request_body(respx_mock)["reasoning_effort"] == "low"

    def test_eden_reported_cost_beats_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=httpx.Response(200, json=_eden_chat_completion()))

        response = litellm.completion(model="edenai/openai/gpt-4.1-nano", messages=MESSAGES, max_tokens=5)

        assert get_response_cost_from_hidden_params(response._hidden_params) == EDEN_REPORTED_COST
        assert (
            response_cost_calculator(
                response_object=response,
                model="openai/gpt-4.1-nano",
                custom_llm_provider="edenai",
                call_type="completion",
                optional_params={},
            )
            == EDEN_REPORTED_COST
        )

    def test_a_body_without_cost_leaves_pricing_to_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=httpx.Response(200, json=_eden_chat_completion(cost=None)))

        response = litellm.completion(model="edenai/openai/gpt-4.1-nano", messages=MESSAGES, max_tokens=5)

        assert response.choices[0].message.content == "OK"
        assert get_response_cost_from_hidden_params(response._hidden_params) is None

    def test_extra_body_forwards_eden_only_fields(self, eden_key, respx_mock):
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=httpx.Response(200, json=_eden_chat_completion()))

        litellm.completion(
            model="edenai/openai/gpt-4.1-nano",
            messages=MESSAGES,
            extra_body={"fallbacks": ["anthropic/claude-sonnet-latest"], "routing": {"sort": "latency"}},
        )

        body = _request_body(respx_mock)
        assert body["fallbacks"] == ["anthropic/claude-sonnet-latest"]
        assert body["routing"] == {"sort": "latency"}
        assert "extra_body" not in body

    def test_unknown_kwargs_ride_along_as_eden_fields(self, eden_key, respx_mock):
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=httpx.Response(200, json=_eden_chat_completion()))

        litellm.completion(model="edenai/openai/gpt-4.1-nano", messages=MESSAGES, routing={"sort": "latency"})

        assert _request_body(respx_mock)["routing"] == {"sort": "latency"}


class TestStreaming:
    def test_include_usage_surfaces_eden_cost_on_the_usage_chunk(self, eden_key, respx_mock):
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=_sse(_eden_stream_frames()))

        chunks = list(
            litellm.completion(
                model="edenai/openai/gpt-4.1-nano",
                messages=MESSAGES,
                stream=True,
                stream_options={"include_usage": True},
            )
        )

        assert _request_body(respx_mock)["stream_options"] == {"include_usage": True}
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "OK"
        usage_chunks = [chunk for chunk in chunks if getattr(chunk, "usage", None) is not None]
        assert len(usage_chunks) == 1
        assert (usage_chunks[0].usage.total_tokens, usage_chunks[0].usage.cost) == (10, EDEN_REPORTED_COST)

    def test_without_include_usage_eden_cost_is_still_tracked_but_hidden(self, eden_key, respx_mock):
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=_sse(_eden_stream_frames()))

        chunks = list(litellm.completion(model="edenai/openai/gpt-4.1-nano", messages=MESSAGES, stream=True))

        assert _request_body(respx_mock)["stream_options"] == {"include_usage": True}
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "OK"
        assert all(getattr(chunk, "usage", None) is None for chunk in chunks)
        hidden_usage = chunks[-1]._hidden_params["usage"]
        assert (hidden_usage.total_tokens, hidden_usage.cost) == (10, EDEN_REPORTED_COST)


class TestStreamingHandler:
    def _parse(self, chunk: dict):
        return EdenAIChatCompletionStreamingHandler(streaming_response=None, sync_stream=True).chunk_parser(chunk)

    def test_moves_top_level_cost_onto_the_usage_object(self):
        parsed = self._parse(_eden_stream_chunk({"content": None}, usage=EDEN_USAGE, cost=EDEN_REPORTED_COST))

        assert parsed.usage is not None
        assert (parsed.usage.prompt_tokens, parsed.usage.cost) == (9, EDEN_REPORTED_COST)

    def test_usage_without_cost_stays_unpriced(self):
        parsed = self._parse(_eden_stream_chunk({"content": None}, usage=EDEN_USAGE))

        assert parsed.usage is not None
        assert getattr(parsed.usage, "cost", None) is None

    def test_content_chunks_are_passed_through(self):
        parsed = self._parse(_eden_stream_chunk({"content": "OK"}))

        assert parsed.choices[0].delta.content == "OK"
        assert getattr(parsed, "usage", None) is None


class TestErrors:
    def test_middleware_401_detail_body_maps_to_authentication_error(self, eden_key, respx_mock):
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=httpx.Response(401, json={"detail": "Invalid token"}))

        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            litellm.completion(model="edenai/openai/gpt-4.1-nano", messages=MESSAGES)

    def test_unknown_model_envelope_maps_to_bad_request(self, eden_key, respx_mock):
        envelope = {
            "error": {
                "message": "Model(s) not found or inactive: openai/does-not-exist",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_parameter",
            }
        }
        respx_mock.post(EDEN_CHAT_URL).mock(return_value=httpx.Response(400, json=envelope))

        with pytest.raises(litellm.BadRequestError, match="not found or inactive"):
            litellm.completion(model="edenai/openai/does-not-exist", messages=MESSAGES)

    def test_429_maps_to_rate_limit_error(self, eden_key, respx_mock):
        envelope = {
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": "rate_limit_exceeded"}
        }
        respx_mock.post(EDEN_CHAT_URL).mock(
            return_value=httpx.Response(429, json=envelope, headers={"Retry-After": "7"})
        )

        with pytest.raises(litellm.RateLimitError, match="Rate limit exceeded"):
            litellm.completion(model="edenai/openai/gpt-4.1-nano", messages=MESSAGES, num_retries=0)

    def test_error_class_is_the_eden_exception(self):
        error = EdenAIChatConfig().get_error_class("boom", 503, {"Content-Type": "application/json"})

        assert isinstance(error, EdenAIException)
        assert isinstance(error, BaseLLMException)
        assert (error.message, error.status_code, error.headers) == ("boom", 503, {"Content-Type": "application/json"})


class TestModelListing:
    CATALOG = {"data": [{"id": "openai/gpt-4.1-nano", "object": "model"}, {"id": "anthropic/claude-sonnet-latest"}]}
    ROUTABLE = ["edenai/openai/gpt-4.1-nano", "edenai/anthropic/claude-sonnet-latest"]

    def test_lists_the_public_catalog_as_routable_model_names(self, eden_key, respx_mock):
        respx_mock.get(f"{EDEN_BASE}/models").mock(return_value=httpx.Response(200, json=self.CATALOG))

        assert EdenAIChatConfig().get_models() == self.ROUTABLE

    def test_lists_from_the_configured_endpoint(self, eden_key, monkeypatch, respx_mock):
        monkeypatch.setenv("EDENAI_API_BASE", EDEN_EU_BASE)
        respx_mock.get(f"{EDEN_EU_BASE}/models").mock(return_value=httpx.Response(200, json=self.CATALOG))

        assert EdenAIChatConfig().get_models() == self.ROUTABLE

    def test_get_valid_models_reads_the_live_catalog(self, eden_key, respx_mock):
        respx_mock.get(f"{EDEN_BASE}/models").mock(return_value=httpx.Response(200, json=self.CATALOG))

        models = litellm.get_valid_models(
            custom_llm_provider="edenai", check_provider_endpoint=True, api_key="listing-key"
        )

        assert models == self.ROUTABLE

    def test_a_rejected_catalog_request_surfaces_edens_status_and_body(self, eden_key, respx_mock):
        """A bad key has to reach the caller as an Eden error, not as a parse failure on the
        rejection body that never held a catalog."""
        respx_mock.get(f"{EDEN_BASE}/models").mock(return_value=httpx.Response(401, json={"detail": "Invalid token"}))

        with pytest.raises(EdenAIException) as rejected:
            EdenAIChatConfig().get_models()

        assert rejected.value.status_code == 401
        assert "Invalid token" in rejected.value.message

    def test_proxy_wildcard_expands_to_the_live_catalog(self, eden_key, monkeypatch, respx_mock):
        monkeypatch.setattr(litellm, "check_provider_endpoint", True)
        respx_mock.get(f"{EDEN_BASE}/models").mock(return_value=httpx.Response(200, json=self.CATALOG))

        models = get_provider_models("edenai", LiteLLM_Params(model="edenai/*", api_key="wildcard-key"))

        assert models == self.ROUTABLE


class TestDashboardRegistration:
    def test_add_model_form_offers_eden_with_a_required_key_and_optional_base(self):
        fields_path = REPO_ROOT / "litellm" / "proxy" / "public_endpoints" / "provider_create_fields.json"
        entries = [e for e in json.loads(fields_path.read_text()) if e["litellm_provider"] == "edenai"]

        assert len(entries) == 1
        entry = entries[0]
        assert (entry["provider"], entry["provider_display_name"]) == ("EDENAI", "Eden AI")
        assert entry["default_model_placeholder"].startswith("edenai/")
        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert (fields["api_key"]["required"], fields["api_key"]["field_type"]) == (True, "password")
        assert (fields["api_base"]["required"], fields["api_base"]["placeholder"]) == (False, EDEN_BASE)

    @pytest.mark.parametrize(
        "matrix_path",
        [
            REPO_ROOT / "provider_endpoints_support.json",
            REPO_ROOT / "litellm" / "provider_endpoints_support_backup.json",
        ],
        ids=["root", "backup"],
    )
    def test_endpoint_matrix_documents_every_served_surface(self, matrix_path):
        entry = json.loads(matrix_path.read_text())["providers"]["edenai"]

        assert entry["url"] == "https://docs.litellm.ai/docs/providers/edenai"
        served = {name for name, flag in entry["endpoints"].items() if flag}
        assert served == {
            "chat_completions",
            "messages",
            "responses",
            "embeddings",
            "image_generations",
            "audio_transcriptions",
            "audio_speech",
            "video_generations",
        }
