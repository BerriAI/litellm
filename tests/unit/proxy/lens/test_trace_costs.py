from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.cost_calculator import completion_cost
from litellm.integrations.otel.model.semconv import resolve_provider
from litellm.proxy.lens.trace_costs import TraceCostInput, TraceCostsRequest, trace_cost, trace_costs
from litellm.types.utils import CacheCreationTokenDetails, ModelResponse, PromptTokensDetailsWrapper, Usage

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")
MODEL: Final = "openai/lens-test-cost"
START_NS: Final = 1767630600000000000


def _register(rates: Mapping[str, object], model: str = MODEL) -> None:
    litellm.register_model({model: {"litellm_provider": "openai", "mode": "chat", **rates}})


def _call(attributes: Mapping[str, str], start_ns: int = START_NS) -> TraceCostInput:
    return TraceCostInput(
        start_ns=start_ns,
        attributes={
            "gen_ai.request.model": MODEL,
            "gen_ai.provider.name": "openai",
            "gen_ai.usage.input_tokens": "100",
            "gen_ai.usage.output_tokens": "10",
            **attributes,
        },
    )


def test_cache_ttl_and_reasoning_use_the_gateway_calculator_without_double_counting() -> None:
    _register(
        {
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "cache_read_input_token_cost": 0.1,
            "cache_creation_input_token_cost": 3,
            "cache_creation_input_token_cost_above_1hr": 7,
            "output_cost_per_reasoning_token": 5,
        }
    )
    call: Final = _call(
        {
            "gen_ai.usage.cache_read.input_tokens": "40",
            "gen_ai.usage.cache_write.input_tokens": "20",
            "anthropic.usage.cache_creation.ephemeral_5m_input_tokens": "15",
            "anthropic.usage.cache_creation.ephemeral_1h_input_tokens": "5",
            "gen_ai.usage.reasoning.output_tokens": "4",
        }
    )
    gateway: Final = completion_cost(
        model=MODEL,
        custom_llm_provider="openai",
        completion_response=ModelResponse(
            model=MODEL,
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                reasoning_tokens=4,
                prompt_tokens_details=PromptTokensDetailsWrapper(
                    cached_tokens=40,
                    cache_write_tokens=20,
                    cache_creation_token_details=CacheCreationTokenDetails(
                        ephemeral_5m_input_tokens=15, ephemeral_1h_input_tokens=5
                    ),
                ),
            ),
        ),
    )
    assert trace_cost(call) == gateway == pytest.approx(40 * 1 + 40 * 0.1 + 15 * 3 + 5 * 7 + 6 * 2 + 4 * 5)


@pytest.mark.parametrize(
    "rates,attributes",
    (
        ({"cache_read_input_token_cost": 0.1}, {}),
        ({"cache_creation_input_token_cost": 3}, {}),
        ({"output_cost_per_reasoning_token": 5}, {}),
        (
            {"cache_creation_input_token_cost": 3, "cache_creation_input_token_cost_above_1hr": 7},
            {"gen_ai.usage.cache_write.input_tokens": "20"},
        ),
    ),
)
def test_missing_usage_that_changes_the_bill_remains_unknown(
    rates: Mapping[str, float], attributes: Mapping[str, str]
) -> None:
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2, **rates})
    assert trace_cost(_call(attributes)) is None


def test_explicit_zero_breakdowns_allow_a_model_with_differentiated_rates() -> None:
    _register(
        {
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "cache_read_input_token_cost": 0.1,
            "cache_creation_input_token_cost": 3,
            "cache_creation_input_token_cost_above_1hr": 7,
            "output_cost_per_reasoning_token": 5,
        }
    )
    assert trace_cost(
        _call(
            {
                "gen_ai.usage.cache_read.input_tokens": "0",
                "gen_ai.usage.cache_write.input_tokens": "0",
                "gen_ai.usage.reasoning.output_tokens": "0",
            }
        )
    ) == pytest.approx(100 * 1 + 10 * 2)


@pytest.mark.parametrize(
    "attributes",
    (
        {"gen_ai.usage.input_tokens": "-1"},
        {"gen_ai.usage.output_tokens": "true"},
        {"gen_ai.usage.input_tokens": "NaN"},
        {"gen_ai.usage.input_tokens": "1.0"},
        {"gen_ai.usage.input_tokens": str(2**32)},
        {"gen_ai.usage.prompt_tokens": "99"},
        {"gen_ai.usage.total_tokens": "100"},
        {"gen_ai.usage.cache_read.input_tokens": "101"},
        {"gen_ai.usage.cache_read.input_tokens": "90", "gen_ai.usage.cache_write.input_tokens": "20"},
        {"gen_ai.usage.reasoning.output_tokens": "11"},
        {
            "gen_ai.usage.cache_write.input_tokens": "20",
            "anthropic.usage.cache_creation.ephemeral_1h_input_tokens": "5",
        },
        {"anthropic.usage.cache_creation.ephemeral_1h_input_tokens": "5"},
        {"gen_ai.usage.audio.input_tokens": "1"},
        {"gen_ai.usage.image.output_tokens": "bad"},
        {"gen_ai.usage.future_meter": "2"},
        {"gen_ai.operation.name": "embeddings"},
        {"gen_ai.output.type": "image"},
        {"gen_ai.system": "anthropic"},
        {"openai.response.service_tier": "unsupported"},
    ),
)
def test_conflicting_or_unrepresented_usage_is_never_priced(attributes: Mapping[str, str]) -> None:
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2})
    assert trace_cost(_call(attributes)) is None


@pytest.mark.parametrize("missing", ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"))
def test_absent_required_counts_do_not_become_zero(missing: str) -> None:
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2})
    complete: Final = _call({})
    assert (
        trace_cost(
            TraceCostInput(
                start_ns=START_NS,
                attributes={key: value for key, value in complete.attributes.items() if key != missing},
            )
        )
        is None
    )


@pytest.mark.parametrize(
    "rates,expected",
    (
        ({"input_cost_per_token": 0, "output_cost_per_token": 0}, 0),
        ({"input_cost_per_token": "0", "output_cost_per_token": "0"}, 0),
        ({}, None),
        ({"output_cost_per_token": 2}, None),
        ({"input_cost_per_token": float("nan"), "output_cost_per_token": 2}, None),
        ({"input_cost_per_token": -1, "output_cost_per_token": 2}, None),
        ({"input_cost_per_token": False, "output_cost_per_token": 2}, None),
        ({"input_cost_per_token": "invalid", "output_cost_per_token": 2}, None),
    ),
)
def test_only_declared_finite_prices_can_establish_free_usage(
    rates: Mapping[str, object], expected: float | None
) -> None:
    _register(rates)
    assert trace_cost(_call({})) == expected
    assert trace_cost(_call({"gen_ai.usage.input_tokens": "0", "gen_ai.usage.output_tokens": "0"})) == expected


def test_served_model_and_tier_win_over_the_request() -> None:
    _register({"input_cost_per_token": 100, "output_cost_per_token": 200})
    served: Final = "openai/lens-test-served"
    _register(
        {
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "input_cost_per_token_priority": 3,
            "output_cost_per_token_priority": 4,
        },
        served,
    )
    assert trace_cost(
        _call(
            {
                "gen_ai.response.model": served,
                "openai.request.service_tier": "auto",
                "openai.response.service_tier": "priority",
            }
        )
    ) == pytest.approx(100 * 3 + 10 * 4)
    assert trace_cost(_call({"gen_ai.response.model": "openai/lens-unknown-served"})) is None


@pytest.mark.parametrize("hour,expected", ((18, 100 * 0.5 + 10 * 1), (12, 100 * 1 + 10 * 2)))
def test_trace_timestamp_selects_existing_off_peak_pricing(hour: int, expected: float) -> None:
    _register(
        {
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "off_peak_pricing": {"hours_utc": "16:00-20:00", "input_cost_per_token": 0.5, "output_cost_per_token": 1},
        }
    )
    instant: Final = datetime(2026, 1, 1, hour, tzinfo=timezone.utc)
    timestamp: Final = int(instant.timestamp()) * 1_000_000_000
    with pinned_billing_time(datetime(2026, 1, 2, 1, tzinfo=timezone.utc)):
        assert trace_cost(_call({}, timestamp)) == pytest.approx(expected)


def test_optional_audio_and_image_rates_do_not_disqualify_text_usage() -> None:
    _register(
        {
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "input_cost_per_audio_token": 8,
            "output_cost_per_image": 7,
            "supports_vision": True,
        }
    )
    assert trace_cost(_call({"gen_ai.usage.audio.input_tokens": "0"})) == pytest.approx(100 * 1 + 10 * 2)


@pytest.mark.parametrize("rate", (0, 3))
def test_tier_only_catalog_prices_use_the_existing_tier_selector(rate: float) -> None:
    _register(
        {
            "tiered_pricing": [
                {"range": [0, 50], "input_cost_per_token": 1, "output_cost_per_token": 2},
                {"range": [50, 1000], "input_cost_per_token": rate, "output_cost_per_token": rate},
            ],
        }
    )
    assert trace_cost(_call({})) == pytest.approx(100 * rate + 10 * rate)


def test_served_text_model_does_not_erase_request_route_search_charges() -> None:
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2, "input_cost_per_query": 3})
    served: Final = "openai/lens-served-search"
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2}, served)
    assert trace_cost(_call({"gen_ai.response.model": served})) is None


@pytest.mark.parametrize(
    "unsupported",
    (
        {"regional_processing_uplift_multiplier_us": 1.1},
        {"regional_processing_uplift_multiplier_eu": 1.1},
        {"regional_endpoint_uplift_multiplier": 1.1},
        {"provider_specific_entry": {"us": 1.1}},
        {"provider_specific_entry": {"fast": 6}},
        {"citation_cost_per_token": 3},
        {"code_interpreter_cost_per_session": 4},
        {"input_cost_per_character": 5},
        {"cost_per_second": 6},
        {"input_cost_per_query": 7},
        {"input_cost_per_request": 8},
    ),
)
def test_missing_billing_dimensions_cannot_be_priced_as_plain_text(unsupported: Mapping[str, object]) -> None:
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2, **unsupported})
    served: Final = "openai/lens-plain-served"
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2}, served)
    assert trace_cost(_call({})) is None
    assert trace_cost(_call({"gen_ai.response.model": served})) is None


def test_identity_multipliers_do_not_require_extra_routing_evidence() -> None:
    _register(
        {
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "regional_processing_uplift_multiplier_us": 1,
            "provider_specific_entry": {"us": 1, "fast": 1},
        }
    )
    assert trace_cost(_call({})) == pytest.approx(100 * 1 + 10 * 2)


@pytest.mark.parametrize("provider", ("azure", "azure_ai"))
@pytest.mark.parametrize("telemetry", ("native", "legacy", "missing", "middleware"))
def test_served_model_cannot_drop_the_azure_router_fee(provider: str, telemetry: str) -> None:
    router: Final = f"{provider}/model_router"
    served_provider: Final = provider if telemetry in ("native", "legacy") else "openai"
    served: Final = f"{provider}/lens-served-model" if telemetry in ("native", "legacy") else "lens-served-model"
    litellm.register_model(
        {
            router: {
                "litellm_provider": provider,
                "mode": "chat",
                "input_cost_per_token": 3,
                "output_cost_per_token": 0,
            },
            served: {
                "litellm_provider": served_provider,
                "mode": "chat",
                "input_cost_per_token": 1,
                "output_cost_per_token": 2,
            },
        }
    )
    identity: Final = {
        "native": {"gen_ai.provider.name": provider},
        "legacy": {"gen_ai.system": provider},
        "missing": {},
        "middleware": {"gen_ai.provider.name": "litellm", "gen_ai.system": "litellm.chat"},
    }[telemetry]
    call: Final = TraceCostInput(
        start_ns=START_NS,
        attributes={
            "gen_ai.request.model": router,
            "gen_ai.response.model": served,
            "gen_ai.usage.input_tokens": "100",
            "gen_ai.usage.output_tokens": "10",
            **identity,
        },
    )
    assert trace_cost(call) is None


def test_direct_moonshot_usage_retains_cache_aware_pricing() -> None:
    model: Final = "moonshot/lens-supported-text"
    litellm.register_model(
        {
            model: {
                "litellm_provider": "moonshot",
                "mode": "chat",
                "input_cost_per_token": 1,
                "output_cost_per_token": 2,
                "cache_read_input_token_cost": 0.1,
            }
        }
    )
    assert trace_cost(
        _call(
            {
                "gen_ai.request.model": model,
                "gen_ai.provider.name": "moonshot",
                "gen_ai.usage.cache_read.input_tokens": "40",
                "gen_ai.usage.cache_write.input_tokens": "20",
            }
        )
    ) == pytest.approx(60 * 1 + 40 * 0.1 + 10 * 2)


@pytest.mark.parametrize(
    "provider,catalog_provider",
    (
        ("openai", "openai"),
        ("text-completion-openai", "text-completion-openai"),
        ("azure", "azure"),
        ("azure_ai", "azure_ai"),
        ("anthropic", "anthropic"),
        ("bedrock", "bedrock"),
        ("bedrock_converse", "bedrock_converse"),
        ("vertex_ai", "vertex_ai"),
        ("vertex_ai_beta", "vertex_ai"),
        ("gemini", "gemini"),
        ("cohere", "cohere"),
        ("cohere_chat", "cohere_chat"),
        ("mistral", "mistral"),
        ("deepseek", "deepseek"),
        ("groq", "groq"),
        ("perplexity", "perplexity"),
        ("xai", "xai"),
        ("watsonx", "watsonx"),
        ("moonshot", "moonshot"),
    ),
)
@pytest.mark.parametrize("form", ("standard", "native", "both"))
def test_semantic_provider_names_and_native_systems_use_the_catalog_identity(
    provider: str, catalog_provider: str, form: str
) -> None:
    model: Final = "lens-provider-identity"
    litellm.register_model(
        {
            f"{catalog_provider}/{model}": {
                "litellm_provider": catalog_provider,
                "mode": "chat",
                "input_cost_per_token": 1,
                "output_cost_per_token": 2,
                "cache_read_input_token_cost": 1,
            }
        }
    )
    assert trace_cost(
        _call(
            {
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider if form == "native" else resolve_provider(provider),
                **({"gen_ai.system": provider} if form == "both" else {}),
            }
        )
    ) == pytest.approx(100 * 1 + 10 * 2)


@pytest.mark.parametrize(
    "qualified,provider,system,expected",
    (
        (False, "aws.bedrock", None, None),
        (True, "aws.bedrock", None, 120),
        (False, "bedrock", None, 120),
        (False, "aws.bedrock", "bedrock", 120),
        (False, "aws.bedrock", "bedrock_converse", 340),
    ),
)
def test_collapsed_provider_aliases_need_unambiguous_catalog_evidence(
    qualified: bool, provider: str, system: str | None, expected: float | None
) -> None:
    model: Final = "lens-provider-ambiguity"
    litellm.register_model(
        {
            f"bedrock/{model}": {
                "litellm_provider": "bedrock",
                "mode": "chat",
                "input_cost_per_token": 1,
                "output_cost_per_token": 2,
            },
            f"bedrock_converse/{model}": {
                "litellm_provider": "bedrock_converse",
                "mode": "chat",
                "input_cost_per_token": 3,
                "output_cost_per_token": 4,
            },
        }
    )
    result: Final = trace_cost(
        _call(
            {
                "gen_ai.request.model": f"bedrock/{model}" if qualified else model,
                "gen_ai.provider.name": provider,
                **({"gen_ai.system": system} if system is not None else {}),
            }
        )
    )
    assert result == expected


@pytest.mark.parametrize("family", (("openai", "text-completion-openai"), ("cohere", "cohere_chat")))
def test_standard_names_that_are_also_native_cannot_choose_between_catalog_variants(family: tuple[str, str]) -> None:
    model: Final = "lens-standard-native-ambiguity"
    litellm.register_model(
        {
            f"{provider}/{model}": {
                "litellm_provider": provider,
                "mode": "chat",
                "input_cost_per_token": index + 1,
                "output_cost_per_token": index + 2,
            }
            for index, provider in enumerate(family)
        }
    )
    assert trace_cost(
        _call({"gen_ai.request.model": model, "gen_ai.provider.name": resolve_provider(family[0])})
    ) is None


def test_vertex_catalog_subtypes_resolve_through_existing_provider_compatibility() -> None:
    model: Final = "lens-vertex-catalog-subtype"
    litellm.register_model(
        {
            model: {
                "litellm_provider": "vertex_ai-language-models",
                "mode": "chat",
                "input_cost_per_token": 1,
                "output_cost_per_token": 2,
                "cache_read_input_token_cost": 1,
            }
        }
    )
    assert trace_cost(_call({"gen_ai.request.model": model, "gen_ai.provider.name": "gcp.vertex_ai"})) == 120


def test_explicit_native_catalog_registration_wins_over_the_unqualified_model() -> None:
    model: Final = "lens-native-catalog-precedence"
    litellm.register_model(
        {
            model: {
                "litellm_provider": "bedrock",
                "mode": "chat",
                "input_cost_per_token": 1,
                "output_cost_per_token": 2,
            },
            f"bedrock_converse/{model}": {
                "litellm_provider": "bedrock_converse",
                "mode": "chat",
                "input_cost_per_token": 3,
                "output_cost_per_token": 4,
            },
        }
    )
    assert trace_cost(
        _call(
            {
                "gen_ai.request.model": model,
                "gen_ai.provider.name": "aws.bedrock",
                "gen_ai.system": "bedrock_converse",
            }
        )
    ) == 340


@pytest.mark.parametrize(
    "recorded,system", (("aws.bedrock", "openai"), ("openai", "bedrock"), ("not-a-provider", "not-a-provider"))
)
def test_provider_conflicts_and_invalid_names_never_guess_a_catalog_route(recorded: str, system: str) -> None:
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2})
    assert trace_cost(_call({"gen_ai.provider.name": recorded, "gen_ai.system": system})) is None


def test_current_registered_prices_are_reused_without_per_call_catalog_mutation() -> None:
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2})
    before: Final = dict(litellm.model_cost[MODEL])
    assert trace_cost(_call({})) == pytest.approx(100 * 1 + 10 * 2)
    assert litellm.model_cost[MODEL] == before
    _register({"input_cost_per_token": 3, "output_cost_per_token": 4})
    assert trace_cost(_call({})) == pytest.approx(100 * 3 + 10 * 4)


def test_batch_keeps_zero_unknown_and_paid_costs_in_input_order() -> None:
    _register({"input_cost_per_token": 1, "output_cost_per_token": 2})
    request: Final = TraceCostsRequest(
        calls=(
            _call({"gen_ai.usage.input_tokens": "0", "gen_ai.usage.output_tokens": "0"}),
            _call({"gen_ai.request.model": "openai/lens-unknown"}),
            _call({}),
        )
    )
    assert trace_costs(request).costs == (0, None, 120)
    assert trace_costs(TraceCostsRequest(calls=())).costs == ()


@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.parametrize("provider", ("huggingface", "ollama", "ollama_chat", "lemonade"))
@pytest.mark.parametrize("registered", (False, True))
def test_dynamic_provider_metadata_is_never_fetched_to_price_a_trace(
    monkeypatch: pytest.MonkeyPatch, provider: str, registered: bool
) -> None:
    import respx

    model: Final = f"{provider}/lens-dynamic-model"
    if registered:
        monkeypatch.setitem(
            litellm.model_cost,
            model,
            {
                "litellm_provider": provider,
                "mode": "chat",
                "input_cost_per_token": 1,
                "output_cost_per_token": 2,
            },
        )
    with respx.mock(assert_all_called=False) as network:
        network.route().respond(200, json={})
        result: Final = trace_cost(_call({"gen_ai.request.model": model, "gen_ai.provider.name": provider}))
        assert len(network.calls) == 0
    assert result is None


@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.parametrize(
    "recorded,system,catalog,expected",
    (
        ("openai", "openai", "openai", 120),
        ("aws.bedrock", "bedrock", "bedrock", 120),
        ("gcp.vertex_ai", "vertex_ai_beta", "vertex_ai", 120),
        ("not-a-provider", "not-a-provider", "openai", None),
    ),
)
def test_catalog_identity_and_recorded_usage_need_no_provider_auth_or_network(
    monkeypatch: pytest.MonkeyPatch, recorded: str, system: str, catalog: str, expected: float | None
) -> None:
    from unittest.mock import Mock

    import respx

    from litellm.proxy.lens import trace_costs as pricing

    model: Final = "lens-offline-provider"
    litellm.register_model(
        {
            f"{catalog}/{model}": {
                "litellm_provider": catalog,
                "mode": "chat",
                "input_cost_per_token": 1,
                "output_cost_per_token": 2,
            }
        }
    )
    provider_auth: Final = Mock(side_effect=AssertionError("Provider authentication is not a pricing operation"))
    calculator: Final = Mock(wraps=pricing.completion_cost)
    monkeypatch.setattr(litellm, "get_llm_provider", provider_auth)
    monkeypatch.setattr(pricing, "completion_cost", calculator)
    with respx.mock(assert_all_called=False) as network:
        network.route().respond(200, json={})
        result: Final = trace_cost(
            _call(
                {
                    "gen_ai.request.model": model,
                    "gen_ai.provider.name": recorded,
                    "gen_ai.system": system,
                }
            )
        )
        assert len(network.calls) == 0
    assert provider_auth.call_count == 0
    assert calculator.call_count == (0 if expected is None else 1)
    assert result == expected


@pytest.mark.parametrize(
    "payload",
    (
        {"calls": [{"start_ns": True, "attributes": {}}]},
        {"calls": [{"start_ns": 2**63, "attributes": {}}]},
        {"calls": [{"start_ns": 0, "attributes": {"gen_ai.usage.input_tokens": 1}}]},
        {"calls": [{"start_ns": 0, "attributes": {"gen_ai.request.model": "x" * 513}}]},
        {"calls": [{"start_ns": 0, "attributes": {"gen_ai.usage." + "x" * 128: "0"}}]},
        {"calls": [{"start_ns": 0, "attributes": {"http.url": "https://example.invalid"}}]},
        {"calls": [{"start_ns": 0, "attributes": {"gen_ai.request.model": "x"}, "rate": 2}]},
        {"calls": [{"start_ns": 0, "attributes": {f"gen_ai.usage.{index}": "0" for index in range(65)}}]},
        {"calls": [{"start_ns": 0, "attributes": {}}] * 129},
    ),
)
def test_wire_rejects_unbounded_or_non_pricing_payloads(payload: Mapping[str, object]) -> None:
    with pytest.raises(ValidationError):
        TraceCostsRequest.model_validate(payload)
