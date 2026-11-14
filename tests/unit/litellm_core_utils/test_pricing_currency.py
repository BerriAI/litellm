import asyncio
from collections.abc import Mapping
from decimal import Decimal
from typing import Final

import pydantic
import pytest

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.pricing_currency import (
    PRICING_SNAPSHOT,
    PRICING_SOURCE,
    USD_PRICING,
    NormalizedPricing,
    PricingCurrencyError,
    frozen_pricing_snapshot,
    normalize_deployment_pricing,
    pricing_configuration_error,
    request_pricing_currency_error,
    routed_pricing_snapshot,
    source_currency_view,
)
from litellm.router_api import ChangeSet
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.types.utils import MONETARY_PRICING_FIELDS, NON_MONETARY_PRICING_FIELDS, CustomPricingLiteLLMParams

EUR_RATE: Final = "0.5"
HUF_RATE: Final = "0.0025"
USD_INPUT: Final = 2e-06
USD_OUTPUT: Final = 4e-06
MESSAGES: Final = [{"role": "user", "content": "price this"}]


def _normalized(litellm_params: Mapping[str, object], model_info: Mapping[str, object]) -> NormalizedPricing:
    result: Final = normalize_deployment_pricing(litellm_params, model_info)
    assert isinstance(result, NormalizedPricing), result
    return result


def _error(
    litellm_params: Mapping[str, object], model_info: Mapping[str, object] | None = None
) -> PricingCurrencyError:
    result: Final = normalize_deployment_pricing(litellm_params, model_info or {})
    assert isinstance(result, PricingCurrencyError), result
    return result


def _sample_price(field: str) -> object:
    match field:
        case "tiered_pricing":
            return [{"range": [0, 1000], "input_cost_per_token": 4.0, "output_cost_per_token": None}]
        case "off_peak_pricing":
            return {"hours_utc": "00:00-06:00", "input_cost_per_token": 4.0}
        case "output_cost_per_video_clip":
            return {"720p": {"5s": 4.0}}
        case "search_context_cost_per_query" | "search_queries_cost_per_query":
            return {"search_context_size_low": 4.0}
        case _:
            return 4.0


def _halved(field: str) -> object:
    match field:
        case "tiered_pricing":
            return [{"range": [0, 1000], "input_cost_per_token": 2.0, "output_cost_per_token": None}]
        case "off_peak_pricing":
            return {"hours_utc": "00:00-06:00", "input_cost_per_token": 2.0}
        case "output_cost_per_video_clip":
            return {"720p": {"5s": 2.0}}
        case "search_context_cost_per_query" | "search_queries_cost_per_query":
            return {"search_context_size_low": 2.0}
        case _:
            return 2.0


@pytest.mark.parametrize("field", sorted(MONETARY_PRICING_FIELDS))
def test_every_declared_price_field_is_converted_with_its_settings_kept(field: str) -> None:
    result: Final = _normalized(
        {"pricing_currency": "EUR", "usd_per_currency_unit": EUR_RATE}, {field: _sample_price(field)}
    )
    assert result.model_info[field] == _halved(field)
    assert result.source is not None
    assert result.source.model_info == {field: _sample_price(field)}


def test_every_custom_pricing_field_is_either_a_price_or_explicitly_not_money() -> None:
    unclassified: Final = sorted(
        field
        for field in CustomPricingLiteLLMParams.model_fields
        if field not in MONETARY_PRICING_FIELDS and field not in NON_MONETARY_PRICING_FIELDS
    )
    assert unclassified == ["output_vector_size"]


@pytest.mark.parametrize("field", [*sorted(NON_MONETARY_PRICING_FIELDS), "output_vector_size", "max_tokens"])
def test_multipliers_and_other_numbers_are_never_converted(field: str) -> None:
    result: Final = _normalized(
        {"pricing_currency": "HUF", "usd_per_currency_unit": HUF_RATE, "input_cost_per_token": 0.0008, field: 1.1},
        {},
    )
    assert result.litellm_params[field] == 1.1
    assert result.litellm_params["input_cost_per_token"] == pytest.approx(USD_INPUT, rel=1e-15)


@pytest.mark.parametrize(
    ("currency", "rate", "input_price", "output_price"),
    [("USD", None, USD_INPUT, USD_OUTPUT), ("EUR", EUR_RATE, 4e-06, 8e-06), ("HUF", HUF_RATE, 0.0008, 0.0016)],
)
def test_equivalent_prices_in_each_currency_normalize_to_the_same_usd(
    currency: str, rate: str | None, input_price: float, output_price: float
) -> None:
    result: Final = _normalized(
        {
            "pricing_currency": currency,
            "usd_per_currency_unit": rate,
            "input_cost_per_token": input_price,
            "output_cost_per_token": output_price,
        },
        {"cache_read_input_token_cost": input_price / 10},
    )
    assert result.litellm_params["input_cost_per_token"] == USD_INPUT
    assert result.litellm_params["output_cost_per_token"] == USD_OUTPUT
    assert result.model_info["cache_read_input_token_cost"] == pytest.approx(USD_INPUT / 10, rel=1e-15)
    assert "pricing_currency" not in result.litellm_params and "usd_per_currency_unit" not in result.litellm_params


def test_explicit_usd_is_identity_and_records_no_provenance() -> None:
    tiers: Final = [{"range": [0, 10], "input_cost_per_token": 0.1}]
    result: Final = _normalized({"pricing_currency": "USD", "input_cost_per_token": 0.3, "tiered_pricing": tiers}, {})
    assert result.litellm_params["input_cost_per_token"] == 0.3
    assert result.litellm_params["tiered_pricing"] is tiers
    assert result.source is None and PRICING_SOURCE not in result.model_info


def test_a_zero_price_stays_zero_and_a_tiny_price_stays_positive() -> None:
    result: Final = _normalized(
        {
            "pricing_currency": "HUF",
            "usd_per_currency_unit": HUF_RATE,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 1e-12,
            "output_cost_per_second": 1_000_000,
        },
        {},
    )
    assert result.litellm_params["input_cost_per_token"] == 0.0
    assert result.litellm_params["output_cost_per_token"] == float(Decimal("1e-12") * Decimal(HUF_RATE))
    assert result.litellm_params["output_cost_per_second"] == 2500.0


@pytest.mark.parametrize(
    ("litellm_params", "kind"),
    [
        ({"pricing_currency": "GBP", "usd_per_currency_unit": 1.2}, "unsupported_currency"),
        ({"pricing_currency": "eur", "usd_per_currency_unit": 1.2}, "unsupported_currency"),
        ({"pricing_currency": "EUR"}, "invalid_rate"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": 0}, "invalid_rate"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": -1.1}, "invalid_rate"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": "nan"}, "invalid_rate"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": "Infinity"}, "invalid_rate"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": True}, "invalid_rate"),
        ({"usd_per_currency_unit": 1.1}, "usd_rate_not_one"),
        ({"pricing_currency": "USD", "usd_per_currency_unit": "0.9"}, "usd_rate_not_one"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": 1.1, "input_cost_per_token": -1e-6}, "invalid_price"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": 1.1, "input_cost_per_token": "cheap"}, "invalid_price"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": 1.1, "input_cost_per_token": True}, "invalid_price"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": 1.1, "input_cost_per_token": "NaN"}, "invalid_price"),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": 10, "input_cost_per_token": 1e308}, "out_of_range"),
        (
            {"pricing_currency": "EUR", "usd_per_currency_unit": "1e999999", "input_cost_per_token": "10"},
            "out_of_range",
        ),
        (
            {"pricing_currency": "HUF", "usd_per_currency_unit": "1e-300", "output_cost_per_token": 1e-300},
            "out_of_range",
        ),
        (
            {"pricing_currency": "HUF", "usd_per_currency_unit": "1e-999999", "output_cost_per_token": "1e-999999"},
            "out_of_range",
        ),
    ],
)
def test_invalid_currency_settings_and_prices_are_refused_with_a_reason(
    litellm_params: Mapping[str, object], kind: str
) -> None:
    assert _error(litellm_params).kind == kind


def test_the_same_rate_written_differently_in_both_sections_is_not_a_conflict() -> None:
    result: Final = _normalized(
        {"pricing_currency": "EUR", "usd_per_currency_unit": 1.2, "input_cost_per_token": 1e-06},
        {"usd_per_currency_unit": "1.20"},
    )
    assert result.litellm_params["input_cost_per_token"] == pytest.approx(1.2e-06, rel=1e-15)


def test_a_price_table_with_non_text_keys_is_refused_with_a_reason() -> None:
    error: Final = _error(
        {"pricing_currency": "EUR", "usd_per_currency_unit": 1.1}, {"output_cost_per_video_clip": {720: {5: 0.1}}}
    )
    assert error.kind == "invalid_price"


def test_conflicting_currency_between_sections_is_refused() -> None:
    error: Final = _error({"pricing_currency": "EUR", "usd_per_currency_unit": 1.1}, {"pricing_currency": "HUF"})
    assert error.kind == "conflicting_currency"


@pytest.mark.parametrize(
    "model_info",
    [
        {"input_cost_per_token_above_9000k_tokens": 1.0},
        {"tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 1.0, "discount_factor": 0.5}]},
        {"off_peak_pricing": {"hours_utc": "00:00-01:00", "night_rate": 1.0}},
    ],
)
def test_a_price_the_schema_does_not_name_is_refused_instead_of_billed_unconverted(
    model_info: Mapping[str, object],
) -> None:
    assert _error({"pricing_currency": "EUR", "usd_per_currency_unit": 1.1}, model_info).kind == "unsupported_field"


def test_nested_tables_keep_their_settings_and_unset_rates() -> None:
    windows: Final = [{"hours_utc": "00:00-24:00", "weekdays": ["sat"]}]
    result: Final = _normalized(
        {
            "pricing_currency": "EUR",
            "usd_per_currency_unit": EUR_RATE,
            "tiered_pricing": [
                {"range": [0, 32000], "input_cost_per_token": 2.0, "cache_read_input_token_cost": None},
                {"range": [32000, 128000], "input_cost_per_token": 6.0},
            ],
        },
        {"off_peak_pricing": {"windows": windows, "weekday_timezone": "Europe/Budapest", "output_cost_per_token": 8.0}},
    )
    assert result.litellm_params["tiered_pricing"] == [
        {"range": [0, 32000], "input_cost_per_token": 1.0, "cache_read_input_token_cost": None},
        {"range": [32000, 128000], "input_cost_per_token": 3.0},
    ]
    assert result.model_info["off_peak_pricing"] == {
        "windows": windows,
        "weekday_timezone": "Europe/Budapest",
        "output_cost_per_token": 4.0,
    }


def _eur_deployment() -> NormalizedPricing:
    return _normalized(
        {"pricing_currency": "EUR", "usd_per_currency_unit": EUR_RATE, "input_cost_per_token": 4e-06},
        {"output_cost_per_token": 8e-06},
    )


def test_rebuilding_a_normalized_deployment_never_converts_twice() -> None:
    first: Final = _eur_deployment()
    again: Final = _normalized(first.litellm_params, first.model_info)
    assert again.litellm_params["input_cost_per_token"] == USD_INPUT
    assert again.model_info["output_cost_per_token"] == USD_OUTPUT
    assert again.source == first.source


@pytest.mark.parametrize(
    "tamper",
    [
        lambda source: {**source, "usd_per_currency_unit": "0.25"},
        lambda source: {**source, "litellm_params": {}, "model_info": {}},
        lambda source: {**source, "model_info": {**source["model_info"], "max_tokens": 10}},
        lambda source: {**source, "currency": "HUF"},
        lambda source: "not a record",
    ],
    ids=["other_rate", "empty", "non_price_key", "other_currency", "malformed"],
)
def test_a_provenance_record_that_does_not_reproduce_the_prices_is_dropped(tamper: object) -> None:
    first: Final = _eur_deployment()
    assert callable(tamper)
    forged_info: Final = {**first.model_info, PRICING_SOURCE: tamper(first.model_info[PRICING_SOURCE])}
    rebuilt: Final = _normalized(first.litellm_params, forged_info)
    assert rebuilt.source is None and PRICING_SOURCE not in rebuilt.model_info
    assert rebuilt.litellm_params["input_cost_per_token"] == USD_INPUT
    assert source_currency_view(first.litellm_params, forged_info)[0] == first.litellm_params


def test_a_provenance_record_missing_a_price_the_deployment_bills_is_dropped() -> None:
    first: Final = _eur_deployment()
    rebuilt: Final = _normalized({**first.litellm_params, "output_cost_per_second": 0.01}, first.model_info)
    assert rebuilt.source is None


def test_a_provenance_record_planted_on_a_usd_deployment_cannot_reinterpret_its_prices() -> None:
    first: Final = _eur_deployment()
    planted: Final = _normalized({"input_cost_per_token": 4e-06}, {PRICING_SOURCE: first.model_info[PRICING_SOURCE]})
    assert planted.source is None
    assert planted.litellm_params["input_cost_per_token"] == 4e-06


def test_the_source_view_round_trips_to_the_same_usd_prices_after_an_unrelated_edit() -> None:
    first: Final = _eur_deployment()
    params, info = source_currency_view(first.litellm_params, first.model_info)
    assert params["input_cost_per_token"] == 4e-06 and params["pricing_currency"] == "EUR"
    assert info["output_cost_per_token"] == 8e-06
    assert info[USD_PRICING] == {"input_cost_per_token": USD_INPUT, "output_cost_per_token": USD_OUTPUT}
    edited: Final = _normalized({**params, "rpm": 10}, {k: v for k, v in info.items() if k != USD_PRICING})
    assert edited.litellm_params["input_cost_per_token"] == USD_INPUT
    assert edited.model_info["output_cost_per_token"] == USD_OUTPUT
    assert edited.source is not None and first.source is not None
    assert edited.source.revision == first.source.revision


def test_configuration_errors_ignore_legacy_usd_and_read_back_projection() -> None:
    assert pricing_configuration_error({"input_cost_per_token": True}, {}) is None
    assert pricing_configuration_error({"pricing_currency": "JPY"}, {}) is not None
    first: Final = _eur_deployment()
    assert pricing_configuration_error(*source_currency_view(first.litellm_params, first.model_info)) is None


@pytest.mark.parametrize(
    ("kwargs", "refused"),
    [
        ({}, False),
        ({"pricing_currency": "USD"}, False),
        ({"pricing_currency": "USD", "usd_per_currency_unit": 1}, False),
        ({"pricing_currency": "EUR", "usd_per_currency_unit": 1.1}, True),
        ({"usd_per_currency_unit": 0.9}, True),
    ],
)
def test_request_level_currency_is_refused_unless_usd(kwargs: Mapping[str, object], refused: bool) -> None:
    assert (request_pricing_currency_error(kwargs) is not None) is refused


def test_a_snapshot_is_only_used_for_the_revision_it_was_captured_for() -> None:
    first: Final = _eur_deployment()
    later: Final = _normalized(
        {"pricing_currency": "EUR", "usd_per_currency_unit": "0.75", "input_cost_per_token": 4e-06}, {}
    )
    captured: Final = frozen_pricing_snapshot({**first.model_info, "input_cost_per_token": USD_INPUT})
    matching: Final = {"metadata": {"model_info": dict(first.model_info), PRICING_SNAPSHOT: captured}}
    stale: Final = {"metadata": {"model_info": dict(later.model_info), PRICING_SNAPSHOT: captured}}
    entry: Final = routed_pricing_snapshot(matching)
    assert entry is not None and entry["input_cost_per_token"] == USD_INPUT
    assert routed_pricing_snapshot(stale) is None
    assert routed_pricing_snapshot({"metadata": {"model_info": dict(first.model_info), PRICING_SNAPSHOT: "{"}}) is None


# --- deployments ------------------------------------------------------------------------------


def test_prices_in_both_sections_keep_litellm_params_precedence() -> None:
    built: Final = Deployment(
        model_name="g",
        litellm_params={  # pyright: ignore[reportArgumentType]  # a config entry
            "model": "openai/currency-demo",
            "pricing_currency": "HUF",
            "usd_per_currency_unit": HUF_RATE,
            "input_cost_per_token": 0.0008,
        },
        model_info={"id": "mixed", "input_cost_per_token": 0.0004, "output_cost_per_second": 400},
    )
    assert built.litellm_params.input_cost_per_token == USD_INPUT
    assert built.model_info.input_cost_per_token == USD_INPUT
    assert built.model_info["output_cost_per_second"] == 1.0
    rebuilt: Final = Deployment(**built.model_dump())
    assert rebuilt.litellm_params.input_cost_per_token == USD_INPUT
    assert rebuilt.model_info["output_cost_per_second"] == 1.0
    params, info = source_currency_view(dict(rebuilt.litellm_params), dict(rebuilt.model_info))
    assert (params["input_cost_per_token"], info["input_cost_per_token"], info["output_cost_per_second"]) == (
        0.0008,
        0.0008,
        400,
    )


@pytest.mark.parametrize("section", [LiteLLM_Params, ModelInfo])
def test_a_boolean_price_beside_a_currency_is_refused_before_float_coercion(section: type[pydantic.BaseModel]) -> None:
    with pytest.raises(pydantic.ValidationError, match="got a boolean for input_cost_per_token"):
        section.model_validate(
            {
                "id": "x",
                "model": "openai/currency-demo",
                "pricing_currency": "EUR",
                "usd_per_currency_unit": 1.1,
                "input_cost_per_token": True,
            }
        )


def test_a_boolean_price_without_a_currency_keeps_the_legacy_coercion() -> None:
    assert (
        LiteLLM_Params.model_validate({"model": "openai/m", "input_cost_per_token": True}).input_cost_per_token == 1.0
    )


@pytest.mark.parametrize("rate", [True, ["0.5"]])
def test_a_non_numeric_rate_is_refused_at_validation(rate: object) -> None:
    with pytest.raises(pydantic.ValidationError):
        LiteLLM_Params.model_validate({"model": "openai/m", "pricing_currency": "EUR", "usd_per_currency_unit": rate})


def test_an_invalid_currency_deployment_fails_router_construction() -> None:
    with pytest.raises(ValueError, match="usd_per_currency_unit must be a finite number above 0"):
        Router(
            model_list=[
                {
                    "model_name": "g",
                    "litellm_params": {
                        "model": "openai/currency-demo",
                        "api_key": "sk-test",
                        "pricing_currency": "EUR",
                        "input_cost_per_token": 1e-6,
                    },
                }
            ]
        )


# --- billing through a router -----------------------------------------------------------------


class _Recorder(CustomLogger):
    def __init__(self, expected: int) -> None:
        super().__init__()
        self.expected = expected
        self.payloads: list[Mapping[str, object]] = []  # mutable-ok: callbacks append as calls finish
        self.done = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:  # noqa: ANN001  # the CustomLogger signature
        self.payloads.append(kwargs["standard_logging_object"])
        if len(self.payloads) >= self.expected:
            self.done.set()


class _HoldFirstCall(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.held = False

    async def async_pre_call_deployment_hook(self, kwargs, call_type):  # noqa: ANN001, ANN201  # the CustomLogger signature
        if not self.held:
            self.held = True
            self.entered.set()
            await self.release.wait()


def _priced(
    group: str, deployment_id: str, **pricing: object
) -> dict[str, object]:  # mutable-ok: a model_list entry  # kwargs-ok: the deployment's prices
    return {
        "model_name": group,
        "litellm_params": {"model": f"openai/{deployment_id}-currency-demo", "api_key": "sk-test", **pricing},
        "model_info": {"id": deployment_id},
    }


def _expected_cost(payload: Mapping[str, object], input_rate: float, output_rate: float) -> float:
    prompt_tokens: Final = payload["prompt_tokens"]
    completion_tokens: Final = payload["completion_tokens"]
    assert isinstance(prompt_tokens, int) and isinstance(completion_tokens, int) and prompt_tokens > 0
    return prompt_tokens * input_rate + completion_tokens * output_rate


@pytest.mark.asyncio
async def test_equivalent_usd_eur_and_huf_deployments_bill_the_same_usd(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: Final = _Recorder(expected=3)
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router: Final = Router(
        model_list=[
            _priced("usd", "usd", input_cost_per_token=USD_INPUT, output_cost_per_token=USD_OUTPUT),
            _priced(
                "eur",
                "eur",
                pricing_currency="EUR",
                usd_per_currency_unit=EUR_RATE,
                input_cost_per_token=4e-06,
                output_cost_per_token=8e-06,
            ),
            _priced(
                "huf",
                "huf",
                pricing_currency="HUF",
                usd_per_currency_unit=HUF_RATE,
                input_cost_per_token=0.0008,
                output_cost_per_token=0.0016,
            ),
        ]
    )
    for group in ("usd", "eur", "huf"):
        await router.acompletion(model=group, messages=MESSAGES, mock_response="the same answer")
    await recorder.done.wait()
    by_group: Final = {str(payload["model_group"]): payload for payload in recorder.payloads}
    for payload in by_group.values():
        assert payload["response_cost"] == pytest.approx(_expected_cost(payload, USD_INPUT, USD_OUTPUT), rel=1e-12)
    breakdowns: Final = {group: payload["cost_breakdown"] for group, payload in by_group.items()}
    assert isinstance(breakdowns["eur"], Mapping) and isinstance(breakdowns["huf"], Mapping)
    assert (breakdowns["eur"]["pricing_currency"], breakdowns["eur"]["usd_per_currency_unit"]) == ("EUR", EUR_RATE)
    assert (breakdowns["huf"]["pricing_currency"], breakdowns["huf"]["usd_per_currency_unit"]) == ("HUF", HUF_RATE)
    assert isinstance(breakdowns["usd"], Mapping) and "pricing_currency" not in breakdowns["usd"]


@pytest.mark.asyncio
async def test_two_deployments_of_one_model_keep_their_own_currency(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: Final = _Recorder(expected=2)
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    shared: Final = {"model": "openai/shared-currency-demo", "api_key": "sk-test", "input_cost_per_token": 4e-06}
    router: Final = Router(
        model_list=[
            {"model_name": "a", "litellm_params": {**shared, "pricing_currency": "EUR", "usd_per_currency_unit": 0.5}},
            {"model_name": "b", "litellm_params": {**shared, "output_cost_per_token": 0.0}},
        ]
    )
    await router.acompletion(model="a", messages=MESSAGES, mock_response="x")
    await router.acompletion(model="b", messages=MESSAGES, mock_response="x")
    await recorder.done.wait()
    by_group: Final = {str(payload["model_group"]): payload for payload in recorder.payloads}
    assert by_group["a"]["response_cost"] == pytest.approx(_expected_cost(by_group["a"], USD_INPUT, 0.0), rel=1e-12)
    assert by_group["b"]["response_cost"] == pytest.approx(_expected_cost(by_group["b"], 4e-06, 0.0), rel=1e-12)


@pytest.mark.asyncio
async def test_a_held_call_bills_the_rate_it_was_routed_under_through_edits_and_a_cost_map_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder: Final = _Recorder(expected=2)
    holder: Final = _HoldFirstCall()
    monkeypatch.setattr(litellm, "callbacks", [holder, recorder])

    def eur(rate: str) -> dict[str, object]:  # mutable-ok: a model_list entry
        return _priced(
            "eur",
            "eur",
            pricing_currency="EUR",
            usd_per_currency_unit=rate,
            input_cost_per_token=4e-06,
            output_cost_per_token=8e-06,
            tiered_pricing=[{"range": [0, 1_000_000], "input_cost_per_token": 4e-06, "output_cost_per_token": 8e-06}],
        )

    router: Final = Router(model_list=[eur(EUR_RATE)])
    held: Final = asyncio.create_task(
        router.acompletion(model="eur", messages=MESSAGES, mock_response="held", litellm_call_id="held-call")
    )
    await holder.entered.wait()
    for rate in ("1", "2", "3", "4", "5"):
        router.apply(ChangeSet(upsert=[Deployment(**eur(rate))]))  # pyright: ignore[reportArgumentType]  # a config entry
    router.replay_model_cost_registrations()
    holder.release.set()
    await held
    await router.acompletion(model="eur", messages=MESSAGES, mock_response="after", litellm_call_id="later-call")
    await recorder.done.wait()
    by_call: Final = {str(payload["id"]): payload for payload in recorder.payloads}
    held_payload, later_payload = by_call["held-call"], by_call["later-call"]
    assert held_payload["response_cost"] == pytest.approx(
        _expected_cost(held_payload, USD_INPUT, USD_OUTPUT), rel=1e-12
    )
    assert later_payload["response_cost"] == pytest.approx(_expected_cost(later_payload, 2e-05, 4e-05), rel=1e-12)
    held_breakdown, later_breakdown = held_payload["cost_breakdown"], later_payload["cost_breakdown"]
    assert isinstance(held_breakdown, Mapping) and isinstance(later_breakdown, Mapping)
    assert (held_breakdown["usd_per_currency_unit"], later_breakdown["usd_per_currency_unit"]) == (EUR_RATE, "5")
    assert held_breakdown["pricing_revision"] != later_breakdown["pricing_revision"]


@pytest.mark.asyncio
async def test_a_direct_call_naming_a_non_usd_currency_is_refused() -> None:
    with pytest.raises(ValueError, match="pricing_currency and usd_per_currency_unit are deployment settings"):
        await litellm.acompletion(
            model="openai/currency-demo",
            messages=MESSAGES,
            mock_response="x",
            input_cost_per_token=1e-6,
            pricing_currency="EUR",
            usd_per_currency_unit=1.1,
        )
