import copy
import itertools
import os
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from queue import Queue
from typing import Final, NoReturn

import pytest

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.litellm_core_utils.get_model_cost_map import _expand_model_aliases, adopt_model_cost_map
from litellm.utils import _invalidate_model_cost_lowercase_map


def _snapshot_model_cost_entries(keys):
    return {key: copy.deepcopy(litellm.model_cost.get(key)) for key in keys}


@pytest.mark.parametrize("stream", (False, True))
def test_sdk_custom_prices_do_not_leak_between_tenants_or_into_deployments(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    model: Final = "openai/m5-tenant-backend"
    deployment_id: Final = "m5-tenant-deployment"
    catalog_row: Final = {"litellm_provider": "openai", "input_cost_per_token": 5.0, "output_cost_per_token": 6.0}
    owned_row: Final = {"litellm_provider": "openai", "input_cost_per_token": 7.0, "output_cost_per_token": 8.0}
    monkeypatch.setitem(litellm.model_cost, model, catalog_row.copy())
    monkeypatch.setitem(litellm.model_cost, deployment_id, owned_row.copy())
    events: Final[Queue[tuple[object, object]]] = Queue()

    def on_success(kwargs: Mapping[str, object], response: object, start_time: datetime, end_time: datetime) -> None:
        if kwargs.get("stream") and kwargs.get("complete_streaming_response") is None:
            return
        events.put((kwargs.get("response_cost"), kwargs.get("complete_streaming_response") or response))

    monkeypatch.setattr(litellm, "success_callback", [on_success])
    for input_rate, output_rate, per_call in (
        (5.0, 6.0, {}),
        (
            1.0,
            4.0,
            {
                "input_cost_per_token": 1.0,
                "output_cost_per_token": 4.0,
                "metadata": {"model_info": {"id": deployment_id}},
            },
        ),
        (5.0, 6.0, {}),
    ):
        result: Final = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            api_key="fake-key",
            mock_response="Hello back",
            stream=stream,
            **per_call,
        )
        if isinstance(result, CustomStreamWrapper):
            assert tuple(result)
        cost, response = events.get(timeout=10)
        assert isinstance(response, litellm.ModelResponse)
        assert isinstance(response.usage, litellm.Usage)
        expected: Final = response.usage.prompt_tokens * input_rate + response.usage.completion_tokens * output_rate
        assert cost == pytest.approx(expected)
        if not stream:
            assert response._hidden_params["response_cost"] == pytest.approx(expected)
        assert litellm.model_cost[model] == catalog_row
        assert litellm.model_cost[deployment_id] == owned_row
    assert litellm.cost_per_token(deployment_id, 10, 20, custom_llm_provider="openai") == pytest.approx((70.0, 160.0))


@pytest.mark.parametrize(
    ("prices", "usage", "service_tier", "expected"),
    (
        (
            {"output_cost_per_reasoning_token": 3.0},
            litellm.Usage(prompt_tokens=10, completion_tokens=10, completion_tokens_details={"reasoning_tokens": 5}),
            "default",
            35.0,
        ),
        (
            {"input_cost_per_audio_token": 3.0, "output_cost_per_audio_token": 4.0},
            litellm.Usage(
                prompt_tokens=10,
                completion_tokens=10,
                prompt_tokens_details={"text_tokens": 6, "audio_tokens": 4},
                completion_tokens_details={"text_tokens": 6, "audio_tokens": 4},
            ),
            "default",
            46.0,
        ),
        (
            {"input_cost_per_token_priority": 3.0, "output_cost_per_token_priority": 4.0},
            litellm.Usage(prompt_tokens=10, completion_tokens=10),
            "priority",
            70.0,
        ),
    ),
    ids=("reasoning", "audio", "priority"),
)
def test_sdk_request_prices_preserve_non_base_token_fields_without_catalog_writes(
    monkeypatch: pytest.MonkeyPatch,
    prices: Mapping[str, float],
    usage: litellm.Usage,
    service_tier: str,
    expected: float,
) -> None:
    model: Final = "openai/m5-request-token-fields"
    row: Final = {"litellm_provider": "openai", "input_cost_per_token": 5.0, "output_cost_per_token": 6.0}
    monkeypatch.setitem(litellm.model_cost, model, row.copy())
    response: Final = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        api_key="fake-key",
        input_cost_per_token=1.0,
        output_cost_per_token=2.0,
        mock_response=litellm.ModelResponse(model=model, usage=usage, service_tier=service_tier),
        **prices,
    )
    assert response._hidden_params["response_cost"] == pytest.approx(expected)
    assert litellm.model_cost[model] == row
    assert litellm.cost_per_token(model, 10, 10, custom_llm_provider="openai") == pytest.approx((50.0, 60.0))


@pytest.mark.parametrize("stream", (False, True))
def test_router_request_prices_override_only_the_current_call(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    model: Final = "openai/m5-routed-request"
    deployment_id: Final = "m5-routed-request-deployment"
    catalog: Final = {"litellm_provider": "openai", "input_cost_per_token": 5.0, "output_cost_per_token": 6.0}
    monkeypatch.setitem(litellm.model_cost, model, catalog.copy())
    events: Final = Queue[tuple[object, object]]()

    class Costs(CustomLogger):
        def log_success_event(
            self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            events.put((kwargs.get("response_cost"), kwargs.get("complete_streaming_response", response_obj)))

    monkeypatch.setattr(litellm, "callbacks", [Costs()])
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "m5-request-group",
                "litellm_params": {"model": model, "api_key": "fake-key"},
                "model_info": {
                    "id": deployment_id,
                    "input_cost_per_token": 7.0,
                    "output_cost_per_token": 8.0,
                },
            }
        ],
        num_retries=0,
    )
    owned: Final = copy.deepcopy(litellm.model_cost[deployment_id])
    try:
        for input_rate, output_rate, per_call in (
            (7.0, 8.0, {}),
            (1.0, 2.0, {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}),
            (0.0, 0.0, {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}),
            (7.0, 8.0, {}),
        ):
            result: Final = router.completion(
                model="m5-request-group",
                messages=[{"role": "user", "content": "hello"}],
                mock_response="Hello back",
                stream=stream,
                **per_call,
            )
            if isinstance(result, CustomStreamWrapper):
                assert tuple(result)
            cost, response = events.get(timeout=10)
            assert isinstance(response, litellm.ModelResponse)
            assert isinstance(response.usage, litellm.Usage)
            assert cost == pytest.approx(
                response.usage.prompt_tokens * input_rate + response.usage.completion_tokens * output_rate
            )
            assert litellm.model_cost[deployment_id] == owned
            assert litellm.model_cost[model] == catalog
    finally:
        router.discard()


def test_request_prices_reach_callback_cache_and_reasoning_breakdown_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.integrations.custom_logger import CustomLogger

    model: Final = "openai/m5-request-breakdown"
    row: Final = {"litellm_provider": "openai", "input_cost_per_token": 5.0, "output_cost_per_token": 6.0}
    monkeypatch.setitem(litellm.model_cost, model, row.copy())
    events: Final = Queue[object]()

    class Costs(CustomLogger):
        def log_success_event(
            self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            events.put(kwargs.get("standard_logging_object"))

    monkeypatch.setattr(litellm, "callbacks", [Costs()])
    response: Final = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        api_key="fake-key",
        input_cost_per_token=1.0,
        output_cost_per_token=2.0,
        cache_read_input_token_cost=0.1,
        cache_creation_input_token_cost=1.25,
        output_cost_per_reasoning_token=3.0,
        model_info={"mode": "chat", "supports_prompt_caching": True, "max_tokens": 8192},
        mock_response=litellm.ModelResponse(
            model=model,
            usage=litellm.Usage(
                prompt_tokens=100,
                completion_tokens=10,
                prompt_tokens_details={"cached_tokens": 60, "cache_creation_tokens": 20},
                completion_tokens_details={"reasoning_tokens": 5},
            ),
        ),
    )
    payload: Final = events.get(timeout=10)
    assert isinstance(payload, dict)
    assert payload["response_cost"] == pytest.approx(76.0)
    assert payload["cost_breakdown"] == pytest.approx(
        {
            "input_cost": 51.0,
            "output_cost": 25.0,
            "total_cost": 76.0,
            "tool_usage_cost": 0.0,
            "service_tier": None,
            "data_residency": None,
            "vertex_location": None,
            "cache_read_cost": 6.0,
            "cache_creation_cost": 25.0,
            "reasoning_cost": 15.0,
            "original_cost": 76.0,
            "discount_percent": 0.0,
            "discount_amount": 0.0,
            "margin_percent": 0.0,
            "margin_fixed_amount": 0.0,
            "margin_total_amount": 0.0,
        }
    )
    assert response._hidden_params["response_cost"] == pytest.approx(76.0)
    assert litellm.model_cost[model] == row


@pytest.mark.parametrize(("cache_price", "expected"), ((None, 0.5), (0.0, 0.0), (0.25, 0.25)))
def test_request_all_cache_usage_retains_unused_rates_and_zero_overrides(
    monkeypatch: pytest.MonkeyPatch, cache_price: float | None, expected: float
) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    model: Final = "openai/m5-request-all-cache"
    row: Final = {
        "litellm_provider": "openai",
        "input_cost_per_token": 5.0,
        "output_cost_per_token": 6.0,
        "cache_read_input_token_cost": 0.5,
    }
    monkeypatch.setitem(litellm.model_cost, model, row.copy())
    when: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    logging_obj: Final = Logging(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        call_type="completion",
        start_time=when,
        litellm_call_id="m5-request-all-cache",
        function_id="f",
        kwargs={
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
            "cache_read_input_token_cost": cache_price,
        },
    )
    with pinned_billing_time(when):
        total: Final = litellm.completion_cost(
            model=model,
            custom_llm_provider="openai",
            custom_pricing=True,
            litellm_logging_obj=logging_obj,
            completion_response=litellm.ModelResponse(
                model=model,
                usage=litellm.Usage(
                    prompt_tokens=100, completion_tokens=0, prompt_tokens_details={"cached_tokens": 100}
                ),
            ),
        )
    assert total == pytest.approx(100 * expected)
    assert logging_obj.billed_token_rates is not None
    assert logging_obj.billed_token_rates.input_cost_per_token == 1.0
    assert logging_obj.billed_token_rates.output_cost_per_token == 2.0
    assert logging_obj.billed_token_rates.cache_read_input_token_cost == expected
    assert litellm.model_cost[model] == row


@pytest.mark.parametrize(("speed", "geo", "factor"), (("standard", "global", 1.0), ("fast", "us", 6.6)))
def test_ephemeral_request_prices_keep_provider_selected_adjustments(
    monkeypatch: pytest.MonkeyPatch, speed: str, geo: str, factor: float
) -> None:
    model: Final = "anthropic/m5-request-adjustments"
    row: Final = {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "input_cost_per_token": 5.0,
        "output_cost_per_token": 6.0,
        "provider_specific_entry": {"fast": 6.0, "us": 1.1},
    }
    monkeypatch.setitem(litellm.model_cost, model, copy.deepcopy(row))
    response: Final = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        api_key="fake-key",
        input_cost_per_token=1.0,
        output_cost_per_token=2.0,
        mock_response=litellm.ModelResponse(
            model=model,
            usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, speed=speed, inference_geo=geo),
        ),
    )
    assert response._hidden_params["response_cost"] == pytest.approx(30.0 * factor)
    assert litellm.model_cost[model] == row


@pytest.mark.parametrize("field", ("input_cost_per_second", "output_cost_per_second", "cost_per_second"))
@pytest.mark.parametrize("rate", (0.0, 0.02))
def test_request_duration_prices_keep_python_time_billing_without_registration(field: str, rate: float) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    model: Final = "openai/m5-request-duration"
    logging_obj: Final = Logging(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="m5-request-duration",
        function_id="f",
        kwargs={field: rate},
    )
    total: Final = litellm.completion_cost(
        model=model,
        custom_llm_provider="openai",
        custom_pricing=True,
        litellm_logging_obj=logging_obj,
        total_time=1500.0,
        completion_response=litellm.ModelResponse(
            model=model, usage=litellm.Usage(prompt_tokens=10, completion_tokens=10)
        ),
    )
    assert total == pytest.approx(rate * 1.5)
    assert logging_obj.cost_breakdown is not None
    assert logging_obj.cost_breakdown["input_cost"] == pytest.approx(rate * 1.5)
    assert logging_obj.cost_breakdown["output_cost"] == 0.0
    assert model not in litellm.model_cost


def _restore_model_cost_entries(original_entries):
    for key, value in original_entries.items():
        if value is None:
            litellm.model_cost.pop(key, None)
        else:
            litellm.model_cost[key] = value
    _invalidate_model_cost_lowercase_map()


def test_register_model_receives_cache_pricing_fields():
    """End-to-end: when register_model is called with a full pricing entry,
    the cache pricing fields should be present in litellm.model_cost."""
    model_key = "openai/test-custom-model-with-cache-pricing"

    litellm.register_model(
        {
            model_key: {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "cache_read_input_token_cost": 0.00025,
                "supports_prompt_caching": True,
                "mode": "chat",
                "max_tokens": 8192,
                "litellm_provider": "openai",
            }
        }
    )

    registered = litellm.model_cost.get(model_key)
    assert registered is not None, f"{model_key} should be in model_cost"
    assert registered["cache_read_input_token_cost"] == 0.00025
    assert registered["supports_prompt_caching"] is True
    assert registered["mode"] == "chat"
    assert registered["max_tokens"] == 8192

    # Cleanup
    litellm.model_cost.pop(model_key, None)


def test_register_model_strips_none_litellm_provider():
    """``get_model_info`` returns ``litellm_provider: None`` for deployments
    registered without a provider (e.g. ``Router.add_deployment`` flows).
    ``register_model`` must not persist that None into ``model_cost``,
    otherwise ``_check_provider_match`` will drop custom pricing on
    subsequent cost lookups.

    Regression test for https://github.com/BerriAI/litellm/issues/28336.
    """
    from litellm.utils import _check_provider_match

    model_key = "test-custom-pricing-no-provider-28336"
    litellm.model_cost.pop(model_key, None)

    try:
        litellm.register_model(
            {
                model_key: {
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                }
            }
        )

        registered = litellm.model_cost.get(model_key)
        assert registered is not None, f"{model_key} should be in model_cost"
        # The key may be absent entirely, but if present it must not be None.
        assert "litellm_provider" not in registered or registered["litellm_provider"] is not None
        # Downstream consumers must accept this entry for any provider,
        # mirroring what the cost calculator does.
        assert _check_provider_match(registered, "openai") is True
        assert _check_provider_match(registered, "anthropic") is True
    finally:
        litellm.model_cost.pop(model_key, None)


def test_register_model_strips_none_provider_from_the_served_row(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.utils import _check_provider_match

    model_key: Final = "test-strip-none-provider-28336"
    row: Final = {"litellm_provider": None, "mode": "chat", "max_tokens": 4096}
    monkeypatch.setitem(litellm.model_cost, model_key, row)

    try:
        litellm.register_model(
            {
                model_key: {
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                }
            }
        )

        registered: Final = litellm.model_cost.get(model_key)
        assert registered is not None, f"{model_key} should be in model_cost"
        assert "litellm_provider" not in registered
        assert registered.get("mode") == "chat"
        assert registered.get("max_tokens") == 4096
        assert registered.get("input_cost_per_token") == 0.001
        assert registered.get("output_cost_per_token") == 0.002
        assert _check_provider_match(registered, "openai") is True
        assert _check_provider_match(registered, "anthropic") is True
    finally:
        litellm.model_cost.pop(model_key, None)


def test_register_model_inherits_builtin_cache_pricing_for_unmapped_key(monkeypatch):
    """Registering a custom override under a key shape that
    ``get_model_info`` cannot resolve (e.g. a triple provider prefix like
    ``bedrock/bedrock/bedrock/us.anthropic.claude-sonnet-4-6``; a double
    prefix now resolves like a routing prefix) must still inherit
    the built-in cache pricing for the underlying model.

    Before the fix ``register_model`` fell back to an empty ``existing_model``
    so the merged entry only carried the fields the user set explicitly
    (input/output cost). ``cache_creation_input_token_cost`` and
    ``cache_read_input_token_cost`` were absent, and the cost calculator
    silently charged 0 for every cache token, dropping the bulk of the bill
    for cache-heavy Anthropic traffic.

    Regression for the cache-pricing dropout under partial overrides.
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    builtin_key = "us.anthropic.claude-sonnet-4-6"
    registered_key = f"bedrock/bedrock/bedrock/{builtin_key}"
    builtin = litellm.model_cost[builtin_key]

    assert builtin["cache_creation_input_token_cost"] > 0
    assert builtin["cache_read_input_token_cost"] > 0

    try:
        litellm.register_model(
            {
                registered_key: {
                    "input_cost_per_token": builtin["input_cost_per_token"],
                    "output_cost_per_token": builtin["output_cost_per_token"],
                    "litellm_provider": "bedrock",
                }
            }
        )

        registered = litellm.model_cost[registered_key]
        assert registered.get("cache_creation_input_token_cost") == builtin["cache_creation_input_token_cost"]
        assert registered.get("cache_read_input_token_cost") == builtin["cache_read_input_token_cost"]
        assert registered["litellm_provider"] == "bedrock"

        usage = Usage(
            prompt_tokens=1100,
            completion_tokens=100,
            total_tokens=1200,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=800,
                text_tokens=100,
            ),
            cache_creation_input_tokens=200,
        )

        input_cost, output_cost = generic_cost_per_token(
            model=registered_key,
            usage=usage,
            custom_llm_provider="bedrock",
        )

        text_only_cost = builtin["input_cost_per_token"] * 100
        expected_input_cost = (
            text_only_cost
            + builtin["cache_read_input_token_cost"] * 800
            + builtin["cache_creation_input_token_cost"] * 200
        )
        assert abs(input_cost - expected_input_cost) < 1e-12
        assert abs(output_cost - builtin["output_cost_per_token"] * 100) < 1e-12
        assert input_cost > text_only_cost + 1e-12
    finally:
        litellm.model_cost.pop(registered_key, None)
        litellm.model_cost = original_model_cost
        os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        from litellm.utils import _invalidate_model_cost_lowercase_map

        _invalidate_model_cost_lowercase_map()


def test_register_model_warns_when_no_builtin_match_for_cache_pricing(caplog):
    """When a custom override is registered under a key that neither
    ``get_model_info`` nor any prefix/region variant can resolve to a
    built-in entry, ``register_model`` must warn that cache cost fields will
    default to 0 instead of silently producing an under-billed entry.
    """
    import logging

    from litellm._logging import verbose_logger

    registered_key = "bedrock/totally-made-up-model-alias-xyz"
    litellm.model_cost.pop(registered_key, None)

    try:
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            litellm.register_model(
                {
                    registered_key: {
                        "input_cost_per_token": 0.001,
                        "output_cost_per_token": 0.002,
                        "litellm_provider": "bedrock",
                    }
                }
            )

        assert any(
            registered_key in record.message and "cache_creation_input_token_cost" in record.message
            for record in caplog.records
        ), "expected a warning naming the unmapped key and the cache cost fields"
    finally:
        litellm.model_cost.pop(registered_key, None)


def test_register_model_no_warning_without_custom_pricing(caplog):
    """LIT-6318: an entry with no custom pricing (e.g. router deployment
    metadata) never drives cost calculation, so registering it under an
    unmatched key must not emit the missing-cache-pricing warning.
    """
    import logging

    from litellm._logging import verbose_logger

    registered_key = "azure/lit6318-deployment-without-pricing"
    litellm.model_cost.pop(registered_key, None)

    try:
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            litellm.register_model(
                {
                    registered_key: {
                        "litellm_provider": "azure",
                        "base_model": "azure/text-embedding-3-large",
                    }
                }
            )

        assert not any("register_model" in record.message for record in caplog.records), (
            "entry without custom pricing must register silently"
        )
    finally:
        litellm.model_cost.pop(registered_key, None)


def test_register_model_no_warning_for_tiered_pricing_without_cache_costs(caplog):
    """LIT-6318: tiered pricing bills cache reads at the tier's input rate when
    cache costs are omitted, so a tiered entry must not trigger the
    cache-defaults-to-0 warning.
    """
    import logging

    from litellm._logging import verbose_logger

    registered_key = "bedrock/lit6318-tiered-priced-model"
    litellm.model_cost.pop(registered_key, None)

    try:
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            litellm.register_model(
                {
                    registered_key: {
                        "litellm_provider": "bedrock",
                        "tiered_pricing": [
                            {
                                "range": [0, 200000],
                                "input_cost_per_token": 1e-06,
                                "output_cost_per_token": 5e-06,
                            }
                        ],
                    }
                }
            )

        assert not any("register_model" in record.message for record in caplog.records), (
            "tiered pricing entry must register silently"
        )
    finally:
        litellm.model_cost.pop(registered_key, None)


def test_router_deployment_without_custom_pricing_registers_silently(caplog):
    """LIT-6318: the router registers every deployment under its hashed id and
    its backend key. Deployments without custom pricing are costed at request
    time from the underlying model name, so startup must not warn about them.
    """
    import logging

    from litellm import Router
    from litellm._logging import verbose_logger

    deployment_model = "azure/lit6318-my-deployment-name"
    deployment_id = "lit6318-no-pricing-deployment"
    snapshot = _snapshot_model_cost_entries([deployment_model, deployment_id])

    try:
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            Router(
                model_list=[
                    {
                        "model_name": "indexing",
                        "litellm_params": {
                            "model": deployment_model,
                            "api_base": "https://example.openai.azure.com",
                            "api_key": "fake-key",
                        },
                        "model_info": {
                            "id": deployment_id,
                            "base_model": "azure/text-embedding-3-large",
                        },
                    }
                ]
            )

        register_warnings = [record.message for record in caplog.records if "register_model" in record.message]
        assert not register_warnings, register_warnings
    finally:
        _restore_model_cost_entries(snapshot)


def test_router_custom_priced_deployment_warning_names_model_not_hash(caplog):
    """LIT-6318: when a custom-priced deployment genuinely lacks cache pricing
    and no built-in entry matches, the warning must name the deployment's
    model rather than its opaque hashed id.
    """
    import logging

    from litellm import Router
    from litellm._logging import verbose_logger

    deployment_model = "bedrock/lit6318-totally-made-up-model"
    deployment_id = "lit6318-custom-priced-deployment-hash"
    snapshot = _snapshot_model_cost_entries([deployment_model, deployment_id])

    try:
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            Router(
                model_list=[
                    {
                        "model_name": "made-up",
                        "litellm_params": {
                            "model": deployment_model,
                            "aws_region_name": "us-east-1",
                            "input_cost_per_token": 1e-06,
                            "output_cost_per_token": 5e-06,
                        },
                        "model_info": {"id": deployment_id},
                    }
                ]
            )

        register_warnings = [record.message for record in caplog.records if "register_model" in record.message]
        assert register_warnings, "expected a warning for missing cache pricing"
        for message in register_warnings:
            assert deployment_id not in message, message
            assert deployment_model in message, message
    finally:
        _restore_model_cost_entries(snapshot)


def test_register_model_router_add_deployment_custom_pricing_applies():
    """End-to-end regression for https://github.com/BerriAI/litellm/issues/28336.

    ``Router.add_deployment`` registers custom pricing without passing
    ``litellm_provider``. Cost calculation must still pick up the custom
    pricing instead of falling back to the default provider price.
    """
    from litellm import Router

    model_key = "router-add-deployment-custom-pricing-28336"
    deployment_model = f"openai/{model_key}"
    litellm.model_cost.pop(model_key, None)
    litellm.model_cost.pop(deployment_model, None)

    router = Router(
        model_list=[
            {
                "model_name": model_key,
                "litellm_params": {
                    "model": deployment_model,
                    "api_key": "fake-key-for-registration",
                    "input_cost_per_token": 0.00042,
                    "output_cost_per_token": 0.00084,
                },
                "model_info": {"id": "deployment-28336"},
            }
        ]
    )

    try:
        # ``add_deployment`` runs as part of ``Router.__init__``; the
        # registered entry must not block ``_check_provider_match`` for
        # the deployment's provider.
        from litellm.utils import _check_provider_match

        registered_keys = [k for k in (deployment_model, model_key) if k in litellm.model_cost]
        assert registered_keys, (
            f"Router.add_deployment did not register custom pricing for {model_key} / {deployment_model}"
        )
        for k in registered_keys:
            assert _check_provider_match(litellm.model_cost[k], "openai") is True, (
                f"custom pricing for {k} was dropped by _check_provider_match"
            )
    finally:
        litellm.model_cost.pop(model_key, None)
        litellm.model_cost.pop(deployment_model, None)
        del router


def test_embedding_router_zero_pricing_does_not_clobber_builtin_pricing():
    """LIT-3991: a router-originated embedding request that carries explicit
    zero custom pricing (e.g. resolved through an ``openai/*`` wildcard
    deployment with ``input_cost_per_token: 0``) must not overwrite the shared
    ``openai/text-embedding-3-small`` entry in ``litellm.model_cost``. Before
    the fix, one call through the wildcard poisoned the shared key and every
    sibling deployment relying on built-in pricing logged $0 until restart.
    """
    shared_key = "openai/text-embedding-3-small"
    deployment_id = "lit3991-wildcard-embed-zero"
    snapshot = _snapshot_model_cost_entries([shared_key, "text-embedding-3-small", deployment_id])
    builtin_input_cost = litellm.get_model_info(model=shared_key)["input_cost_per_token"]
    assert builtin_input_cost > 0

    try:
        litellm.register_model(
            {
                deployment_id: {
                    "litellm_provider": "openai",
                    "mode": "embedding",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                }
            }
        )
        litellm.embedding(
            model=shared_key,
            input=["hello"],
            api_key="fake-key",
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            model_info={"id": deployment_id},
            metadata={"model_info": {"id": deployment_id}},
            mock_response=[0.1, 0.2],
        )

        assert litellm.get_model_info(model=shared_key)["input_cost_per_token"] == builtin_input_cost, (
            "wildcard deployment's zero pricing leaked into the shared model_cost key"
        )
        assert litellm.model_cost[deployment_id]["input_cost_per_token"] == 0.0
        assert litellm.model_cost[deployment_id]["output_cost_per_token"] == 0.0

        sibling_response = litellm.embedding(
            model=shared_key,
            input=["hello"],
            api_key="fake-key",
            mock_response=[0.1, 0.2],
        )
        sibling_cost = litellm.completion_cost(completion_response=sibling_response, call_type="embedding")
        assert sibling_cost == pytest.approx(10 * builtin_input_cost)
    finally:
        _restore_model_cost_entries(snapshot)


def test_embedding_router_custom_pricing_costs_request_via_deployment_id():
    """The request that carries custom pricing must still be costed with that
    pricing (via its deployment id entry), while the shared backend key keeps
    the built-in rate for siblings.
    """
    shared_key = "openai/text-embedding-3-small"
    deployment_id = "lit3991-wildcard-embed-custom"
    override_input_cost = 5e-05
    snapshot = _snapshot_model_cost_entries([shared_key, "text-embedding-3-small", deployment_id])
    builtin_input_cost = litellm.get_model_info(model=shared_key)["input_cost_per_token"]
    assert builtin_input_cost != override_input_cost

    try:
        litellm.register_model(
            {
                deployment_id: {
                    "litellm_provider": "openai",
                    "mode": "embedding",
                    "input_cost_per_token": override_input_cost,
                    "output_cost_per_token": override_input_cost * 2,
                }
            }
        )
        response = litellm.embedding(
            model=shared_key,
            input=["hello"],
            api_key="fake-key",
            input_cost_per_token=override_input_cost,
            output_cost_per_token=override_input_cost * 2,
            model_info={"id": deployment_id},
            metadata={"model_info": {"id": deployment_id}},
            mock_response=[0.1, 0.2],
        )

        request_cost = litellm.completion_cost(
            completion_response=response,
            model=shared_key,
            custom_llm_provider="openai",
            call_type="embedding",
            custom_pricing=True,
            router_model_id=deployment_id,
        )
        assert request_cost == pytest.approx(10 * override_input_cost)
        assert litellm.get_model_info(model=shared_key)["input_cost_per_token"] == builtin_input_cost
    finally:
        _restore_model_cost_entries(snapshot)


def test_completion_router_zero_pricing_does_not_clobber_builtin_pricing():
    """Same isolation as the embedding path, exercised through completion()."""
    shared_key = "openai/gpt-4o-mini"
    deployment_id = "lit3991-wildcard-chat-zero"
    snapshot = _snapshot_model_cost_entries([shared_key, "gpt-4o-mini", deployment_id])
    builtin_input_cost = litellm.get_model_info(model=shared_key)["input_cost_per_token"]
    assert builtin_input_cost > 0

    try:
        litellm.register_model(
            {
                deployment_id: {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                }
            }
        )
        litellm.completion(
            model=shared_key,
            messages=[{"role": "user", "content": "hello"}],
            api_key="fake-key",
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            model_info={"id": deployment_id},
            metadata={"model_info": {"id": deployment_id}},
            mock_response="hello back",
        )

        assert litellm.get_model_info(model=shared_key)["input_cost_per_token"] == builtin_input_cost, (
            "wildcard deployment's zero pricing leaked into the shared model_cost key"
        )
        assert litellm.model_cost[deployment_id]["input_cost_per_token"] == 0.0
    finally:
        _restore_model_cost_entries(snapshot)


def test_embedding_direct_sdk_custom_pricing_does_not_register_shared_key():
    model_key = "openai/lit3991-direct-sdk-embed-model"
    override_input_cost = 3e-05
    try:
        response = litellm.embedding(
            model=model_key,
            input=["hello"],
            api_key="fake-key",
            input_cost_per_token=override_input_cost,
            output_cost_per_token=override_input_cost * 2,
            mock_response=[0.1, 0.2],
        )

        assert model_key not in litellm.model_cost
        assert response._hidden_params["response_cost"] == pytest.approx(10 * override_input_cost)
        cost = litellm.completion_cost(
            completion_response=response,
            model=model_key,
            custom_llm_provider="openai",
            call_type="embedding",
            custom_pricing=True,
            custom_cost_per_token={
                "input_cost_per_token": override_input_cost,
                "output_cost_per_token": override_input_cost * 2,
            },
        )
        assert cost == pytest.approx(10 * override_input_cost)
    finally:
        litellm.model_cost.pop(model_key, None)
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize("route", ["", "converse/", "invoke/"])
def test_register_model_prices_a_geo_alias_from_its_builtin_when_the_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    from litellm.types.utils import ModelResponse, Usage

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    _invalidate_model_cost_lowercase_map()
    canonical_key: Final = "au.anthropic.claude-sonnet-5"
    alias: Final = f"bedrock/{route}{canonical_key}"
    canonical: Final = litellm.model_cost[canonical_key]

    def lookup_tripped_by_a_concurrent_registration(model: str, custom_llm_provider: str | None = None) -> NoReturn:
        raise RuntimeError("dictionary changed size during iteration")

    with pytest.MonkeyPatch.context() as lookup_patch:
        lookup_patch.setattr(litellm.utils, "get_model_info", lookup_tripped_by_a_concurrent_registration)
        litellm.register_model(
            {alias: {"litellm_provider": "bedrock", "mode": "chat", "supports_vision": True}},
            persist_across_reloads=False,
        )

    usage: Final = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)
    alias_cost: Final = litellm.completion_cost(
        completion_response=ModelResponse(model=alias, usage=usage), model=alias, custom_llm_provider="bedrock"
    )
    canonical_cost: Final = litellm.completion_cost(
        completion_response=ModelResponse(model=canonical_key, usage=usage),
        model=canonical_key,
        custom_llm_provider="bedrock",
    )
    try:
        assert alias_cost == canonical_cost > 0
        assert litellm.get_model_info(alias)["input_cost_per_token"] == canonical["input_cost_per_token"]
        assert litellm.get_model_info(alias)["supports_vision"] is True
    finally:
        _invalidate_model_cost_lowercase_map()


def test_register_model_keeps_a_catalog_rows_own_prices_when_the_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    geo_key: Final = "au.anthropic.lit-5853-geo-model"
    geo_row: Final = {
        "litellm_provider": "bedrock_converse",
        "mode": "chat",
        "input_cost_per_token": 3.3e-06,
        "output_cost_per_token": 1.65e-05,
        "cache_read_input_token_cost": 3.3e-07,
    }
    sibling_row: Final = {
        "litellm_provider": "bedrock_converse",
        "mode": "chat",
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_read_input_token_cost": 3e-07,
    }
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.get_model_cost_map(url=""),
            geo_key: dict(geo_row),
            "anthropic.lit-5853-geo-model": sibling_row,
        },
    )
    _invalidate_model_cost_lowercase_map()

    def lookup_tripped_by_a_concurrent_registration(model: str, custom_llm_provider: str | None = None) -> NoReturn:
        raise RuntimeError("dictionary changed size during iteration")

    with pytest.MonkeyPatch.context() as lookup_patch:
        lookup_patch.setattr(litellm.utils, "get_model_info", lookup_tripped_by_a_concurrent_registration)
        litellm.register_model({geo_key: {"mode": "chat"}}, persist_across_reloads=False)

    try:
        for field in ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost"):
            assert litellm.model_cost[geo_key][field] == geo_row[field]
            assert litellm.get_model_info(geo_key)[field] == geo_row[field]
        assert litellm.model_cost[geo_key]["litellm_provider"] == geo_row["litellm_provider"]
    finally:
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize("route", ["", "converse/", "invoke/"])
def test_register_model_keeps_caller_pricing_independent_when_the_lookup_trips_once(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    catalog_key: Final = "au.anthropic.lit-5853-geo-model"
    alias: Final = f"bedrock/{route}{catalog_key}"
    repriced_input_cost: Final = 9e-06
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.get_model_cost_map(url=""),
            catalog_key: {
                "litellm_provider": "bedrock_converse",
                "mode": "chat",
                "input_cost_per_token": 3.3e-06,
                "output_cost_per_token": 1.65e-05,
            },
        },
    )
    _invalidate_model_cost_lowercase_map()
    real_lookup: Final = litellm.utils.get_model_info
    lookups: Final = itertools.count()

    def lookup_tripped_once_by_a_concurrent_registration(model: str, custom_llm_provider: str | None = None) -> object:
        if next(lookups) == 0:
            raise RuntimeError("dictionary changed size during iteration")
        return real_lookup(model=model, custom_llm_provider=custom_llm_provider)

    with pytest.MonkeyPatch.context() as lookup_patch:
        lookup_patch.setattr(litellm.utils, "get_model_info", lookup_tripped_once_by_a_concurrent_registration)
        litellm.register_model({alias: {"litellm_provider": "bedrock", "mode": "chat"}}, persist_across_reloads=False)
    assert litellm.model_cost[catalog_key]["input_cost_per_token"] == 3.3e-06
    litellm.register_model({catalog_key: {"input_cost_per_token": repriced_input_cost}}, persist_across_reloads=False)

    try:
        assert litellm.get_model_info(alias)["key"] == alias
        assert litellm.get_model_info(alias)["input_cost_per_token"] == 3.3e-06
        assert litellm.model_cost[catalog_key]["input_cost_per_token"] == repriced_input_cost
    finally:
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize("key_prefix", ["", "bedrock/", "bedrock/converse/"])
def test_register_model_bills_a_catalog_rows_tier_rates_the_same_when_the_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, key_prefix: str
) -> None:
    from litellm.types.utils import ModelResponse, Usage

    catalog_key: Final = "au.anthropic.lit-5853-tiered-geo-model"
    registered_key: Final = f"{key_prefix}{catalog_key}"
    catalog_row: Final = {
        "litellm_provider": "bedrock_converse",
        "mode": "chat",
        "input_cost_per_token": 3.3e-06,
        "output_cost_per_token": 1.65e-05,
        "input_cost_per_token_above_200k_tokens": 6.6e-06,
        "output_cost_per_token_above_200k_tokens": 2.475e-05,
    }
    builtin_cost_map: Final = litellm.get_model_cost_map(url="")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def lookup_tripped_by_a_concurrent_registration(model: str, custom_llm_provider: str | None = None) -> NoReturn:
        raise RuntimeError("dictionary changed size during iteration")

    def long_prompt_cost_after_registering_with(lookup: Callable[..., object]) -> float:
        monkeypatch.setattr(litellm, "model_cost", {**builtin_cost_map, catalog_key: dict(catalog_row)})
        _invalidate_model_cost_lowercase_map()
        with pytest.MonkeyPatch.context() as lookup_patch:
            lookup_patch.setattr(litellm.utils, "get_model_info", lookup)
            litellm.register_model(
                {
                    registered_key: {
                        "litellm_provider": "bedrock",
                        "input_cost_per_token": 1e-06,
                        "output_cost_per_token": 2e-06,
                    }
                },
                persist_across_reloads=False,
            )
        return litellm.completion_cost(
            completion_response=ModelResponse(
                model=registered_key,
                usage=Usage(prompt_tokens=250_000, completion_tokens=1_000, total_tokens=251_000),
            ),
            model=registered_key,
            custom_llm_provider="bedrock",
        )

    try:
        assert long_prompt_cost_after_registering_with(lookup_tripped_by_a_concurrent_registration) == pytest.approx(
            long_prompt_cost_after_registering_with(litellm.utils.get_model_info)
        )
    finally:
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize("lookup_trips", [False, True])
def test_register_model_prices_a_provider_less_azure_ai_key_without_repricing_the_openai_row(
    monkeypatch: pytest.MonkeyPatch, lookup_trips: bool
) -> None:
    openai_key: Final = "lit-5853-openai-model"
    azure_ai_key: Final = f"azure_ai/{openai_key}"
    openai_row: Final = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 1.5e-05,
    }
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://lit-5853.openai.azure.com")
    monkeypatch.setattr(litellm, "model_cost", {**litellm.get_model_cost_map(url=""), openai_key: dict(openai_row)})
    monkeypatch.setattr(
        litellm, "open_ai_chat_completion_models", {*litellm.open_ai_chat_completion_models, openai_key}
    )
    _invalidate_model_cost_lowercase_map()

    def lookup_tripped_by_a_concurrent_registration(model: str, custom_llm_provider: str | None = None) -> NoReturn:
        raise RuntimeError("dictionary changed size during iteration")

    with pytest.MonkeyPatch.context() as lookup_patch:
        if lookup_trips:
            lookup_patch.setattr(litellm.utils, "get_model_info", lookup_tripped_by_a_concurrent_registration)
        litellm.register_model(
            {azure_ai_key: {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}},
            persist_across_reloads=False,
        )

    try:
        assert litellm.model_cost[openai_key] == openai_row
        assert litellm.cost_per_token(model=azure_ai_key, prompt_tokens=1000, completion_tokens=1000) == pytest.approx(
            (1e-03, 2e-03)
        )
    finally:
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize("persist", [False, True])
def test_register_model_keeps_the_prices_a_reload_adopts_while_it_registers(
    monkeypatch: pytest.MonkeyPatch, persist: bool
) -> None:
    from litellm._catalog import _register
    from litellm.rust_bridge._native import RegistrationTarget

    catalog_key: Final = "au.anthropic.lit-5853-reloaded-model"
    deployment_key: Final = f"bedrock/{catalog_key}"
    catalog_row: Final = {
        "litellm_provider": "bedrock_converse",
        "mode": "chat",
        "input_cost_per_token": 3.3e-06,
        "output_cost_per_token": 1.65e-05,
    }
    reloaded_input_cost: Final = 6.6e-06
    builtin_cost_map: Final = litellm.get_model_cost_map(url="")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    adopt_model_cost_map({**builtin_cost_map, catalog_key: dict(catalog_row)})
    live_cost_map: Final = litellm.model_cost
    _invalidate_model_cost_lowercase_map()
    reloaded_cost_map: Final = {
        **builtin_cost_map,
        catalog_key: {**catalog_row, "input_cost_per_token": reloaded_input_cost},
    }
    real_target: Final = _register.registration_target
    lookups: Final = itertools.count()

    def target_racing_a_price_data_reload(
        rows: dict[str, object], key: str, *, provider: str | None = None
    ) -> RegistrationTarget:
        target: Final = real_target(rows, key, provider=provider)
        if next(lookups) == 0:
            adopt_model_cost_map(reloaded_cost_map)
        return target

    with pytest.MonkeyPatch.context() as lookup_patch:
        lookup_patch.setattr(_register, "registration_target", target_racing_a_price_data_reload)
        litellm.register_model({deployment_key: {"litellm_provider": "bedrock"}}, persist_across_reloads=persist)

    try:
        assert litellm.model_cost is live_cost_map
        assert litellm.model_cost[catalog_key]["input_cost_per_token"] == reloaded_input_cost
        assert litellm.get_model_info(deployment_key)["input_cost_per_token"] == reloaded_input_cost
        adopt_model_cost_map(reloaded_cost_map)
        assert (deployment_key in litellm.model_cost) is persist
        if persist:
            assert litellm.get_model_info(deployment_key)["input_cost_per_token"] == reloaded_input_cost
    finally:
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize("persist", [False, True])
def test_register_model_keeps_its_price_when_a_reload_adopts_a_map_as_it_writes(
    monkeypatch: pytest.MonkeyPatch, persist: bool,
) -> None:
    from pydantic import JsonValue

    from litellm._catalog import _install

    catalog_key: Final = "au.anthropic.lit-5853-reloaded-as-written-model"
    deployment_key: Final = f"bedrock/{catalog_key}"
    catalog_row: Final = {
        "litellm_provider": "bedrock_converse",
        "mode": "chat",
        "input_cost_per_token": 3.3e-06,
        "output_cost_per_token": 1.65e-05,
    }
    reloaded_input_cost: Final = 6.6e-06
    registered_output_cost: Final = 2e-05
    builtin_cost_map: Final = litellm.get_model_cost_map(url="")
    reloaded_cost_map: Final = {
        **builtin_cost_map,
        catalog_key: {**catalog_row, "input_cost_per_token": reloaded_input_cost},
    }

    writes: Final = itertools.count()
    encode: Final = _install._encoded_row

    def encode_racing_a_price_data_reload(row: Mapping[str, object]) -> dict[str, JsonValue]:
        encoded: Final = encode(row)
        if row.get("output_cost_per_token") == registered_output_cost and next(writes) == 0:
            adopt_model_cost_map(reloaded_cost_map)
        return encoded

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    adopt_model_cost_map({**builtin_cost_map, catalog_key: dict(catalog_row)})
    monkeypatch.setattr(_install, "_encoded_row", encode_racing_a_price_data_reload)
    _invalidate_model_cost_lowercase_map()

    live_cost_map: Final = litellm.model_cost

    litellm.register_model(
        {deployment_key: {"litellm_provider": "bedrock", "output_cost_per_token": registered_output_cost}},
        persist_across_reloads=persist,
    )

    try:
        assert litellm.model_cost is live_cost_map
        assert litellm.get_model_info(deployment_key)["input_cost_per_token"] == reloaded_input_cost
        assert litellm.get_model_info(deployment_key)["output_cost_per_token"] == registered_output_cost
        adopt_model_cost_map(reloaded_cost_map)
        assert (deployment_key in litellm.model_cost) is persist
        if persist:
            assert litellm.get_model_info(deployment_key)["input_cost_per_token"] == reloaded_input_cost
            assert litellm.get_model_info(deployment_key)["output_cost_per_token"] == registered_output_cost
    finally:
        _invalidate_model_cost_lowercase_map()


def test_register_model_leaves_the_entry_a_lookup_is_walking_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_key: Final = "lit-5853-live-entry-model"
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.get_model_cost_map(url=""),
            catalog_key: {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 2e-06},
        },
    )
    _invalidate_model_cost_lowercase_map()
    walk: Final = iter(litellm.model_cost[catalog_key].items())
    first_field: Final = next(walk)

    try:
        litellm.register_model(
            {catalog_key: {"litellm_provider": "openai", "input_cost_per_token": 1e-06, "supports_vision": True}},
            persist_across_reloads=False,
        )
        assert dict((first_field, *walk)) == {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 2e-06,
        }
        assert litellm.get_model_info(catalog_key)["input_cost_per_token"] == 1e-06
        assert litellm.get_model_info(catalog_key)["supports_vision"] is True
    finally:
        _invalidate_model_cost_lowercase_map()


_CANONICAL_KEY: Final = "lit-5853-canonical-model"
_ALIAS_KEY: Final = "lit-5853-alias-model"


def _install_aliased_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        _expand_model_aliases(
            {
                **litellm.get_model_cost_map(url=""),
                _CANONICAL_KEY: {
                    "aliases": [_ALIAS_KEY],
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 2e-06,
                    "output_cost_per_token": 8e-06,
                },
            }
        ),
    )
    _invalidate_model_cost_lowercase_map()


def _token_prices(name: str) -> tuple[float | None, float | None]:
    info: Final = litellm.get_model_info(name)
    return info["input_cost_per_token"], info["output_cost_per_token"]


@pytest.mark.parametrize("registered_key", [_CANONICAL_KEY, _ALIAS_KEY])
def test_register_model_prices_every_alias_of_the_registered_entry(
    monkeypatch: pytest.MonkeyPatch, registered_key: str
) -> None:
    _install_aliased_cost_map(monkeypatch)

    try:
        litellm.register_model(
            {registered_key: {"litellm_provider": "openai", "input_cost_per_token": 1e-06}},
            persist_across_reloads=False,
        )
        litellm.register_model(
            {registered_key: {"litellm_provider": "openai", "output_cost_per_token": 4e-06}},
            persist_across_reloads=False,
        )
        assert {name: _token_prices(name) for name in (_CANONICAL_KEY, _ALIAS_KEY)} == {
            _CANONICAL_KEY: (1e-06, 4e-06),
            _ALIAS_KEY: (1e-06, 4e-06),
        }
    finally:
        _invalidate_model_cost_lowercase_map()


def test_register_model_leaves_an_alias_repointed_at_its_own_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_aliased_cost_map(monkeypatch)
    litellm.model_cost[_ALIAS_KEY] = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 6e-06,
    }

    try:
        litellm.register_model(
            {_CANONICAL_KEY: {"litellm_provider": "openai", "input_cost_per_token": 1e-06}},
            persist_across_reloads=False,
        )
        assert {name: _token_prices(name) for name in (_CANONICAL_KEY, _ALIAS_KEY)} == {
            _CANONICAL_KEY: (1e-06, 8e-06),
            _ALIAS_KEY: (5e-06, 6e-06),
        }
    finally:
        _invalidate_model_cost_lowercase_map()


def test_register_model_inherits_builtin_token_pricing_for_unmapped_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    _invalidate_model_cost_lowercase_map()
    builtin_key: Final = "us.anthropic.claude-sonnet-4-6"
    registered_key: Final = f"bedrock/bedrock/bedrock/{builtin_key}"
    builtin: Final = litellm.model_cost[builtin_key]

    try:
        litellm.register_model(
            {registered_key: {"litellm_provider": "bedrock", "mode": "chat"}}, persist_across_reloads=False
        )
        info: Final = litellm.get_model_info(registered_key)
        assert info["input_cost_per_token"] == builtin["input_cost_per_token"] > 0
        assert info["output_cost_per_token"] == builtin["output_cost_per_token"] > 0
        assert info["cache_read_input_token_cost"] == builtin["cache_read_input_token_cost"]
    finally:
        _invalidate_model_cost_lowercase_map()


def test_register_model_keeps_the_capabilities_an_unregistered_region_alias_resolves_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.litellm_core_utils.fallback_generalizations import (
        get_fallback_generalization_rules,
        set_fallback_generalizations,
    )

    sibling_key: Final = "anthropic.lit-5853-flagless-model"
    region_alias: Final = f"bedrock/ap-northeast-1.{sibling_key}"
    sibling_row: Final = {
        "litellm_provider": "bedrock_converse",
        "mode": "chat",
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
    }
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", {**litellm.get_model_cost_map(url=""), sibling_key: sibling_row})
    _invalidate_model_cost_lowercase_map()
    previous_rules: Final = list(get_fallback_generalization_rules())
    set_fallback_generalizations(
        [
            {
                "name": "lit-5853-flagless",
                "pattern": "lit-5853-flagless",
                "model_info": {"supports_system_messages": True, "supports_parallel_function_calling": True},
            }
        ]
    )

    try:
        unregistered: Final = litellm.get_model_info(region_alias)
        litellm.register_model(
            {region_alias: {"litellm_provider": "bedrock", "mode": "chat"}}, persist_across_reloads=False
        )
        registered: Final = litellm.get_model_info(region_alias)
        assert registered["supports_system_messages"] is unregistered["supports_system_messages"] is True
        assert (
            registered["supports_parallel_function_calling"]
            is unregistered["supports_parallel_function_calling"]
            is True
        )
        assert registered["input_cost_per_token"] == sibling_row["input_cost_per_token"]
    finally:
        set_fallback_generalizations(previous_rules)
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize(
    ("builtin_prices", "custom_prices", "usage", "expected_cost"),
    [
        pytest.param(
            {
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 1.5e-05,
                "input_cost_per_token_above_200k_tokens": 6e-06,
                "output_cost_per_token_above_200k_tokens": 2.25e-05,
            },
            {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06},
            {"prompt_tokens": 250_000, "completion_tokens": 1_000},
            250_000 * 1e-06 + 1_000 * 2e-06,
            id="above-200k-tier",
        ),
        pytest.param(
            {
                "tiered_pricing": [
                    {"input_cost_per_token": 5e-08, "output_cost_per_token": 4e-07, "range": [0, 256_000]},
                    {"input_cost_per_token": 2.5e-07, "output_cost_per_token": 2e-06, "range": [256_000, 1_000_000]},
                ]
            },
            {"input_cost_per_token": 1e-06},
            {"prompt_tokens": 300_000, "completion_tokens": 1_000},
            300_000 * 1e-06 + 1_000 * 2e-06,
            id="tiered-pricing-input-only",
        ),
        pytest.param(
            {"input_cost_per_token": 3e-06, "output_cost_per_token": 1.5e-05, "output_cost_per_reasoning_token": 3e-05},
            {"output_cost_per_token": 2e-06},
            {"prompt_tokens": 100, "completion_tokens": 1_000, "reasoning_tokens": 600},
            100 * 3e-06 + 1_000 * 2e-06,
            id="reasoning-rate-output-only",
        ),
        pytest.param(
            {
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 1.5e-05,
                "off_peak_pricing": {
                    "input_cost_per_token": 1e-06,
                    "output_cost_per_token": 5e-06,
                    "hours_utc": "00:00-00:00",
                },
            },
            {"output_cost_per_token": 2e-06},
            {"prompt_tokens": 100, "completion_tokens": 1_000},
            100 * 1e-06 + 1_000 * 2e-06,
            id="off-peak-output-only",
        ),
    ],
)
def test_register_model_keeps_custom_flat_prices_over_inherited_rate_variants(
    monkeypatch: pytest.MonkeyPatch,
    builtin_prices: dict[str, object],
    custom_prices: dict[str, float],
    usage: dict[str, int],
    expected_cost: float,
) -> None:
    from litellm.types.utils import CompletionTokensDetailsWrapper, ModelResponse, Usage

    builtin_key: Final = "anthropic.lit-5853-tiered-model-v1:0"
    alias: Final = f"bedrock/ap-northeast-1.{builtin_key}"
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.get_model_cost_map(url=""),
            builtin_key: {"litellm_provider": "bedrock", "mode": "chat", **builtin_prices},
        },
    )
    _invalidate_model_cost_lowercase_map()

    try:
        litellm.register_model(
            {alias: {"litellm_provider": "bedrock", **custom_prices}},
            persist_across_reloads=False,
        )
        response_usage: Final = Usage(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["prompt_tokens"] + usage["completion_tokens"],
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=usage.get("reasoning_tokens", 0)),
        )
        cost: Final = litellm.completion_cost(
            completion_response=ModelResponse(model=alias, usage=response_usage),
            model=alias,
            custom_llm_provider="bedrock",
        )
        assert cost == pytest.approx(expected_cost)
    finally:
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize(
    ("base_key", "caller", "provider", "served_provider", "keeps_variant"),
    (
        ("registration-shape-model", "openai/registration-shape-model", "openai", "openai", True),
        ("us.anthropic.shape-model", "bedrock/us.anthropic.shape-model", "bedrock", "bedrock", True),
        ("us.anthropic.shape-model", "bedrock/bedrock/bedrock/us.anthropic.shape-model", "bedrock", None, True),
        ("anthropic.shape-model", "bedrock/ap-northeast-1.anthropic.shape-model", "bedrock", None, False),
        ("registration-shape-model", "azure_ai/registration-shape-model", "openai", None, True),
    ),
)
def test_registration_keeps_primary_and_fallback_field_shaping_across_reload(
    monkeypatch: pytest.MonkeyPatch,
    base_key: str,
    caller: str,
    provider: str,
    served_provider: str | None,
    keeps_variant: bool,
) -> None:
    row: Final = {
        "litellm_provider": provider,
        "mode": "chat",
        "input_cost_per_token": 0.001,
        "input_cost_per_token_above_200k_tokens": 0.004,
        "cache_read_input_token_cost": 0.0001,
        "supports_vision": True,
    }
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://registration.openai.azure.com")
    adopt_model_cost_map({base_key: row})
    litellm.register_model({caller: {"input_cost_per_token": 0.002}})
    expected: Final = {
        "mode": "chat",
        "input_cost_per_token": 0.002,
        "cache_read_input_token_cost": row["cache_read_input_token_cost"],
        "supports_vision": row["supports_vision"],
        **({"litellm_provider": served_provider} if served_provider is not None else {}),
        **(
            {"input_cost_per_token_above_200k_tokens": row["input_cost_per_token_above_200k_tokens"]}
            if keeps_variant
            else {}
        ),
    }
    assert litellm.model_cost[base_key] == row
    assert litellm.model_cost[caller] == expected
    expected_cost: Final = 250_000 * (0.004 if keeps_variant else 0.002)
    assert litellm.cost_per_token(model=caller, prompt_tokens=250_000, completion_tokens=0) == pytest.approx(
        (expected_cost, 0)
    )
    adopt_model_cost_map({base_key: row})
    assert litellm.model_cost[base_key] == row
    assert litellm.model_cost[caller] == expected
    assert litellm.cost_per_token(model=caller, prompt_tokens=250_000, completion_tokens=0) == pytest.approx(
        (expected_cost, 0)
    )


def test_register_model_never_inherits_pricing_across_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    openai_key: Final = "lit-5853-openai-only-model"
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.get_model_cost_map(url=""),
            openai_key: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 2.5e-06,
                "output_cost_per_token": 1e-05,
                "cache_read_input_token_cost": 1.25e-06,
            },
        },
    )
    _invalidate_model_cost_lowercase_map()
    registered_key: Final = f"bedrock/{openai_key}"

    try:
        litellm.register_model(
            {registered_key: {"litellm_provider": "bedrock", "mode": "chat"}}, persist_across_reloads=False
        )
        registered: Final = litellm.model_cost[registered_key]
        assert "cache_read_input_token_cost" not in registered
        assert "input_cost_per_token" not in registered
        assert "output_cost_per_token" not in registered
    finally:
        _invalidate_model_cost_lowercase_map()


def test_update_dictionary_merges_nested_dicts_without_aliasing():
    """A nested dict must be merged copy-on-write: the pre-existing nested dict
    object stays untouched, and the caller's incoming nested dict is never
    inserted by reference into the merged result.
    """
    from typing import cast

    from litellm.utils import update_dictionary

    existing_nested = cast(dict[str, object], {"hours_utc": "01:00-02:00", 1: "existing"})
    existing = {"off_peak_pricing": existing_nested}
    incoming_nested = cast(
        dict[str, object],
        {"windows": [{"hours_utc": "16:00-19:00", "weekdays": [2]}], 2: "new"},
    )
    incoming = {"off_peak_pricing": incoming_nested}

    merged = update_dictionary(existing, incoming)

    assert merged["off_peak_pricing"] == {
        "hours_utc": "01:00-02:00",
        "windows": [{"hours_utc": "16:00-19:00", "weekdays": [2]}],
        1: "existing",
        2: "new",
    }
    assert existing_nested == {"hours_utc": "01:00-02:00", 1: "existing"}
    assert merged["off_peak_pricing"] is not incoming_nested

    fresh = update_dictionary({}, incoming)
    assert fresh["off_peak_pricing"] == incoming_nested
    assert fresh["off_peak_pricing"] is not incoming_nested


def test_router_deployments_sharing_backend_keep_their_own_off_peak_pricing():
    """Two deployments of the same backend model with different
    ``off_peak_pricing`` blocks must each keep their own schedule under their
    unique model id, and neither block may leak onto the shared backend keys.

    Before the fix, ``register_model`` inserted the first deployment's block by
    reference into the built-in ``gpt-4o-mini`` entry, and the second
    deployment's registration merged its keys into that same object, corrupting
    the first deployment's schedule and polluting the built-in entry.
    """
    from litellm import Router

    active_block = {
        "windows": [{"hours_utc": "16:00-19:00", "weekdays": [2]}],
        "input_cost_per_token": 5e-07,
        "output_cost_per_token": 1e-06,
    }
    inactive_block = {
        "hours_utc": "05:00-06:00",
        "input_cost_per_token": 5e-07,
        "output_cost_per_token": 1e-06,
    }
    shared_keys = ["gpt-4o-mini", "openai/gpt-4o-mini"]
    deployment_ids = ["offpeak-alias-dep-1", "offpeak-alias-dep-2"]
    original_entries = _snapshot_model_cost_entries(shared_keys)

    router = Router(
        model_list=[
            {
                "model_name": "offpeak-active-weekday",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fake-key-for-registration",
                },
                "model_info": {
                    "id": deployment_ids[0],
                    "input_cost_per_token": 1e-06,
                    "output_cost_per_token": 2e-06,
                    "off_peak_pricing": dict(active_block),
                },
            },
            {
                "model_name": "offpeak-inactive-hours",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fake-key-for-registration",
                },
                "model_info": {
                    "id": deployment_ids[1],
                    "input_cost_per_token": 1e-06,
                    "output_cost_per_token": 2e-06,
                    "off_peak_pricing": dict(inactive_block),
                },
            },
        ]
    )

    try:
        registered_first = litellm.model_cost[deployment_ids[0]]["off_peak_pricing"]
        registered_second = litellm.model_cost[deployment_ids[1]]["off_peak_pricing"]
        assert registered_first == active_block
        assert registered_second == inactive_block
        for shared_key in shared_keys:
            shared_entry = litellm.model_cost.get(shared_key) or {}
            assert not shared_entry.get("off_peak_pricing")
    finally:
        for deployment_id in deployment_ids:
            litellm.model_cost.pop(deployment_id, None)
        _restore_model_cost_entries(original_entries)
        del router


def test_router_off_peak_only_deployment_inherits_builtin_base_rates():
    """A deployment that sets only ``off_peak_pricing`` on its model_info must
    still be costed from its deployment-scoped entry: the base token rates are
    inherited from the backend model's built-in cost map entry, since the
    shared backend key deliberately never carries the off-peak block.
    """
    from litellm import Router

    block = {
        "hours_utc": "00:00-00:00",
        "input_cost_per_token": 5e-05,
        "output_cost_per_token": 1e-04,
    }
    shared_keys = ["gpt-4o-mini", "openai/gpt-4o-mini"]
    deployment_id = "offpeak-only-dep-1"
    original_entries = _snapshot_model_cost_entries(shared_keys + [deployment_id])
    builtin_info = litellm.get_model_info(model="openai/gpt-4o-mini")

    router = Router(
        model_list=[
            {
                "model_name": "offpeak-only",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fake-key-for-registration",
                },
                "model_info": {"id": deployment_id, "off_peak_pricing": dict(block)},
            }
        ]
    )

    try:
        entry = litellm.model_cost[deployment_id]
        assert entry["off_peak_pricing"] == block
        assert entry["input_cost_per_token"] is not None
        assert entry["input_cost_per_token"] == builtin_info["input_cost_per_token"]
        assert entry["output_cost_per_token"] == builtin_info["output_cost_per_token"]
        for shared_key in shared_keys:
            shared_entry = litellm.model_cost.get(shared_key) or {}
            assert not shared_entry.get("off_peak_pricing")
    finally:
        _restore_model_cost_entries(original_entries)
        del router


def test_use_custom_pricing_for_model_sees_off_peak_only_model_info():
    from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model

    block = {"hours_utc": "00:00-00:00", "input_cost_per_token": 5e-05}
    assert use_custom_pricing_for_model({"metadata": {"model_info": {"off_peak_pricing": block}}}) is True
    assert use_custom_pricing_for_model({"metadata": {"model_info": {"off_peak_pricing": None}}}) is False
    assert use_custom_pricing_for_model({"metadata": {"model_info": {"id": "some-id"}}}) is False


def test_completion_cost_applies_off_peak_only_deployment_pricing():
    """End to end through the cost calculator: with ``custom_pricing`` set and
    a ``router_model_id`` whose entry carries only an always-on off-peak block,
    the request bills at the block's rates rather than the shared backend rate.
    """
    from litellm import Router
    from litellm.types.utils import ModelResponse, Usage

    block = {
        "hours_utc": "00:00-00:00",
        "input_cost_per_token": 5e-05,
        "output_cost_per_token": 1e-04,
    }
    shared_keys = ["gpt-4o-mini", "openai/gpt-4o-mini"]
    deployment_id = "offpeak-only-dep-2"
    original_entries = _snapshot_model_cost_entries(shared_keys + [deployment_id])

    router = Router(
        model_list=[
            {
                "model_name": "offpeak-only",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fake-key-for-registration",
                },
                "model_info": {"id": deployment_id, "off_peak_pricing": dict(block)},
            }
        ]
    )

    try:
        response = ModelResponse(
            model="gpt-4o-mini",
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        cost = litellm.completion_cost(
            completion_response=response,
            model="openai/gpt-4o-mini",
            custom_llm_provider="openai",
            custom_pricing=True,
            router_model_id=deployment_id,
        )
        assert cost == pytest.approx(100 * 5e-05 + 50 * 1e-04)
    finally:
        _restore_model_cost_entries(original_entries)
        del router


def test_completion_uses_ephemeral_cost_per_second_pricing():
    model_key: Final = "openai/test-cost-per-second-registration"
    original_entries: Final = _snapshot_model_cost_entries([model_key])

    try:
        response: Final = litellm.completion(
            model=model_key,
            messages=[{"role": "user", "content": "hello"}],
            api_key="fake-key",
            cost_per_second=0.02,
            mock_response="hello back",
        )

        assert model_key not in litellm.model_cost
        assert response._hidden_params["response_cost"] is not None
        assert litellm.completion_cost(
            litellm.ModelResponse(model=model_key),
            model=model_key,
            custom_cost_per_second=0.02,
            total_time=2000,
        ) == pytest.approx(0.04)
    finally:
        _restore_model_cost_entries(original_entries)


def test_update_model_cost():
    try:
        litellm.register_model(
            {
                "gpt-4": {
                    "max_tokens": 8192,
                    "input_cost_per_token": 0.00002,
                    "output_cost_per_token": 0.00006,
                    "litellm_provider": "openai",
                    "mode": "chat",
                },
            }
        )
        assert litellm.model_cost["gpt-4"]["input_cost_per_token"] == 0.00002
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


def test_register_model_scopes_builtin_match_to_the_given_provider():
    """Registering under an id that collides with another provider's catalog
    key keeps the entry provider-less under the id instead of merging into the
    baseten row."""
    colliding_id: Final = "baseten/zai-org/glm-5.2"
    builtin_key: Final = "baseten/zai-org/GLM-5.2"
    model_cost_entries: Final = _snapshot_model_cost_entries((colliding_id, builtin_key))
    builtin_row_before: Final = copy.deepcopy(litellm.model_cost[builtin_key])
    try:
        litellm.register_model(
            {
                colliding_id: {
                    "input_cost_per_token": 0.00000096,
                    "output_cost_per_token": 0.00000302,
                    "cache_read_input_token_cost": 0.00000010,
                    "mode": "chat",
                }
            },
            custom_llm_provider="openai",
        )

        entry: Final = litellm.model_cost[colliding_id]
        assert "litellm_provider" not in entry
        assert entry["input_cost_per_token"] == 0.00000096
        assert entry["output_cost_per_token"] == 0.00000302
        assert entry["cache_read_input_token_cost"] == 0.00000010
        assert litellm.model_cost[builtin_key] == builtin_row_before
    finally:
        _restore_model_cost_entries(model_cost_entries)


def test_per_request_custom_pricing_does_not_write_a_colliding_deployment_id():
    deployment_id: Final = "baseten/zai-org/glm-5.2"
    builtin_key: Final = "baseten/zai-org/GLM-5.2"
    shared_key: Final = "openai/zai-org/GLM-5.2"
    model_cost_entries: Final = _snapshot_model_cost_entries((deployment_id, builtin_key, shared_key))
    builtin_row_before: Final = copy.deepcopy(litellm.model_cost[builtin_key])
    openai_models_before: Final = frozenset(litellm.open_ai_chat_completion_models)
    try:
        response: Final = litellm.completion(
            model=shared_key,
            messages=[{"role": "user", "content": "hello"}],
            api_key="fake-key",
            input_cost_per_token=0.00000096,
            output_cost_per_token=0.00000302,
            metadata={"model_info": {"id": deployment_id}},
            mock_response=litellm.ModelResponse(
                model=shared_key, usage=litellm.Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            ),
        )

        assert response._hidden_params["response_cost"] == pytest.approx(10 * 0.00000096 + 20 * 0.00000302)
        assert litellm.model_cost.get(deployment_id) == model_cost_entries[deployment_id]
        assert litellm.model_cost[builtin_key] == builtin_row_before
    finally:
        _restore_model_cost_entries(model_cost_entries)
        litellm.open_ai_chat_completion_models.intersection_update(openai_models_before)
