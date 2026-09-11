"""Tests for the /cost/estimate/cache-switch endpoint."""

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.proxy._types import (
    CacheSwitchCostEstimateArmRequest,
    CacheSwitchCostEstimateRequest,
    CacheSwitchPredictionRequest,
)
from litellm.proxy.management_endpoints.cache_switch_cost_estimate import (
    estimate_cache_switch_cost,
    predict_cache_switch_cost,
)
from litellm.litellm_core_utils.prompt_cache_prediction import make_plan, observe


class TestCacheSwitchCostEstimateEndpoint:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("missing_field", "request_overrides", "expected_label"),
        (
            (
                "cache_read_input_token_cost",
                {"cache_read_input_tokens": 1},
                "cache read",
            ),
            (
                "cache_creation_input_token_cost",
                {"cache_creation_input_tokens_5m": 1},
                "5-minute cache write",
            ),
            (
                "cache_creation_input_token_cost_above_1hr",
                {"cache_creation_input_tokens_1h": 1},
                "1-hour cache write",
            ),
        ),
    )
    async def test_rejects_nonzero_cache_bucket_without_its_rate(
        self, monkeypatch, missing_field, request_overrides, expected_label
    ):
        model = "openai/cache-rate-gap"
        pricing = {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 0.1e-6,
            "cache_creation_input_token_cost": 1.25e-6,
            "cache_creation_input_token_cost_above_1hr": 2e-6,
            "litellm_provider": "openai",
            "mode": "chat",
        }
        pricing.pop(missing_field)
        monkeypatch.setitem(litellm.model_cost, model, pricing)
        arm = CacheSwitchCostEstimateArmRequest(
            model=model,
            uncached_input_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens_5m=0,
            cache_creation_input_tokens_1h=0,
        ).model_copy(update=request_overrides)

        with patch(  # test-quality-ok: inject router singleton to exercise the public endpoint
            "litellm.proxy.proxy_server.llm_router", None
        ):  # test-quality-ok: proxy router is the endpoint dependency injection seam
            from fastapi import HTTPException

            with pytest.raises(HTTPException, match=expected_label) as exc_info:
                await estimate_cache_switch_cost(
                    request=CacheSwitchCostEstimateRequest(stay=arm, switch=arm),
                    user_api_key_dict=MagicMock(),
                )

        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_prices_warm_stay_against_cold_switch_with_ttl_split(self, monkeypatch):
        monkeypatch.setitem(
            litellm.model_cost,
            "anthropic/stay-model",
            {
                "input_cost_per_token": 5e-6,
                "output_cost_per_token": 25e-6,
                "cache_read_input_token_cost": 0.5e-6,
                "cache_creation_input_token_cost": 6.25e-6,
                "cache_creation_input_token_cost_above_1hr": 10e-6,
                "litellm_provider": "anthropic",
                "mode": "chat",
            },
        )
        monkeypatch.setitem(
            litellm.model_cost,
            "anthropic/switch-model",
            {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 5e-6,
                "cache_read_input_token_cost": 0.1e-6,
                "cache_creation_input_token_cost": 1.25e-6,
                "cache_creation_input_token_cost_above_1hr": 2e-6,
                "litellm_provider": "anthropic",
                "mode": "chat",
            },
        )
        request = CacheSwitchCostEstimateRequest(
            stay=CacheSwitchCostEstimateArmRequest(
                model="anthropic/stay-model",
                uncached_input_tokens=0,
                cache_read_input_tokens=100_000,
                cache_creation_input_tokens_5m=2_000,
                cache_creation_input_tokens_1h=0,
                assumed_cache_state="warm",
            ),
            switch=CacheSwitchCostEstimateArmRequest(
                model="anthropic/switch-model",
                uncached_input_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens_5m=42_000,
                cache_creation_input_tokens_1h=60_000,
                assumed_cache_state="cold",
            ),
        )

        with patch(  # test-quality-ok: inject router singleton to exercise the public endpoint
            "litellm.proxy.proxy_server.llm_router", None
        ):  # test-quality-ok: proxy router is the endpoint dependency injection seam
            response = await estimate_cache_switch_cost(request=request, user_api_key_dict=MagicMock())

        assert response.stay.input_cost == pytest.approx(0.0625)
        assert response.switch.cache_creation_input_cost_5m == pytest.approx(0.0525)
        assert response.switch.cache_creation_input_cost_1h == pytest.approx(0.12)
        assert response.switch.input_cost == pytest.approx(0.1725)
        assert response.switch_cost_delta == pytest.approx(0.11)
        assert response.stay.assumed_cache_state == "warm"
        assert response.switch.assumed_cache_state == "cold"

    @pytest.mark.asyncio
    async def test_rejects_ambiguous_model_group_without_model_id(self):
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {"model_info": {"id": "one"}, "litellm_params": {"model": "openai/gpt-4o"}},
            {"model_info": {"id": "two"}, "litellm_params": {"model": "openai/gpt-4o-mini"}},
        ]
        arm = CacheSwitchCostEstimateArmRequest(
            model="ambiguous-group",
            uncached_input_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens_5m=0,
            cache_creation_input_tokens_1h=0,
        )

        with patch(  # test-quality-ok: inject router singleton to exercise the public endpoint
            "litellm.proxy.proxy_server.llm_router", mock_router
        ):  # test-quality-ok: proxy router is the endpoint dependency injection seam
            from fastapi import HTTPException

            with pytest.raises(HTTPException, match="multiple deployments") as exc_info:
                await estimate_cache_switch_cost(
                    request=CacheSwitchCostEstimateRequest(stay=arm, switch=arm),
                    user_api_key_dict=MagicMock(),
                )

        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_model_id_must_belong_to_the_named_group(self):
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {"model_info": {"id": "one"}, "litellm_params": {"model": "openai/gpt-4o"}}
        ]
        arm = CacheSwitchCostEstimateArmRequest(
            model="group",
            model_id="other",
            uncached_input_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens_5m=0,
            cache_creation_input_tokens_1h=0,
        )

        with patch(  # test-quality-ok: inject router singleton to exercise the public endpoint
            "litellm.proxy.proxy_server.llm_router", mock_router
        ):  # test-quality-ok: proxy router is the endpoint dependency injection seam
            from fastapi import HTTPException

            with pytest.raises(HTTPException, match="does not uniquely identify") as exc_info:
                await estimate_cache_switch_cost(
                    request=CacheSwitchCostEstimateRequest(stay=arm, switch=arm),
                    user_api_key_dict=MagicMock(),
                )

        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_same_model_and_buckets_cost_the_same_for_every_cache_state_label(self, monkeypatch):
        model = "anthropic/state-label-model"
        monkeypatch.setitem(
            litellm.model_cost,
            model,
            {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 5e-6,
                "cache_read_input_token_cost": 0.1e-6,
                "cache_creation_input_token_cost": 1.25e-6,
                "cache_creation_input_token_cost_above_1hr": 2e-6,
                "litellm_provider": "anthropic",
                "mode": "chat",
            },
        )
        base = CacheSwitchCostEstimateArmRequest(
            model=model,
            uncached_input_tokens=100,
            cache_read_input_tokens=800,
            cache_creation_input_tokens_5m=50,
            cache_creation_input_tokens_1h=50,
            assumed_cache_state="warm",
        )

        with patch(  # test-quality-ok: inject router singleton to exercise the public endpoint
            "litellm.proxy.proxy_server.llm_router", None
        ):  # test-quality-ok: proxy router is the endpoint dependency injection seam
            response = await estimate_cache_switch_cost(
                request=CacheSwitchCostEstimateRequest(
                    stay=base,
                    switch=base.model_copy(update={"assumed_cache_state": "stale"}),
                ),
                user_api_key_dict=MagicMock(),
            )

        assert response.stay.input_cost == response.switch.input_cost
        assert response.switch_cost_delta == 0.0

    @pytest.mark.asyncio
    async def test_model_id_prices_the_selected_deployment(self):
        mock_router = MagicMock()
        deployments = [
            {
                "model_info": {"id": "expensive"},
                "litellm_params": {
                    "model": "custom/model",
                    "custom_llm_provider": "openai",
                    "input_cost_per_token": 2e-6,
                    "output_cost_per_token": 0.0,
                },
            },
            {
                "model_info": {"id": "cheap"},
                "litellm_params": {
                    "model": "custom/model",
                    "custom_llm_provider": "openai",
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 0.0,
                },
            },
        ]
        mock_router.get_model_list.return_value = deployments
        mock_router.get_deployment_model_info.side_effect = lambda model_id, model_name: {
            "input_cost_per_token": 2e-6 if model_id == "expensive" else 1e-6,
            "output_cost_per_token": 0.0,
            "cache_read_input_token_cost": 2e-6 if model_id == "expensive" else 1e-6,
            "cache_creation_input_token_cost": 2e-6 if model_id == "expensive" else 1e-6,
            "litellm_provider": "openai",
            "mode": "chat",
        }
        stay = CacheSwitchCostEstimateArmRequest(
            model="pool",
            model_id="expensive",
            uncached_input_tokens=1_000,
            cache_read_input_tokens=0,
            cache_creation_input_tokens_5m=0,
            cache_creation_input_tokens_1h=0,
        )
        switch = stay.model_copy(update={"model_id": "cheap"})

        with patch(  # test-quality-ok: inject router singleton to exercise the public endpoint
            "litellm.proxy.proxy_server.llm_router", mock_router
        ):  # test-quality-ok: proxy router is the endpoint dependency injection seam
            response = await estimate_cache_switch_cost(
                request=CacheSwitchCostEstimateRequest(stay=stay, switch=switch),
                user_api_key_dict=MagicMock(),
            )

        assert response.stay.resolved_model_id == "expensive"
        assert response.switch.resolved_model_id == "cheap"
        assert response.stay.input_cost == pytest.approx(0.002)
        assert response.switch.input_cost == pytest.approx(0.001)
        assert response.switch_cost_delta == pytest.approx(-0.001)

    def test_http_route_returns_the_signed_comparison(self):
        from fastapi.testclient import TestClient

        from litellm.proxy.proxy_server import app

        with patch(  # test-quality-ok: inject router singleton to exercise the public endpoint
            "litellm.proxy.proxy_server.llm_router", None
        ):  # test-quality-ok: proxy router is the endpoint dependency injection seam
            response = TestClient(app).post(
                "/cost/estimate/cache-switch",
                headers={"Authorization": "Bearer sk-1234"},
                json={
                    "stay": {
                        "model": "anthropic/claude-opus-4-8",
                        "uncached_input_tokens": 0,
                        "cache_read_input_tokens": 100_000,
                        "cache_creation_input_tokens_5m": 2_000,
                        "cache_creation_input_tokens_1h": 0,
                    },
                    "switch": {
                        "model": "anthropic/claude-haiku-4-5",
                        "uncached_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens_5m": 102_000,
                        "cache_creation_input_tokens_1h": 0,
                    },
                },
            )

        assert response.status_code == 200
        assert response.json()["stay"]["input_cost"] == pytest.approx(0.0625)
        assert response.json()["switch"]["input_cost"] == pytest.approx(0.1275)
        assert response.json()["switch_cost_delta"] == pytest.approx(0.065)

    @pytest.mark.asyncio
    async def test_predict_uses_scoped_observed_cache_instead_of_caller_buckets(self):
        from datetime import datetime, timezone
        from litellm._internal_context import pinned_billing_time
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm import Router

        now = datetime(2026, 9, 12, tzinfo=timezone.utc)
        router = Router(
            model_list=[
                {
                    "model_name": "source",
                    "litellm_params": {"model": "anthropic/claude-opus-4-8", "api_key": "test"},
                    "model_info": {"id": "source-id"},
                },
                {
                    "model_name": "target",
                    "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "test"},
                    "model_info": {"id": "target-id"},
                },
            ]
        )
        prompt = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "stable prefix " * 3000, "cache_control": {"type": "ephemeral"}}
                    ],
                }
            ]
        }
        request = CacheSwitchPredictionRequest.model_validate(
            {
                "stay": {"model": "source", "prompt": prompt},
                "switch": {"model": "target", "prompt": prompt},
            }
        )
        plan = make_plan({**prompt, "model": "claude-opus-4-8"})
        assert plan is not None
        provider_tokens = plan.final_boundary.estimated_tokens
        await observe(
            router.cache,
            "caller-hash",
            "source-id",
            plan,
            {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": provider_tokens,
                "prompt_tokens_details": {
                    "cache_creation_tokens": provider_tokens,
                    "cache_creation_token_details": {
                        "ephemeral_5m_input_tokens": provider_tokens,
                        "ephemeral_1h_input_tokens": 0,
                    },
                },
            },
            now.timestamp() - 1,
        )
        with patch(  # test-quality-ok: inject router singleton to exercise the public endpoint
            "litellm.proxy.proxy_server.llm_router", router
        ):  # test-quality-ok: inject real Router into proxy singleton
            with pinned_billing_time(now):
                response = await predict_cache_switch_cost(request, UserAPIKeyAuth(api_key="caller-hash"))
                other = await predict_cache_switch_cost(request, UserAPIKeyAuth(api_key="other-hash"))
        assert response.stay.cache_state == "warm"
        assert response.stay.cache_read_input_tokens == provider_tokens
        assert response.stay.cache_creation_input_tokens_5m == 0
        assert response.stay.estimated_input_cost is not None
        assert response.switch.cache_state == "unknown"
        assert response.switch.estimated_input_cost is None
        assert response.switch.cold_input_cost > response.switch.warm_input_cost
        assert response.switch_cost_delta is None
        assert other.stay.cache_state == "unknown"
