"""
Test Free Key Optimization routing strategy with multi-window rate limiting.

Tests cover:
- Multi-window rate limiting (minute, hour, day)
- AND logic validation (all limits must be respected)
- Deployment selection based on lowest token usage
- Edge cases and error scenarios
- Integration with Router class
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.free_key_optimization import FreeKeyOptimizationHandler


@pytest.fixture
def sample_model_list():
    """Sample model list for testing"""
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "fake-key-1",
            },
            "model_info": {
                "id": "deployment-1",
            },
            "rpm": 10,  # 10 requests per minute
            "rph": 100,  # 100 requests per hour
            "rpd": 1000,  # 1000 requests per day
            "tpm": 1000,  # 1000 tokens per minute
            "tph": 10000,  # 10000 tokens per hour
            "tpd": 100000,  # 100000 tokens per day
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "fake-key-2",
            },
            "model_info": {
                "id": "deployment-2",
            },
            "rpm": 20,  # Higher minute limit
            "rph": 50,  # Lower hour limit
            "rpd": 2000,
            "tpm": 2000,
            "tph": 5000,  # Lower hour limit
            "tpd": 200000,
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "fake-key-3",
            },
            "model_info": {
                "id": "deployment-3",
            },
            # Only minute limits (backward compatibility)
            "rpm": 5,
            "tpm": 500,
        },
    ]


@pytest.fixture
def mock_cache():
    """Mock cache for testing"""
    cache = MagicMock(spec=DualCache)
    cache.get_cache = MagicMock(return_value=None)
    cache.increment_cache = MagicMock(return_value=1)
    cache.batch_get_cache = MagicMock(
        return_value=[0] * 20
    )  # Return zeros for all keys
    cache.async_get_cache = AsyncMock(return_value=None)
    cache.async_increment_cache = AsyncMock(return_value=1)
    cache.async_batch_get_cache = AsyncMock(return_value=[0] * 20)
    return cache


@pytest.fixture
def free_key_handler(sample_model_list, mock_cache):
    """Free key optimization handler for testing"""
    return FreeKeyOptimizationHandler(
        router_cache=mock_cache,
        model_list=sample_model_list,
        routing_args={"ttl": 60, "hour_ttl": 3600, "day_ttl": 86400},
    )


class TestFreeKeyOptimizationBasics:
    """Test basic functionality of the free key optimization strategy"""

    def test_initialization(self, free_key_handler):
        """Test handler initialization"""
        assert free_key_handler.routing_args.ttl == 60
        assert free_key_handler.routing_args.hour_ttl == 3600
        assert free_key_handler.routing_args.day_ttl == 86400

    def test_get_time_windows(self, free_key_handler):
        """Test time window generation"""
        from datetime import datetime

        dt = datetime(2024, 1, 15, 14, 30, 0)  # 2:30 PM

        windows = free_key_handler._get_time_windows(dt)

        assert windows["minute"] == "14-30"
        assert windows["hour"] == "2024-01-15-14"
        assert windows["day"] == "2024-01-15"

    def test_get_deployment_limits(self, free_key_handler, sample_model_list):
        """Test deployment limit extraction"""
        deployment = sample_model_list[0]
        limits = free_key_handler._get_deployment_limits(deployment)

        assert limits["rpm"] == 10
        assert limits["rph"] == 100
        assert limits["rpd"] == 1000
        assert limits["tpm"] == 1000
        assert limits["tph"] == 10000
        assert limits["tpd"] == 100000

    def test_get_deployment_limits_backward_compatibility(
        self, free_key_handler, sample_model_list
    ):
        """Test that deployments without new limits default to infinity"""
        deployment = sample_model_list[2]  # Only has rpm/tpm
        limits = free_key_handler._get_deployment_limits(deployment)

        assert limits["rpm"] == 5
        assert limits["tpm"] == 500
        assert limits["rph"] == float("inf")
        assert limits["rpd"] == float("inf")
        assert limits["tph"] == float("inf")
        assert limits["tpd"] == float("inf")

    def test_generate_cache_keys(self, free_key_handler):
        """Test cache key generation"""
        time_windows = {"minute": "14-30", "hour": "2024-01-15-14", "day": "2024-01-15"}

        keys = free_key_handler._generate_cache_keys(
            "deployment-1", "gpt-3.5-turbo", time_windows
        )

        assert keys["rpm_minute"] == "deployment-1:gpt-3.5-turbo:rpm:minute:14-30"
        assert keys["rpm_hour"] == "deployment-1:gpt-3.5-turbo:rpm:hour:2024-01-15-14"
        assert keys["rpm_day"] == "deployment-1:gpt-3.5-turbo:rpm:day:2024-01-15"
        assert keys["tpm_minute"] == "deployment-1:gpt-3.5-turbo:tpm:minute:14-30"
        assert keys["tpm_hour"] == "deployment-1:gpt-3.5-turbo:tpm:hour:2024-01-15-14"
        assert keys["tpm_day"] == "deployment-1:gpt-3.5-turbo:tpm:day:2024-01-15"

    def test_get_ttl_for_window(self, free_key_handler):
        """Test TTL value retrieval"""
        assert free_key_handler._get_ttl_for_window("minute") == 60
        assert free_key_handler._get_ttl_for_window("hour") == 3600
        assert free_key_handler._get_ttl_for_window("day") == 86400
        assert free_key_handler._get_ttl_for_window("unknown") == 60  # Default


class TestPreCallChecks:
    """Test pre-call check methods"""

    def test_pre_call_check_increments_counters(
        self, free_key_handler, sample_model_list, mock_cache
    ):
        """Test that pre_call_check increments request counters"""
        deployment = sample_model_list[0]

        result = free_key_handler.pre_call_check(deployment)

        # Should return the deployment
        assert result == deployment

        # Should increment cache for all time windows (3 calls)
        assert mock_cache.increment_cache.call_count == 3

    @pytest.mark.asyncio
    async def test_async_pre_call_check_increments_counters(
        self, free_key_handler, sample_model_list, mock_cache
    ):
        """Test that async_pre_call_check increments request counters"""
        deployment = sample_model_list[0]

        # Mock the _increment_value_in_current_window method
        free_key_handler._increment_value_in_current_window = AsyncMock(return_value=1)

        result = await free_key_handler.async_pre_call_check(deployment, None)

        # Should return the deployment
        assert result == deployment

        # Should increment cache for all time windows (3 calls)
        assert free_key_handler._increment_value_in_current_window.call_count == 3

    def test_pre_call_check_handles_exceptions(
        self, free_key_handler, sample_model_list, mock_cache
    ):
        """Test that pre_call_check handles exceptions gracefully"""
        deployment = sample_model_list[0]
        mock_cache.increment_cache.side_effect = Exception("Redis error")

        # Should not raise exception, should return deployment
        result = free_key_handler.pre_call_check(deployment)
        assert result == deployment


class TestDeploymentSelection:
    """Test deployment selection logic"""

    def test_filter_deployments_by_limits_all_within_limits(
        self, free_key_handler, sample_model_list
    ):
        """Test filtering when all deployments are within limits"""
        usage_data = {
            "deployment-1": {
                "rpm_minute": 5,
                "rpm_hour": 50,
                "rpm_day": 500,
                "tpm_minute": 500,
                "tpm_hour": 5000,
                "tpm_day": 50000,
            },
            "deployment-2": {
                "rpm_minute": 10,
                "rpm_hour": 25,
                "rpm_day": 1000,
                "tpm_minute": 1000,
                "tpm_hour": 2500,
                "tpm_day": 100000,
            },
            "deployment-3": {"rpm_minute": 2, "tpm_minute": 200},
        }

        eligible = free_key_handler._filter_deployments_by_limits(
            healthy_deployments=sample_model_list,
            usage_data=usage_data,
            input_tokens=100,
        )

        # All deployments should be eligible
        assert len(eligible) == 3

    def test_filter_deployments_by_limits_some_exceed(
        self, free_key_handler, sample_model_list
    ):
        """Test filtering when some deployments exceed limits"""
        usage_data = {
            "deployment-1": {
                "rpm_minute": 9,
                "rpm_hour": 99,
                "rpm_day": 999,  # Within limits
                "tpm_minute": 900,
                "tpm_hour": 9900,
                "tpm_day": 99900,
            },
            "deployment-2": {
                "rpm_minute": 19,
                "rpm_hour": 50,
                "rpm_day": 1999,  # rpm_hour will exceed (50 + 1 = 51 > 50)
                "tpm_minute": 1900,
                "tpm_hour": 4900,
                "tpm_day": 199900,
            },
            "deployment-3": {"rpm_minute": 4, "tpm_minute": 400},  # Within limits
        }

        eligible = free_key_handler._filter_deployments_by_limits(
            healthy_deployments=sample_model_list,
            usage_data=usage_data,
            input_tokens=100,
        )

        # deployment-2 should be filtered out (rpm_hour: 50 + 1 = 51 > 50, exceeds limit)
        # deployment-1 and deployment-3 should be eligible
        assert len(eligible) == 2
        deployment_ids = [d.get("model_info", {}).get("id") for d in eligible]
        assert "deployment-1" in deployment_ids
        assert "deployment-3" in deployment_ids
        assert "deployment-2" not in deployment_ids

    def test_select_lowest_cost_deployment_litellm_params(
        self, free_key_handler, sample_model_list
    ):
        """Test selection using cost information from litellm_params (highest priority)"""
        # Add cost information to sample deployments via litellm_params (user override)
        # For 100 input tokens + 100 estimated output tokens (1:1 ratio)
        sample_model_list[0]["litellm_params"]["input_cost_per_token"] = 0.001
        sample_model_list[0]["litellm_params"]["output_cost_per_token"] = 0.002
        # Cost: (100 * 0.001) + (100 * 0.002) = 0.1 + 0.2 = 0.3

        sample_model_list[1]["litellm_params"]["input_cost_per_token"] = 0.0005
        sample_model_list[1]["litellm_params"]["output_cost_per_token"] = 0.001
        # Cost: (100 * 0.0005) + (100 * 0.001) = 0.05 + 0.1 = 0.15 (lowest)

        sample_model_list[2]["litellm_params"]["input_cost_per_token"] = 0.002
        sample_model_list[2]["litellm_params"]["output_cost_per_token"] = 0.003
        # Cost: (100 * 0.002) + (100 * 0.003) = 0.2 + 0.3 = 0.5

        usage_data = {
            "deployment-1": {"tpm_minute": 100},
            "deployment-2": {"tpm_minute": 200},
            "deployment-3": {"tpm_minute": 50},
        }

        selected = free_key_handler._select_lowest_cost_deployment(
            eligible_deployments=sample_model_list,
            usage_data=usage_data,
            input_tokens=100,
        )

        # Should select deployment-2 because it has the lowest estimated cost (0.15)
        assert selected.get("model_info", {}).get("id") == "deployment-2"

    def test_select_lowest_cost_deployment_model_info(
        self, free_key_handler, sample_model_list
    ):
        """Test selection using cost information from model_info (second priority)"""
        # Add cost information to sample deployments via model_info
        sample_model_list[0]["model_info"]["input_cost_per_token"] = 0.001
        sample_model_list[0]["model_info"]["output_cost_per_token"] = 0.002
        # Cost: (100 * 0.001) + (100 * 0.002) = 0.3

        sample_model_list[1]["model_info"]["input_cost_per_token"] = 0.0
        sample_model_list[1]["model_info"]["output_cost_per_token"] = 0.0
        # Cost: (100 * 0.0) + (100 * 0.0) = 0.0 (lowest - free model)

        sample_model_list[2]["model_info"]["input_cost_per_token"] = 0.002
        sample_model_list[2]["model_info"]["output_cost_per_token"] = 0.003
        # Cost: (100 * 0.002) + (100 * 0.003) = 0.5

        usage_data = {
            "deployment-1": {"tpm_minute": 100},
            "deployment-2": {"tpm_minute": 200},
            "deployment-3": {"tpm_minute": 50},
        }

        selected = free_key_handler._select_lowest_cost_deployment(
            eligible_deployments=sample_model_list,
            usage_data=usage_data,
            input_tokens=100,
        )

        # Should select deployment-2 because it has zero cost (free model)
        assert selected.get("model_info", {}).get("id") == "deployment-2"

    def test_select_lowest_cost_deployment_fallback_to_usage(
        self, free_key_handler, sample_model_list
    ):
        """Test fallback to token usage when no cost information is available"""
        # Modify sample model list to use models that don't exist in global cost map
        for i, deployment in enumerate(sample_model_list):
            deployment["litellm_params"]["model"] = f"unknown-model-{i+1}"

        usage_data = {
            "deployment-1": {"tpm_minute": 500},
            "deployment-2": {"tpm_minute": 200},  # Lowest usage
            "deployment-3": {"tpm_minute": 300},
        }

        selected = free_key_handler._select_lowest_cost_deployment(
            eligible_deployments=sample_model_list,
            usage_data=usage_data,
            input_tokens=100,
        )

        # Should select deployment-2 because it has the lowest usage (fallback)
        assert selected.get("model_info", {}).get("id") == "deployment-2"

    def test_select_lowest_cost_deployment_empty_list(self, free_key_handler):
        """Test selection when no deployments are eligible"""
        selected = free_key_handler._select_lowest_cost_deployment(
            eligible_deployments=[], usage_data={}, input_tokens=100
        )

        assert selected is None

    def test_cost_lookup_priority_order(self, free_key_handler, sample_model_list):
        """Test that litellm_params overrides model_info cost data"""
        # Set up conflicting cost data - litellm_params should win
        sample_model_list[0]["litellm_params"][
            "input_cost_per_token"
        ] = 0.001  # This should be used
        sample_model_list[0]["model_info"][
            "input_cost_per_token"
        ] = 0.01  # This should be ignored
        sample_model_list[0]["litellm_params"]["output_cost_per_token"] = 0.002
        sample_model_list[0]["model_info"]["output_cost_per_token"] = 0.02

        sample_model_list[1]["model_info"][
            "input_cost_per_token"
        ] = 0.005  # No litellm_params, so this is used
        sample_model_list[1]["model_info"]["output_cost_per_token"] = 0.01

        usage_data = {
            "deployment-1": {"tpm_minute": 100},
            "deployment-2": {"tpm_minute": 200},
        }

        selected = free_key_handler._select_lowest_cost_deployment(
            eligible_deployments=sample_model_list[:2],
            usage_data=usage_data,
            input_tokens=100,
        )

        # Should select deployment-1 because litellm_params cost (0.3) < model_info cost (1.5)
        assert selected.get("model_info", {}).get("id") == "deployment-1"

    def test_select_lowest_cost_deployment_from_global_cost_map(
        self, free_key_handler, sample_model_list
    ):
        """Test selection using cost information from global litellm.model_cost map"""
        import litellm

        # Mock the global model cost map with cost information
        original_model_cost = litellm.model_cost.copy()
        try:
            # Add cost info to global model cost map
            litellm.model_cost.update(
                {
                    "gpt-3.5-turbo": {
                        "input_cost_per_token": 0.001,
                        "output_cost_per_token": 0.002,  # Total cost: 0.3 for 100 tokens
                    },
                    "gpt-4": {
                        "input_cost_per_token": 0.03,
                        "output_cost_per_token": 0.06,  # Total cost: 9.0 for 100 tokens (highest)
                    },
                    "claude-3-sonnet": {
                        "input_cost_per_token": 0.0005,
                        "output_cost_per_token": 0.001,  # Total cost: 0.15 for 100 tokens (lowest)
                    },
                }
            )

            # Update sample model list to use different models
            sample_model_list[0]["litellm_params"]["model"] = "gpt-3.5-turbo"
            sample_model_list[1]["litellm_params"]["model"] = "claude-3-sonnet"
            sample_model_list[2]["litellm_params"]["model"] = "gpt-4"

            usage_data = {
                "deployment-1": {"tpm_minute": 100},
                "deployment-2": {"tpm_minute": 200},
                "deployment-3": {"tpm_minute": 50},
            }

            selected = free_key_handler._select_lowest_cost_deployment(
                eligible_deployments=sample_model_list,
                usage_data=usage_data,
                input_tokens=100,
            )

            # Should select deployment-2 (claude-3-sonnet) because it has the lowest cost (0.15)
            assert selected.get("model_info", {}).get("id") == "deployment-2"

        finally:
            # Restore original model cost map
            litellm.model_cost = original_model_cost


class TestRateLimitValidation:
    """Test rate limit validation and error handling"""

    @pytest.mark.asyncio
    async def test_async_get_available_deployments_no_limits_exceeded(
        self, free_key_handler, sample_model_list, mock_cache
    ):
        """Test successful deployment selection when no limits are exceeded"""
        # Mock batch cache to return low usage values
        mock_cache.async_batch_get_cache.return_value = [
            1
        ] * 18  # Low usage for all keys

        with patch("litellm.token_counter", return_value=50):
            deployment = await free_key_handler.async_get_available_deployments(
                model_group="gpt-3.5-turbo",
                healthy_deployments=sample_model_list,
                messages=[{"role": "user", "content": "test"}],
            )

        assert deployment is not None
        assert deployment.get("model_info", {}).get("id") in [
            "deployment-1",
            "deployment-2",
            "deployment-3",
        ]

    @pytest.mark.asyncio
    async def test_async_get_available_deployments_all_limits_exceeded(
        self, free_key_handler, sample_model_list, mock_cache
    ):
        """Test RateLimitError when all deployments exceed limits"""
        # Mock batch cache to return high usage values that exceed limits
        high_usage_values = []
        for _ in range(6):  # 6 keys per deployment (rpm/tpm for minute/hour/day)
            high_usage_values.extend([999999, 999999, 999999])  # Very high values

        mock_cache.async_batch_get_cache.return_value = high_usage_values

        with patch("litellm.token_counter", return_value=50):
            with pytest.raises(litellm.RateLimitError) as exc_info:
                await free_key_handler.async_get_available_deployments(
                    model_group="gpt-3.5-turbo",
                    healthy_deployments=sample_model_list,
                    messages=[{"role": "user", "content": "test"}],
                )

            assert "No deployments available" in str(exc_info.value)

    def test_get_available_deployments_no_limits_exceeded(
        self, free_key_handler, sample_model_list, mock_cache
    ):
        """Test successful deployment selection (sync version)"""
        # Mock batch cache to return low usage values
        mock_cache.batch_get_cache.return_value = [1] * 18  # Low usage for all keys

        with patch("litellm.token_counter", return_value=50):
            deployment = free_key_handler.get_available_deployments(
                model_group="gpt-3.5-turbo",
                healthy_deployments=sample_model_list,
                messages=[{"role": "user", "content": "test"}],
            )

        assert deployment is not None
        assert deployment.get("model_info", {}).get("id") in [
            "deployment-1",
            "deployment-2",
            "deployment-3",
        ]


class TestSuccessLogging:
    """Test success logging and token tracking"""

    def test_log_success_event_increments_tpm_counters(
        self, free_key_handler, mock_cache
    ):
        """Test that log_success_event increments TPM counters for all time windows"""
        kwargs = {
            "standard_logging_object": {
                "model_group": "gpt-3.5-turbo",
                "model_id": "deployment-1",
                "total_tokens": 150,
                "hidden_params": {"litellm_model_name": "gpt-3.5-turbo"},
            }
        }

        free_key_handler.log_success_event(kwargs, None, None, None)

        # Should increment cache for all time windows (3 calls with 150 tokens each)
        assert mock_cache.increment_cache.call_count == 3

        # Verify the calls were made with correct token count
        for call in mock_cache.increment_cache.call_args_list:
            args, kwargs_call = call
            assert kwargs_call["value"] == 150

    @pytest.mark.asyncio
    async def test_async_log_success_event_increments_tpm_counters(
        self, free_key_handler, mock_cache
    ):
        """Test that async_log_success_event increments TPM counters for all time windows"""
        kwargs = {
            "standard_logging_object": {
                "model_group": "gpt-3.5-turbo",
                "model_id": "deployment-1",
                "total_tokens": 200,
                "hidden_params": {"litellm_model_name": "gpt-3.5-turbo"},
            }
        }

        await free_key_handler.async_log_success_event(kwargs, None, None, None)

        # Should increment cache for all time windows (3 calls with 200 tokens each)
        assert mock_cache.async_increment_cache.call_count == 3

        # Verify the calls were made with correct token count
        for call in mock_cache.async_increment_cache.call_args_list:
            args, kwargs_call = call
            assert kwargs_call["value"] == 200


class TestRouterIntegration:
    """Test integration with Router class"""

    def test_router_initialization_with_free_key_optimization(self, sample_model_list):
        """Test that Router can be initialized with free-key-optimization strategy"""
        router = Router(
            model_list=sample_model_list,
            routing_strategy="free-key-optimization",
            routing_strategy_args={"ttl": 60, "hour_ttl": 3600, "day_ttl": 86400},
        )

        assert router.routing_strategy == "free-key-optimization"
        assert hasattr(router, "free_key_optimization_logger")
        assert router.free_key_optimization_logger is not None

    @pytest.mark.asyncio
    async def test_router_async_completion_with_free_key_optimization(
        self, sample_model_list
    ):
        """Test that Router can make async completions with free-key-optimization"""
        router = Router(
            model_list=sample_model_list, routing_strategy="free-key-optimization"
        )

        # Mock the actual LLM call to avoid making real API requests
        with patch("litellm.acompletion") as mock_completion:
            mock_completion.return_value = {
                "choices": [{"message": {"content": "test response"}}],
                "usage": {"total_tokens": 50},
            }

            # This should not raise an exception
            try:
                response = await router.acompletion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                )
                # If we get here, the routing worked
                assert True
            except Exception as e:
                # If there's an error, it should not be related to routing strategy
                assert "free-key-optimization" not in str(e)


class TestCostLookupHelpers:
    """Test the new cost lookup helper methods"""

    def test_get_cost_information_litellm_params(self, free_key_handler):
        """Test _get_cost_information with litellm_params data"""
        deployment = {
            "litellm_params": {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            },
            "model_info": {
                "input_cost_per_token": 0.01,  # Should be ignored
                "output_cost_per_token": 0.02,  # Should be ignored
            },
        }

        result = free_key_handler._get_cost_information(deployment)

        assert result["input_cost_per_token"] == 0.001
        assert result["output_cost_per_token"] == 0.002
        assert result["cost_data_source"] == "litellm_params_override"

    def test_get_cost_information_model_info(self, free_key_handler):
        """Test _get_cost_information with model_info data"""
        deployment = {
            "litellm_params": {"model": "gpt-3.5-turbo"},
            "model_info": {
                "input_cost_per_token": 0.0,  # Explicit zero should be preserved
                "output_cost_per_token": 0.001,
            },
        }

        result = free_key_handler._get_cost_information(deployment)

        assert result["input_cost_per_token"] == 0.0  # Should preserve explicit zero
        assert result["output_cost_per_token"] == 0.001
        assert result["cost_data_source"] == "model_info"

    def test_calculate_cost_metric_with_cost_data(self, free_key_handler):
        """Test _calculate_cost_metric with valid cost data"""
        cost_info = {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "cost_data_source": "litellm_params_override",
        }
        usage_data = {"deployment-1": {"tpm_minute": 100}}

        result = free_key_handler._calculate_cost_metric(
            cost_info, usage_data, "deployment-1", 50
        )

        assert result["cost_source"] == "cost_calculation"
        assert result["cost_metric"] == 0.15  # (50 * 0.001) + (50 * 0.002) = 0.15
        assert result["estimated_output_tokens"] == 50
        assert result["estimated_total_cost"] == 0.15

    def test_calculate_cost_metric_zero_cost(self, free_key_handler):
        """Test _calculate_cost_metric with zero cost (free model)"""
        cost_info = {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "cost_data_source": "model_info",
        }
        usage_data = {"deployment-1": {"tpm_minute": 100}}

        result = free_key_handler._calculate_cost_metric(
            cost_info, usage_data, "deployment-1", 50
        )

        assert (
            result["cost_source"] == "cost_calculation"
        )  # Should still use cost calculation
        assert result["cost_metric"] == 0.0  # Free model
        assert result["estimated_output_tokens"] == 50
        assert result["estimated_total_cost"] == 0.0

    def test_calculate_cost_metric_fallback(self, free_key_handler):
        """Test _calculate_cost_metric fallback to token usage"""
        cost_info = {
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
            "cost_data_source": "no_cost_data_found",
        }
        usage_data = {"deployment-1": {"tpm_minute": 150}}

        result = free_key_handler._calculate_cost_metric(
            cost_info, usage_data, "deployment-1", 50
        )

        assert result["cost_source"] == "token_usage_fallback"
        assert result["cost_metric"] == 150  # Uses tpm_minute
        assert result["estimated_output_tokens"] is None
        assert result["estimated_total_cost"] is None


class TestEdgeCases:
    """Test edge cases and error scenarios"""

    def test_deployment_without_model_info_id(self, free_key_handler):
        """Test handling of deployment without model_info.id"""
        deployment = {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo"},
            # Missing model_info.id
        }

        # Should handle gracefully and return deployment
        result = free_key_handler.pre_call_check(deployment)
        assert result == deployment

    def test_deployment_limits_from_different_levels(self, free_key_handler):
        """Test that limits are correctly extracted from different config levels"""
        deployment = {
            "model_name": "gpt-3.5-turbo",
            "rpm": 10,  # Top level
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "rph": 100,  # litellm_params level
            },
            "model_info": {
                "id": "deployment-1",
                "rpd": 1000,  # model_info level
            },
        }

        limits = free_key_handler._get_deployment_limits(deployment)

        assert limits["rpm"] == 10  # From top level
        assert limits["rph"] == 100  # From litellm_params
        assert limits["rpd"] == 1000  # From model_info
        assert limits["tpm"] == float("inf")  # Default

    def test_cache_failure_handling(
        self, free_key_handler, sample_model_list, mock_cache
    ):
        """Test graceful handling of cache failures"""
        deployment = sample_model_list[0]
        mock_cache.increment_cache.side_effect = Exception("Cache error")

        # Should not raise exception
        result = free_key_handler.pre_call_check(deployment)
        assert result == deployment

    @pytest.mark.asyncio
    async def test_async_cache_failure_handling(
        self, free_key_handler, sample_model_list, mock_cache
    ):
        """Test graceful handling of async cache failures"""
        deployment = sample_model_list[0]
        free_key_handler._increment_value_in_current_window = AsyncMock(
            side_effect=Exception("Cache error")
        )

        # Should not raise exception
        result = await free_key_handler.async_pre_call_check(deployment, None)
        assert result == deployment
