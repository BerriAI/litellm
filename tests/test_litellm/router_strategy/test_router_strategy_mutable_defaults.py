from unittest.mock import MagicMock

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2


class TestRouterStrategyInstanceIsolation:
    """Behavioral regression tests ensuring strategy handlers maintain isolated state when initialized with default arguments."""

    @pytest.fixture
    def mock_cache(self) -> MagicMock:
        return MagicMock(spec=DualCache)

    def test_lowest_cost_isolated_routing_args(self, mock_cache: MagicMock) -> None:
        """Verify LowestCostLoggingHandler instances do not share routing_args dict."""
        h1 = LowestCostLoggingHandler(router_cache=mock_cache)
        h2 = LowestCostLoggingHandler(router_cache=mock_cache)

        assert h1.routing_args is not h2.routing_args
        assert h1.routing_args == {}

        h1.routing_args["ttl"] = 999
        assert "ttl" not in h2.routing_args

    def test_lowest_latency_isolated_routing_args(self, mock_cache: MagicMock) -> None:
        """Verify LowestLatencyLoggingHandler instances initialize independent RoutingArgs objects."""
        h1 = LowestLatencyLoggingHandler(router_cache=mock_cache)
        h2 = LowestLatencyLoggingHandler(router_cache=mock_cache)

        assert h1.routing_args is not h2.routing_args
        assert h1.routing_args.ttl == 3600

        h1.routing_args.ttl = 120
        assert h2.routing_args.ttl == 3600

    def test_lowest_tpm_isolated_routing_args(self, mock_cache: MagicMock) -> None:
        """Verify LowestTPMLoggingHandler instances initialize independent RoutingArgs objects."""
        h1 = LowestTPMLoggingHandler(router_cache=mock_cache)
        h2 = LowestTPMLoggingHandler(router_cache=mock_cache)

        assert h1.routing_args is not h2.routing_args
        assert h1.routing_args.ttl == 60

        h1.routing_args.ttl = 300
        assert h2.routing_args.ttl == 60

    @pytest.mark.asyncio
    async def test_lowest_tpm_v2_isolated_routing_args(self, mock_cache: MagicMock) -> None:
        """Verify LowestTPMLoggingHandler_v2 instances initialize independent RoutingArgs objects."""
        h1 = LowestTPMLoggingHandler_v2(router_cache=mock_cache)
        h2 = LowestTPMLoggingHandler_v2(router_cache=mock_cache)

        try:
            assert h1.routing_args is not h2.routing_args
            assert h1.routing_args.ttl == 60

            h1.routing_args.ttl = 300
            assert h2.routing_args.ttl == 60
        finally:
            await h1.cleanup()
            await h2.cleanup()
