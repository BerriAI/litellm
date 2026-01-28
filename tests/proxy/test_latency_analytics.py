
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
import sys
import os
from datetime import datetime

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

# MOCK PRISMA MODULE BEFORE IMPORT to avoid runtime errors
mock_prisma_module = MagicMock()
sys.modules["prisma"] = mock_prisma_module
mock_prisma_module.Prisma = MagicMock() 

try:
    from litellm.proxy.analytics.latency_analytics import get_latency_analytics
except ImportError:
    print("Could not import get_latency_analytics")
    sys.exit(1)

class TestLatencyAnalytics(unittest.IsolatedAsyncioTestCase):
    async def test_get_latency_analytics(self):
        # 1. Mock Request and App State
        mock_request = MagicMock()
        mock_prisma_client = MagicMock()
        mock_request.app.state.prisma_client = mock_prisma_client
        
        # 2. Mock DB Response
        # The query returns rows with model, request_count, p50_ms, etc.
        mock_rows = [
            {
                "model": "gpt-4",
                "request_count": 100,
                "p50_ms": 500.0,
                "p95_ms": 1200.0,
                "p99_ms": 2000.0
            },
            {
                "model": "claude-3",
                "request_count": 50,
                "p50_ms": 300.0,
                "p95_ms": 800.0,
                "p99_ms": 1500.0
            }
        ]
        mock_prisma_client.db.query_raw = AsyncMock(return_value=mock_rows)
        
        # 3. Call Endpoint Handler
        response = await get_latency_analytics(
            request=mock_request,
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-02T00:00:00"
        )
        
        # 4. Verify Response
        self.assertEqual(response.total_models, 2)
        self.assertEqual(len(response.latency_percentiles), 2)
        
        # Check first item
        item1 = response.latency_percentiles[0]
        self.assertEqual(item1.model, "gpt-4")
        self.assertEqual(item1.p50_ms, 500.0)
        
        # Verify SQL Query execution
        mock_prisma_client.db.query_raw.assert_called_once()
        call_args = mock_prisma_client.db.query_raw.call_args
        sql_query = call_args[0][0]
        self.assertIn("PERCENTILE_CONT", sql_query)
        self.assertIn("LiteLLM_SpendLogs", sql_query)

if __name__ == "__main__":
    unittest.main()
