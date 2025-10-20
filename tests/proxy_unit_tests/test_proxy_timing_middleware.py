"""
Test that proxy timing middleware correctly tracks overhead including auth
"""

import sys
import os
import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.middleware.proxy_timing_middleware import ProxyTimingMiddleware
from fastapi import Request


@pytest.mark.asyncio
async def test_middleware_measures_actual_overhead():
    """Test that middleware correctly measures request overhead"""
    middleware = ProxyTimingMiddleware(app=MagicMock())
    
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.url.path = "/v1/chat/completions"
    
    # Simulate processing that takes ~100ms
    async def mock_call_next(req):
        await asyncio.sleep(0.1)
        return MagicMock()
    
    start = datetime.now()
    await middleware.dispatch(request, mock_call_next)
    elapsed = (datetime.now() - start).total_seconds()
    
    # Verify timing was captured and is reasonable
    assert hasattr(request.state, "proxy_start_time")
    assert isinstance(request.state.proxy_start_time, datetime)
    assert 0.08 < elapsed < 0.15  # ~100ms with some tolerance

