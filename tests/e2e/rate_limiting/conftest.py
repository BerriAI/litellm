from __future__ import annotations

import pytest

from rate_limiting_client import RateLimitingClient, build_client


@pytest.fixture
def client() -> RateLimitingClient:
    return build_client()
