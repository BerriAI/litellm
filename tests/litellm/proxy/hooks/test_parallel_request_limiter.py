import os
import sys

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.hooks.parallel_request_limiter import _PROXY_MaxParallelRequestsHandler
from litellm.proxy.utils import InternalUsageCache
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_async_pre_call_hook():
  dual_cache = DualCache()
  internal_usage_cache = InternalUsageCache(dual_cache=dual_cache)
  handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=internal_usage_cache)

  data={"metadata": {"global_max_parallel_requests": 1}}

  user_api_key_dict = UserAPIKeyAuth()

  # The first parallel request succeeds
  await handler.async_pre_call_hook(
    user_api_key_dict=user_api_key_dict,
    cache=dual_cache,
    data=data,
    call_type="completion",
  )

  # The second parallel request fails
  try:
    await handler.async_pre_call_hook(
      user_api_key_dict=user_api_key_dict,
      cache=dual_cache,
      data=data,
      call_type="completion",
    )
    assert False
  except HTTPException as e:
    pass