from collections.abc import Callable
from datetime import datetime, timezone
from typing import Final

import pytest
from fastapi import Request

from litellm.caching.dual_cache import DualCache
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.proxy._types import LiteLLM_JWTAuth
from litellm.proxy.auth.host.python import HttpCall, NoRegisteredAgents, PythonAuthHost
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.model_max_budget_limiter import PROXY_VirtualKeyModelMaxBudgetLimiter
from litellm.proxy.utils import ProxyLogging


@pytest.fixture
def auth_host() -> Callable[..., PythonAuthHost]:
    def build(api_key: str | None = None) -> PythonAuthHost:
        cache: Final = UserApiKeyCache()
        request: Final = Request({"type": "http", "method": "POST", "path": "/chat/completions", "headers": []})
        return PythonAuthHost(
            route="/chat/completions",
            parent_otel_span=None,
            http=HttpCall(request=request, request_data={}, start_time=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            prisma_client=None,
            user_api_key_cache=cache,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=cache),
            llm_router=None,
            llm_model_list=None,
            model_max_budget_limiter=PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache()),
            litellm_proxy_admin_name="default_user_id",
            litellm_jwtauth=LiteLLM_JWTAuth(),
            agent_lookup=NoRegisteredAgents(),
            http_handler=HTTPHandler(),
            api_key=api_key,
        )

    return build
