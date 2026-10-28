from dataclasses import dataclass
from typing import Final
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

import litellm.proxy.proxy_server as proxy_server
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    end_user_cache_key,
    project_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router

_ProxyObject = (
    LiteLLM_TeamTableCachedObj
    | LiteLLM_UserTable
    | LiteLLM_EndUserTable
    | LiteLLM_ProjectTableCachedObj
    | LiteLLM_TeamMembership
)


def _cache_key(row: _ProxyObject) -> str:
    match row:
        case LiteLLM_TeamTableCachedObj():
            return f"team_id:{row.team_id}"
        case LiteLLM_UserTable():
            return row.user_id
        case LiteLLM_EndUserTable():
            return end_user_cache_key(row.user_id)
        case LiteLLM_ProjectTableCachedObj():
            return project_cache_key(row.project_id)
        case LiteLLM_TeamMembership():
            return team_membership_reservation_cache_key(user_id=row.user_id, team_id=row.team_id)


@dataclass(frozen=True, slots=True)
class ProxyAuthState:
    """A running proxy with a master key, so the centralized auth checks run, and its auth cache to seed."""

    cache: UserApiKeyCache
    monkeypatch: pytest.MonkeyPatch

    def seed(self, *rows: _ProxyObject) -> None:
        for row in rows:
            stored: Final[BaseModel] = row
            self.cache.set_cache(key=_cache_key(row), value=stored)

    def serve(self, router: Router) -> None:
        self.monkeypatch.setattr(proxy_server, "llm_router", router)


@pytest.fixture
def proxy_auth_state(monkeypatch: pytest.MonkeyPatch) -> ProxyAuthState:
    cache: Final = UserApiKeyCache()
    proxy_globals: Final = {
        "master_key": "sk-test-master",
        "prisma_client": MagicMock(),
        "user_api_key_cache": cache,
        "spend_counter_cache": DualCache(),
        "proxy_logging_obj": ProxyLogging(user_api_key_cache=cache),
        "general_settings": {},
        "user_custom_auth": None,
        "litellm_proxy_admin_name": "admin",
        "llm_router": None,
        "llm_model_list": None,
    }
    for name, value in proxy_globals.items():
        monkeypatch.setattr(proxy_server, name, value)
    return ProxyAuthState(cache=cache, monkeypatch=monkeypatch)
