import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from litellm.router_utils.search_api_router import SearchAPIRouter


def test_resolve_credentials_team_metadata_overrides_tool_params():
    api_key, api_base = SearchAPIRouter._resolve_search_provider_credentials(
        search_provider="tavily",
        tool_litellm_params={
            "api_key": "tool-key",
            "api_base": "https://tool.example.com",
        },
        team_metadata={
            "search_provider_config": {
                "tavily": {
                    "api_key": "team-key",
                    "api_base": "https://team.example.com",
                }
            }
        },
    )
    assert api_key == "team-key"
    assert api_base == "https://team.example.com"


def test_resolve_credentials_request_metadata_has_highest_precedence():
    api_key, api_base = SearchAPIRouter._resolve_search_provider_credentials(
        search_provider="tavily",
        tool_litellm_params={
            "api_key": "tool-key",
            "api_base": "https://tool.example.com",
        },
        request_metadata={
            "search_provider_config": {
                "tavily": {
                    "api_key": "request-key",
                    "api_base": "https://request.example.com",
                }
            }
        },
        team_metadata={
            "search_provider_config": {
                "tavily": {
                    "api_key": "team-key",
                    "api_base": "https://team.example.com",
                }
            }
        },
    )
    assert api_key == "request-key"
    assert api_base == "https://request.example.com"


def test_resolve_credentials_from_default_team_settings():
    with patch(
        "litellm.default_team_settings",
        [
            {
                "team_id": "team-a",
                "search_provider_config": {
                    "tavily": {
                        "api_key": "team-settings-key",
                        "api_base": "https://team-settings.example.com",
                    }
                },
            }
        ],
    ):
        team_config = SearchAPIRouter._get_team_config_from_default_settings("team-a")
        api_key, api_base = SearchAPIRouter._resolve_search_provider_credentials(
            search_provider="tavily",
            tool_litellm_params={},
            team_config=team_config,
        )
        assert api_key == "team-settings-key"
        assert api_base == "https://team-settings.example.com"


@pytest.mark.asyncio
async def test_search_endpoint_injects_team_metadata():
    captured_metadata = {}

    async def _mock_process(self, **kwargs):
        nonlocal captured_metadata
        captured_metadata = self.data.get("metadata", {})
        return {"object": "search", "results": []}

    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-user",
        team_id="team-test",
        team_metadata={
            "search_provider_config": {
                "tavily": {"api_key": "team-test-key"},
            }
        },
    )

    try:
        with patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            new=_mock_process,
        ):
            client = TestClient(app)
            response = client.post(
                "/v1/search",
                json={
                    "search_tool_name": "tool-a",
                    "search_provider": "tavily",
                    "query": "latest ai news",
                },
            )
            assert response.status_code == 200
            assert captured_metadata.get("user_api_key_team_id") == "team-test"
            assert (
                captured_metadata.get("user_api_key_team_metadata", {})
                .get("search_provider_config", {})
                .get("tavily", {})
                .get("api_key")
                == "team-test-key"
            )
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)
