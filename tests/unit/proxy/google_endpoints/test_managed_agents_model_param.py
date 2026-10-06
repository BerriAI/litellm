"""
Tests verifying that managed-agent proxy endpoints never pass the agent name
as the ``model`` parameter to ``base_process_llm_request``.

Passing ``model=<agent_name>`` would cause ``common_processing_pre_call_logic``
to write the agent name into ``self.data["model"]``, which triggers spurious
model-alias mapping, rate-limiting lookups, and logging tied to a
non-existent model deployment.  The agent name is already carried in
``data["name"]`` and must not pollute the ``model`` slot.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _build_agents_client():
    """Build a TestClient whose auth dependency is overridden to a PROXY_ADMIN
    user. Using ``dependency_overrides`` is the only reliable way to bypass the
    real ``user_api_key_auth`` for FastAPI route tests — patching the module-
    level name does not affect the function reference captured by ``Depends``.
    The PROXY_ADMIN role also bypasses the caller-supplied-api_key guard so
    these tests can focus on the ``model=None`` invariant.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.google_endpoints.agents_endpoints import router as agents_router

    app = FastAPI()
    app.include_router(agents_router)

    async def _fake_user_api_key_auth():
        return UserAPIKeyAuth(
            api_key="sk-test",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

    app.dependency_overrides[user_api_key_auth] = _fake_user_api_key_auth
    return TestClient(app)


def _patch_proxy_server_imports(client=None):
    """Return a context-manager that stubs _proxy_server_imports so tests
    don't need a running proxy."""
    mock_srv = {
        "general_settings": {},
        "llm_router": MagicMock(),
        "proxy_config": MagicMock(),
        "proxy_logging_obj": MagicMock(),
        "select_data_generator": None,
        "user_api_base": None,
        "user_max_tokens": None,
        "user_model": None,
        "user_request_timeout": None,
        "user_temperature": None,
        "version": "0.0.0",
    }
    return patch(
        "litellm.proxy.google_endpoints.agents_endpoints._proxy_server_imports",
        return_value=mock_srv,
    )


def _patch_base_process(return_value=None):
    if return_value is None:
        return_value = {"name": "agents/my-agent", "displayName": "My Agent"}
    return patch(
        "litellm.proxy.google_endpoints.agents_endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
        return_value=return_value,
    )


def _patch_auth():
    """Deprecated no-op kept for call-site compatibility.

    ``_build_agents_client`` now installs a FastAPI ``dependency_overrides``
    entry that injects a PROXY_ADMIN ``UserAPIKeyAuth``, so individual tests
    no longer need to patch the module-level ``user_api_key_auth`` name.
    """
    return patch("os.getpid")


class TestManagedAgentsModelParam:
    """Endpoints must pass model=None, not the agent name, to base_process_llm_request."""

    def test_create_agent_passes_model_none(self):
        """POST /v1beta/agents: model kwarg must be None, not the name field."""
        try:
            client = _build_agents_client()
        except ImportError as exc:
            pytest.skip(f"Skipping: missing dependency {exc}")

        with (
            _patch_proxy_server_imports(),
            _patch_base_process() as mock_process,
            _patch_auth(),
        ):
            client.post(
                "/v1beta/agents",
                json={
                    "name": "my-custom-slides-agent",
                    "base_agent": "waverunner",
                    "instructions": "Be helpful.",
                },
            )

        mock_process.assert_called_once()
        kwargs = mock_process.call_args.kwargs
        assert kwargs["model"] is None, (
            f"create_gemini_agent must not pass model={kwargs['model']!r}; "
            "the agent name must stay in data['name'], not pollute data['model']"
        )
        assert kwargs["route_type"] == "acreate_agent"

    def test_get_agent_passes_model_none(self):
        """GET /v1beta/agents/{name}: model kwarg must be None."""
        try:
            client = _build_agents_client()
        except ImportError as exc:
            pytest.skip(f"Skipping: missing dependency {exc}")

        with (
            _patch_proxy_server_imports(),
            _patch_base_process() as mock_process,
            _patch_auth(),
        ):
            client.get("/v1beta/agents/my-custom-slides-agent")

        mock_process.assert_called_once()
        kwargs = mock_process.call_args.kwargs
        assert (
            kwargs["model"] is None
        ), f"get_gemini_agent must not pass model={kwargs['model']!r}"
        assert kwargs["route_type"] == "aget_agent"

    def test_delete_agent_passes_model_none(self):
        """DELETE /v1beta/agents/{name}: model kwarg must be None."""
        try:
            client = _build_agents_client()
        except ImportError as exc:
            pytest.skip(f"Skipping: missing dependency {exc}")

        with (
            _patch_proxy_server_imports(),
            _patch_base_process() as mock_process,
            _patch_auth(),
        ):
            client.delete("/v1beta/agents/my-custom-slides-agent")

        mock_process.assert_called_once()
        kwargs = mock_process.call_args.kwargs
        assert (
            kwargs["model"] is None
        ), f"delete_gemini_agent must not pass model={kwargs['model']!r}"
        assert kwargs["route_type"] == "adelete_agent"

    def test_list_agent_versions_passes_model_none(self):
        """GET /v1beta/agents/{name}/versions: model kwarg must be None."""
        try:
            client = _build_agents_client()
        except ImportError as exc:
            pytest.skip(f"Skipping: missing dependency {exc}")

        with (
            _patch_proxy_server_imports(),
            _patch_base_process() as mock_process,
            _patch_auth(),
        ):
            client.get("/v1beta/agents/my-custom-slides-agent/versions")

        mock_process.assert_called_once()
        kwargs = mock_process.call_args.kwargs
        assert (
            kwargs["model"] is None
        ), f"list_gemini_agent_versions must not pass model={kwargs['model']!r}"
        assert kwargs["route_type"] == "alist_agent_versions"

    def test_list_agents_already_passes_model_none(self):
        """GET /v1beta/agents: existing list endpoint already passes model=None — keep it so."""
        try:
            client = _build_agents_client()
        except ImportError as exc:
            pytest.skip(f"Skipping: missing dependency {exc}")

        with (
            _patch_proxy_server_imports(),
            _patch_base_process(return_value={"agents": []}) as mock_process,
            _patch_auth(),
        ):
            client.get("/v1beta/agents")

        mock_process.assert_called_once()
        kwargs = mock_process.call_args.kwargs
        assert kwargs["model"] is None
        assert kwargs["route_type"] == "alist_agents"
