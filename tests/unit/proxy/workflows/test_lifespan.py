from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy._types import ConfigGeneralSettings
from litellm.proxy.workflows import lifespan


def test_workflows_default_to_disabled_and_require_explicit_enablement(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ConfigGeneralSettings().enable_workflows is False
    monkeypatch.setattr(proxy_server, "general_settings", {})
    assert not lifespan.workflows_enabled()
    with pytest.raises(HTTPException) as rejected:
        lifespan.require_workflows_enabled()
    assert rejected.value.status_code == 404
    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": True})
    assert ConfigGeneralSettings(enable_workflows=True).enable_workflows is True
    assert lifespan.workflows_enabled()
    lifespan.require_workflows_enabled()


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [{}, {"enable_workflows": False}])
async def test_disabled_workflows_do_not_create_repositories_or_start_runner(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, bool]
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", settings)
    definitions = AsyncMock(side_effect=AssertionError("Disabled workflows must not initialize storage"))
    monkeypatch.setattr(lifespan, "get_workflow_definition_repository", definitions)
    async with lifespan.workflow_lifespan(Mock()):
        pass
    definitions.assert_not_awaited()


@pytest.mark.asyncio
async def test_enabled_workflows_start_runner_and_close_it_on_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": True})
    monkeypatch.setattr(lifespan, "get_workflow_definition_repository", AsyncMock(return_value=Mock()))
    monkeypatch.setattr(lifespan, "get_workflow_execution_repository", AsyncMock(return_value=Mock()))
    runner = Mock(start=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(lifespan, "WorkflowRunner", Mock(return_value=runner))
    async with lifespan.workflow_lifespan(Mock()):
        runner.start.assert_awaited_once()
        runner.close.assert_not_awaited()
    runner.close.assert_awaited_once()
