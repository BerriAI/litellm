from typing import Final

import pytest
from fastapi import HTTPException, Request

from litellm.proxy import proxy_server
from litellm.proxy._types import ConfigGeneralSettings
from litellm.proxy.utils import PrismaClient
from litellm.proxy.workflows.lifespan import get_workflow_runner, require_workflows_enabled, workflow_lifespan
from litellm.proxy.workflows.runner import WorkflowRunner
from litellm.repositories.database import Database


def test_workflows_default_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ConfigGeneralSettings().enable_workflows is False
    monkeypatch.setattr(proxy_server, "general_settings", {})
    with pytest.raises(HTTPException) as rejected:
        require_workflows_enabled()
    assert rejected.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_opt_in_preserves_request_owned_runner_context(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": enabled})
    prisma: Final = PrismaClient("postgresql://unused", proxy_server.proxy_logging_obj)
    database: Final = await Database.open(prisma)
    try:
        async with workflow_lifespan(database) as runner:
            assert isinstance(runner, WorkflowRunner) is enabled
            request: Final = Request({"type": "http", "state": {"workflow_runner": runner}})
            assert get_workflow_runner(request) is runner
        async with workflow_lifespan(None) as absent:
            assert absent is None
    finally:
        await database.aclose()
