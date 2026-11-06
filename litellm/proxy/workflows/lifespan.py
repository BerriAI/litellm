from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import Request

from litellm.proxy.deps.repositories import get_workflow_definition_repository, get_workflow_execution_repository
from litellm.proxy.workflows.gateway import execute_chat_node
from litellm.proxy.workflows.runner import WorkflowRunner
from litellm.repositories.database import Database


@asynccontextmanager
async def workflow_lifespan(database: Database | None) -> AsyncGenerator[WorkflowRunner | None]:
    definitions: Final = await get_workflow_definition_repository(database)
    executions: Final = await get_workflow_execution_repository(database)
    runner: Final = (
        WorkflowRunner(executions, definitions, execute_chat_node)
        if definitions is not None and executions is not None
        else None
    )
    try:
        yield runner
    finally:
        if runner is not None:
            await runner.close()


def get_workflow_runner(request: Request) -> WorkflowRunner | None:
    runner: Final[object] = getattr(request.state, "workflow_runner", None)
    return runner if isinstance(runner, WorkflowRunner) else None
