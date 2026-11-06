from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from litellm.proxy.deps.repositories import (
    get_workflow_definition_repository,
    get_workflow_execution_repository,
)
from litellm.proxy.workflows.gateway import execute_chat_node
from litellm.proxy.workflows.runner import WorkflowRunner
from litellm.repositories.database import Database


@asynccontextmanager
async def workflow_lifespan(database: Database | None) -> AsyncGenerator[None]:
    if database is None:
        yield
        return
    definitions: Final = await get_workflow_definition_repository(database)
    executions: Final = await get_workflow_execution_repository(database)
    if definitions is None or executions is None:
        yield
        return
    runner: Final = WorkflowRunner(executions, definitions, execute_chat_node)
    await runner.start()
    try:
        yield
    finally:
        await runner.close()
