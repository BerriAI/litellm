import asyncio
from contextvars import Context
from typing import Any, TypeVar

_T = TypeVar("_T")

# Create a global constant to store background tasks
# See: https://docs.python.org/3/library/asyncio-task.html#create_background_task
_background_tasks: set[asyncio.Task[Any]] = set()


def create_background_task(coro, *, name: str | None = None, context: Context | None = None):
    """Create background task, keeping a strong reference to it until it's done."""

    task = asyncio.create_task(coro, name=name, context=context)

    # Add task to the background tasks set to create a strong reference
    _background_tasks.add(task)

    # Make the task remove its own reference from the set after completion
    task.add_done_callback(_background_tasks.discard)
