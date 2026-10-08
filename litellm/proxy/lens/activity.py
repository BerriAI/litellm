import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final

from .analysis import ModelCall, ReportProgress
from .models import Activity, ActivityOperation, ActivityPhase, ModelRequest, ModelResult, ToolCount


class ActivityTracker:
    def __init__(self, activity: Activity, progress: ReportProgress | None) -> None:
        self.activity: Activity = activity
        self.progress: Final = progress
        self.lock: Final = asyncio.Lock()

    async def publish(self) -> None:
        if self.progress is not None:
            await self.progress(None, None, None, None, self.activity)

    async def change(self, operation: ActivityOperation, started: bool) -> None:
        async with self.lock:
            current: Final = self.activity
            operations: Final = (
                (*current.operations, operation)
                if started
                else current.operations[: current.operations.index(operation)]
                + current.operations[current.operations.index(operation) + 1 :]
            )
            previous: Final = next((tool.calls for tool in current.tool_calls if tool.name == operation), 0)
            counts: Final = (
                tuple(tool for tool in current.tool_calls if tool.name != operation)
                + (ToolCount(name=operation, calls=previous + 1),)
                if started and operation != "model"
                else current.tool_calls
            )
            self.activity = current.model_copy(
                update=MappingProxyType({"operations": operations, "tool_calls": counts})
            )
            await self.publish()


@asynccontextmanager
async def track_activity(
    progress: ReportProgress | None,
    *,
    identity: str,
    phase: ActivityPhase,
    label: str,
    execution_ids: tuple[str, ...],
) -> AsyncGenerator[ActivityTracker]:
    tracker: Final = ActivityTracker(
        Activity(
            id=identity,
            phase=phase,
            label=label,
            execution_ids=execution_ids,
            started_at=datetime.now(timezone.utc),
        ),
        progress,
    )
    try:
        await tracker.publish()
        yield tracker
    finally:
        tracker.activity = tracker.activity.model_copy(update=MappingProxyType({"operations": (), "finished": True}))
        await tracker.publish()


@asynccontextmanager
async def observe_operation(
    tracker: ActivityTracker | None, operation: ActivityOperation | None
) -> AsyncGenerator[None]:
    if tracker is None or operation is None:
        yield
        return
    await tracker.change(operation, True)
    try:
        yield
    finally:
        await tracker.change(operation, False)


def observed_model(model: ModelCall, tracker: ActivityTracker | None) -> ModelCall:
    if tracker is None:
        return model

    async def call(request: ModelRequest) -> ModelResult:
        async with observe_operation(tracker, "model"):
            return await model(request)

    return call
