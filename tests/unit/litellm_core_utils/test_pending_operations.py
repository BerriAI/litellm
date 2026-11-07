import asyncio
from threading import BoundedSemaphore
from typing import Final

import pytest

from litellm.litellm_core_utils.pending_operations import (
    defer_until_operations_finish,
    finish_pending_operation,
    register_pending_operation,
)


@pytest.mark.asyncio
async def test_late_child_cannot_release_a_drained_parent_scope_twice() -> None:
    admission: Final = BoundedSemaphore(1)
    assert admission.acquire(blocking=False)
    proceed: Final = asyncio.Event()

    async def late_child() -> None:
        await proceed.wait()
        operation: Final = register_pending_operation()
        assert operation is None
        finish_pending_operation(operation)

    with defer_until_operations_finish(admission.release):
        child: Final = asyncio.create_task(late_child())
    assert admission.acquire(blocking=False)
    proceed.set()
    await child
    assert not admission.acquire(blocking=False)
    admission.release()
