"""The row-lock budget every spend rollup transaction runs under.

A rollup upsert that waits on a row another pod holds keeps its pooled
connection for as long as it waits. ``lock_timeout`` bounds that wait, so a
long holder costs the fleet one aborted statement per waiter instead of a
connection pinned until the chain drains. The aborted statement never applied,
so the caller requeues its rows (SQLSTATE 55P03, see
``PrismaDBExceptionHandler.is_lock_timeout_error``).

``set_config(..., true)`` is ``SET LOCAL``: it lasts for the enclosing
transaction only, and takes the value as a bind parameter, which ``SET``
cannot. Issued as the first statement of every rollup transaction.
"""

from typing import Final, Protocol

from typing_extensions import LiteralString

from litellm.constants import SPEND_ROLLUP_LOCK_TIMEOUT_MS

ROLLUP_LOCK_TIMEOUT_SQL: Final = "SELECT set_config('lock_timeout', $1::text, true)"


def rollup_lock_timeout_setting() -> str:
    return f"{SPEND_ROLLUP_LOCK_TIMEOUT_MS}ms"


class _RawExecutor(Protocol):
    async def execute_raw(self, query: LiteralString, *args: object) -> int: ...


async def apply_rollup_lock_timeout(transaction: _RawExecutor) -> None:
    """Bound the row-lock waits of the open ``transaction`` for the rest of its life."""
    _ = await transaction.execute_raw(ROLLUP_LOCK_TIMEOUT_SQL, rollup_lock_timeout_setting())
