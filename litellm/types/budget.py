"""Shared budget types.

Kept outside `litellm.proxy._types` so `litellm_core_utils.duration_parser` can
import it without pulling in the proxy package.
"""

from typing import Final, Literal

BudgetResetAlignment = Literal["calendar", "rolling"]
"""How a `budget_duration` is turned into the next `budget_reset_at`.

"calendar" snaps to a shared boundary (start of day / Monday / 1st of the month),
so every budget on the proxy resets at the same instant. "rolling" measures the
duration from the budget's own anchor, keeping the original time of day.

A `Literal` rather than a `str` Enum because these values are persisted to TEXT
columns: `str(SomeStrEnum.MEMBER)` renders as "SomeStrEnum.MEMBER" on Python 3.11+,
and `model_dump()` would hand Prisma an enum member instead of the value.
"""

DEFAULT_BUDGET_RESET_ALIGNMENT: Final[BudgetResetAlignment] = "calendar"
