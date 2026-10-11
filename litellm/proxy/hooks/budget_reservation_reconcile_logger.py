"""Reconcile in-memory budget reservations when no database logger is registered.

See https://github.com/BerriAI/litellm/issues/45895.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    budget_reservation_from_metadata,
    get_litellm_metadata_from_kwargs,
)

class BudgetReservationReconcileLogger(CustomLogger):
    """Settle auth's spend-counter reservation on success when ProxyDBLogger is absent.

    With ``custom_auth_run_common_checks`` and no ``database_url``, auth still reserves
    spend in the in-memory counter and ``bind_budget_reservation_to_callbacks`` marks
    ``callback_bound=True``. ``ProxyDBLogger`` (the only success path that reconciled)
    is registered only when ``prisma_client is not None``, so the reservation stuck
    until TTL and every later request on that key failed with ``BudgetExceededError``.
    """

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        del response_obj, start_time, end_time
        metadata: Final = get_litellm_metadata_from_kwargs(kwargs=dict(kwargs))
        if not isinstance(metadata, Mapping):
            return
        budget_reservation: Final = budget_reservation_from_metadata(metadata)
        if not isinstance(budget_reservation, dict) or budget_reservation.get("finalized") is True:
            return

        sl_object = kwargs.get("standard_logging_object")
        response_cost: object
        if isinstance(sl_object, Mapping):
            response_cost = sl_object.get("response_cost")
        else:
            response_cost = kwargs.get("response_cost")
        actual_cost: Final = float(response_cost or 0.0)

        from litellm.proxy.spend_tracking.budget_reservation import reconcile_budget_reservation

        try:
            await reconcile_budget_reservation(
                budget_reservation=budget_reservation,
                actual_cost=actual_cost,
            )
        except Exception:
            verbose_proxy_logger.exception(
                "Failed to reconcile budget reservation without a database logger; releasing instead"
            )
            from litellm.proxy.spend_tracking.budget_reservation import (
                release_or_invalidate_budget_reservation,
            )

            await release_or_invalidate_budget_reservation(budget_reservation=budget_reservation)
