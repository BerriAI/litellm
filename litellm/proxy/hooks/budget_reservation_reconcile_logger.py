"""Reconcile in-memory budget reservations when no database logger is registered.

See https://github.com/BerriAI/litellm/issues/45895.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import BACKGROUND_INTERACTION_COST_POLLING_ENABLED
from litellm.integrations.custom_logger import CustomLogger
from litellm.interactions.background_cost_polling import is_pollable_background_interaction
from litellm.litellm_core_utils.core_helpers import (
    budget_reservation_from_metadata,
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # shared helper has a legacy untyped signature
)
from litellm.types.interactions import InteractionsAPIResponse


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
        metadata: Final = get_litellm_metadata_from_kwargs(kwargs=dict(kwargs))  # pyright: ignore[reportUnknownVariableType]  # shared helper returns untyped metadata
        if not isinstance(metadata, Mapping):
            return
        budget_reservation: Final = budget_reservation_from_metadata(metadata)  # pyright: ignore[reportUnknownArgumentType]  # helper only reads known keys from the checked mapping
        if not isinstance(budget_reservation, dict) or budget_reservation.get("finalized") is True:
            return

        sl_object: Final = kwargs.get("standard_logging_object")
        response_cost: Final[object] = (  # pyright: ignore[reportUnknownVariableType]  # callback payloads are legacy untyped mappings
            sl_object.get("response_cost")  # pyright: ignore[reportUnknownMemberType]  # float below validates the untyped cost value
            if isinstance(sl_object, Mapping)
            else kwargs.get("response_cost")
        )
        if (
            response_cost is None
            and BACKGROUND_INTERACTION_COST_POLLING_ENABLED
            and isinstance(response_obj, InteractionsAPIResponse)
            and response_obj.usage is None
            and is_pollable_background_interaction(response_obj)
        ):
            return
        actual_cost: Final = float(response_cost or 0.0)  # pyright: ignore[reportArgumentType]  # float validates legacy callback values at runtime

        from litellm.proxy.spend_tracking.budget_reservation import (
            reconcile_budget_reservation,  # pyright: ignore[reportUnknownVariableType]  # shared reconciler accepts legacy reservation dictionaries
        )

        try:
            await reconcile_budget_reservation(
                budget_reservation=budget_reservation,
                actual_cost=actual_cost,
            )
        except Exception:  # noqa: BLE001  # a counter backend failure must still release the reservation
            verbose_proxy_logger.exception(
                "Failed to reconcile budget reservation without a database logger; releasing instead"
            )
            from litellm.proxy.spend_tracking.budget_reservation import (
                release_or_invalidate_budget_reservation,  # pyright: ignore[reportUnknownVariableType]  # shared release accepts legacy reservation dictionaries
            )

            await release_or_invalidate_budget_reservation(budget_reservation=budget_reservation)
