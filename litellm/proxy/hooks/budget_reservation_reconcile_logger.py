"""Reconcile in-memory budget reservations and unreserved proxy spend when no database logger is registered.

See https://github.com/BerriAI/litellm/issues/45895.
"""

import math
from collections.abc import Buffer, Mapping, Sequence
from datetime import datetime
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import BACKGROUND_INTERACTION_COST_POLLING_ENABLED
from litellm.integrations.custom_logger import CustomLogger
from litellm.interactions.background_cost_polling import is_pollable_background_interaction
from litellm.litellm_core_utils.core_helpers import (
    budget_reservation_from_metadata,
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # shared helper has a legacy untyped signature
)
from litellm.types.interactions import InteractionsAPIResponse


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


_OBJECT_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_OBJECT_LIST_ADAPTER: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])


def _as_object_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return _OBJECT_MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _string_list_or_none(
    value: object,
) -> list[str] | None:  # mutable-ok: increment_spend_counters takes list[str] | None
    if not isinstance(value, list):
        return None
    try:
        items: Final = _OBJECT_LIST_ADAPTER.validate_python(value)
    except ValidationError:
        return None
    tags: Final = [item for item in items if isinstance(item, str) and item]
    return tags or None


class _PROXY_BudgetReservationReconcileLogger(CustomLogger):
    """Settle auth's spend-counter reservation and unreserved proxy turns on success when ProxyDBLogger is absent.

    Named with ``_PROXY_`` so ``Logging.should_run_callback`` still invokes this
    accounting hook when a request sets ``no-log: true``. ``callback_bound=True``
    also skips request-end cleanup, so a suppressed callback would pin a shared
    key's reservation until TTL.

    With ``custom_auth_run_common_checks`` and no ``database_url``, auth still reserves
    spend in the in-memory counter and ``bind_budget_reservation_to_callbacks`` marks
    ``callback_bound=True``. ``ProxyDBLogger`` (the only success path that reconciled)
    is registered only when ``prisma_client is not None``, so the reservation stuck
    until TTL and every later request on that key failed with ``BudgetExceededError``.

    Multi-turn WebSocket ``/v1/responses`` sessions also detach
    ``user_api_key_budget_reservation`` after turn 1 so later turns charge their
    actual cost directly via ``increment_spend_counters``.
    """

    @staticmethod
    def _parse_response_cost(response_cost: object) -> float | None:
        if not isinstance(response_cost, (int, float, str, Buffer)) or isinstance(response_cost, bool):
            return None
        try:
            return float(response_cost)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _should_defer_background_interaction(response_cost: object, response_obj: object) -> bool:
        if response_cost is not None or not BACKGROUND_INTERACTION_COST_POLLING_ENABLED:
            return False
        if not isinstance(response_obj, InteractionsAPIResponse) or response_obj.usage is not None:
            return False
        return bool(is_pollable_background_interaction(response_obj))

    async def _reconcile_reserved_spend(
        self,
        budget_reservation: dict[str, object],  # mutable-ok: reconcile stamps finalized on the shared reservation dict
        actual_cost: float,
    ) -> None:
        from litellm.proxy.spend_tracking.budget_reservation import (
            reconcile_budget_reservation,  # pyright: ignore[reportUnknownVariableType]  # shared reconciler accepts legacy reservation dictionaries
        )

        try:
            await reconcile_budget_reservation(
                budget_reservation=budget_reservation,
                actual_cost=actual_cost,
            )
        except Exception:  # noqa: BLE001  # counter failure: refund only when no known spend
            if actual_cost > 0:
                verbose_proxy_logger.exception(
                    "Failed to reconcile budget reservation without a database logger; "
                    "retaining known spend %s for retry",
                    actual_cost,
                )
                return
            verbose_proxy_logger.exception(
                "Failed to reconcile budget reservation without a database logger; releasing instead"
            )
            from litellm.proxy.spend_tracking.budget_reservation import (
                release_or_invalidate_budget_reservation,  # pyright: ignore[reportUnknownVariableType]  # shared release accepts legacy reservation dictionaries
            )

            await release_or_invalidate_budget_reservation(budget_reservation=budget_reservation)

    async def _charge_unreserved_spend(
        self,
        kwargs: Mapping[str, object],
        metadata: Mapping[str, object],
        sl_object: Mapping[str, object] | None,
        response_cost: float,
        start_time: datetime,
    ) -> None:
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.customer_spend import settle_customer_spend
        from litellm.proxy.db.db_spend_update_writer import debitable_model_access_groups, get_llm_router
        from litellm.proxy.proxy_server import (
            customer_accounting_counters,
            increment_spend_counters,
            spend_counter_store,
        )
        from litellm.proxy.spend_tracking.customer_cost_margin import (
            apply_customer_cost_margin_to_estimate,
            request_customer_cost_margin,
        )
        from litellm.proxy.spend_tracking.spend_tracking_utils import get_request_model_access_groups
        from litellm.proxy.utils import ProxyUpdateSpend
        from litellm.types.proxy.auth.customer_hierarchy import (
            customer_budget_scopes_from_request,
            customer_hierarchy_ids_from_request,
        )
        from litellm.utils import get_end_user_id_for_cost_tracking

        if ProxyUpdateSpend.disable_spend_updates() is True:
            return

        litellm_params: Final[Mapping[str, object]] = _as_object_mapping(kwargs.get("litellm_params")) or {}
        end_user_id: Final = get_end_user_id_for_cost_tracking(litellm_params)
        user_api_key_auth: Final = metadata.get("user_api_key_auth")
        raw_user_api_key: Final = _string_or_none(metadata.get("user_api_key"))
        token: Final[str | None] = (
            user_api_key_auth.token
            if isinstance(user_api_key_auth, UserAPIKeyAuth) and _string_or_none(user_api_key_auth.token) is not None
            else raw_user_api_key
        )
        user_id: Final = _string_or_none(metadata.get("user_api_key_user_id"))
        team_id: Final = _string_or_none(metadata.get("user_api_key_team_id"))
        org_id: Final = _string_or_none(metadata.get("user_api_key_org_id"))
        project_id: Final = _string_or_none(metadata.get("user_api_key_project_id"))
        customer_hierarchy: Final = metadata.get("user_api_key_customer_hierarchy")
        agent_id: Final = _string_or_none(metadata.get("billing_agent_id")) or _string_or_none(metadata.get("agent_id"))
        app_id: Final = _string_or_none(metadata.get("user_api_key_app_id"))

        if not any((token, user_id, team_id, org_id, project_id, end_user_id, customer_hierarchy, agent_id, app_id)):
            return

        tags: Final = _string_list_or_none(
            sl_object.get("request_tags") if sl_object is not None else None
        ) or _string_list_or_none(metadata.get("tags"))
        served_model_id: Final = _string_or_none(sl_object.get("model_id")) if sl_object is not None else None
        model_access_groups: Final[Sequence[str] | None] = debitable_model_access_groups(
            attributed=get_request_model_access_groups(dict(kwargs)),
            served_model_id=served_model_id,
            router=get_llm_router(),
        )

        margin: Final = request_customer_cost_margin(metadata)
        breakdown: Final = sl_object.get("cost_breakdown") if sl_object is not None else None
        already_margined: Final = (sl_object is not None and bool(sl_object.get("cost_finalized", False))) or (
            isinstance(breakdown, Mapping) and "customer_margin_amount" in breakdown
        )
        charged_cost: Final = (
            float(apply_customer_cost_margin_to_estimate(response_cost, margin) or response_cost)
            if margin is not None and not already_margined
            else response_cost
        )

        try:
            customer_counters: Final = (
                await customer_accounting_counters(
                    customer_hierarchy_ids_from_request(kwargs),
                    customer_budget_scopes_from_request(kwargs),
                    started_at=start_time,
                )
                if customer_hierarchy is not None
                else ()
            )
            if customer_hierarchy is not None:
                await settle_customer_spend(metadata, charged_cost, spend_counter_store)
            await increment_spend_counters(
                token=token,
                team_id=team_id,
                user_id=user_id,
                response_cost=charged_cost,
                org_id=org_id,
                budget_reservation=None,
                end_user_id=end_user_id,
                tags=tags,
                request_started_at=start_time,
                model_access_groups=model_access_groups,
                project_id=project_id,
                customer_counters=customer_counters,
            )
        except Exception:  # noqa: BLE001  # counter failure in callback must not crash caller
            verbose_proxy_logger.exception(
                "Failed to increment unreserved spend counters without a database logger for cost %s",
                charged_cost,
            )

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        metadata: Final = _as_object_mapping(
            get_litellm_metadata_from_kwargs(kwargs=dict(kwargs))  # pyright: ignore[reportUnknownArgumentType]  # shared helper accepts untyped dict
        )
        if metadata is None:
            return
        budget_reservation: Final = budget_reservation_from_metadata(metadata)
        sl_object: Final = _as_object_mapping(kwargs.get("standard_logging_object"))
        response_cost: Final[object] = (
            sl_object.get("response_cost", kwargs.get("response_cost"))
            if sl_object is not None
            else kwargs.get("response_cost")
        )

        parsed_cost: Final = self._parse_response_cost(response_cost)

        if isinstance(budget_reservation, dict):
            if budget_reservation.get("finalized") is True:
                return
            if self._should_defer_background_interaction(response_cost=response_cost, response_obj=response_obj):
                return
            actual_cost: Final = parsed_cost if parsed_cost is not None else 0.0
            await self._reconcile_reserved_spend(
                budget_reservation=budget_reservation,
                actual_cost=actual_cost,
            )
            return

        if parsed_cost is None or not math.isfinite(parsed_cost) or parsed_cost <= 0:
            return
        await self._charge_unreserved_spend(
            kwargs=kwargs,
            metadata=metadata,
            sl_object=sl_object,
            response_cost=parsed_cost,
            start_time=start_time,
        )


BudgetReservationReconcileLogger: Final = _PROXY_BudgetReservationReconcileLogger
