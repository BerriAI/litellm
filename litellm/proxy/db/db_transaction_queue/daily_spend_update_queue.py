import asyncio
from collections.abc import Coroutine
from copy import deepcopy
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_ASYNCIO_QUEUE_MAXSIZE
from litellm.proxy._types import BaseDailySpendTransaction
from litellm.proxy.db.db_transaction_queue.base_update_queue import (
    BaseUpdateQueue,
    service_logger_obj,
)
from litellm.types.services import ServiceTypes


class DailySpendUpdateQueue(BaseUpdateQueue):
    """
    In memory buffer for daily spend updates that should be committed to the database

    To add a new daily spend update transaction, use the following format:
        daily_spend_update_queue.add_update({
            "user1_date_api_key_model_custom_llm_provider": {
                "spend": 10,
                "prompt_tokens": 100,
                "completion_tokens": 100,
            }
        })

    Queue contains a list of daily spend update transactions

    eg
        queue = [
            {
                "user1_date_api_key_model_custom_llm_provider": {
                    "spend": 10,
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                    "api_requests": 100,
                    "successful_requests": 100,
                    "failed_requests": 100,
                }
            },
            {
                "user2_date_api_key_model_custom_llm_provider": {
                    "spend": 10,
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                    "api_requests": 100,
                    "successful_requests": 100,
                    "failed_requests": 100,
                }
            }
        ]
    """

    def __init__(self):
        super().__init__()
        self.update_queue: asyncio.Queue[dict[str, BaseDailySpendTransaction]] = asyncio.Queue(
            maxsize=LITELLM_ASYNCIO_QUEUE_MAXSIZE
        )
        self.interrupted_commits: set[asyncio.Task[None]] = (
            set()
        )  # mutable-ok: registry of in-flight commit outcomes, entries leave via their done callback

    def track_interrupted_commit(self, settle: Coroutine[object, object, None]) -> None:
        task: Final = asyncio.ensure_future(settle)
        self.interrupted_commits.add(task)
        task.add_done_callback(self.interrupted_commits.discard)

    async def settle_interrupted_commits(self) -> None:
        while self.interrupted_commits:
            await asyncio.wait(tuple(self.interrupted_commits))

    async def add_update(self, update: dict[str, BaseDailySpendTransaction]):
        """Enqueue an update."""
        verbose_proxy_logger.debug("Adding update to queue: %s", update)
        await self.update_queue.put(update)
        if self.update_queue.qsize() >= self.MAX_SIZE_IN_MEMORY_QUEUE:
            verbose_proxy_logger.warning(
                "Spend update queue is full. Aggregating all entries in queue to concatenate entries."
            )
            await self.aggregate_queue_updates()

    async def aggregate_queue_updates(self):
        """
        Combine all updates in the queue into a single update.
        This is used to reduce the size of the in-memory queue.
        """
        updates: Final[list[dict[str, BaseDailySpendTransaction]]] = await self.flush_all_updates_from_in_memory_queue()
        aggregated_updates: Final = self.get_aggregated_daily_spend_update_transactions(updates)
        await self.update_queue.put(aggregated_updates)

    async def flush_and_get_aggregated_daily_spend_update_transactions(
        self,
    ) -> dict[str, BaseDailySpendTransaction]:
        """Get all updates from the queue and return all updates aggregated by daily_transaction_key. Works for both user and team spend updates."""
        await self.settle_interrupted_commits()
        updates: Final = await self.flush_all_updates_from_in_memory_queue()
        if len(updates) > 0:
            verbose_proxy_logger.info(
                "Spend tracking - flushed %d daily spend update items from in-memory queue",
                len(updates),
            )
        aggregated_daily_spend_update_transactions: Final = (
            DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(updates)
        )
        verbose_proxy_logger.debug(
            "Aggregated daily spend update transactions: %s",
            aggregated_daily_spend_update_transactions,
        )
        return aggregated_daily_spend_update_transactions

    @staticmethod
    def get_aggregated_daily_spend_update_transactions(
        updates: list[dict[str, BaseDailySpendTransaction]],
    ) -> dict[str, BaseDailySpendTransaction]:
        """Aggregate updates by daily_transaction_key."""
        aggregated_daily_spend_update_transactions: Final[dict[str, BaseDailySpendTransaction]] = {}
        for _update in updates:
            for _key, payload in _update.items():
                if _key in aggregated_daily_spend_update_transactions:
                    daily_transaction = aggregated_daily_spend_update_transactions[_key]
                    daily_transaction["spend"] += payload["spend"]
                    daily_transaction["prompt_tokens"] += payload["prompt_tokens"]
                    daily_transaction["completion_tokens"] += payload["completion_tokens"]
                    daily_transaction["api_requests"] += payload["api_requests"]
                    daily_transaction["successful_requests"] += payload["successful_requests"]
                    daily_transaction["failed_requests"] += payload["failed_requests"]

                    # Add optional metrics cache_read_input_tokens and cache_creation_input_tokens
                    daily_transaction["cache_read_input_tokens"] = (
                        payload.get("cache_read_input_tokens", 0) or 0
                    ) + daily_transaction.get("cache_read_input_tokens", 0)

                    daily_transaction["cache_creation_input_tokens"] = (
                        payload.get("cache_creation_input_tokens", 0) or 0
                    ) + daily_transaction.get("cache_creation_input_tokens", 0)

                    daily_transaction["compression_saved_tokens"] = (
                        payload.get("compression_saved_tokens", 0) or 0
                    ) + daily_transaction.get("compression_saved_tokens", 0)

                    daily_transaction["compression_savings_spend"] = (
                        payload.get("compression_savings_spend", 0) or 0
                    ) + daily_transaction.get("compression_savings_spend", 0)

                    daily_transaction["prompt_caching_savings_spend"] = (
                        payload.get("prompt_caching_savings_spend", 0) or 0
                    ) + daily_transaction.get("prompt_caching_savings_spend", 0)

                    daily_transaction["gateway_injected_caching_savings_spend"] = (
                        payload.get("gateway_injected_caching_savings_spend", 0) or 0
                    ) + daily_transaction.get("gateway_injected_caching_savings_spend", 0)

                    daily_transaction["autorouter_savings_spend"] = (
                        payload.get("autorouter_savings_spend", 0) or 0
                    ) + daily_transaction.get("autorouter_savings_spend", 0)

                    daily_transaction["total_response_time_ms"] = (
                        payload.get("total_response_time_ms", 0) or 0
                    ) + daily_transaction.get("total_response_time_ms", 0)

                    daily_transaction["timed_requests"] = (
                        payload.get("timed_requests", 0) or 0
                    ) + daily_transaction.get("timed_requests", 0)
                    aggregated_daily_spend_update_transactions[_key] = {
                        **daily_transaction,
                        "autorouter_accounted_requests": daily_transaction.get("autorouter_accounted_requests", 0)
                        + payload.get("autorouter_accounted_requests", 0),
                        "autorouter_requests": daily_transaction.get("autorouter_requests", 0)
                        + payload.get("autorouter_requests", 0),
                        "autorouter_llm_spend": daily_transaction.get("autorouter_llm_spend", 0.0)
                        + payload.get("autorouter_llm_spend", 0.0),
                        "autorouter_classifier_cost": daily_transaction.get("autorouter_classifier_cost", 0.0)
                        + payload.get("autorouter_classifier_cost", 0.0),
                        "autorouter_classifier_cost_recorded_requests": daily_transaction.get(
                            "autorouter_classifier_cost_recorded_requests", 0
                        )
                        + payload.get("autorouter_classifier_cost_recorded_requests", 0),
                        "autorouter_estimated_requests": daily_transaction.get("autorouter_estimated_requests", 0)
                        + payload.get("autorouter_estimated_requests", 0),
                        "autorouter_estimated_actual_spend": daily_transaction.get(
                            "autorouter_estimated_actual_spend", 0.0
                        )
                        + payload.get("autorouter_estimated_actual_spend", 0.0),
                    }

                else:
                    aggregated_daily_spend_update_transactions[_key] = deepcopy(payload)
        return aggregated_daily_spend_update_transactions

    async def _emit_new_item_added_to_queue_event(
        self,
        queue_size: int | None = None,
    ):
        asyncio.create_task(
            service_logger_obj.async_service_success_hook(
                service=ServiceTypes.IN_MEMORY_DAILY_SPEND_UPDATE_QUEUE,
                duration=0,
                call_type="_emit_new_item_added_to_queue_event",
                event_metadata={
                    "gauge_labels": ServiceTypes.IN_MEMORY_DAILY_SPEND_UPDATE_QUEUE,
                    "gauge_value": queue_size,
                },
            )
        )
