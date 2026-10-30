from collections.abc import Callable
from typing import Final

from litellm.accounting import CostEvent, DeliveryTarget, OutboundUsageEvent, PricingSnapshot, Unpriced
from litellm.proxy.db.accounting_ledger import AccountingLedger


class LoggingCostCompiler:
    def __init__(
        self,
        ledger: AccountingLedger,
        price_usage: Callable[[str], PricingSnapshot | Unpriced],
        delivery_targets: tuple[DeliveryTarget, ...],
    ) -> None:
        self._ledger: Final = ledger
        self._price_usage: Final = price_usage
        self._delivery_targets: Final = delivery_targets

    async def compile(self, event: OutboundUsageEvent) -> CostEvent | Unpriced:
        return await self._ledger.compile_cost(event, self._price_usage, self._delivery_targets)
