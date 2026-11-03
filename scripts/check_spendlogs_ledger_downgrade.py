import argparse
import asyncio
from typing import Final

from prisma import Prisma

from litellm.repositories.accounting_delivery_repository import AccountingDeliveryRepository


async def check_downgrade(*, writers_disabled: bool, buffers_drained: bool) -> int:
    if not writers_disabled or not buffers_drained:
        print("Blocked: confirm the flag is off on every writer and all ledger buffers are drained")
        return 1
    client: Final = Prisma()
    await client.connect()
    try:
        metrics: Final = await AccountingDeliveryRepository(client).metrics()
        if metrics.pending_count or metrics.parked_count:
            print(f"Blocked: {metrics.pending_count} pending and {metrics.parked_count} parked deliveries")
            return 1
        print(
            "Database gate passed: no pending or parked deliveries; writer and buffer checks are operator attestations"
        )
        return 0
    finally:
        await client.disconnect()


def main() -> int:
    parser: Final = argparse.ArgumentParser(
        description="Check prerequisites before downgrading below the ledger writer"
    )
    parser.add_argument("--all-writers-ledger-disabled", action="store_true")
    parser.add_argument("--ledger-buffers-drained", action="store_true")
    arguments: Final = parser.parse_args()
    return asyncio.run(
        check_downgrade(
            writers_disabled=arguments.all_writers_ledger_disabled,
            buffers_drained=arguments.ledger_buffers_drained,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
