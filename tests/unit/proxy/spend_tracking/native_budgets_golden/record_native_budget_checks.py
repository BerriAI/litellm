from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Final

import pytest

import litellm
from litellm.proxy import proxy_server
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from tests.unit.proxy.spend_tracking.test_native_budget_checks import CASES, GOLDEN, _outcome, outcome_record


def main() -> None:
    commit: Final = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
    ).stdout.strip()
    with pytest.MonkeyPatch.context() as patch:
        store: Final = SpendCounterStore(connect(None).spend_counters())
        patch.setattr(proxy_server, "spend_counter_store", store)
        patch.setattr(proxy_server, "prisma_client", None)
        patch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.25)
        cases: Final = {
            name: outcome_record(asyncio.run(_outcome(store, case))) for name, case in CASES.items()
        }
    GOLDEN.write_text(json.dumps({"recorded_from": commit, "python": cases}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
